"""Feature registry, tuning config, and the F-generic CPU column builder.

The production TDEW anomaly model fits a **fixed** feature set
``[const, TMIN_anom, TD_anom_lag1, TD_anom_lag2, TMIN_anom_lag1]`` with a fixed DOY
half-window ``h=11`` (see ``tdew_estimation.anomaly_train`` and ``HPC_code.gpu_train``).
This module makes the feature set *declarative* so tuning can search over a candidate
pool without re-preparing data.

A feature column name is one of:
  * ``"const"``                 -> the intercept column (all ones).
  * ``"<VAR>_anom"``            -> the contemporaneous anomaly ``VAR - VAR_clim``.
  * ``"<VAR>_anom_lag<k>"``     -> ``groupby(ID)[<VAR>_anom].shift(k)`` (k >= 1).

``VAR`` is a base variable present in the bucketed training shards. Only ``TMIN`` and
``TD`` are available today (shards carry ``[ID, FECHA, TD, TMIN, doy]``); ``TD_anom`` is
always the regression **target**, so ``TD`` enters the feature pool only through its lags.
TMAX/PREC candidates are deferred (they need a shard-builder change — plan P7).

Both the feature recipe **and** the DOY half-window ``h`` are selected per SENAMHI
climate **zone x doy** (the zones come from ``zones.py``). There is no "season" bucketing.

The canonical fixed-5 recipe is exposed as :data:`CANONICAL_FEATURES` so the F-generic
path can be checked byte-for-byte against ``HPC_code.gpu_train`` (plan P1 parity test).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd

from tdew_estimation.anomaly_train import AnomalyTrainingConfig

# The production fixed feature set, in the exact column order gpu_train emits.
CANONICAL_FEATURES: Tuple[str, ...] = (
    "const",
    "TMIN_anom",
    "TD_anom_lag1",
    "TD_anom_lag2",
    "TMIN_anom_lag1",
)

# Plan-locked candidate pool (no data re-prep needed): TMIN/TD anoms + lags only.
DEFAULT_CANDIDATE_POOL: Tuple[str, ...] = (
    "const",
    "TMIN_anom",
    "TMIN_anom_lag1",
    "TMIN_anom_lag2",
    "TMIN_anom_lag7",
    "TMIN_anom_lag30",
    "TD_anom_lag1",
    "TD_anom_lag2",
    "TD_anom_lag3",
    "TD_anom_lag7",
    "TD_anom_lag30",
)

TARGET_COL = "TD_anom"  # the regressand (never a feature)

_LAG_RE = re.compile(r"^(?P<var>[A-Za-z][A-Za-z0-9]*)_anom_lag(?P<k>\d+)$")
_ANOM_RE = re.compile(r"^(?P<var>[A-Za-z][A-Za-z0-9]*)_anom$")


# ---------------------------------------------------------------------------------------
# Feature name parsing
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """Parsed feature descriptor. ``kind`` is 'const' | 'anom' | 'lag'."""

    name: str
    kind: str
    var: str | None = None
    lag: int = 0


def parse_feature(name: str) -> FeatureSpec:
    """Parse a feature column name into a :class:`FeatureSpec`."""
    if name == "const":
        return FeatureSpec(name, "const")
    m = _LAG_RE.match(name)
    if m:
        return FeatureSpec(name, "lag", m.group("var"), int(m.group("k")))
    m = _ANOM_RE.match(name)
    if m:
        return FeatureSpec(name, "anom", m.group("var"), 0)
    raise ValueError(
        f"unrecognised feature {name!r}; expected 'const', '<VAR>_anom', "
        f"or '<VAR>_anom_lag<k>'"
    )


class FeatureRegistry:
    """A resolved, ordered feature set. ``const`` is always index 0."""

    def __init__(self, feature_cols: Sequence[str]):
        cols = list(feature_cols)
        if not cols or cols[0] != "const":
            raise ValueError("feature_cols must start with 'const'")
        if len(set(cols)) != len(cols):
            raise ValueError(f"duplicate feature columns: {cols}")
        self.feature_cols: Tuple[str, ...] = tuple(cols)
        self.specs: Tuple[FeatureSpec, ...] = tuple(parse_feature(c) for c in cols)

    def __len__(self) -> int:
        return len(self.feature_cols)

    def index(self, name: str) -> int:
        return self.feature_cols.index(name)

    def base_vars(self) -> List[str]:
        """Base variables whose anomaly series are needed (always includes TD target)."""
        vs = {s.var for s in self.specs if s.kind != "const" and s.var is not None}
        vs.add("TD")  # target TD_anom
        return sorted(vs)

    def max_lag(self) -> int:
        return max((s.lag for s in self.specs), default=0)


# ---------------------------------------------------------------------------------------
# Tuning configuration
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TuningConfig:
    """Configuration for the feature/h tuning search.

    Parameters
    ----------
    base:
        The underlying :class:`AnomalyTrainingConfig` (supplies ``kernel``, ``min_samples``,
        ``train_year_range`` and the default ``h``).
    candidate_pool:
        Ordered feature superset assembled once per zone; backward-stepwise may drop any
        non-``const`` member. Must start with ``const``.
    h_grid:
        DOY half-windows to search. Both ``h`` and the feature set are chosen per
        ``zone x doy`` (or per ``zone`` when ``granularity == "zone"``).
    tol:
        Backward-stepwise stop threshold: keep dropping while the best candidate drop
        improves LOYOCV cosine skill by at least ``tol``.
    granularity:
        Recipe granularity: ``"doy"`` (per zone x doy) or ``"zone"`` (one recipe + one h
        per zone, shared across all doys).
    per_zone_n:
        IDs sampled per zone for the selection search.
    id_chunk:
        Number of IDs assembled/convolved on the GPU at once during selection. Bounds
        peak device memory ``O(id_chunk * n_years * 366 * F^2)`` (see README). Cosine
        segment-sums are additive across ID chunks, so chunking is exact.
    seed:
        RNG seed for the stratified sample.
    """

    base: AnomalyTrainingConfig
    candidate_pool: Tuple[str, ...] = DEFAULT_CANDIDATE_POOL
    h_grid: Tuple[int, ...] = (7, 11, 15, 21)
    tol: float = 0.01
    granularity: str = "doy"
    per_zone_n: int = 2000
    id_chunk: int = 96
    seed: int = 0

    def registry(self) -> FeatureRegistry:
        return FeatureRegistry(self.candidate_pool)


# ---------------------------------------------------------------------------------------
# F-generic CPU feature engineering
# ---------------------------------------------------------------------------------------
def build_feature_frame(
    train_df: pd.DataFrame,
    clim_df: pd.DataFrame,
    registry: FeatureRegistry,
) -> pd.DataFrame:
    """Prepare a per-sample feature frame for a set of IDs (F-generic).

    Replicates the CPU/GPU prep exactly (merge climatology on ``(ID, doy)``, sort each ID
    series by ``FECHA``, form anomalies and global lags), then keeps only rows where the
    target and every requested feature is finite (the per-doy ``dropna``). Adds a ``year``
    column for the LOYOCV year axis.

    Returns a frame with columns ``[ID, doy, year, TD_anom, <feature_cols...>]`` (the
    ``const`` feature is materialised implicitly as ones at matrix-build time, not here).
    """
    df = train_df.copy()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ID"]).copy()
    df["ID"] = df["ID"].astype(int)
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["doy"] = pd.to_numeric(df["doy"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["doy"]).copy()
    df["doy"] = df["doy"].astype(int)

    clim = clim_df.copy()
    clim["ID"] = pd.to_numeric(clim["ID"], errors="coerce").astype("Int64")
    clim = clim.dropna(subset=["ID"]).copy()
    clim["ID"] = clim["ID"].astype(int)
    clim["doy"] = pd.to_numeric(clim["doy"], errors="coerce").astype("Int64")
    clim = clim.dropna(subset=["doy"]).copy()
    clim["doy"] = clim["doy"].astype(int)

    base_vars = registry.base_vars()
    clim_cols = [f"{v}_clim" for v in base_vars]
    df = df.merge(clim[["ID", "doy", *clim_cols]], on=["ID", "doy"], how="left")
    df = df.sort_values(["ID", "FECHA"]).reset_index(drop=True)

    # Contemporaneous anomalies for every base var (TD_anom is the target).
    for v in base_vars:
        df[f"{v}_anom"] = df[v] - df[f"{v}_clim"]

    # Declarative lags: shift within each ID series (matches gpu_train global shift).
    g = df.groupby("ID", sort=False)
    for spec in registry.specs:
        if spec.kind == "lag":
            df[spec.name] = g[f"{spec.var}_anom"].shift(spec.lag)

    feature_cols = [c for c in registry.feature_cols if c != "const"]
    keep_mask = df[[TARGET_COL, *feature_cols]].notna().all(axis=1)
    df = df.loc[keep_mask].copy()
    df["year"] = df["FECHA"].dt.year.astype(int)

    return df[["ID", "doy", "year", TARGET_COL, *feature_cols]].reset_index(drop=True)
