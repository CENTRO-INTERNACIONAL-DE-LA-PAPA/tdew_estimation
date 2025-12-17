"""
tdew_estimation.anomaly_train

Anomaly-model training utilities for TD (dewpoint temperature) estimation.

This module is extracted/refactored from the Colab-derived script
`tdew_estimation_arimax.py` and is designed to be:

- Path-agnostic: no hard-coded filesystem paths.
- Rerun-friendly: supports training only a subset of DOYs (days of year),
  which enables repairing incomplete DOYs discovered by checks.

Scope (for now)
---------------
This module provides the *core* pieces needed to:
1) build anomalies (TD_anom, TMIN_anom) by merging with daily climatology
2) build lag features for anomalies
3) fit a weighted least squares (WLS) regression per DOY (local regression over a DOY neighborhood)
4) optionally restrict training to a list/set of DOYs, for fast reruns

It intentionally does NOT implement a CLI. The package-level `main.py` (example)
should orchestrate:
climatology -> train anomaly coeffs -> checks -> rerun doys -> patch.

Type-checking notes
-------------------
This project may be checked with strict type checking. Some third-party library
stubs (pandas/statsmodels) are not always precise for our dynamic usage (e.g.,
`read_parquet` returning a DataFrame, and statsmodels accepting array-like weights).
To keep the file readable and robust, we use a few small, explicit casts and
lightweight runtime checks.

Inputs expected (parquet)
-------------------------
The original pipeline reads parquet files with columns:
- ID: int-like
- FECHA: datetime-like
- Value: float-like

Variables:
- td      (dewpoint temperature) -> becomes TD
- tmin_v1 (minimum temperature)  -> becomes TMIN

The file naming convention in the original environment was:
  {base_path}/{variable}/Outputs/{variable}_daily_YYYY_MM.parquet
and for tmin_v1 sometimes:
  tmin_daily_YYYY_MM.parquet

This module keeps the naming logic configurable through `find_parquet_files()`.

Outputs
-------
A coefficient table with one row per (ID, doy), containing:
- ID
- doy
- const_anom
- TMIN_anom_coeff
- TD_anom_lag1
- TD_anom_lag2
- TMIN_anom_lag1
- r_squared_anom

(Columns match the original refactor naming from the notebook.)

Performance notes
-----------------
- This implementation is faithful to the original logic, not "optimal".
- It is intended to support correctness and reruns first.
- Improvements (I/O patterns, batching, distributed strategy) can be discussed later.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

PathLike = Union[str, Path]


# --------------------------------------------------------------------------------------
# Configuration dataclasses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyTrainingConfig:
    """
    Configuration for anomaly coefficient training.

    Parameters
    ----------
    base_path:
        Root path containing variable folders and Outputs.
    td_var:
        Variable name for dewpoint parquet folder (default 'td').
    tmin_var:
        Variable name for tmin parquet folder (default 'tmin_v1').
    train_year_range:
        Inclusive year range (start_year, end_year) used for training.
        This is used to select parquet months.
    h:
        Neighborhood half-width in DOY units. The DOY window is computed with circular wrap.
    kernel:
        Weighting kernel. Supported: "Tricube" or "Gaussian".
    min_samples:
        Minimum number of neighborhood samples required to fit a DOY regression.
        (Original notebook used 15.)
    """

    base_path: Path
    td_var: str = "td"
    tmin_var: str = "tmin_v1"
    train_year_range: Tuple[int, int] = (1981, 2016)
    h: int = 11
    kernel: str = "Tricube"
    min_samples: int = 15


# --------------------------------------------------------------------------------------
# File discovery / loading
# --------------------------------------------------------------------------------------


def find_parquet_files(
    base_path: PathLike,
    variable: str,
    year_range: Tuple[int, int],
    *,
    outputs_subdir: str = "Outputs",
    tmin_v1_legacy_name: bool = True,
) -> List[Path]:
    """
    Find parquet files for a variable over a year range using the original monthly convention.

    Parameters
    ----------
    base_path:
        Root directory containing per-variable folders.
    variable:
        Variable folder name, e.g. 'td' or 'tmin_v1'.
    year_range:
        Inclusive year range, e.g. (1981, 2016).
    outputs_subdir:
        Subdirectory under each variable folder where parquet files live.
    tmin_v1_legacy_name:
        If True, and variable == 'tmin_v1', also allow `tmin_daily_YYYY_MM.parquet`
        naming from the older notebook.

    Returns
    -------
    List[Path]
        Sorted list of parquet files.
    """
    base = Path(base_path).expanduser().resolve()
    start_year, end_year = year_range
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    months = pd.date_range(start_date, end_date, freq="MS").strftime("%Y_%m").unique()

    files: List[Path] = []
    var_dir = base / variable / outputs_subdir

    for ym in months:
        if variable == "tmin_v1" and tmin_v1_legacy_name:
            # legacy file name
            candidate = var_dir / f"tmin_daily_{ym}.parquet"
            if candidate.exists():
                files.append(candidate)
                continue

        candidate = var_dir / f"{variable}_daily_{ym}.parquet"
        if candidate.exists():
            files.append(candidate)

    return sorted(files)


def _read_parquet_for_id(
    files: Sequence[Path],
    *,
    location_id: int,
    required_cols: Sequence[str],
) -> List[pd.DataFrame]:
    """
    Read parquet files filtered to a single ID. Returns only non-empty frames.

    Notes on typing:
    - pandas.read_parquet type stubs can be overly broad (DataFrame | Series | ndarray | ...).
      We force the result into a DataFrame and then operate only on DataFrame APIs.
    """
    dfs: List[pd.DataFrame] = []
    for p in files:
        df_any: Any
        try:
            df_any = pd.read_parquet(
                p, filters=[("ID", "==", location_id)], columns=list(required_cols)
            )
        except TypeError:
            # Some parquet engines may not support `filters` or `columns` in this combination.
            # Fallback: read all and filter.
            df_any = pd.read_parquet(p)

        # Force to DataFrame (both runtime and static typing)
        # Some stubs for pandas.read_parquet are overly broad; cast to DataFrame explicitly.
        df = df_any if isinstance(df_any, pd.DataFrame) else pd.DataFrame(df_any)
        df = pd.DataFrame(df)

        if df.empty:
            continue

        if "ID" in df.columns:
            df = pd.DataFrame(df[df["ID"] == location_id])

        if df.empty:
            continue

        # Ensure required columns exist before selecting
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            continue

        df = pd.DataFrame(df[list(required_cols)])

        if not df.empty:
            dfs.append(df)

    return dfs


# --------------------------------------------------------------------------------------
# Core anomaly feature engineering + training
# --------------------------------------------------------------------------------------


def _compute_weights(
    doy_series: pd.Series,
    doy_target: int,
    h: int,
    kernel: str,
) -> np.ndarray:
    """
    Compute kernel weights for each sample based on circular DOY distance to target.
    """
    distance = (doy_series - doy_target).abs()
    distance = np.minimum(distance, 366 - distance)

    if kernel.lower().startswith("tri"):
        # Tricube kernel (as in the notebook)
        scaled = np.clip(distance / h, 0, 1)
        w = (1 - np.abs(scaled) ** 3) ** 3
        return np.asarray(w, dtype=float)
    elif kernel.lower().startswith("gau"):
        # Gaussian kernel
        dist = np.asarray(distance, dtype=float)
        w = np.exp(-(dist**2) / (2 * (h**2)))
        return np.asarray(w, dtype=float)
    else:
        raise ValueError(
            f"Unsupported kernel: {kernel!r}. Use 'Tricube' or 'Gaussian'."
        )


def _doy_neighborhood_mask(doy: pd.Series, doy_target: int, h: int) -> pd.Series:
    """
    Boolean mask selecting DOYs in the circular neighborhood around doy_target with half-width h.
    """
    # Wrap-around logic (circular DOY window)
    lower_bound = (doy_target - h - 1) % 366 + 1
    upper_bound = (doy_target + h - 1) % 366 + 1
    if lower_bound < upper_bound:
        return (doy >= lower_bound) & (doy <= upper_bound)
    return (doy >= lower_bound) | (doy <= upper_bound)


def train_anomaly_coeffs_for_one_id(
    location_id: int,
    *,
    config: AnomalyTrainingConfig,
    climatology_df: pd.DataFrame,
    doys: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    """
    Train anomaly coefficients for a single spatial ID.

    Parameters
    ----------
    location_id:
        The ID to train for.
    config:
        Training configuration.
    climatology_df:
        Full climatology dataframe containing at least columns: ID, doy, TD_clim, TMIN_clim.
    doys:
        Optional subset of DOYs to train. If provided, only those DOYs are fitted,
        which is the key feature to support reruns after detecting incomplete DOYs.

    Returns
    -------
    Optional[pd.DataFrame]
        Coefficients dataframe for this ID. Columns include ID, doy, and coefficients.
        Returns None if no models could be trained.
    """
    # Validate doys set
    if doys is not None:
        doys = {int(d) for d in doys if 1 <= int(d) <= 366}
        if not doys:
            return None

    # Filter climatology for this ID once
    clim_id = climatology_df[climatology_df["ID"] == location_id].copy()
    if clim_id.empty:
        return None

    required_cols = ["ID", "FECHA", "Value"]

    # Discover parquet files
    td_files = find_parquet_files(
        config.base_path, config.td_var, config.train_year_range
    )
    tmin_files = find_parquet_files(
        config.base_path, config.tmin_var, config.train_year_range
    )

    if not td_files or not tmin_files:
        return None

    # Load only this ID
    td_list = _read_parquet_for_id(
        td_files, location_id=location_id, required_cols=required_cols
    )
    tmin_list = _read_parquet_for_id(
        tmin_files, location_id=location_id, required_cols=required_cols
    )

    if not td_list or not tmin_list:
        return None

    df_td = pd.concat(td_list, ignore_index=True).rename(columns={"Value": "TD"})
    df_tmin = pd.concat(tmin_list, ignore_index=True).rename(columns={"Value": "TMIN"})

    if df_td.empty or df_tmin.empty:
        return None

    train_df = pd.merge(df_td, df_tmin, on=["FECHA", "ID"], how="inner")
    if train_df.empty:
        return None

    train_df["FECHA"] = pd.to_datetime(train_df["FECHA"])
    train_df["doy"] = train_df["FECHA"].dt.dayofyear

    # Merge climatology to create anomalies
    train_df = pd.merge(train_df, clim_id, on=["ID", "doy"], how="left")
    train_df = train_df.sort_values("FECHA").reset_index(drop=True)

    # Anomaly and lag features
    train_df["TD_anom"] = train_df["TD"] - train_df["TD_clim"]
    train_df["TMIN_anom"] = train_df["TMIN"] - train_df["TMIN_clim"]
    train_df["TD_anom_lag1"] = train_df["TD_anom"].shift(1)
    train_df["TD_anom_lag2"] = train_df["TD_anom"].shift(2)
    train_df["TMIN_anom_lag1"] = train_df["TMIN_anom"].shift(1)

    feature_cols = ["TMIN_anom", "TD_anom_lag1", "TD_anom_lag2", "TMIN_anom_lag1"]

    doys_to_run: Iterable[int] = doys if doys is not None else range(1, 367)

    results: List[Dict[str, Any]] = []
    for doy_target in doys_to_run:
        doy_series = train_df["doy"]
        if isinstance(doy_series, pd.DataFrame):
            # Defensive: shouldn't happen, but keeps static checkers happy.
            doy_series = doy_series.iloc[:, 0]
        mask = _doy_neighborhood_mask(doy_series, int(doy_target), config.h)
        df_nb = train_df.loc[mask].copy()
        df_nb = df_nb.dropna(subset=["TD_anom"] + feature_cols)

        if len(df_nb) < config.min_samples:
            continue

        weights = _compute_weights(
            df_nb["doy"], int(doy_target), config.h, config.kernel
        )

        y = df_nb["TD_anom"]
        X = sm.add_constant(df_nb[feature_cols], has_constant="add")
        X = pd.DataFrame(X)

        try:
            # statsmodels accepts array-like weights at runtime.
            # Some type checkers think WLS expects float weights; ignore that mismatch.
            model = sm.WLS(y, X, weights=weights).fit()  # type: ignore[arg-type]
        except Exception:
            # Fail gracefully at the DOY-level
            continue

        x_cols = list(X.columns)
        row: Dict[str, Any] = {
            col: float(model.params.get(col, np.nan)) for col in x_cols
        }
        row["doy"] = int(doy_target)
        row["r_squared_anom"] = float(getattr(model, "rsquared", np.nan))
        results.append(row)

    if not results:
        return None

    out = pd.DataFrame(results)
    # Match the original naming conventions used later in forecasting code
    out = out.rename(
        columns={
            "const": "const_anom",
            "TMIN_anom": "TMIN_anom_coeff",
        }
    )
    out["ID"] = int(location_id)
    return out


# --------------------------------------------------------------------------------------
# NOTE: Orchestration removed (Dask-only)
# --------------------------------------------------------------------------------------
#
# This module intentionally contains ONLY the core per-ID training logic:
# - AnomalyTrainingConfig
# - train_anomaly_coeffs_for_one_id
#
# Multi-ID orchestration (batching, chunk writes, combination, and DOY reruns)
# is implemented in `tdew_estimation.anomaly_dask`.
