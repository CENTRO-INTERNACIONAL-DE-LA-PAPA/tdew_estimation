"""F-generic GPU assembly of per-(ID, year, doy) sufficient statistics + raw scoring rows.

This is the tunable-feature analogue of ``HPC_code.gpu_train._assemble_day_sums``. Two
generalisations over the production path:

1. **Arbitrary feature count F** (not the hard-coded 5). The Gram tensor is assembled for
   the *full* candidate superset once; every candidate subset is an index-selected
   sub-block (plan trick #1, "assemble-once, subset-by-indexing").
2. **A ``year`` axis** kept *before* the DOY convolution, so leave-one-year-out is a
   subtraction ``A_full - A_{year}`` rather than a re-scan (plan trick #2).

The convolution (:func:`convolve_doy`) is the same circular tricube/gaussian used in
production, generalised to run along the DOY axis of a ``(N, Y, 366, ...)`` tensor. It is
linear, so ``A_full = convolve(S).sum(year) == convolve(S.sum(year))``.

Raw per-sample rows (``X[M, F]``, ``y[M]`` and the ``(id, doy, year)`` index arrays) are
also returned on the GPU so held-out one-step predictions can be gathered without the
recursive forecaster (plan trick #3).

Everything is float64 to match statsmodels / the production GPU trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

import cupy as cp

from .feature_spec import CANONICAL_FEATURES, FeatureRegistry, TARGET_COL, build_feature_frame
# Reuse the production scalar tricube weight so h behaves identically.
from HPC_code.gpu_train import _tricube  # type: ignore  # noqa: E402

DOY_AXIS = 366  # day-of-year bins 0..365 for doy 1..366; circular modulo 366


@dataclass
class AssembledGeneric:
    """Per-(ID, year, doy) day-sums for a set of IDs, plus raw scoring rows (GPU).

    Day-sum tensors carry a leading ID axis, a ``year`` axis and the 366-long DOY axis.
    ``S_xx[N, Y, 366, F, F]``, ``S_xy[N, Y, 366, F]``, ``S_yy[N, Y, 366]``,
    ``cnt[N, Y, 366]``. Feature order matches ``feature_cols`` (``const`` at index 0).
    """

    S_xx: cp.ndarray
    S_xy: cp.ndarray
    S_yy: cp.ndarray
    cnt: cp.ndarray
    id_values: np.ndarray
    year_values: np.ndarray
    feature_cols: Tuple[str, ...]
    # Raw per-sample rows (GPU), aligned; M = number of kept samples.
    X: cp.ndarray          # [M, F]
    y: cp.ndarray          # [M]
    id_idx: cp.ndarray     # [M] int32, into id_values
    doy_idx: cp.ndarray    # [M] int32, 0..365
    year_idx: cp.ndarray   # [M] int32, into year_values

    @property
    def n_id(self) -> int:
        return int(self.id_values.shape[0])

    @property
    def n_year(self) -> int:
        return int(self.year_values.shape[0])

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def assemble_day_sums_generic(
    train_df: pd.DataFrame,
    clim_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = CANONICAL_FEATURES,
    year_values: np.ndarray | None = None,
) -> AssembledGeneric:
    """Assemble day-sums (year, doy axes) + raw rows for the given feature superset.

    ``feature_cols`` defaults to the canonical fixed-5 set so this reproduces the
    production assembly (summed over the year axis) byte-for-byte. Pass an explicit
    ``year_values`` to force a shared year axis across ID chunks (so per-chunk tensors
    line up); otherwise the years present in ``train_df`` are used.
    """
    registry = FeatureRegistry(feature_cols)
    df = build_feature_frame(train_df, clim_df, registry)
    return assemble_from_feature_frame(df, registry, year_values=year_values)


def assemble_from_feature_frame(
    df: pd.DataFrame,
    registry: FeatureRegistry,
    *,
    year_values: np.ndarray | None = None,
) -> AssembledGeneric:
    """Assemble day-sums + raw rows from an already-built feature frame.

    ``df`` must be a frame produced by :func:`feature_spec.build_feature_frame` for the
    same ``registry`` (columns ``[ID, doy, year, TD_anom, <non-const features...>]``).
    Selection builds the frame once per zone and re-assembles per ``h``/round from it, so
    the (disk-bound) climatology merge and lag construction happen only once.
    """
    nf = len(registry)
    non_const = [c for c in registry.feature_cols if c != "const"]

    id_values = np.sort(df["ID"].unique()) if len(df) else np.empty(0, dtype="int64")
    if year_values is None:
        year_values = np.sort(df["year"].unique()) if len(df) else np.empty(0, dtype="int64")
    else:
        year_values = np.asarray(year_values, dtype="int64")

    n_id = int(id_values.shape[0])
    n_year = int(year_values.shape[0])

    if n_id == 0 or n_year == 0:
        z = cp.zeros
        return AssembledGeneric(
            S_xx=z((0, n_year, DOY_AXIS, nf, nf)),
            S_xy=z((0, n_year, DOY_AXIS, nf)),
            S_yy=z((0, n_year, DOY_AXIS)),
            cnt=z((0, n_year, DOY_AXIS)),
            id_values=id_values,
            year_values=year_values,
            feature_cols=registry.feature_cols,
            X=z((0, nf)),
            y=z((0,)),
            id_idx=cp.zeros((0,), dtype=cp.int32),
            doy_idx=cp.zeros((0,), dtype=cp.int32),
            year_idx=cp.zeros((0,), dtype=cp.int32),
        )

    id_idx = np.searchsorted(id_values, df["ID"].to_numpy())
    year_idx = np.searchsorted(year_values, df["year"].to_numpy())
    doy_idx = df["doy"].to_numpy().astype(np.int64) - 1  # doy 1..366 -> 0..365

    # Feature matrix X (M, F): const first, then the non-const features in order.
    ones = np.ones(len(df), dtype=np.float64)
    X = np.column_stack([ones, *(df[c].to_numpy(np.float64) for c in non_const)])
    y = df[TARGET_COL].to_numpy(np.float64)

    Xg = cp.asarray(X)
    yg = cp.asarray(y)
    # Flat (id, year, doy) scatter index.
    lin = cp.asarray(((id_idx * n_year + year_idx) * DOY_AXIS + doy_idx).astype(np.int64))

    cells = n_id * n_year * DOY_AXIS
    S_xx = cp.zeros((cells, nf, nf))
    S_xy = cp.zeros((cells, nf))
    S_yy = cp.zeros((cells,))
    cnt = cp.zeros((cells,))

    xx = Xg[:, :, None] * Xg[:, None, :]  # (M, F, F)
    cp.add.at(S_xx, lin, xx)
    cp.add.at(S_xy, lin, Xg * yg[:, None])
    cp.add.at(S_yy, lin, yg * yg)
    cp.add.at(cnt, lin, cp.ones(len(df)))

    return AssembledGeneric(
        S_xx=S_xx.reshape(n_id, n_year, DOY_AXIS, nf, nf),
        S_xy=S_xy.reshape(n_id, n_year, DOY_AXIS, nf),
        S_yy=S_yy.reshape(n_id, n_year, DOY_AXIS),
        cnt=cnt.reshape(n_id, n_year, DOY_AXIS),
        id_values=id_values,
        year_values=year_values,
        feature_cols=registry.feature_cols,
        X=Xg,
        y=yg,
        id_idx=cp.asarray(id_idx.astype(np.int32)),
        doy_idx=cp.asarray(doy_idx.astype(np.int32)),
        year_idx=cp.asarray(year_idx.astype(np.int32)),
    )


def assemble_grams_noyear(
    df: pd.DataFrame,
    registry: FeatureRegistry,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, np.ndarray]:
    """Assemble per-(ID, doy) day-sums (no year axis) for training.

    Returns ``(S_xx[N,366,F,F], S_xy[N,366,F], S_yy[N,366], cnt[N,366], id_values[N])``.
    This is the F-generic, year-free analogue used by ``train_zoned`` (LOYOCV is not
    needed once the recipe is fixed), so it keeps peak memory to ``O(N*366*F^2)``.
    """
    nf = len(registry)
    non_const = [c for c in registry.feature_cols if c != "const"]
    id_values = np.sort(df["ID"].unique()) if len(df) else np.empty(0, dtype="int64")
    n_id = int(id_values.shape[0])
    if n_id == 0:
        z = cp.zeros
        return (z((0, DOY_AXIS, nf, nf)), z((0, DOY_AXIS, nf)),
                z((0, DOY_AXIS)), z((0, DOY_AXIS)), id_values)

    id_idx = np.searchsorted(id_values, df["ID"].to_numpy())
    doy_idx = df["doy"].to_numpy().astype(np.int64) - 1
    ones = np.ones(len(df), dtype=np.float64)
    X = np.column_stack([ones, *(df[c].to_numpy(np.float64) for c in non_const)])
    y = df[TARGET_COL].to_numpy(np.float64)

    Xg = cp.asarray(X)
    yg = cp.asarray(y)
    lin = cp.asarray((id_idx * DOY_AXIS + doy_idx).astype(np.int64))
    cells = n_id * DOY_AXIS
    S_xx = cp.zeros((cells, nf, nf))
    S_xy = cp.zeros((cells, nf))
    S_yy = cp.zeros((cells,))
    cnt = cp.zeros((cells,))
    cp.add.at(S_xx, lin, Xg[:, :, None] * Xg[:, None, :])
    cp.add.at(S_xy, lin, Xg * yg[:, None])
    cp.add.at(S_yy, lin, yg * yg)
    cp.add.at(cnt, lin, cp.ones(len(df)))
    return (
        S_xx.reshape(n_id, DOY_AXIS, nf, nf),
        S_xy.reshape(n_id, DOY_AXIS, nf),
        S_yy.reshape(n_id, DOY_AXIS),
        cnt.reshape(n_id, DOY_AXIS),
        id_values,
    )


def convolve_doy(
    S_xx: cp.ndarray,
    S_xy: cp.ndarray,
    S_yy: cp.ndarray,
    cnt: cp.ndarray,
    *,
    h: int,
    kernel: str,
    axis: int = 2,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Circular tricube/gaussian convolution along the DOY ``axis``.

    Generalises ``HPC_code.gpu_train._circular_convolve`` to any array shape (the DOY axis
    is given explicitly, default 2 for ``(N, Y, 366, ...)`` tensors). ``A``/``b``/``Syy_w``
    use the kernel weight ``w(delta)``; ``nbr`` uses a box over the same ``|delta| <= h``
    offsets to match the CPU raw neighborhood count (the ``min_samples`` gate).
    """
    A = cp.zeros_like(S_xx)
    b = cp.zeros_like(S_xy)
    Syy_w = cp.zeros_like(S_yy)
    nbr = cp.zeros_like(cnt)

    use_tricube = kernel.lower().startswith("tri")
    for delta in range(-h, h + 1):
        shift = -delta  # roll so target d gathers source (d+delta) mod 366
        nbr += cp.roll(cnt, shift, axis=axis)
        ad = abs(delta)
        if use_tricube:
            w = _tricube(ad, h)
        else:  # gaussian
            w = float(np.exp(-(ad ** 2) / (2 * (h ** 2))))
        if w != 0.0:
            A += w * cp.roll(S_xx, shift, axis=axis)
            b += w * cp.roll(S_xy, shift, axis=axis)
            Syy_w += w * cp.roll(S_yy, shift, axis=axis)
    return A, b, Syy_w, nbr
