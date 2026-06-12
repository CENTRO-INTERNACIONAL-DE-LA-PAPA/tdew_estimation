#!/usr/bin/env python3
"""
HPC_code.gpu_train

GPU batched weighted-least-squares (WLS) anomaly trainer, numerically equivalent to the
CPU per-(ID, doy) ``statsmodels.WLS`` path (``tdew_estimation.anomaly_train.
fit_anomaly_coeffs_for_prepared_id``). One bucket is fit in a few GPU launches instead of
a Python loop over (ID, doy).

Key idea: the WLS weight depends only on the circular day-of-year offset ``δ`` between a
row and the target day, so the per-target normal equations are a **circular convolution**
over DOY. We accumulate per-(ID, day) sufficient statistics, convolve them along the DOY
axis with the tricube weight (and a box kernel for the raw neighborhood count), then solve
one 5x5 SPD system per surviving (ID, doy).

Two solvers share the same assembled ``A``/``b``:
  * ``solve_bucket_reference``  — array-level CuPy oracle (``cp.linalg.solve``), readable.
  * ``solve_bucket_rawkernel``  — fused ``RawKernel``, one thread per fit (production path).

Everything is float64 to match statsmodels. Requires CuPy + a CUDA device.

Output coefficient schema is byte-identical to the CPU path:
``const_anom, TMIN_anom_coeff, TD_anom_lag1, TD_anom_lag2, TMIN_anom_lag1, doy,
r_squared_anom, ID``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import cupy as cp

# Allow `import tdew_estimation` regardless of cwd / install state (mirror entrypoint).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from tdew_estimation.anomaly_dask import (  # noqa: E402
    BucketTrainingSummary,
    _error_bucket_summary,
    _failure_row,
)
from tdew_estimation.anomaly_train import AnomalyTrainingConfig  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402
from tdew_estimation.parquet_io import as_path, read_parquet_any  # noqa: E402

logger = logging.getLogger(__name__)

DOY_AXIS = 366  # day-of-year bins, indices 0..365 for doy 1..366; circular modulo 366
NF = 5  # features: [const, TMIN_anom, TD_anom_lag1, TD_anom_lag2, TMIN_anom_lag1]

# Output column order must match anomaly_train.fit_anomaly_coeffs_for_prepared_id.
COEFF_COLUMNS = [
    "const_anom",
    "TMIN_anom_coeff",
    "TD_anom_lag1",
    "TD_anom_lag2",
    "TMIN_anom_lag1",
    "doy",
    "r_squared_anom",
    "ID",
]


# ---------------------------------------------------------------------------------------
# Fused RawKernel: one thread per valid (ID, doy) fit.
# In-register 5x5 Cholesky solve of the SPD normal equations, plus weighted R².
# ---------------------------------------------------------------------------------------
_SOLVE_SRC = r"""
extern "C" __global__
void solve5(const double* A, const double* b, const double* syy,
            double* beta, double* r2, const int M) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const double* Am = A + (long)m * 25;   // row-major 5x5
    const double* bm = b + (long)m * 5;

    double a[25];
    for (int i = 0; i < 25; ++i) a[i] = Am[i];

    // Cholesky: a = L L^T (L lower-triangular, stored in l).
    double l[25];
    for (int i = 0; i < 25; ++i) l[i] = 0.0;
    bool ok = true;
    for (int j = 0; j < 5 && ok; ++j) {
        double d = a[j * 5 + j];
        for (int k = 0; k < j; ++k) d -= l[j * 5 + k] * l[j * 5 + k];
        if (d <= 0.0) { ok = false; break; }
        double ljj = sqrt(d);
        l[j * 5 + j] = ljj;
        for (int i = j + 1; i < 5; ++i) {
            double s = a[i * 5 + j];
            for (int k = 0; k < j; ++k) s -= l[i * 5 + k] * l[j * 5 + k];
            l[i * 5 + j] = s / ljj;
        }
    }
    if (!ok) {
        for (int i = 0; i < 5; ++i) beta[(long)m * 5 + i] = nan("");
        r2[m] = nan("");
        return;
    }

    // Solve L z = b (forward), then L^T be = z (backward).
    double z[5];
    for (int i = 0; i < 5; ++i) {
        double s = bm[i];
        for (int k = 0; k < i; ++k) s -= l[i * 5 + k] * z[k];
        z[i] = s / l[i * 5 + i];
    }
    double be[5];
    for (int i = 4; i >= 0; --i) {
        double s = z[i];
        for (int k = i + 1; k < 5; ++k) s -= l[k * 5 + i] * be[k];
        be[i] = s / l[i * 5 + i];
    }

    // Weighted R² from the same sums: ssr = Σwy² - 2 bᵀβ + βᵀAβ ; tss = Σwy² - (Σwy)²/Σw.
    double Abe[5];
    for (int i = 0; i < 5; ++i) {
        double s = 0.0;
        for (int k = 0; k < 5; ++k) s += a[i * 5 + k] * be[k];
        Abe[i] = s;
    }
    double bdotbe = 0.0, beAbe = 0.0;
    for (int i = 0; i < 5; ++i) { bdotbe += bm[i] * be[i]; beAbe += be[i] * Abe[i]; }
    double ssr = syy[m] - 2.0 * bdotbe + beAbe;
    double tss = syy[m] - bm[0] * bm[0] / a[0];   // Σw = A[0,0], Σwy = b[0]

    for (int i = 0; i < 5; ++i) beta[(long)m * 5 + i] = be[i];
    r2[m] = 1.0 - ssr / tss;
}
"""

_SOLVE_KERNEL: Optional[cp.RawKernel] = None


def _solve_kernel() -> cp.RawKernel:
    global _SOLVE_KERNEL
    if _SOLVE_KERNEL is None:
        _SOLVE_KERNEL = cp.RawKernel(_SOLVE_SRC, "solve5")
    return _SOLVE_KERNEL


# ---------------------------------------------------------------------------------------
# Assembly: per-(ID, day) sufficient statistics.
# ---------------------------------------------------------------------------------------
def _assemble_day_sums(
    train_df: pd.DataFrame,
    clim_df: pd.DataFrame,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, np.ndarray]:
    """Return (S_xx[N,366,5,5], S_xy[N,366,5], S_yy[N,366], cnt[N,366], id_values[N]).

    Replicates the CPU prep exactly: merge clim on (ID,doy), sort each ID series by date,
    anomalies + global lags (shift 1/2), then keep only rows where ``y`` and all four
    features are non-NaN (the CPU per-doy dropna), scatter-added into per-(ID,day) bins.
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

    df = df.merge(clim[["ID", "doy", "TD_clim", "TMIN_clim"]], on=["ID", "doy"], how="left")
    df = df.sort_values(["ID", "FECHA"]).reset_index(drop=True)

    df["TD_anom"] = df["TD"] - df["TD_clim"]
    df["TMIN_anom"] = df["TMIN"] - df["TMIN_clim"]
    g = df.groupby("ID", sort=False)
    df["TD_anom_lag1"] = g["TD_anom"].shift(1)
    df["TD_anom_lag2"] = g["TD_anom"].shift(2)
    df["TMIN_anom_lag1"] = g["TMIN_anom"].shift(1)

    feature_cols = ["TMIN_anom", "TD_anom_lag1", "TD_anom_lag2", "TMIN_anom_lag1"]
    valid = df[["TD_anom"] + feature_cols].notna().all(axis=1)
    df = df.loc[valid]

    id_values = np.sort(df["ID"].unique())
    n_id = int(id_values.shape[0])
    if n_id == 0:
        z = cp.zeros
        return (
            z((0, DOY_AXIS, NF, NF)),
            z((0, DOY_AXIS, NF)),
            z((0, DOY_AXIS)),
            z((0, DOY_AXIS)),
            id_values,
        )

    id_idx = np.searchsorted(id_values, df["ID"].to_numpy())
    day_idx = df["doy"].to_numpy().astype(np.int64) - 1  # doy 1..366 -> 0..365

    # Feature matrix X (M,5) with const first, target y (M,).
    ones = np.ones(len(df), dtype=np.float64)
    X = np.column_stack(
        [
            ones,
            df["TMIN_anom"].to_numpy(np.float64),
            df["TD_anom_lag1"].to_numpy(np.float64),
            df["TD_anom_lag2"].to_numpy(np.float64),
            df["TMIN_anom_lag1"].to_numpy(np.float64),
        ]
    )
    y = df["TD_anom"].to_numpy(np.float64)

    Xg = cp.asarray(X)
    yg = cp.asarray(y)
    lin = cp.asarray(id_idx.astype(np.int64) * DOY_AXIS + day_idx)  # flat (ID,day) index

    S_xx = cp.zeros((n_id * DOY_AXIS, NF, NF))
    S_xy = cp.zeros((n_id * DOY_AXIS, NF))
    S_yy = cp.zeros((n_id * DOY_AXIS,))
    cnt = cp.zeros((n_id * DOY_AXIS,))

    xx = Xg[:, :, None] * Xg[:, None, :]  # (M,5,5)
    cp.add.at(S_xx, lin, xx)
    cp.add.at(S_xy, lin, Xg * yg[:, None])
    cp.add.at(S_yy, lin, yg * yg)
    cp.add.at(cnt, lin, cp.ones(len(df)))

    return (
        S_xx.reshape(n_id, DOY_AXIS, NF, NF),
        S_xy.reshape(n_id, DOY_AXIS, NF),
        S_yy.reshape(n_id, DOY_AXIS),
        cnt.reshape(n_id, DOY_AXIS),
        id_values,
    )


def _tricube(delta_abs: int, h: int) -> float:
    scaled = min(delta_abs / h, 1.0)
    return (1.0 - scaled ** 3) ** 3


def _circular_convolve(
    S_xx: cp.ndarray,
    S_xy: cp.ndarray,
    S_yy: cp.ndarray,
    cnt: cp.ndarray,
    *,
    h: int,
    kernel: str,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Circular convolution over the DOY axis.

    ``A``/``b``/``Syy_w`` use the kernel weight ``w(δ)`` (tricube/gaussian); ``nbr_count``
    uses a box (all-ones) over the *same* offset set ``|δ| ≤ h`` so it equals the CPU raw
    neighborhood count (the ``min_samples`` gate), independent of the zero weight at δ=h.
    """
    A = cp.zeros_like(S_xx)
    b = cp.zeros_like(S_xy)
    Syy_w = cp.zeros_like(S_yy)
    nbr_count = cp.zeros_like(cnt)

    use_tricube = kernel.lower().startswith("tri")

    for delta in range(-h, h + 1):
        shift = -delta  # roll so that target d gathers source (d+δ) mod 366
        nbr_count += cp.roll(cnt, shift, axis=1)

        ad = abs(delta)
        if use_tricube:
            w = _tricube(ad, h)
        else:  # gaussian
            w = float(np.exp(-(ad ** 2) / (2 * (h ** 2))))
        if w != 0.0:
            A += w * cp.roll(S_xx, shift, axis=1)
            b += w * cp.roll(S_xy, shift, axis=1)
            Syy_w += w * cp.roll(S_yy, shift, axis=1)

    return A, b, Syy_w, nbr_count


# ---------------------------------------------------------------------------------------
# Solvers (operate on the gathered valid fits).
# ---------------------------------------------------------------------------------------
def solve_bucket_reference(
    A_v: cp.ndarray, b_v: cp.ndarray, syy_v: cp.ndarray
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Array-level oracle: stacked ``cp.linalg.solve`` + weighted R². Returns (beta, r2)."""
    beta = cp.linalg.solve(A_v, b_v[..., None])[..., 0]  # (M,5)
    Abeta = cp.einsum("mij,mj->mi", A_v, beta)
    ssr = syy_v - 2.0 * cp.einsum("mi,mi->m", b_v, beta) + cp.einsum("mi,mi->m", beta, Abeta)
    tss = syy_v - b_v[:, 0] ** 2 / A_v[:, 0, 0]
    r2 = 1.0 - ssr / tss
    return beta, r2


def solve_bucket_rawkernel(
    A_v: cp.ndarray, b_v: cp.ndarray, syy_v: cp.ndarray, *, block: int = 128
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Fused one-thread-per-fit kernel (in-register 5x5 Cholesky). Returns (beta, r2)."""
    m = int(A_v.shape[0])
    beta = cp.empty((m, NF), dtype=cp.float64)
    r2 = cp.empty((m,), dtype=cp.float64)
    if m == 0:
        return beta, r2
    A_c = cp.ascontiguousarray(A_v, dtype=cp.float64)
    b_c = cp.ascontiguousarray(b_v, dtype=cp.float64)
    syy_c = cp.ascontiguousarray(syy_v, dtype=cp.float64)
    grid = ((m + block - 1) // block,)
    _solve_kernel()(grid, (block,), (A_c, b_c, syy_c, beta, r2, np.int32(m)))
    return beta, r2


# ---------------------------------------------------------------------------------------
# Bucket-level API.
# ---------------------------------------------------------------------------------------
def fit_anomaly_coeffs_for_bucket_gpu(
    train_df: pd.DataFrame,
    clim_df: pd.DataFrame,
    config: AnomalyTrainingConfig,
    *,
    doys: Optional[Sequence[int]] = None,
    backend: str = "rawkernel",
) -> pd.DataFrame:
    """Fit every (ID, doy) in a bucket on the GPU. Returns a coeffs frame (CPU schema).

    ``backend`` is ``"rawkernel"`` (production) or ``"reference"`` (array-level oracle).
    """
    if train_df.empty or clim_df.empty:
        return pd.DataFrame(columns=COEFF_COLUMNS)

    S_xx, S_xy, S_yy, cnt, id_values = _assemble_day_sums(train_df, clim_df)
    if id_values.shape[0] == 0:
        return pd.DataFrame(columns=COEFF_COLUMNS)

    A, b, Syy_w, nbr_count = _circular_convolve(
        S_xx, S_xy, S_yy, cnt, h=config.h, kernel=config.kernel
    )

    n_id = id_values.shape[0]
    valid = (nbr_count >= config.min_samples).reshape(n_id * DOY_AXIS)
    idx = cp.where(valid)[0]
    if int(idx.shape[0]) == 0:
        return pd.DataFrame(columns=COEFF_COLUMNS)

    A_v = A.reshape(n_id * DOY_AXIS, NF, NF)[idx]
    b_v = b.reshape(n_id * DOY_AXIS, NF)[idx]
    syy_v = Syy_w.reshape(n_id * DOY_AXIS)[idx]

    if backend == "reference":
        beta, r2 = solve_bucket_reference(A_v, b_v, syy_v)
    elif backend == "rawkernel":
        beta, r2 = solve_bucket_rawkernel(A_v, b_v, syy_v)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (use 'rawkernel' or 'reference').")

    idx_h = cp.asnumpy(idx)
    beta_h = cp.asnumpy(beta)
    r2_h = cp.asnumpy(r2)
    id_idx_h = (idx_h // DOY_AXIS).astype(int)
    doy_h = (idx_h % DOY_AXIS).astype(int) + 1

    out = pd.DataFrame(
        {
            "const_anom": beta_h[:, 0],
            "TMIN_anom_coeff": beta_h[:, 1],
            "TD_anom_lag1": beta_h[:, 2],
            "TD_anom_lag2": beta_h[:, 3],
            "TMIN_anom_lag1": beta_h[:, 4],
            "doy": doy_h,
            "r_squared_anom": r2_h,
            "ID": id_values[id_idx_h].astype(int),
        }
    )

    if doys is not None:
        keep = {int(d) for d in doys if 1 <= int(d) <= DOY_AXIS}
        out = out[out["doy"].isin(keep)]

    out = out.sort_values(["ID", "doy"]).reset_index(drop=True)
    return out[COEFF_COLUMNS]


def _train_anomaly_bucket_task_gpu(
    *,
    bucket_id: int,
    prepared_training_root,
    bucketed_climatology_root,
    coeffs_output_root,
    config: AnomalyTrainingConfig,
    doys: Optional[Sequence[int]] = None,
    failure_output_root=None,
    overwrite: bool = False,
    backend: str = "rawkernel",
) -> BucketTrainingSummary:
    """GPU analogue of ``anomaly_dask._train_anomaly_bucket_task`` (same I/O contract)."""
    bucket_training_dir = bucket_dir(prepared_training_root, bucket_id)
    bucket_clim_dir = bucket_dir(bucketed_climatology_root, bucket_id)
    coeffs_dir = bucket_dir(coeffs_output_root, bucket_id)
    coeffs_dir.mkdir(parents=True, exist_ok=True)
    coeffs_file = coeffs_dir / "coeffs.parquet"

    failures_file = None
    if failure_output_root is not None:
        failures_dir = bucket_dir(failure_output_root, bucket_id)
        failures_dir.mkdir(parents=True, exist_ok=True)
        failures_file = failures_dir / "failures.parquet"

    if coeffs_file.exists() and not overwrite:
        return BucketTrainingSummary(
            bucket_id=int(bucket_id),
            id_count=0,
            coeff_rows=0,
            failure_rows=0,
            status="skipped",
            coeffs_path=coeffs_file,
            failures_path=failures_file if failures_file and failures_file.exists() else None,
        )

    failures: List[dict[str, Any]] = []
    try:
        train_df = read_parquet_any(bucket_training_dir)
    except Exception as exc:
        failures.append(
            _failure_row(
                phase="read_training_bucket",
                bucket_id=bucket_id,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
        )
        train_df = pd.DataFrame()

    clim_file = bucket_clim_dir / "climatology.parquet"
    if clim_file.exists():
        clim_df = pd.read_parquet(clim_file)
    else:
        failures.append(
            _failure_row(
                phase="read_climatology_bucket",
                bucket_id=bucket_id,
                message=f"Missing climatology shard: {clim_file}",
            )
        )
        clim_df = pd.DataFrame()

    id_count = int(train_df["ID"].nunique()) if not train_df.empty else 0
    coeffs_df = pd.DataFrame(columns=COEFF_COLUMNS)
    if not train_df.empty and not clim_df.empty:
        try:
            coeffs_df = fit_anomaly_coeffs_for_bucket_gpu(
                train_df, clim_df, config, doys=doys, backend=backend
            )
        except Exception as exc:
            failures.append(
                _failure_row(
                    phase="fit_bucket_gpu",
                    bucket_id=bucket_id,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            )

    if not coeffs_df.empty:
        coeffs_df = coeffs_df.sort_values(["ID", "doy"]).reset_index(drop=True)
        coeffs_df.to_parquet(coeffs_file, engine="pyarrow", index=False)
    elif overwrite and coeffs_file.exists():
        coeffs_file.unlink()

    failures_df = pd.DataFrame(failures)
    if failures_file is not None and not failures_df.empty:
        failures_df.to_parquet(failures_file, engine="pyarrow", index=False)

    return BucketTrainingSummary(
        bucket_id=int(bucket_id),
        id_count=id_count,
        coeff_rows=int(len(coeffs_df)),
        failure_rows=int(len(failures_df)),
        status="ok",
        coeffs_path=coeffs_file,
        failures_path=failures_file if (failures_file and not failures_df.empty) else None,
    )


# ---------------------------------------------------------------------------------------
# Runner: GPU analogue of anomaly_dask.run_bucketed_anomaly_training_dask.
# ---------------------------------------------------------------------------------------
def run_bucketed_anomaly_training_gpu(
    *,
    prepared_training_root,
    bucketed_climatology_root,
    coeffs_output_root,
    config: AnomalyTrainingConfig,
    bucket_ids: Optional[Sequence[int]] = None,
    doys: Optional[Sequence[int]] = None,
    failure_output_root=None,
    overwrite: bool = False,
    client: Optional[Any] = None,
    backend: str = "rawkernel",
    max_in_flight: int = 4,
) -> List[BucketTrainingSummary]:
    """Run GPU bucket training across many buckets (client-agnostic).

    Mirrors :func:`tdew_estimation.anomaly_dask.run_bucketed_anomaly_training_dask` but
    fits each bucket with the GPU batched-WLS task (:func:`_train_anomaly_bucket_task_gpu`).

    ``client is None``
        Process buckets **sequentially in-process** on the single visible GPU — no dask.
        This is the dev-box / single-GPU verification path and needs no ``dask-cuda``.
    ``client`` provided
        Submit each bucket via the injected dask client (one worker per GPU, typically
        built by :func:`hpc.make_local_cuda_cluster`) using the same sliding-window
        ``as_completed`` structure as the CPU runner. A bucket whose task fails on a
        worker is recorded as an error summary and skipped rather than aborting the run.

    Returns summaries sorted by ``bucket_id``.
    """
    prepared_root_p = as_path(prepared_training_root)
    clim_root_p = as_path(bucketed_climatology_root)
    coeffs_root_p = as_path(coeffs_output_root)
    coeffs_root_p.mkdir(parents=True, exist_ok=True)
    failure_root_p = as_path(failure_output_root) if failure_output_root is not None else None
    if failure_root_p is not None:
        failure_root_p.mkdir(parents=True, exist_ok=True)

    if bucket_ids is None:
        buckets_to_run = discover_bucket_ids(prepared_root_p)
    else:
        buckets_to_run = sorted({int(b) for b in bucket_ids})
    if not buckets_to_run:
        raise ValueError(f"No buckets found under {prepared_root_p}")

    def _task_kwargs(bucket_id: int) -> dict:
        return dict(
            bucket_id=int(bucket_id),
            prepared_training_root=prepared_root_p,
            bucketed_climatology_root=clim_root_p,
            coeffs_output_root=coeffs_root_p,
            config=config,
            doys=doys,
            failure_output_root=failure_root_p,
            overwrite=overwrite,
            backend=backend,
        )

    summaries: List[BucketTrainingSummary] = []

    # --- Single-GPU, no-dask path: fit buckets one at a time in this process. ---
    if client is None:
        for bid in buckets_to_run:
            try:
                summaries.append(_train_anomaly_bucket_task_gpu(**_task_kwargs(bid)))
            except Exception as exc:  # one bad bucket must not abort the run
                logger.warning("Bucket %s failed: %s: %s", bid, type(exc).__name__, exc)
                summaries.append(_error_bucket_summary(bid, coeffs_root_p, exc))
        return sorted(summaries, key=lambda item: item.bucket_id)

    # --- Distributed path: submit via the injected dask(-cuda) client. ---
    from distributed import as_completed  # imported lazily

    n_workers = len(getattr(client, "scheduler_info", lambda: {})().get("workers", {})) or 1
    window = max(int(max_in_flight) if max_in_flight else 0, n_workers)

    def _submit(bucket_id: int):
        return client.submit(
            _train_anomaly_bucket_task_gpu,
            **_task_kwargs(bucket_id),
            pure=False,
        )

    pending: dict[Any, int] = {}
    next_idx = 0
    ac = as_completed()
    # Prime the sliding window.
    while next_idx < len(buckets_to_run) and len(pending) < window:
        bid = buckets_to_run[next_idx]
        fut = _submit(bid)
        pending[fut] = bid
        ac.add(fut)
        next_idx += 1

    for fut in ac:
        bid = pending.pop(fut, -1)
        try:
            summaries.append(fut.result())
        except Exception as exc:  # KilledWorker etc. must not abort the run
            logger.warning("Bucket %s failed: %s: %s", bid, type(exc).__name__, exc)
            summaries.append(_error_bucket_summary(bid, coeffs_root_p, exc))

        # Top up the window with the next bucket, if any remain.
        if next_idx < len(buckets_to_run):
            bid = buckets_to_run[next_idx]
            fut = _submit(bid)
            pending[fut] = bid
            ac.add(fut)
            next_idx += 1

    return sorted(summaries, key=lambda item: item.bucket_id)
