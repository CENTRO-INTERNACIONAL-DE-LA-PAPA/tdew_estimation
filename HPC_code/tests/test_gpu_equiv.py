"""
Three-way numerical-equivalence test for the GPU batched WLS trainer (D3).

For a synthetic single bucket, compares per-(ID, doy):
  CPU  `fit_anomaly_coeffs_for_prepared_id` (statsmodels.WLS, per ID)
  vs GPU array-level CuPy reference  (`backend="reference"`)
  vs GPU fused RawKernel             (`backend="rawkernel"`)

Asserts identical (ID, doy) row sets and `max|Δβ| < 1e-6`, `|ΔR²| < 1e-6`. Also checks
the bucket task writes a `coeffs.parquet` matching the CPU bucket task's output.

Skipped automatically when CuPy or a CUDA device is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make HPC_code/ importable (its modules are scripts, not an installed package).
_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

cp = pytest.importorskip("cupy")


def _has_gpu() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="no CUDA device")

import _synth  # noqa: E402
import gpu_train  # noqa: E402
from tdew_estimation.anomaly_dask import _train_anomaly_bucket_task  # noqa: E402
from tdew_estimation.anomaly_train import (  # noqa: E402
    AnomalyTrainingConfig,
    fit_anomaly_coeffs_for_prepared_id,
)
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402
from tdew_estimation.parquet_io import read_parquet_any  # noqa: E402

NUMERIC_COLS = [
    "const_anom",
    "TMIN_anom_coeff",
    "TD_anom_lag1",
    "TD_anom_lag2",
    "TMIN_anom_lag1",
    "r_squared_anom",
]


def _cfg() -> AnomalyTrainingConfig:
    return AnomalyTrainingConfig(
        base_path=Path("/unused"),
        td_var="td",
        tmin_var="tmin_v1",
        train_year_range=(2010, 2014),
        h=11,
        kernel="Tricube",
        min_samples=15,
    )


def _cpu_bucket_coeffs(train_df: pd.DataFrame, clim_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Reference CPU result: per-ID statsmodels fits, concatenated (bucket-task logic)."""
    frames = []
    for location_id, id_df in train_df.groupby("ID", sort=True):
        coeffs_df, _ = fit_anomaly_coeffs_for_prepared_id(
            int(location_id),
            prepared_df=id_df,
            climatology_df=clim_df[clim_df["ID"] == int(location_id)].copy(),
            config=cfg,
        )
        if coeffs_df is not None and not coeffs_df.empty:
            frames.append(coeffs_df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ID", "doy"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def synth_bucket(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    results = tmp_path_factory.mktemp("results")
    _synth.build_synthetic_results(
        base, results, n_ids=24, year_range=(2010, 2014), num_buckets=8, seed=7
    )
    bucket_ids = discover_bucket_ids(results / "bucketed_training_data")
    bid = bucket_ids[0]
    train_df = read_parquet_any(bucket_dir(results / "bucketed_training_data", bid))
    clim_df = pd.read_parquet(
        bucket_dir(results / "climatology_by_bucket", bid) / "climatology.parquet"
    )
    return {"results": results, "bid": bid, "train": train_df, "clim": clim_df}


def _assert_equiv(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    merged = a.merge(b, on=["ID", "doy"], suffixes=("_a", "_b"))
    assert len(merged) == len(a) == len(b), (
        f"{label}: row sets differ (a={len(a)}, b={len(b)}, common={len(merged)})"
    )
    for col in NUMERIC_COLS:
        delta = np.abs(merged[f"{col}_a"].to_numpy() - merged[f"{col}_b"].to_numpy())
        assert np.nanmax(delta) < 1e-6, f"{label}: {col} max|Δ|={np.nanmax(delta):.3e}"


def test_three_way_equivalence(synth_bucket):
    cfg = _cfg()
    train_df, clim_df = synth_bucket["train"], synth_bucket["clim"]

    cpu = _cpu_bucket_coeffs(train_df, clim_df, cfg)
    ref = gpu_train.fit_anomaly_coeffs_for_bucket_gpu(
        train_df, clim_df, cfg, backend="reference"
    )
    raw = gpu_train.fit_anomaly_coeffs_for_bucket_gpu(
        train_df, clim_df, cfg, backend="rawkernel"
    )

    assert len(cpu) > 0 and len(ref) > 0 and len(raw) > 0
    _assert_equiv(cpu, ref, "CPU vs CuPy-reference")
    _assert_equiv(ref, raw, "CuPy-reference vs RawKernel")
    _assert_equiv(cpu, raw, "CPU vs RawKernel")


def test_bucket_task_parquet_parity(synth_bucket, tmp_path):
    cfg = _cfg()
    results, bid = synth_bucket["results"], synth_bucket["bid"]
    prepared = results / "bucketed_training_data"
    clim = results / "climatology_by_bucket"

    cpu_root = tmp_path / "cpu_coeffs"
    gpu_root = tmp_path / "gpu_coeffs"

    _train_anomaly_bucket_task(
        bucket_id=bid,
        prepared_training_root=prepared,
        bucketed_climatology_root=clim,
        coeffs_output_root=cpu_root,
        config=cfg,
        overwrite=True,
    )
    gpu_train._train_anomaly_bucket_task_gpu(
        bucket_id=bid,
        prepared_training_root=prepared,
        bucketed_climatology_root=clim,
        coeffs_output_root=gpu_root,
        config=cfg,
        overwrite=True,
        backend="rawkernel",
    )

    cpu_df = pd.read_parquet(bucket_dir(cpu_root, bid) / "coeffs.parquet")
    gpu_df = pd.read_parquet(bucket_dir(gpu_root, bid) / "coeffs.parquet")
    assert list(cpu_df.columns) == list(gpu_df.columns) == gpu_train.COEFF_COLUMNS
    _assert_equiv(
        cpu_df.sort_values(["ID", "doy"]).reset_index(drop=True),
        gpu_df.sort_values(["ID", "doy"]).reset_index(drop=True),
        "bucket-task parquet",
    )
