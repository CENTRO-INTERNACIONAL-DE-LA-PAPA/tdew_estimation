"""
Smoke test for the multi-bucket GPU runner (D3 Phase B).

Builds a small synthetic multi-bucket dataset and runs
``gpu_train.run_bucketed_anomaly_training_gpu(client=None, overwrite=True)`` — the
single-GPU, no-dask path that needs no ``dask-cuda``. Asserts:
  * a ``coeffs.parquet`` is written for every non-empty bucket,
  * those buckets report ``coeff_rows > 0`` and ``status == "ok"``,
  * for one bucket the GPU coeffs equal the CPU ``_train_anomaly_bucket_task`` output
    within 1e-6.

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
from tdew_estimation.anomaly_train import AnomalyTrainingConfig  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402

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


@pytest.fixture(scope="module")
def synth_results(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    results = tmp_path_factory.mktemp("results")
    _synth.build_synthetic_results(
        base, results, n_ids=24, year_range=(2010, 2014), num_buckets=4, seed=11
    )
    return results


def test_gpu_runner_writes_all_buckets(synth_results):
    cfg = _cfg()
    prepared = synth_results / "bucketed_training_data"
    clim = synth_results / "climatology_by_bucket"
    coeffs_root = synth_results / "gpu_coeffs"

    bucket_ids = discover_bucket_ids(prepared)
    assert len(bucket_ids) > 0

    summaries = gpu_train.run_bucketed_anomaly_training_gpu(
        prepared_training_root=prepared,
        bucketed_climatology_root=clim,
        coeffs_output_root=coeffs_root,
        config=cfg,
        client=None,
        overwrite=True,
    )

    # One summary per discovered bucket, returned sorted by bucket_id.
    assert [s.bucket_id for s in summaries] == sorted(bucket_ids)
    assert all(s.status == "ok" for s in summaries), [s.status for s in summaries]

    # Every bucket with coeffs must have written its parquet; total coeffs must be > 0.
    total_rows = 0
    for s in summaries:
        coeffs_file = bucket_dir(coeffs_root, s.bucket_id) / "coeffs.parquet"
        if s.coeff_rows > 0:
            assert coeffs_file.exists(), f"missing coeffs.parquet for bucket {s.bucket_id}"
            df = pd.read_parquet(coeffs_file)
            assert list(df.columns) == gpu_train.COEFF_COLUMNS
            assert len(df) == s.coeff_rows
            total_rows += s.coeff_rows
    assert total_rows > 0, "runner produced no coefficients across any bucket"


def test_gpu_runner_matches_cpu_for_one_bucket(synth_results, tmp_path):
    cfg = _cfg()
    prepared = synth_results / "bucketed_training_data"
    clim = synth_results / "climatology_by_bucket"

    bucket_ids = discover_bucket_ids(prepared)
    # Pick the first bucket that actually yields coefficients on the GPU.
    gpu_root = synth_results / "gpu_coeffs"
    summaries = gpu_train.run_bucketed_anomaly_training_gpu(
        prepared_training_root=prepared,
        bucketed_climatology_root=clim,
        coeffs_output_root=gpu_root,
        config=cfg,
        client=None,
        overwrite=True,
    )
    nonempty = [s.bucket_id for s in summaries if s.coeff_rows > 0]
    assert nonempty, "no non-empty bucket to compare against CPU"
    bid = nonempty[0]

    cpu_root = tmp_path / "cpu_coeffs"
    _train_anomaly_bucket_task(
        bucket_id=bid,
        prepared_training_root=prepared,
        bucketed_climatology_root=clim,
        coeffs_output_root=cpu_root,
        config=cfg,
        overwrite=True,
    )

    cpu_df = pd.read_parquet(bucket_dir(cpu_root, bid) / "coeffs.parquet")
    gpu_df = pd.read_parquet(bucket_dir(gpu_root, bid) / "coeffs.parquet")

    cpu_df = cpu_df.sort_values(["ID", "doy"]).reset_index(drop=True)
    gpu_df = gpu_df.sort_values(["ID", "doy"]).reset_index(drop=True)

    merged = cpu_df.merge(gpu_df, on=["ID", "doy"], suffixes=("_cpu", "_gpu"))
    assert len(merged) == len(cpu_df) == len(gpu_df), (
        f"row sets differ (cpu={len(cpu_df)}, gpu={len(gpu_df)}, common={len(merged)})"
    )
    for col in NUMERIC_COLS:
        delta = np.abs(merged[f"{col}_cpu"].to_numpy() - merged[f"{col}_gpu"].to_numpy())
        assert np.nanmax(delta) < 1e-6, f"{col} max|Δ|={np.nanmax(delta):.3e}"
