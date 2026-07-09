"""P1 parity: F-generic assembly reproduces the production fixed-5 path, and an
arbitrary-F assembly matches an explicit NumPy weighted normal-equations reference.

Skipped when CuPy / a CUDA device is unavailable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]  # repo root
_HPC = _ROOT / "HPC_code"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HPC))

cp = pytest.importorskip("cupy")


def _has_gpu() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="no CUDA device")

import _synth  # noqa: E402
import gpu_train  # noqa: E402
from HPC_code_tunning.assemble_generic import (  # noqa: E402
    assemble_day_sums_generic,
    convolve_doy,
)
from HPC_code_tunning.feature_spec import (  # noqa: E402
    CANONICAL_FEATURES,
    FeatureRegistry,
    build_feature_frame,
)
from tdew_estimation.anomaly_train import AnomalyTrainingConfig  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402
from tdew_estimation.parquet_io import read_parquet_any  # noqa: E402

H = 11
KERNEL = "Tricube"
MIN_SAMPLES = 15


def _cfg() -> AnomalyTrainingConfig:
    return AnomalyTrainingConfig(
        base_path=Path("/unused"), td_var="td", tmin_var="tmin_v1",
        train_year_range=(2010, 2014), h=H, kernel=KERNEL, min_samples=MIN_SAMPLES,
    )


@pytest.fixture(scope="module")
def synth_bucket(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    results = tmp_path_factory.mktemp("results")
    _synth.build_synthetic_results(
        base, results, n_ids=24, year_range=(2010, 2014), num_buckets=8, seed=7
    )
    bid = discover_bucket_ids(results / "bucketed_training_data")[0]
    train = read_parquet_any(bucket_dir(results / "bucketed_training_data", bid))
    clim = pd.read_parquet(
        bucket_dir(results / "climatology_by_bucket", bid) / "climatology.parquet"
    )
    return {"train": train, "clim": clim}


def _generic_canonical_coeffs(train, clim) -> pd.DataFrame:
    """Solve the canonical 5-feature model through the F-generic path (year-summed)."""
    S = assemble_day_sums_generic(train, clim, feature_cols=CANONICAL_FEATURES)
    # Sum the year axis, then convolve along the DOY axis (now axis=1).
    Sxx = S.S_xx.sum(1)
    Sxy = S.S_xy.sum(1)
    Syy = S.S_yy.sum(1)
    cnt = S.cnt.sum(1)
    A, b, syyw, nbr = convolve_doy(Sxx, Sxy, Syy, cnt, h=H, kernel=KERNEL, axis=1)
    n_id = S.n_id
    valid = (nbr >= MIN_SAMPLES).reshape(n_id * 366)
    idx = cp.where(valid)[0]
    beta, r2 = gpu_train.solve_bucket_reference(
        A.reshape(n_id * 366, 5, 5)[idx],
        b.reshape(n_id * 366, 5)[idx],
        syyw.reshape(n_id * 366)[idx],
    )
    idx_h = cp.asnumpy(idx)
    return pd.DataFrame(
        {
            "const_anom": cp.asnumpy(beta)[:, 0],
            "TMIN_anom_coeff": cp.asnumpy(beta)[:, 1],
            "TD_anom_lag1": cp.asnumpy(beta)[:, 2],
            "TD_anom_lag2": cp.asnumpy(beta)[:, 3],
            "TMIN_anom_lag1": cp.asnumpy(beta)[:, 4],
            "doy": (idx_h % 366).astype(int) + 1,
            "r_squared_anom": cp.asnumpy(r2),
            "ID": S.id_values[(idx_h // 366).astype(int)].astype(int),
        }
    ).sort_values(["ID", "doy"]).reset_index(drop=True)


def test_canonical_equals_production_fixed5(synth_bucket):
    train, clim = synth_bucket["train"], synth_bucket["clim"]
    ref = gpu_train.fit_anomaly_coeffs_for_bucket_gpu(train, clim, _cfg(), backend="reference")
    mine = _generic_canonical_coeffs(train, clim)

    merged = ref.merge(mine, on=["ID", "doy"], suffixes=("_ref", "_mine"))
    assert len(merged) == len(ref) == len(mine) > 0
    for col in ["const_anom", "TMIN_anom_coeff", "TD_anom_lag1", "TD_anom_lag2",
                "TMIN_anom_lag1", "r_squared_anom"]:
        d = np.abs(merged[f"{col}_ref"].to_numpy() - merged[f"{col}_mine"].to_numpy())
        assert np.nanmax(d) < 1e-6, f"{col}: max|Δ|={np.nanmax(d):.3e}"


def test_arbitrary_F_matches_numpy_weighted_lstsq(synth_bucket):
    """A 6-feature assembly's normal equations match an explicit NumPy WLS at one (ID,doy)."""
    train, clim = synth_bucket["train"], synth_bucket["clim"]
    feats = ("const", "TMIN_anom", "TD_anom_lag1", "TD_anom_lag2", "TMIN_anom_lag1",
             "TD_anom_lag3")
    reg = FeatureRegistry(feats)
    frame = build_feature_frame(train, clim, reg)
    S = assemble_day_sums_generic(train, clim, feature_cols=feats)
    Sxx, Sxy = S.S_xx.sum(1), S.S_xy.sum(1)
    A, b, _syy, nbr = convolve_doy(Sxx, Sxy, S.S_yy.sum(1), S.cnt.sum(1), h=H, kernel=KERNEL, axis=1)

    # Pick a well-populated (ID, doy).
    nbr_h = cp.asnumpy(nbr)
    i, d0 = np.unravel_index(int(np.argmax(nbr_h)), nbr_h.shape)
    target_id = int(S.id_values[i])
    doy = int(d0) + 1

    # NumPy reference: circular tricube weights over |Δ| <= H around `doy`.
    fdf = frame[frame["ID"] == target_id]
    dist = np.minimum((fdf["doy"] - doy).abs(), 366 - (fdf["doy"] - doy).abs())
    m = dist <= H
    sub = fdf[m]
    scaled = np.clip(dist[m].to_numpy() / H, 0, 1)
    w = (1 - scaled ** 3) ** 3
    Xn = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy() for c in feats[1:]])
    yn = sub["TD_anom"].to_numpy()
    A_np = Xn.T @ (w[:, None] * Xn)
    b_np = Xn.T @ (w * yn)
    beta_np = np.linalg.solve(A_np, b_np)

    beta_gpu = cp.asnumpy(cp.linalg.solve(A[i, d0], b[i, d0]))
    assert np.max(np.abs(A_np - cp.asnumpy(A[i, d0]))) < 1e-6
    assert np.max(np.abs(beta_np - beta_gpu)) < 1e-6
