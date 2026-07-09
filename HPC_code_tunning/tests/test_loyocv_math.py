"""P2: the leave-one-year-out subtraction identity equals an independent re-assembly.

``A_loo = A_full - A_year[j]`` (kept-year-axis subtraction) must equal convolving day-sums
re-assembled from the raw samples with year ``j`` removed — the honest brute-force refit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
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
from HPC_code_tunning.assemble_generic import (  # noqa: E402
    assemble_day_sums_generic,
    assemble_grams_noyear,
    convolve_doy,
)
from HPC_code_tunning.feature_spec import (  # noqa: E402
    CANONICAL_FEATURES,
    FeatureRegistry,
    build_feature_frame,
)
from HPC_code_tunning.loyocv import ConvolvedChunk, leave_one_year_normal_equations  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402
from tdew_estimation.parquet_io import read_parquet_any  # noqa: E402

H = 11
KERNEL = "Tricube"


@pytest.fixture(scope="module")
def synth_bucket(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    results = tmp_path_factory.mktemp("results")
    _synth.build_synthetic_results(
        base, results, n_ids=16, year_range=(2010, 2015), num_buckets=4, seed=3
    )
    bid = discover_bucket_ids(results / "bucketed_training_data")[0]
    train = read_parquet_any(bucket_dir(results / "bucketed_training_data", bid))
    clim = pd.read_parquet(
        bucket_dir(results / "climatology_by_bucket", bid) / "climatology.parquet"
    )
    return {"train": train, "clim": clim}


def test_subtraction_identity_equals_reassembly(synth_bucket):
    train, clim = synth_bucket["train"], synth_bucket["clim"]
    reg = FeatureRegistry(CANONICAL_FEATURES)
    S = assemble_day_sums_generic(train, clim, feature_cols=CANONICAL_FEATURES)
    conv = ConvolvedChunk(S, h=H, kernel=KERNEL)

    frame = build_feature_frame(train, clim, reg)
    years = S.year_values.tolist()
    assert len(years) >= 3

    for j, yr in enumerate(years):
        A_loo, b_loo = leave_one_year_normal_equations(
            conv.A_full, conv.A_year, conv.b_full, conv.b_year, j
        )
        # Independent re-assembly: drop year yr from the raw frame, then convolve.
        reduced = frame[frame["year"] != yr]
        Sxx, Sxy, Syy, cnt, _ = assemble_grams_noyear(reduced, reg)
        A_b, b_b, _syy_b, _nbr = convolve_doy(Sxx, Sxy, Syy, cnt, h=H, kernel=KERNEL, axis=1)

        da = float(cp.max(cp.abs(A_loo - A_b)))
        db = float(cp.max(cp.abs(b_loo - b_b)))
        assert da < 1e-6, f"year {yr}: max|ΔA|={da:.3e}"
        assert db < 1e-6, f"year {yr}: max|Δb|={db:.3e}"


def test_loo_beta_matches_reduced_refit(synth_bucket):
    """β from the subtracted normal equations equals β from the reduced-data refit."""
    train, clim = synth_bucket["train"], synth_bucket["clim"]
    reg = FeatureRegistry(CANONICAL_FEATURES)
    S = assemble_day_sums_generic(train, clim, feature_cols=CANONICAL_FEATURES)
    conv = ConvolvedChunk(S, h=H, kernel=KERNEL)
    frame = build_feature_frame(train, clim, reg)

    j = 1
    yr = int(S.year_values[j])
    A_loo, b_loo = leave_one_year_normal_equations(
        conv.A_full, conv.A_year, conv.b_full, conv.b_year, j
    )
    Sxx, Sxy, Syy, cnt, _ = assemble_grams_noyear(frame[frame["year"] != yr], reg)
    A_b, b_b, _s, nbr = convolve_doy(Sxx, Sxy, Syy, cnt, h=H, kernel=KERNEL, axis=1)

    n_id = S.n_id
    valid = (nbr >= 15).reshape(n_id * 366)
    idx = cp.where(valid)[0]
    beta_loo = cp.linalg.solve(A_loo.reshape(n_id * 366, 5, 5)[idx],
                               b_loo.reshape(n_id * 366, 5)[idx][..., None])[..., 0]
    beta_b = cp.linalg.solve(A_b.reshape(n_id * 366, 5, 5)[idx],
                             b_b.reshape(n_id * 366, 5)[idx][..., None])[..., 0]
    assert float(cp.max(cp.abs(beta_loo - beta_b))) < 1e-6
