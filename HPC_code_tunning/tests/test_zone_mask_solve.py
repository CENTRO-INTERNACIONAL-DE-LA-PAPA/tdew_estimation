"""P4: the zero-column mask solve equals the true reduced sub-block solve.

``train_zoned`` keeps a uniform F×F batched solve by zeroing the columns of dropped
features (diag->1, row/col + rhs -> 0). The retained sub-block must solve exactly and the
dropped coefficients must come out as 0, with the weighted R² matching a genuine reduced
solve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

import gpu_train  # noqa: E402


def _masked_solve(A, b, syy, keep):
    """Reproduce train_zoned's zero-column trick and solve."""
    F = A.shape[1]
    km = keep.astype(cp.float64)
    A_m = A * km[:, :, None] * km[:, None, :]
    diag = cp.arange(F)
    A_m[:, diag, diag] += (1.0 - km)
    b_m = b * km
    return gpu_train.solve_bucket_reference(A_m, b_m, syy)


def test_masked_uniform_solve_equals_reduced_subblock():
    rng = np.random.default_rng(0)
    m, F = 200, 7
    # Well-conditioned SPD systems.
    R = rng.standard_normal((m, F, F))
    A = R @ np.transpose(R, (0, 2, 1)) + F * np.eye(F)
    b = rng.standard_normal((m, F))
    # syy chosen large enough that tss > 0 for the R² formula.
    syy = (b[:, 0] ** 2) / A[:, 0, 0] + rng.uniform(5.0, 20.0, size=m)

    # Random keep masks; feature 0 (const) always kept, >=1 extra kept.
    keep = rng.random((m, F)) > 0.4
    keep[:, 0] = True
    for i in range(m):
        if keep[i].sum() < 2:
            keep[i, 1] = True

    A_g, b_g, syy_g = cp.asarray(A), cp.asarray(b), cp.asarray(syy)
    keep_g = cp.asarray(keep)
    beta_m, r2_m = _masked_solve(A_g, b_g, syy_g, keep_g)
    beta_m = cp.asnumpy(beta_m)
    r2_m = cp.asnumpy(r2_m)

    # Reference: per-row reduced solve on the retained sub-block only.
    for i in range(m):
        s = np.where(keep[i])[0]
        A_r = A[i][np.ix_(s, s)]
        b_r = b[i][s]
        beta_r = np.linalg.solve(A_r, b_r)
        # dropped coeffs must be exactly zero
        dropped = np.where(~keep[i])[0]
        assert np.allclose(beta_m[i, dropped], 0.0, atol=1e-12)
        # retained coeffs must match the reduced solve
        assert np.max(np.abs(beta_m[i, s] - beta_r)) < 1e-8, f"row {i}"
        # weighted R² from the reduced systems
        ssr = syy[i] - 2.0 * b_r @ beta_r + beta_r @ A_r @ beta_r
        tss = syy[i] - b[i, 0] ** 2 / A[i, 0, 0]
        r2_ref = 1.0 - ssr / tss
        assert abs(r2_m[i] - r2_ref) < 1e-8, f"row {i} R²"
