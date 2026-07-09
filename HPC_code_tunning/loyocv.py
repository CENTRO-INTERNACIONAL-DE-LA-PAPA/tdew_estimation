"""Leave-one-year-out cross-validation (LOYOCV) cosine-skill primitive (GPU-batched).

The tuning objective is the multitask "cosine skill" of the SubseasonalRodeo MultiLLR
paper (arXiv:1809.07394): for each target date ``(doy, year)`` the held-out one-step
predictions across all sampled IDs form a **spatial vector**, scored by uncentred cosine
similarity against the observed anomalies. Skill for a ``(zone, doy)`` recipe is the mean
of that cosine over held-out years.

Three plan tricks make this cheap:

* **LOYOCV by subtraction** — with a ``year`` axis kept before the DOY convolution, the
  training normal equations for held-out year ``y*`` are ``A_full - A_{y*}`` (no re-scan).
  :func:`leave_one_year_normal_equations` is that subtraction; the P2 math test checks it
  equals a brute-force per-fold re-assembly.
* **One-step scoring** — held-out predictions are a gathered ``X . beta``; the recursive
  ``forecast.py`` is never used here.
* **ID-chunk additivity** — the per-target segment sums (``sum p*a``, ``sum p^2``,
  ``sum a^2``) are additive over IDs, so a large ID sample is processed in memory-bounded
  chunks and the cosine is formed once from the accumulated sums. See
  :func:`accumulate_segment_sums` / :func:`cosine_from_segments`.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

import cupy as cp

from .assemble_generic import DOY_AXIS, AssembledGeneric, convolve_doy


# ---------------------------------------------------------------------------------------
# Convolution wrapper: full (year-summed) + per-year, over the full feature superset.
# ---------------------------------------------------------------------------------------
class ConvolvedChunk:
    """Convolved Gram tensors for one ID chunk at one ``h`` (full feature superset).

    ``A_full``/``b_full``/``cnt_full`` are summed over years (all training data); the
    ``*_year`` tensors keep the year axis for the leave-one-year-out subtraction.
    """

    __slots__ = (
        "A_full", "b_full", "cnt_full", "A_year", "b_year", "cnt_year",
    )

    def __init__(self, assembled: AssembledGeneric, *, h: int, kernel: str):
        A_year, b_year, _syy_year, cnt_year = convolve_doy(
            assembled.S_xx, assembled.S_xy, assembled.S_yy, assembled.cnt,
            h=h, kernel=kernel, axis=2,
        )
        self.A_year = A_year      # [N, Y, 366, F, F]
        self.b_year = b_year      # [N, Y, 366, F]
        self.cnt_year = cnt_year  # [N, Y, 366]
        # Full = sum over the year axis (convolution is linear).
        self.A_full = A_year.sum(axis=1)      # [N, 366, F, F]
        self.b_full = b_year.sum(axis=1)      # [N, 366, F]
        self.cnt_full = cnt_year.sum(axis=1)  # [N, 366]


def leave_one_year_normal_equations(
    A_full: cp.ndarray, A_year: cp.ndarray, b_full: cp.ndarray, b_year: cp.ndarray, j: int
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Return ``(A_loo, b_loo)`` for held-out year index ``j`` by subtraction.

    ``A_full``/``b_full`` are the all-year convolved sums ``[N, 366, F, F]``/``[N, 366, F]``;
    ``A_year``/``b_year`` carry the year axis ``[N, Y, 366, ...]``. This is the identity the
    P2 test pins against a brute-force refit that excludes year ``j``.
    """
    return A_full - A_year[:, j], b_full - b_year[:, j]


# ---------------------------------------------------------------------------------------
# Batched solve that never raises on singular systems (invalid fits -> NaN, masked later).
# ---------------------------------------------------------------------------------------
def _batched_solve(A: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    """Solve a batch of SPD systems ``A[m] beta = b[m]``; NaN rows on singular A.

    Uses a single batched ``cp.linalg.solve`` on the happy path (all systems
    non-singular, which the ``min_samples`` gate makes the common case) and only falls
    back to a per-row solve when the batch contains a singular system.
    """
    m = int(A.shape[0])
    f = int(A.shape[1])
    if m == 0:
        return cp.empty((0, f), dtype=cp.float64)
    try:
        return cp.linalg.solve(A, b[..., None])[..., 0]
    except cp.linalg.LinAlgError:
        out = cp.full((m, f), cp.nan, dtype=cp.float64)
        for i in range(m):
            try:
                out[i] = cp.linalg.solve(A[i], b[i])
            except cp.linalg.LinAlgError:
                pass  # leave NaN; masked out by finiteness in the segment sums
        return out


# ---------------------------------------------------------------------------------------
# Per-target segment sums (additive across ID chunks).
# ---------------------------------------------------------------------------------------
def accumulate_segment_sums(
    assembled: AssembledGeneric,
    conv: ConvolvedChunk,
    subset: cp.ndarray,
    *,
    min_samples: int,
    spy: cp.ndarray,
    sxx: cp.ndarray,
    syy: cp.ndarray,
) -> None:
    """Add this chunk's LOYOCV cosine segment sums into ``spy``/``sxx``/``syy`` in place.

    Each is a flat ``[366 * Y]`` buffer keyed by ``doy_idx * Y + year_idx`` (the held-out
    target date). ``subset`` is the int array of retained feature indices (must include 0,
    ``const``). For every held-out sample the training system is the LOO subtraction
    restricted to ``subset``; the one-step prediction is ``X[:, subset] . beta``.
    """
    m = int(assembled.X.shape[0])
    if m == 0:
        return
    n_year = assembled.n_year
    ii, jj, dd = assembled.id_idx, assembled.year_idx, assembled.doy_idx

    # Gather per-sample LOO normal equations (never materialise a dense [N,Y,...] block).
    A_loo = conv.A_full[ii, dd] - conv.A_year[ii, jj, dd]          # [M, F, F]
    b_loo = conv.b_full[ii, dd] - conv.b_year[ii, jj, dd]          # [M, F]
    nbr_loo = conv.cnt_full[ii, dd] - conv.cnt_year[ii, jj, dd]    # [M]

    # Restrict to the candidate subset (sub-block of the assembled superset).
    A_s = A_loo[:, subset][:, :, subset]   # [M, f, f]
    b_s = b_loo[:, subset]                 # [M, f]
    X_s = assembled.X[:, subset]           # [M, f]

    valid = nbr_loo >= float(min_samples)
    beta = cp.full((m, int(subset.shape[0])), cp.nan, dtype=cp.float64)
    vidx = cp.where(valid)[0]
    if int(vidx.shape[0]) > 0:
        beta[vidx] = _batched_solve(A_s[vidx], b_s[vidx])

    pred = cp.sum(X_s * beta, axis=1)      # [M]
    actual = assembled.y                   # [M]

    good = cp.isfinite(pred) & cp.isfinite(actual)
    key = (dd.astype(cp.int64) * n_year + jj.astype(cp.int64))
    kg = key[good]
    pg = pred[good]
    ag = actual[good]
    cp.add.at(spy, kg, pg * ag)
    cp.add.at(sxx, kg, pg * pg)
    cp.add.at(syy, kg, ag * ag)


def cosine_from_segments(
    spy: cp.ndarray, sxx: cp.ndarray, syy: cp.ndarray, n_year: int
) -> cp.ndarray:
    """Form the ``[366, Y]`` uncentred cosine matrix from accumulated segment sums.

    ``cosine = sum(p*a) / sqrt(sum(p^2) * sum(a^2))``; entries with a zero-norm vector
    are NaN (mirrors ``compare_datasets._cosine``).
    """
    denom = cp.sqrt(sxx * syy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = cp.where(denom > 0, spy / denom, cp.nan)
    return cos.reshape(DOY_AXIS, n_year)


def skill_from_cosine(cos: cp.ndarray, doy_mask: Optional[Iterable[int]] = None) -> float:
    """Mean cosine over held-out years and target doys (optionally masked to some doys).

    ``cos`` is ``[366, Y]`` (doy index 0..365 == doy 1..366). ``doy_mask`` restricts to a
    set of doys (0-based indices); ``None`` uses all. NaN entries are ignored.
    """
    if doy_mask is not None:
        rows = cp.asarray(sorted(int(d) for d in doy_mask), dtype=cp.int64)
        if int(rows.shape[0]) == 0:
            return float("nan")
        sub = cos[rows]
    else:
        sub = cos
    vals = sub[cp.isfinite(sub)]
    if int(vals.shape[0]) == 0:
        return float("nan")
    return float(cp.mean(vals))


def new_segment_buffers(n_year: int) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Allocate zeroed flat ``[366 * Y]`` segment-sum buffers."""
    size = DOY_AXIS * int(n_year)
    return cp.zeros(size), cp.zeros(size), cp.zeros(size)
