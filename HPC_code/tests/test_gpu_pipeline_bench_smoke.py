"""
Smoke test for the full-pipeline GPU roofline benchmark (benchmark_gpu_pipeline).

Runs a tiny two-N sweep on the GPU and checks the CSV schema, the four stages per N
(assemble/convolve/solve/total), and that the derived roofline quantities are sane:
GFLOPS > 0, arithmetic intensity > 0 and low (memory-bound), M = N*366, an empirical
bandwidth was measured, and per-stage flops/bytes sum to the 'total' row.

CuPy is importorskip-ed so the suite still runs on machines without a GPU.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

cp = pytest.importorskip("cupy")  # noqa: F841

_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

import benchmark_gpu_pipeline as bp  # noqa: E402


def test_pipeline_benchmark_csv(tmp_path):
    out = tmp_path / "gpu_pipeline.csv"
    rc = bp.main(
        [
            "--n-ids-list", "8,16",
            "--years", "4",
            "--reps", "2",
            "--warmup", "1",
            "--out-csv", str(out),
        ]
    )
    assert rc == 0 and out.exists()

    with out.open() as fh:
        rows = list(csv.DictReader(fh))

    assert [c for c in rows[0].keys()] == bp.CSV_COLUMNS
    # 4 stages per N, 2 Ns.
    assert len(rows) == 8
    stages = {(int(r["n_ids"]), r["stage"]) for r in rows}
    for n in (8, 16):
        for st in ("assemble", "convolve", "solve", "total"):
            assert (n, st) in stages

    for r in rows:
        assert int(r["M_fits"]) == int(r["n_ids"]) * bp.DOY
        assert float(r["time_ms_median"]) > 0
        assert float(r["gflops"]) > 0
        assert float(r["meas_bw_gbs"]) > 0
        ai = float(r["ai_flop_per_byte"])
        assert 0 < ai < 5.0  # below the A100 ridge -> memory-bound

    # 'total' flops/bytes equal the sum of the three stages (per N).
    for n in (8, 16):
        sub = {r["stage"]: r for r in rows if int(r["n_ids"]) == n}
        tot_f = sum(int(sub[s]["flops"]) for s in ("assemble", "convolve", "solve"))
        tot_b = sum(int(sub[s]["bytes"]) for s in ("assemble", "convolve", "solve"))
        assert int(sub["total"]["flops"]) == tot_f
        assert int(sub["total"]["bytes"]) == tot_b

    # solve has the highest AI of the three (fixed 5x5, least streaming).
    for n in (8, 16):
        sub = {r["stage"]: float(r["ai_flop_per_byte"]) for r in rows if int(r["n_ids"]) == n}
        assert sub["solve"] > sub["convolve"]
