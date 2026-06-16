"""
Smoke test for the single-GPU kernel micro-benchmark (benchmark_gpu_kernel.py).

Runs a tiny block + size sweep and checks the CSV is well-formed: both studies present,
every swept block/M represented, and finite positive throughput. Skipped without CuPy/CUDA.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

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

import benchmark_gpu_kernel as bk  # noqa: E402


def test_kernel_benchmark_csv(tmp_path):
    out = tmp_path / "k.csv"
    rc = bk.main(
        [
            "--m", "2000",
            "--blocks", "64,128",
            "--m-list", "500,2000",
            "--best-block", "128",
            "--reps", "3",
            "--warmup", "1",
            "--out-csv", str(out),
        ]
    )
    assert rc == 0 and out.exists()

    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [c for c in rows[0].keys()] == bk.CSV_COLUMNS

    block_rows = [r for r in rows if r["study"] == "block"]
    size_rows = [r for r in rows if r["study"] == "size"]
    assert {int(r["block"]) for r in block_rows} == {64, 128}
    assert {int(r["M"]) for r in size_rows} == {500, 2000}
    assert all(int(r["M"]) == 2000 for r in block_rows)  # block sweep at fixed M

    for r in rows:
        assert float(r["fits_per_s"]) > 0
        assert float(r["kernel_ms_median"]) > 0
        assert int(r["num_regs"]) > 0
        assert float(r["gflops"]) > 0
        assert float(r["t_per_fit_us"]) > 0
        assert float(r["ai_flop_per_byte"]) == pytest.approx(
            bk.FLOPS_PER_FIT / bk.BYTES_PER_FIT, abs=1e-3
        )
