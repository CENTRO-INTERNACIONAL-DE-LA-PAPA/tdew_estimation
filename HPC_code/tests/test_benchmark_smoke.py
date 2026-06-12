"""
Fast, RAPIDS-free smoke test for the D4 benchmark/scaling harness.

Generates a tiny synthetic dataset via the real prep pipeline, runs the CPU benchmark
driver for a couple of processor counts, and checks that the CSV + analysis artifacts
are produced with the expected schema. No SLURM, no GPU.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# Make HPC_code/ importable (its modules are scripts, not an installed package).
_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

import analyze_scaling  # noqa: E402
import benchmark_scaling  # noqa: E402

EXPECTED_COLUMNS = [
    "dataset",
    "mode",
    "hw",
    "p",
    "n_ids",
    "B",
    "phase",
    "trial",
    "wall_s",
    "timestamp",
]


def test_cpu_strong_smoke(tmp_path: Path) -> None:
    base = tmp_path / "base"
    results = tmp_path / "results"
    csv_path = tmp_path / "scaling.csv"

    benchmark_scaling.main(
        [
            "--base", str(base),
            "--results", str(results),
            "--hw", "cpu",
            "--mode", "strong",
            "--p-list", "1,2",
            "--trials", "1",
            "--num-buckets", "8",
            "--phases", "train",
            "--dataset-label", "synth",
            "--synth",
            "--synth-ids", "24",
            "--train-start", "2010",
            "--train-end", "2013",
            "--min-samples", "5",
            "--out-csv", str(csv_path),
        ]
    )

    assert csv_path.exists(), "benchmark CSV was not written"
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "CSV has no data rows"
    assert list(rows[0].keys()) == EXPECTED_COLUMNS
    # one row per (p, trial, phase): p in {1,2}, 1 trial, 1 phase -> 2 rows
    assert len(rows) == 2
    ps = sorted(int(r["p"]) for r in rows)
    assert ps == [1, 2]
    assert all(float(r["wall_s"]) >= 0 for r in rows)
    assert all(int(r["n_ids"]) > 0 for r in rows)

    # Analysis stage: tables + plots
    out_dir = tmp_path / "plots"
    md_out = tmp_path / "tables.md"
    analyze_scaling.main(
        ["--csv", str(csv_path), "--out-dir", str(out_dir), "--md-out", str(md_out)]
    )
    assert md_out.exists() and md_out.read_text().strip(), "tables.md empty/missing"
    speedup_pngs = list(out_dir.glob("speedup_*.png"))
    assert speedup_pngs, "no speedup plot produced"
    assert list(out_dir.glob("efficiency_*.png")), "no efficiency plot produced"


def test_cluster_defaults_local(tmp_path: Path) -> None:
    args = benchmark_scaling.build_parser().parse_args(
        [
            "--base", str(tmp_path / "b"),
            "--results", str(tmp_path / "r"),
            "--out-csv", str(tmp_path / "x.csv"),
        ]
    )
    assert args.cluster == "local"


def test_slurm_requires_queue() -> None:
    import argparse

    ns = argparse.Namespace(cluster="slurm", slurm_queue=None)
    with pytest.raises(SystemExit):
        benchmark_scaling.make_bench_client(ns, 1)


def test_gpu_guard(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        benchmark_scaling.main(
            [
                "--base", str(tmp_path / "b"),
                "--results", str(tmp_path / "r"),
                "--hw", "gpu",
                "--p-list", "1",
                "--out-csv", str(tmp_path / "x.csv"),
            ]
        )
