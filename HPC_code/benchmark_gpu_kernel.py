#!/usr/bin/env python3
"""
HPC_code.benchmark_gpu_kernel

Single-GPU micro-benchmark for the fused WLS solve kernel (``gpu_train.solve5``). On a
one-GPU machine (e.g. KHIPU's single A100) true multi-processor weak/strong scaling
(Q = number of GPUs) is not available, so the GPU performance story is two curves this
script produces, plus the CPU-vs-GPU comparison from ``benchmark_scaling.py``:

  * **block sweep** — CUDA threads-per-block ``Q`` ∈ {32,64,128,256,512} at a fixed number
    of fits ``M``: a launch-config / occupancy-tuning curve (kernel time vs block). The
    kernel is one thread per (ID, doy) fit, so the result is identical across blocks — only
    the wall time changes. Picks the fastest block.
  * **size sweep** — fixed (best) block, growing ``M``: throughput vs problem size. On a GPU
    the CUDA *grid* grows with ``M`` while the block is fixed; the A100's SMs are the real
    parallel units. This is the honest single-GPU "scaling" view (throughput, fits/s).

Timing uses ``cupy.cuda.Event`` around the kernel (warmup + median of ``--reps``), NOT wall
clock around pandas/IO — we are measuring the kernel, not the data pipeline. Inputs are
freshly generated well-conditioned SPD 5x5 systems (the kernel's actual input); block size
provably does not change results (see tests/test_gpu_equiv.py), so synthetic systems are
fine for timing.

CSV schema (one row per measured point)::

    study, block, M, reps, kernel_ms_median, kernel_ms_min, fits_per_s, num_regs, gpu

Example::

    python HPC_code/benchmark_gpu_kernel.py \
        --m 200000 --blocks 32,64,128,256,512 \
        --m-list 1000,5000,20000,50000,100000,200000,500000 --best-block 128 \
        --reps 30 --out-csv results/gpu_kernel_a100.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

# Allow `import gpu_train` (sibling script) regardless of cwd / install state.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import cupy as cp  # noqa: E402

import gpu_train  # noqa: E402

log = logging.getLogger("benchmark_gpu_kernel")

CSV_COLUMNS = [
    "study",
    "block",
    "M",
    "reps",
    "kernel_ms_median",
    "kernel_ms_min",
    "fits_per_s",
    "num_regs",
    "gpu",
]


def _parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def make_spd_systems(m: int, *, seed: int = 0):
    """Build ``m`` well-conditioned SPD 5x5 systems (A, b, syy) on the GPU.

    ``A = R Rᵀ + 5 I`` is symmetric positive-definite, so the in-kernel Cholesky succeeds
    for every fit — the timing path matches the real (non-degenerate) workload.
    """
    rng = np.random.default_rng(seed)
    nf = gpu_train.NF
    r = rng.standard_normal((m, nf, nf))
    a = r @ np.transpose(r, (0, 2, 1)) + 5.0 * np.eye(nf)
    b = rng.standard_normal((m, nf))
    syy = rng.uniform(1.0, 10.0, size=m)
    return cp.asarray(a), cp.asarray(b), cp.asarray(syy)


def time_kernel(A, b, syy, *, block: int, reps: int, warmup: int) -> tuple[float, float]:
    """Median and min kernel time (ms) over ``reps`` launches, after ``warmup`` launches."""
    for _ in range(warmup):
        gpu_train.solve_bucket_rawkernel(A, b, syy, block=block)
    cp.cuda.Device().synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    times = []
    for _ in range(reps):
        start.record()
        gpu_train.solve_bucket_rawkernel(A, b, syy, block=block)
        end.record()
        end.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(start, end)))  # ms
    return float(np.median(times)), float(np.min(times))


def _gpu_name() -> str:
    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "unknown"


def _num_regs() -> int:
    try:
        return int(gpu_train._solve_kernel().attributes["num_regs"])
    except Exception:
        return -1


def run(args: argparse.Namespace) -> list[dict]:
    blocks = _parse_int_list(args.blocks)
    m_list = _parse_int_list(args.m_list)
    gpu = _gpu_name()
    num_regs = _num_regs()
    log.info("[gpu-kernel] device=%s  kernel num_regs=%s", gpu, num_regs)

    rows: list[dict] = []

    def _row(study: str, block: int, m: int, med: float, mn: float) -> dict:
        return {
            "study": study,
            "block": block,
            "M": m,
            "reps": args.reps,
            "kernel_ms_median": round(med, 6),
            "kernel_ms_min": round(mn, 6),
            "fits_per_s": round(m / (med / 1000.0), 1) if med > 0 else float("nan"),
            "num_regs": num_regs,
            "gpu": gpu,
        }

    # --- block sweep at fixed M ---
    A, b, syy = make_spd_systems(args.m, seed=args.seed)
    for block in blocks:
        med, mn = time_kernel(A, b, syy, block=block, reps=args.reps, warmup=args.warmup)
        rows.append(_row("block", block, args.m, med, mn))
        log.info("[block]  block=%-4d M=%-8d  %.4f ms  %.3e fits/s", block, args.m, med, rows[-1]["fits_per_s"])
    del A, b, syy

    # --- size sweep at the best block ---
    best = args.best_block
    biggest = max(m_list)
    A, b, syy = make_spd_systems(biggest, seed=args.seed + 1)
    for m in sorted(m_list):
        med, mn = time_kernel(A[:m], b[:m], syy[:m], block=best, reps=args.reps, warmup=args.warmup)
        rows.append(_row("size", best, m, med, mn))
        log.info("[size]   block=%-4d M=%-8d  %.4f ms  %.3e fits/s", best, m, med, rows[-1]["fits_per_s"])

    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("[gpu-kernel] wrote %s rows -> %s", len(rows), out_csv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-GPU micro-benchmark of the fused WLS solve kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--m", type=int, default=200000, help="Fixed #fits for the block sweep.")
    p.add_argument("--blocks", default="32,64,128,256,512", help="Threads-per-block to sweep.")
    p.add_argument(
        "--m-list",
        default="1000,5000,20000,50000,100000,200000,500000",
        help="#fits to sweep for the throughput curve.",
    )
    p.add_argument("--best-block", type=int, default=128, help="Block used for the size sweep.")
    p.add_argument("--reps", type=int, default=30, help="Timed launches per point (median).")
    p.add_argument("--warmup", type=int, default=5, help="Untimed warmup launches per point.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    rows = run(args)
    write_csv(rows, args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
