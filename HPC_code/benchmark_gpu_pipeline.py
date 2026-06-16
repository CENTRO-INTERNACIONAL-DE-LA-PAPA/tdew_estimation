#!/usr/bin/env python3
"""
HPC_code.benchmark_gpu_pipeline

Single-GPU **roofline** study of the full GPU training pipeline for one bucket
(``gpu_train``): the three GPU stages that turn a bucket's rows into fitted coefficients —

  1. **assemble**  — scatter raw rows into per-(ID, doy) 5x5 sufficient statistics
                     (``S_xx, S_xy, S_yy, cnt``), the ``cp.add.at`` outer-product accumulate.
  2. **convolve**  — ``2h+1`` weighted circular rolls over the DOY axis (the local window).
  3. **solve**     — the fused one-thread-per-fit Cholesky kernel (``solve5``).

The experiment varies **N = IDs per bucket** (``--n-ids-list``); the resulting number of
fits is **M ≈ N · 366** (the x-axis the plots use). For each (N, stage) it times the GPU
work with ``cupy.cuda.Event`` (warmup + median of ``--reps``) and reports, from a documented
analytic FLOP/byte model, the achieved **GFLOPS** and **arithmetic intensity (AI)**. All three
stages have low AI (≈ 0.1–0.6 FLOP/byte) — far below the A100 roofline ridge (~5) — so the
pipeline is **memory-bandwidth bound**, which the roofline plot (``analyze_gpu.py``) shows.

The device's HBM bandwidth is also measured empirically (a large copy) and emitted, so the
roofline's memory ceiling is grounded on the actual device / MIG slice rather than a spec sheet.

CSV schema (one row per (N, stage))::

    n_ids, years, M_fits, stage, time_ms_median, flops, bytes, gflops,
    ai_flop_per_byte, fits_per_s, meas_bw_gbs, gpu

Example::

    python HPC_code/benchmark_gpu_pipeline.py \
        --n-ids-list 64,128,293,512 --years 40 --reps 10 \
        --out-csv results/gpu_pipeline_a100.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import cupy as cp  # noqa: E402

import gpu_train  # noqa: E402
from benchmark_gpu_kernel import BYTES_PER_FIT, FLOPS_PER_FIT  # noqa: E402

log = logging.getLogger("benchmark_gpu_pipeline")

DOY = gpu_train.DOY_AXIS  # 366
NF = gpu_train.NF         # 5

CSV_COLUMNS = [
    "n_ids",
    "years",
    "M_fits",
    "stage",
    "time_ms_median",
    "flops",
    "bytes",
    "gflops",
    "ai_flop_per_byte",
    "fits_per_s",
    "meas_bw_gbs",
    "gpu",
]


def _parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


# --------------------------------------------------------------------------------------- #
# Synthetic single-bucket GPU inputs (sizes match a real bucket; values only need to be
# non-degenerate). All prep that the production code does on the CPU (merge/sort/lags) is
# done here ONCE, outside the timing loop, so each stage times only its GPU work.
# --------------------------------------------------------------------------------------- #
def make_gpu_inputs(n_ids: int, years: int, *, seed: int = 0):
    """Return GPU arrays for the three stages of one synthetic bucket of ``n_ids`` IDs.

    Each ID has ``years · 366`` rows; every (ID, doy) therefore sees ``years`` samples, well
    above ``min_samples`` after the window convolution, so all ``n_ids · 366`` fits are valid
    (``M = n_ids · 366``).
    """
    rng = cp.random.RandomState(seed)
    rows_per_id = years * DOY
    R = n_ids * rows_per_id

    # Feature matrix X (R,5) with const first; target y (R,). Well-scaled so XᵀX is SPD.
    X = rng.standard_normal((R, NF))
    X[:, 0] = 1.0
    y = rng.standard_normal((R,))

    # Flat (ID, doy) bin index for the scatter.
    id_idx = cp.repeat(cp.arange(n_ids), rows_per_id)
    day_idx = cp.tile(cp.tile(cp.arange(DOY), years), n_ids)
    lin = id_idx * DOY + day_idx

    return {"X": X, "y": y, "lin": lin, "n_id": n_ids, "R": int(R)}


def assemble(inp) -> tuple:
    """Stage 1 — scatter rows into per-(ID,doy) sufficient stats (the cp.add.at accumulate)."""
    n_id = inp["n_id"]
    X, y, lin = inp["X"], inp["y"], inp["lin"]
    S_xx = cp.zeros((n_id * DOY, NF, NF))
    S_xy = cp.zeros((n_id * DOY, NF))
    S_yy = cp.zeros((n_id * DOY,))
    cnt = cp.zeros((n_id * DOY,))
    xx = X[:, :, None] * X[:, None, :]
    cp.add.at(S_xx, lin, xx)
    cp.add.at(S_xy, lin, X * y[:, None])
    cp.add.at(S_yy, lin, y * y)
    cp.add.at(cnt, lin, cp.ones(X.shape[0]))
    return (
        S_xx.reshape(n_id, DOY, NF, NF),
        S_xy.reshape(n_id, DOY, NF),
        S_yy.reshape(n_id, DOY),
        cnt.reshape(n_id, DOY),
    )


def time_gpu(fn, *, reps: int, warmup: int) -> float:
    """Median GPU time (ms) of ``fn`` over ``reps`` launches after ``warmup`` (CUDA events)."""
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(start, end)))
    return float(np.median(times))


def measure_bw_gbs(*, mb: int = 256, reps: int = 20) -> float:
    """Empirical HBM bandwidth (GB/s) from a large device-to-device copy (read+write)."""
    n = (mb * 1024 * 1024) // 8
    src = cp.ones(n, dtype=cp.float64)
    dst = cp.empty_like(src)
    ms = time_gpu(lambda: cp.copyto(dst, src), reps=reps, warmup=3)
    bytes_moved = 2 * src.nbytes  # read src + write dst
    return float(bytes_moved / (ms / 1000.0) / 1e9)


# --------------------------------------------------------------------------------------- #
# Analytic FLOP / byte model per stage (float64, 8 B). Constants documented inline; the
# qualitative conclusion (low AI -> memory-bound) is robust to the exact counts.
# --------------------------------------------------------------------------------------- #
def assemble_cost(R: int) -> tuple[int, int]:
    """FLOPs, bytes for the scatter-assemble over ``R`` rows.

    Per row: outer product XᵀX (25 mul) + Xy (5 mul) + y² (1) ≈ 31 mul, plus the scatter
    read-modify-write adds into 25+5+1+1 = 32 accumulators. Bytes: atomic RMW of those 32
    doubles (read+write) + reading the 5 feature values ≈ (32·2 + 5)·8.
    """
    flops = 63 * R
    nbytes = (32 * 2 + 5) * 8 * R
    return flops, nbytes


def convolve_cost(n_id: int, h: int) -> tuple[int, int]:
    """FLOPs, bytes for the ``2h+1`` weighted rolls over [n_id,366,(25+5+1)] tensors.

    Per shift: mul+add over the 31 stat channels of every (ID,doy) cell. Bytes: each shift
    reads the source + reads/writes the accumulator (~3 touches) of those 31 channels.
    """
    cells = n_id * DOY
    shifts = 2 * h + 1
    flops = shifts * cells * 31 * 2
    nbytes = shifts * cells * 31 * 3 * 8
    return flops, nbytes


def solve_cost(m_fits: int) -> tuple[int, int]:
    """FLOPs, bytes for the solve kernel over ``m_fits`` fits (reuse the kernel model)."""
    return FLOPS_PER_FIT * m_fits, BYTES_PER_FIT * m_fits


def run(args: argparse.Namespace) -> list[dict]:
    gpu = _gpu_name()
    bw = measure_bw_gbs()
    log.info("[gpu-pipeline] device=%s  measured HBM BW=%.1f GB/s", gpu, bw)

    n_list = _parse_int_list(args.n_ids_list)
    h, kernel = args.h, args.kernel
    rows: list[dict] = []

    def _row(n_ids, m_fits, stage, ms, flops, nbytes) -> dict:
        sec = ms / 1000.0
        return {
            "n_ids": n_ids,
            "years": args.years,
            "M_fits": m_fits,
            "stage": stage,
            "time_ms_median": round(ms, 6),
            "flops": int(flops),
            "bytes": int(nbytes),
            "gflops": round(flops / sec / 1e9, 3) if ms > 0 else float("nan"),
            "ai_flop_per_byte": round(flops / nbytes, 4) if nbytes else float("nan"),
            "fits_per_s": round(m_fits / sec, 1) if ms > 0 else float("nan"),
            "meas_bw_gbs": round(bw, 1),
            "gpu": gpu,
        }

    for n_ids in n_list:
        inp = make_gpu_inputs(n_ids, args.years, seed=args.seed)
        R, m_fits = inp["R"], n_ids * DOY

        # Stage 1: assemble (scatter).
        t_asm = time_gpu(lambda: assemble(inp), reps=args.reps, warmup=args.warmup)
        f, by = assemble_cost(R)
        rows.append(_row(n_ids, m_fits, "assemble", t_asm, f, by))

        # Pre-assemble once for the next two stages (not timed here).
        S_xx, S_xy, S_yy, cnt = assemble(inp)

        # Stage 2: convolve.
        t_conv = time_gpu(
            lambda: gpu_train._circular_convolve(S_xx, S_xy, S_yy, cnt, h=h, kernel=kernel),
            reps=args.reps, warmup=args.warmup,
        )
        f, by = convolve_cost(n_ids, h)
        rows.append(_row(n_ids, m_fits, "convolve", t_conv, f, by))

        # Stage 3: solve (all fits valid -> gather is the full set).
        A, b, Syy_w, _ = gpu_train._circular_convolve(S_xx, S_xy, S_yy, cnt, h=h, kernel=kernel)
        A_v = A.reshape(n_ids * DOY, NF, NF)
        b_v = b.reshape(n_ids * DOY, NF)
        syy_v = Syy_w.reshape(n_ids * DOY)
        t_solve = time_gpu(
            lambda: gpu_train.solve_bucket_rawkernel(A_v, b_v, syy_v, block=args.block),
            reps=args.reps, warmup=args.warmup,
        )
        f, by = solve_cost(m_fits)
        rows.append(_row(n_ids, m_fits, "solve", t_solve, f, by))

        # Total = sum of the three stages.
        t_tot = t_asm + t_conv + t_solve
        f_a, by_a = assemble_cost(R)
        f_c, by_c = convolve_cost(n_ids, h)
        f_s, by_s = solve_cost(m_fits)
        rows.append(_row(n_ids, m_fits, "total", t_tot, f_a + f_c + f_s, by_a + by_c + by_s))

        log.info(
            "[pipeline] N=%-5d M=%-7d  assemble %.3f ms | convolve %.3f ms | solve %.3f ms",
            n_ids, m_fits, t_asm, t_conv, t_solve,
        )
        del S_xx, S_xy, S_yy, cnt, A, b, Syy_w, A_v, b_v, syy_v, inp
        cp.get_default_memory_pool().free_all_blocks()

    return rows


def _gpu_name() -> str:
    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "unknown"


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("[gpu-pipeline] wrote %s rows -> %s", len(rows), out_csv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-GPU roofline of the full bucket pipeline (assemble/convolve/solve).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-ids-list", default="64,128,293,512", help="IDs/bucket to sweep (=> M=N*366).")
    p.add_argument("--years", type=int, default=40, help="Years of daily data per ID.")
    p.add_argument("--h", type=int, default=11, help="DOY half-window (sets 2h+1 convolution shifts).")
    p.add_argument("--kernel", default="Tricube", help="Window kernel (Tricube|Gaussian).")
    p.add_argument("--block", type=int, default=128, help="Threads-per-block for the solve kernel.")
    p.add_argument("--reps", type=int, default=10, help="Timed launches per stage (median).")
    p.add_argument("--warmup", type=int, default=3, help="Untimed warmup launches per stage.")
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
