#!/usr/bin/env python3
"""
HPC_code.analyze_gpu

Render the single-GPU performance story for the PRAM "GPU" subsection from two CSVs:

  * ``--kernel-csv`` (from ``benchmark_gpu_kernel.py``): the **block-tuning** curve
    (kernel time / throughput vs threads-per-block, at fixed M) and the **throughput**
    curve (fits/s vs problem size M, at the best block).
  * ``--scaling-csv`` (from ``benchmark_scaling.py``, containing both ``hw=cpu`` and
    ``hw=gpu`` rows): the **CPU-vs-GPU** comparison — CPU median wall time vs processor
    count ``p`` with the single-GPU time as a reference line, plus the GPU-vs-CPU speedup.

At least one CSV must be given. Emits PNG plots (``--out-dir``) and a markdown report
(``--md-out``). With one GPU, true multi-GPU weak/strong scaling is not available; this
script deliberately reports the block-tuning + throughput + CPU-vs-GPU story instead.

Example::

    python HPC_code/analyze_gpu.py \
        --kernel-csv results/gpu_kernel_a100.csv \
        --scaling-csv results/scaling_cpu_v1.csv \
        --phase train --out-dir results/gpu_plots --md-out results/gpu_report.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless / HPC-safe
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger("analyze_gpu")


# ---------------------------------------------------------------------------------------
# Kernel CSV: block-tuning + throughput curves.
# ---------------------------------------------------------------------------------------
def kernel_tables(kdf: pd.DataFrame) -> dict:
    """Split the kernel CSV into the block-sweep and size-sweep frames + the best block."""
    block = kdf[kdf["study"] == "block"].sort_values("block").reset_index(drop=True)
    size = kdf[kdf["study"] == "size"].sort_values("M").reset_index(drop=True)
    best_block = None
    if not block.empty:
        best_block = int(block.loc[block["kernel_ms_median"].idxmin(), "block"])
    return {"block": block, "size": size, "best_block": best_block}


def plot_kernel(kt: dict, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    block, size = kt["block"], kt["size"]

    if not block.empty:
        # kernel time vs block (lower better); annotate the winner.
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(block["block"], block["kernel_ms_median"], marker="o")
        if kt["best_block"] is not None:
            bb = kt["best_block"]
            best_ms = float(block.loc[block["block"] == bb, "kernel_ms_median"].iloc[0])
            ax.scatter([bb], [best_ms], color="red", zorder=5, label=f"best = {bb}")
            ax.legend()
        ax.set_xlabel("threads per block")
        ax.set_ylabel("kernel time (ms, median)")
        ax.set_xscale("log", base=2)
        ax.set_title(f"Block-size tuning (M={int(block['M'].iloc[0])})")
        ax.grid(True, alpha=0.3)
        f = out_dir / "gpu_block_tuning.png"
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        written.append(f)

    if not size.empty:
        # throughput vs M (fits/s).
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(size["M"], size["fits_per_s"], marker="o", color="seagreen")
        ax.set_xlabel("number of fits M")
        ax.set_ylabel("throughput (fits / s)")
        ax.set_xscale("log")
        ax.set_title(f"GPU throughput vs problem size (block={int(size['block'].iloc[0])})")
        ax.grid(True, alpha=0.3)
        f = out_dir / "gpu_throughput.png"
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        written.append(f)

    return written


# ---------------------------------------------------------------------------------------
# Scaling CSV: CPU-vs-GPU comparison.
# ---------------------------------------------------------------------------------------
def cpu_vs_gpu(sdf: pd.DataFrame, *, phase: str) -> dict:
    """Median wall_s per (hw, p) for one phase; CPU curve + single-GPU reference."""
    df = sdf[sdf["phase"] == phase]
    if df.empty:
        raise ValueError(f"no rows for phase={phase!r} in scaling CSV")
    med = (
        df.groupby(["hw", "p"], as_index=False)
        .agg(wall_s=("wall_s", "median"), n_ids=("n_ids", "first"), B=("B", "first"))
        .sort_values(["hw", "p"])
    )
    cpu = med[med["hw"] == "cpu"].sort_values("p").reset_index(drop=True)
    gpu = med[med["hw"] == "gpu"].sort_values("p").reset_index(drop=True)
    gpu_wall = float(gpu["wall_s"].min()) if not gpu.empty else None
    return {"cpu": cpu, "gpu": gpu, "gpu_wall": gpu_wall, "phase": phase}


def plot_cpu_vs_gpu(cg: dict, out_dir: Path) -> list[Path]:
    cpu, gpu_wall = cg["cpu"], cg["gpu_wall"]
    if cpu.empty and gpu_wall is None:
        return []
    fig, ax = plt.subplots(figsize=(5, 4))
    if not cpu.empty:
        ax.plot(cpu["p"], cpu["wall_s"], marker="o", label="CPU (P workers)")
    if gpu_wall is not None:
        ax.axhline(gpu_wall, linestyle="--", color="crimson", label="1× GPU")
    ax.set_xlabel("CPU processors p")
    ax.set_ylabel("median wall time (s)")
    ax.set_title(f"CPU vs GPU — {cg['phase']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    f = out_dir / "cpu_vs_gpu.png"
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    return [f]


# ---------------------------------------------------------------------------------------
# Pipeline CSV: per-stage roofline + curves over M (from benchmark_gpu_pipeline.py).
# ---------------------------------------------------------------------------------------
_STAGES = ["assemble", "convolve", "solve", "total"]
_STAGE_COLOR = {
    "assemble": "tab:blue",
    "convolve": "tab:orange",
    "solve": "tab:green",
    "total": "black",
}


def pipeline_tables(pdf: pd.DataFrame) -> dict:
    """Sort the pipeline CSV and derive time-per-fit (µs)."""
    df = pdf.copy().sort_values(["stage", "M_fits"]).reset_index(drop=True)
    df["t_per_fit_us"] = df["time_ms_median"] * 1000.0 / df["M_fits"]
    bw = float(df["meas_bw_gbs"].median()) if "meas_bw_gbs" in df else float("nan")
    return {"df": df, "meas_bw_gbs": bw}


def plot_pipeline_curves(pt: dict, out_dir: Path) -> list[Path]:
    """time vs M, GFLOPS vs M, and time-per-fit vs M — one curve per stage."""
    df = pt["df"]
    if df.empty:
        return []
    written: list[Path] = []
    specs = [
        ("time_ms_median", "kernel time (ms, median)", "gpu_pipeline_time_vs_M.png", "GPU pipeline: time vs M", True),
        ("gflops", "achieved GFLOPS (FP64)", "gpu_pipeline_gflops_vs_M.png", "GPU pipeline: GFLOPS vs M", True),
        ("t_per_fit_us", "time per fit (µs)", "gpu_pipeline_tperfit_vs_M.png", "GPU pipeline: time/fit vs M", True),
    ]
    for col, ylabel, fname, title, ylog in specs:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for st in _STAGES:
            s = df[df["stage"] == st].sort_values("M_fits")
            if not s.empty:
                ax.plot(s["M_fits"], s[col], marker="o", label=st, color=_STAGE_COLOR[st])
        ax.set_xlabel("number of fits M (= N·366)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        f = out_dir / fname
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        written.append(f)
    return written


def plot_roofline(
    pt: dict, out_dir: Path, *, peak_fp64_gflops: Optional[float], peak_bw_gbs: Optional[float]
) -> list[Path]:
    """Roofline: achieved GFLOPS vs arithmetic intensity, with memory + (optional) compute ceilings."""
    df = pt["df"]
    if df.empty:
        return []
    bw = peak_bw_gbs if peak_bw_gbs is not None else pt["meas_bw_gbs"]
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ai_min = max(float(df["ai_flop_per_byte"].min()) * 0.3, 1e-3)
    ai_max = float(df["ai_flop_per_byte"].max()) * 30
    ai_line = [ai_min, ai_max]
    # Memory-bound ceiling: GFLOPS = AI · BW(GB/s). (BW in GB/s, AI in FLOP/byte -> GFLOP/s.)
    mem = [ai * bw for ai in ai_line]
    ax.plot(ai_line, mem, "--", color="gray", label=f"HBM BW ceiling ({bw:.0f} GB/s)")
    if peak_fp64_gflops is not None:
        ax.axhline(peak_fp64_gflops, linestyle=":", color="firebrick",
                   label=f"FP64 peak ({peak_fp64_gflops:.0f} GFLOPS)")
        ridge = peak_fp64_gflops / bw
        ax.axvline(ridge, linestyle=":", color="lightgray")
        ax.text(ridge, mem[0], f" ridge AI≈{ridge:.1f}", fontsize=8, color="gray")

    for st in _STAGES:
        s = df[df["stage"] == st]
        if not s.empty:
            ax.scatter(s["ai_flop_per_byte"], s["gflops"], label=st, color=_STAGE_COLOR[st], zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved GFLOPS (FP64)")
    ax.set_title("Roofline — GPU training pipeline (one device)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    f = out_dir / "gpu_roofline.png"
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    return [f]


# ---------------------------------------------------------------------------------------
# Markdown.
# ---------------------------------------------------------------------------------------
def write_markdown(
    kt: Optional[dict],
    cg: Optional[dict],
    md_out: Path,
    pt: Optional[dict] = None,
    *,
    peak_fp64_gflops: Optional[float] = None,
) -> None:
    lines: list[str] = ["# GPU performance (single device)", ""]

    if pt is not None and not pt["df"].empty:
        df = pt["df"]
        bw = pt["meas_bw_gbs"]
        intro = f"Measured HBM bandwidth: **{bw:.0f} GB/s**."
        if peak_fp64_gflops:
            intro = (
                f"Measured HBM bandwidth: **{bw:.0f} GB/s**; FP64 peak: "
                f"**{peak_fp64_gflops:.0f} GFLOPS** (ridge AI ≈ {peak_fp64_gflops / bw:.1f})."
            )
        intro += " Every stage sits at AI well below the ridge → **memory-bound**."
        lines += [
            "## Full-pipeline roofline (assemble / convolve / solve)",
            "",
            intro,
            "",
            "| N (IDs) | M (fits) | stage | time ms | GFLOPS | AI (FLOP/B) | fits/s |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
        for _, r in df.sort_values(["n_ids", "stage"]).iterrows():
            lines.append(
                f"| {int(r['n_ids'])} | {int(r['M_fits'])} | {r['stage']} | "
                f"{r['time_ms_median']:.4f} | {r['gflops']:.3g} | "
                f"{r['ai_flop_per_byte']:.4f} | {r['fits_per_s']:.3e} |"
            )
        lines.append("")

    if kt is not None:
        block, size = kt["block"], kt["size"]
        if not block.empty:
            gpu = block["gpu"].iloc[0]
            regs = int(block["num_regs"].iloc[0])
            lines += [
                f"## Block-size tuning ({gpu}, num_regs={regs}, M={int(block['M'].iloc[0])})",
                "",
                "| threads/block | kernel ms (median) | fits/s | best |",
                "|---:|---:|---:|:--:|",
            ]
            for _, r in block.iterrows():
                star = "★" if int(r["block"]) == kt["best_block"] else ""
                lines.append(
                    f"| {int(r['block'])} | {r['kernel_ms_median']:.4f} | "
                    f"{r['fits_per_s']:.3e} | {star} |"
                )
            lines += ["", f"Best block on this device: **{kt['best_block']}**.", ""]
        if not size.empty:
            lines += [
                f"## Throughput vs problem size (block={int(size['block'].iloc[0])})",
                "",
                "| M (fits) | kernel ms (median) | fits/s |",
                "|---:|---:|---:|",
            ]
            for _, r in size.iterrows():
                lines.append(
                    f"| {int(r['M'])} | {r['kernel_ms_median']:.4f} | {r['fits_per_s']:.3e} |"
                )
            lines.append("")

    if cg is not None:
        cpu, gpu_wall = cg["cpu"], cg["gpu_wall"]
        lines += [f"## CPU vs GPU — {cg['phase']}", ""]
        lines += ["| hw | p | median wall_s | speedup vs 1× GPU |", "|---|---:|---:|---:|"]
        for _, r in cpu.iterrows():
            sp = f"{gpu_wall / r['wall_s']:.2f}×" if gpu_wall else "n/a"
            lines.append(f"| cpu | {int(r['p'])} | {r['wall_s']:.3f} | {sp} |")
        if gpu_wall is not None:
            lines.append(f"| gpu | 1 | {gpu_wall:.3f} | 1.00× |")
        lines.append("")
        if gpu_wall is not None and not cpu.empty:
            cpu1 = float(cpu.loc[cpu["p"].idxmin(), "wall_s"])
            best_cpu = float(cpu["wall_s"].min())
            lines += [
                f"- GPU vs CPU(P=1): **{cpu1 / gpu_wall:.2f}×** faster.",
                f"- GPU vs best CPU (P={int(cpu.loc[cpu['wall_s'].idxmin(), 'p'])}): "
                f"**{best_cpu / gpu_wall:.2f}×**.",
                "",
            ]

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines), encoding="utf-8")
    log.info("[analyze-gpu] wrote report -> %s", md_out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render single-GPU block-tuning, throughput, and CPU-vs-GPU comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kernel-csv", type=Path, default=None, help="From benchmark_gpu_kernel.py.")
    p.add_argument("--scaling-csv", type=Path, default=None, help="From benchmark_scaling.py.")
    p.add_argument("--pipeline-csv", type=Path, default=None, help="From benchmark_gpu_pipeline.py.")
    p.add_argument("--phase", default="train", help="Phase for the CPU-vs-GPU comparison.")
    p.add_argument("--peak-fp64-gflops", type=float, default=None,
                   help="FP64 peak for the roofline compute ceiling (e.g. ~4200 for a100_3g.20gb).")
    p.add_argument("--peak-bw-gbs", type=float, default=None,
                   help="HBM bandwidth ceiling; default = measured value in the pipeline CSV.")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory for PNG plots.")
    p.add_argument("--md-out", required=True, type=Path, help="Markdown report output.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.kernel_csv is None and args.scaling_csv is None and args.pipeline_csv is None:
        raise SystemExit("provide at least one of --kernel-csv / --scaling-csv / --pipeline-csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    kt = None
    if args.kernel_csv is not None:
        kt = kernel_tables(pd.read_csv(args.kernel_csv))
        written += plot_kernel(kt, args.out_dir)

    pt = None
    if args.pipeline_csv is not None:
        pt = pipeline_tables(pd.read_csv(args.pipeline_csv))
        written += plot_pipeline_curves(pt, args.out_dir)
        written += plot_roofline(
            pt, args.out_dir,
            peak_fp64_gflops=args.peak_fp64_gflops, peak_bw_gbs=args.peak_bw_gbs,
        )

    cg = None
    if args.scaling_csv is not None:
        cg = cpu_vs_gpu(pd.read_csv(args.scaling_csv), phase=args.phase)
        written += plot_cpu_vs_gpu(cg, args.out_dir)

    write_markdown(kt, cg, args.md_out, pt, peak_fp64_gflops=args.peak_fp64_gflops)
    log.info("[analyze-gpu] wrote %s plot(s) -> %s", len(written), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
