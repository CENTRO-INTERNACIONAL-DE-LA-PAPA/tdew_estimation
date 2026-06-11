#!/usr/bin/env python3
"""
HPC_code.analyze_scaling

Turn the scaling CSV(s) emitted by ``benchmark_scaling.py`` into the metrics, markdown
tables, and PNG plots that fill the PRAM "Reporting" section.

Metrics (median ``wall_s`` over trials, per (dataset, hw, mode, phase, p)):
  * strong: speedup ``S(p) = T(1) / T(p)``, efficiency ``E(p) = S(p) / p``.
  * weak:   weak efficiency ``E_weak(p) = T(1, n0) / T(p, n0*p)`` (ideal == 1).

Plots, one set per (dataset, hw, mode, phase):
  * exec-time vs p
  * speedup vs p (strong) with the ideal ``S = p`` line
  * efficiency vs p

Example::

    python HPC_code/analyze_scaling.py \
        --csv results/scaling_cpu_v1.csv \
        --out-dir results/scaling_plots --md-out results/scaling_tables.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / HPC-safe
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger("analyze_scaling")

GROUP_KEYS = ["dataset", "hw", "mode", "phase"]


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def median_by_p(df: pd.DataFrame) -> pd.DataFrame:
    """Median wall_s over trials, keeping n_ids/B (constant within a (group, p))."""
    agg = (
        df.groupby(GROUP_KEYS + ["p"], as_index=False)
        .agg(wall_s=("wall_s", "median"), n_ids=("n_ids", "first"), B=("B", "first"))
        .sort_values(GROUP_KEYS + ["p"])
        .reset_index(drop=True)
    )
    return agg


def compute_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    """Add speedup/efficiency columns per group, baselined at the smallest p."""
    out = []
    for _, g in agg.groupby(GROUP_KEYS, sort=False):
        g = g.sort_values("p").copy()
        baseline = g.iloc[0]  # smallest p (normally p=1)
        t1 = float(baseline["wall_s"])
        mode = g.iloc[0]["mode"]
        if mode == "strong":
            g["speedup"] = t1 / g["wall_s"]
            g["efficiency"] = g["speedup"] / g["p"]
        else:  # weak
            # E_weak(p) = T(1, n0) / T(p, n0*p); ideal flat at 1.0
            g["speedup"] = pd.NA
            g["efficiency"] = t1 / g["wall_s"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _slug(*parts) -> str:
    return "_".join(str(p) for p in parts)


def write_markdown(metrics: pd.DataFrame, md_out: Path) -> None:
    lines: list[str] = ["# Scaling results", ""]
    for keys, g in metrics.groupby(GROUP_KEYS, sort=True):
        dataset, hw, mode, phase = keys
        g = g.sort_values("p")
        lines.append(f"## {dataset} — {hw} — {mode} — {phase}")
        lines.append("")
        if mode == "strong":
            lines.append("| p | n_ids | B | median wall_s | speedup S(p) | efficiency E(p) |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for _, r in g.iterrows():
                lines.append(
                    f"| {int(r['p'])} | {int(r['n_ids'])} | {int(r['B'])} | "
                    f"{r['wall_s']:.3f} | {r['speedup']:.3f} | {r['efficiency']:.3f} |"
                )
        else:
            lines.append("| p | n_ids | B | median wall_s | weak efficiency E_weak(p) |")
            lines.append("|---:|---:|---:|---:|---:|")
            for _, r in g.iterrows():
                lines.append(
                    f"| {int(r['p'])} | {int(r['n_ids'])} | {int(r['B'])} | "
                    f"{r['wall_s']:.3f} | {r['efficiency']:.3f} |"
                )
        lines.append("")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines), encoding="utf-8")
    log.info("[analyze] wrote tables -> %s", md_out)


def make_plots(metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for keys, g in metrics.groupby(GROUP_KEYS, sort=True):
        dataset, hw, mode, phase = keys
        g = g.sort_values("p")
        ps = g["p"].to_numpy()
        base = _slug(dataset, hw, mode, phase)

        # 1) execution time vs p
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ps, g["wall_s"], marker="o")
        ax.set_xlabel("processors p")
        ax.set_ylabel("median wall time (s)")
        ax.set_title(f"Execution time — {dataset}/{hw}/{mode}/{phase}")
        ax.grid(True, alpha=0.3)
        f1 = out_dir / f"time_{base}.png"
        fig.tight_layout()
        fig.savefig(f1, dpi=120)
        plt.close(fig)
        written.append(f1)

        # 2) speedup vs p (strong only) with ideal line
        if mode == "strong":
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(ps, g["speedup"], marker="o", label="measured")
            ax.plot(ps, ps, linestyle="--", color="gray", label="ideal S=p")
            ax.set_xlabel("processors p")
            ax.set_ylabel("speedup S(p)")
            ax.set_title(f"Speedup — {dataset}/{hw}/{phase}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            f2 = out_dir / f"speedup_{base}.png"
            fig.tight_layout()
            fig.savefig(f2, dpi=120)
            plt.close(fig)
            written.append(f2)

        # 3) efficiency vs p
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ps, g["efficiency"], marker="o")
        ax.axhline(1.0, linestyle="--", color="gray", label="ideal")
        ax.set_xlabel("processors p")
        ax.set_ylabel("weak efficiency" if mode == "weak" else "efficiency E(p)")
        ax.set_ylim(0, 1.2)
        ax.set_title(f"Efficiency — {dataset}/{hw}/{mode}/{phase}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        f3 = out_dir / f"efficiency_{base}.png"
        fig.tight_layout()
        fig.savefig(f3, dpi=120)
        plt.close(fig)
        written.append(f3)

    log.info("[analyze] wrote %s plot(s) -> %s", len(written), out_dir)
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute scaling metrics and render tables/plots from benchmark CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, nargs="+", type=Path, help="One or more CSVs.")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory for PNG plots.")
    p.add_argument("--md-out", required=True, type=Path, help="Markdown tables output.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    df = load_csvs(args.csv)
    agg = median_by_p(df)
    metrics = compute_metrics(agg)
    write_markdown(metrics, args.md_out)
    make_plots(metrics, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
