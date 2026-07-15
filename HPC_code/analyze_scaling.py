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
import numpy as np  # noqa: E402
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


# ---------------------------------------------------------------------------------------
# Theoretical reference: the memory-bandwidth-contention model (derived in
# tdew_estimation_pram.qmd, "Theoretical Efficiency" section). Each unit of work needs
# compute time t_c and moves q bytes over the memory bus shared by all p cores, so
#   T_p ≈ W · (t_c / p + q / β)
# and, with γ = (q/β) / t_c (memory-time : compute-time ratio per unit of work),
#   S(p) = p(1+γ)/(1+γp),  E(p) = (1+γ)/(1+γp) = O(1/p),  S(∞) = 1 + 1/γ.
# The problem size N cancels, so a single γ describes every N — which is why the
# per-N efficiency curves superimpose.
# ---------------------------------------------------------------------------------------
def theory_efficiency(p, gamma: float):
    """E(p) = (1+γ)/(1+γp) for the shared-bandwidth contention model."""
    return (1.0 + gamma) / (1.0 + gamma * np.asarray(p, dtype=float))


def fit_gamma(p, eff) -> float:
    """Least-squares fit of γ in E(p)=(1+γ)/(1+γp) over pooled (p, efficiency) points."""
    p = np.asarray(p, dtype=float)
    eff = np.asarray(eff, dtype=float)
    grid = np.geomspace(1e-4, 10.0, 400)
    best = grid[0]
    for _ in range(3):  # coarse-to-fine 1-D search (keeps us scipy-free)
        sse = np.array([np.square(eff - theory_efficiency(p, g)).sum() for g in grid])
        i = int(np.argmin(sse))
        best = float(grid[i])
        grid = np.linspace(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)], 400)
    return best


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
        if mode == "strong" and len(ps) >= 3:
            gamma = fit_gamma(ps, g["efficiency"])
            pg = np.geomspace(ps.min(), ps.max(), 128)
            ax.plot(pg, theory_efficiency(pg, gamma), "--", color="red",
                    label=rf"theory $E=(1+\gamma)/(1+\gamma p)$, $\gamma$={gamma:.3f} $\to O(1/p)$")
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


# ---------------------------------------------------------------------------------------
# Family view (--by-size): one curve per problem size N, over the processor axis p.
# Fix N, sweep p; increase N, sweep p again -> a family. Weak scaling is then the locus
# across curves where work-per-worker N/p is constant.
# ---------------------------------------------------------------------------------------
FAMILY_KEYS = ["dataset", "hw", "phase"]


def family_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Strong-mode rows, per (dataset,hw,phase,B): median wall_s + speedup/efficiency vs p."""
    d = df[df["mode"] == "strong"]
    if d.empty:
        raise ValueError("--by-size needs strong-mode rows (multiple --num-buckets N).")
    g = (d.groupby(FAMILY_KEYS + ["B", "p"], as_index=False)
           .agg(wall_s=("wall_s", "median"), n_ids=("n_ids", "first")))
    out = []
    for _, s in g.groupby(FAMILY_KEYS + ["B"], sort=True):
        s = s.sort_values("p").copy()
        t1 = float(s["wall_s"].iloc[0])  # smallest p
        s["speedup"] = t1 / s["wall_s"]
        s["efficiency"] = s["speedup"] / s["p"]
        out.append(s)
    return pd.concat(out, ignore_index=True)


def make_family_plots(fam: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Time / Speedup / Efficiency vs p, one line per N (=B), per (dataset,hw,phase)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for keys, g in fam.groupby(FAMILY_KEYS, sort=True):
        dataset, hw, phase = keys
        sizes = sorted(g["B"].unique())
        colors = plt.cm.viridis([i / max(len(sizes) - 1, 1) * 0.7 + 0.15 for i in range(len(sizes))])
        base = _slug(dataset, hw, phase)
        # One γ fitted over ALL problem sizes at once: the contention model is N-free,
        # so a single curve should describe the whole family.
        gamma = fit_gamma(g["p"], g["efficiency"]) if len(g) >= 3 else None

        def _label(B):
            nid = int(g[g.B == B]["n_ids"].iloc[0])
            return f"N={nid} IDs (B={int(B)})"

        specs = [
            ("wall_s", "median wall time (s)", "log", "Time vs p", f"family_time_{base}.png", None),
            ("speedup", "speedup S(p)=T(N,1)/T(N,p)", "log", "Speedup vs p", f"family_speedup_{base}.png", "ideal"),
            ("efficiency", "efficiency E(p)=S/p", "linear", "Efficiency vs p", f"family_efficiency_{base}.png", "one"),
        ]
        for col, ylabel, yscale, title, fname, ideal in specs:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            pmax = int(g["p"].max())
            if ideal == "ideal":
                ax.plot([1, pmax], [1, pmax], "--", color="gray", label="ideal S=p")
            elif ideal == "one":
                ax.axhline(1.0, linestyle="--", color="gray", label="ideal E=1")
            for B, c in zip(sizes, colors):
                s = g[g.B == B].sort_values("p")
                ax.plot(s["p"], s[col], marker="o", color=c, label=_label(B))
            if gamma is not None and col in ("speedup", "efficiency"):
                pg = np.geomspace(1, pmax, 128)
                if col == "efficiency":
                    ax.plot(pg, theory_efficiency(pg, gamma), "--", color="red",
                            label=rf"theory $E=(1+\gamma)/(1+\gamma p)$, $\gamma$={gamma:.3f} $\to O(1/p)$")
                else:
                    ax.plot(pg, pg * theory_efficiency(pg, gamma), "--", color="red",
                            label=rf"theory $S(p)$, $S(\infty)=1+1/\gamma \approx${1 + 1 / gamma:.1f}")
            ax.set_xscale("log", base=2)
            if yscale == "log":
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1.15)
            ax.set_xlabel("processors p")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {dataset}/{hw}/{phase} (per N)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            f = out_dir / fname
            fig.tight_layout()
            fig.savefig(f, dpi=120)
            plt.close(fig)
            written.append(f)
    log.info("[analyze] wrote %s family plot(s) -> %s", len(written), out_dir)
    return written


def write_family_markdown(fam: pd.DataFrame, md_out: Path) -> None:
    lines: list[str] = ["# Scaling results — family by problem size N", ""]
    for keys, g in fam.groupby(FAMILY_KEYS, sort=True):
        dataset, hw, phase = keys
        lines.append(f"## {dataset} — {hw} — {phase}")
        for B, s in g.groupby("B", sort=True):
            s = s.sort_values("p")
            nid = int(s["n_ids"].iloc[0])
            lines += ["", f"### N = {nid} IDs (B={int(B)})", "",
                      "| p | wall_s | S(p) | E(p) |", "|---:|---:|---:|---:|"]
            for _, r in s.iterrows():
                lines.append(f"| {int(r['p'])} | {r['wall_s']:.2f} | {r['speedup']:.2f} | {r['efficiency']:.3f} |")
        # weak-scaling diagonal: constant work-per-worker n0 = B/p
        piv = g.pivot_table(index="B", columns="p", values="wall_s")
        sizes = sorted(g["B"].unique()); ps = sorted(int(p) for p in g["p"].unique())
        lines += ["", "### Weak diagonals (constant N/p): E_weak(p)=T(n0,1)/T(n0·p,p)", ""]
        for n0 in sizes:
            pts = [(p, n0 * p, float(piv.loc[n0 * p, p])) for p in ps
                   if (n0 * p) in piv.index and p in piv.columns and not pd.isna(piv.loc[n0 * p, p])]
            if len(pts) < 2:
                continue
            t0 = pts[0][2]
            chain = ", ".join(f"p={p}(B={B}) E={t0/t:.3f}" for p, B, t in pts)
            lines.append(f"- n0={n0} buckets/worker → {chain}")
        lines.append("")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines), encoding="utf-8")
    log.info("[analyze] wrote family tables -> %s", md_out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute scaling metrics and render tables/plots from benchmark CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, nargs="+", type=Path, help="One or more CSVs.")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory for PNG plots.")
    p.add_argument("--md-out", required=True, type=Path, help="Markdown tables output.")
    p.add_argument(
        "--by-size",
        action="store_true",
        help="Family view: one curve per problem size N (=B), over p, for Time/Speedup/Efficiency "
        "(use with a CSV containing strong-mode runs at several --num-buckets). Weak scaling is "
        "read off the constant-N/p diagonals.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    df = load_csvs(args.csv)
    if args.by_size:
        fam = family_metrics(df)
        write_family_markdown(fam, args.md_out)
        make_family_plots(fam, args.out_dir)
        return 0
    agg = median_by_p(df)
    metrics = compute_metrics(agg)
    write_markdown(metrics, args.md_out)
    make_plots(metrics, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
