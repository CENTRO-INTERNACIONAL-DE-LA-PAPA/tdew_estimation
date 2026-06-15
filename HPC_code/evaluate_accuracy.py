#!/usr/bin/env python3
"""
HPC_code.evaluate_accuracy

Score TDEW forecast **skill against observed TD** — the ground-truth axis that
``compare_datasets.py`` deliberately omits (it only diffs two prediction sets against each
other). For each model run this loads its predictions, aligns them with the *observed* TD
for the same (ID, FECHA), and reports RMSE / MAE / bias / Pearson r (overall and per month).
Given two runs (e.g. PISCOt v1 vs v2) it tabulates both and says which is more skilful.

Use it on a **held-out** window: train coeffs on years ≤ Y, forecast Y+1.., then evaluate
the forecast against the observed TD of Y+1.. (the observed values were never used to fit
those coeffs). That turns the D5 v1-vs-v2 comparison from "the coefficients differ" into
"version X predicts held-out TD better."

Inputs are layout-agnostic (single parquet or a directory tree), reusing
``compare_datasets.load_dataset``:
  * predictions: columns ``ID, FECHA, TD_predicted`` (per-bucket ``pred.parquet`` is fine),
  * observed:    raw TD monthly parquet ``ID, FECHA, Value`` (``{base}/td/Outputs/...``) or
    any parquet with those columns; ``--obs-value`` names the value column (default Value).

Outputs: a markdown report (``--md-out``) and PNG plots (``--out-dir``) — residual
histogram and predicted-vs-observed scatter per run, and a per-month RMSE bar chart.

Example::

    python HPC_code/evaluate_accuracy.py \
        --pred-a /data/run_v1/results/predictions \
        --pred-b /data/run_v2/results/predictions \
        --obs /data/base/td/Outputs --obs-value Value \
        --label-a v1 --label-b v2 \
        --out-dir results/accuracy_plots --md-out results/accuracy_v1_v2.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless / HPC-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Allow `import compare_datasets` (sibling script) regardless of cwd / install state.
_HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from compare_datasets import _safe_corr, load_dataset  # noqa: E402

log = logging.getLogger("evaluate_accuracy")

PRED_VALUE = "TD_predicted"
KEYS = ["ID", "FECHA"]


def load_observed(path: Path, *, value_col: str = "Value", label: str = "obs") -> pd.DataFrame:
    """Load observed TD as columns ``ID, FECHA, td_obs`` from a parquet file or dir tree."""
    df = load_dataset(path, label=label)
    if value_col not in df.columns:
        raise ValueError(f"[{label}] observed value column {value_col!r} not in {list(df.columns)}")
    out = df[["ID", "FECHA", value_col]].rename(columns={value_col: "td_obs"}).copy()
    out["FECHA"] = pd.to_datetime(out["FECHA"])
    return out.drop_duplicates(KEYS)


def score(pred: pd.DataFrame, obs: pd.DataFrame, *, label: str) -> dict:
    """Skill of one prediction set vs observed TD. Returns overall metrics + monthly frame.

    Error convention: ``error = predicted − observed`` (positive bias = over-prediction).
    """
    if PRED_VALUE not in pred.columns:
        raise ValueError(f"[{label}] predictions missing {PRED_VALUE!r}")
    p = pred[KEYS + [PRED_VALUE]].copy()
    p["FECHA"] = pd.to_datetime(p["FECHA"])
    p = p.drop_duplicates(KEYS)

    merged = p.merge(obs, on=KEYS, how="inner")
    pv = merged[PRED_VALUE].to_numpy(dtype=float)
    ov = merged["td_obs"].to_numpy(dtype=float)
    err = pv - ov
    m = np.isfinite(err)
    overall = {
        "label": label,
        "n_pred": int(len(p)),
        "n_scored": int(m.sum()),
        "rmse": float(np.sqrt(np.nanmean(err[m] ** 2))) if m.any() else float("nan"),
        "mae": float(np.nanmean(np.abs(err[m]))) if m.any() else float("nan"),
        "bias": float(np.nanmean(err[m])) if m.any() else float("nan"),
        "pearson_r": _safe_corr(pv, ov),
    }

    merged = merged.assign(err=err)
    merged["month"] = merged["FECHA"].dt.month
    monthly = (
        merged.groupby("month")
        .apply(lambda g: pd.Series({
            "rmse": float(np.sqrt(np.nanmean(g["err"].to_numpy() ** 2))),
            "mae": float(np.nanmean(np.abs(g["err"].to_numpy()))),
            "bias": float(np.nanmean(g["err"].to_numpy())),
        }), include_groups=False)
        .reset_index()
        .sort_values("month")
    )
    return {"overall": overall, "monthly": monthly, "merged": merged}


# ---------------------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------------------
def write_markdown(results: list[dict], md_out: Path) -> None:
    lines: list[str] = ["# Forecast accuracy vs observed TD", ""]

    lines += ["## Overall skill", "", "| run | n scored | RMSE | MAE | bias (pred−obs) | Pearson r |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for res in results:
        o = res["overall"]
        lines.append(
            f"| {o['label']} | {o['n_scored']} | {o['rmse']:.4g} | {o['mae']:.4g} | "
            f"{o['bias']:.4g} | {o['pearson_r']:.4f} |"
        )
    lines.append("")

    if len(results) == 2:
        a, b = results[0]["overall"], results[1]["overall"]
        better = a["label"] if a["rmse"] <= b["rmse"] else b["label"]
        d_rmse = b["rmse"] - a["rmse"]
        lines += [
            f"**Lower RMSE: `{better}`.** "
            f"Δ RMSE ({b['label']}−{a['label']}) = {d_rmse:+.4g} "
            f"(ΔMAE = {b['mae'] - a['mae']:+.4g}, Δbias = {b['bias'] - a['bias']:+.4g}).",
            "",
        ]

    for res in results:
        o = res["overall"]
        lines += [f"## Monthly skill — {o['label']}", "", "| month | RMSE | MAE | bias |", "|---:|---:|---:|---:|"]
        for _, r in res["monthly"].iterrows():
            lines.append(
                f"| {int(r['month'])} | {r['rmse']:.4g} | {r['mae']:.4g} | {r['bias']:.4g} |"
            )
        lines.append("")

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines), encoding="utf-8")
    log.info("[accuracy] wrote report -> %s", md_out)


def make_plots(results: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for res in results:
        label = res["overall"]["label"]
        merged = res["merged"]
        err = merged["err"].to_numpy(dtype=float)
        err = err[np.isfinite(err)]
        if err.size:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(err, bins=60, color="slateblue", alpha=0.85)
            ax.axvline(0.0, linestyle="--", color="gray")
            ax.set_xlabel("error (predicted − observed)")
            ax.set_ylabel("count")
            ax.set_title(f"Residuals — {label}")
            ax.grid(True, alpha=0.3)
            f = out_dir / f"residual_hist_{label}.png"
            fig.tight_layout()
            fig.savefig(f, dpi=120)
            plt.close(fig)
            written.append(f)

        pv = merged[PRED_VALUE].to_numpy(dtype=float)
        ov = merged["td_obs"].to_numpy(dtype=float)
        mfin = np.isfinite(pv) & np.isfinite(ov)
        if mfin.any():
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(ov[mfin], pv[mfin], s=4, alpha=0.3, color="teal")
            lo = float(min(ov[mfin].min(), pv[mfin].min()))
            hi = float(max(ov[mfin].max(), pv[mfin].max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="y = x")
            ax.set_xlabel("observed TD")
            ax.set_ylabel("predicted TD")
            ax.set_title(f"Predicted vs observed — {label}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            f = out_dir / f"pred_vs_obs_{label}.png"
            fig.tight_layout()
            fig.savefig(f, dpi=120)
            plt.close(fig)
            written.append(f)

    # Per-month RMSE, runs side by side.
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / max(len(results), 1)
    for i, res in enumerate(results):
        mo = res["monthly"]
        ax.bar(mo["month"] + i * width, mo["rmse"], width=width, label=res["overall"]["label"])
    ax.set_xlabel("month")
    ax.set_ylabel("RMSE")
    ax.set_title("Monthly RMSE")
    ax.set_xticks(range(1, 13))
    ax.legend()
    ax.grid(True, alpha=0.3)
    f = out_dir / "monthly_rmse.png"
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    written.append(f)

    log.info("[accuracy] wrote %s plot(s) -> %s", len(written), out_dir)
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score TDEW forecast skill against observed TD (one or two runs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-a", required=True, type=Path, help="Run A predictions (file or dir).")
    p.add_argument("--pred-b", type=Path, default=None, help="Run B predictions (file or dir).")
    p.add_argument("--obs", required=True, type=Path, help="Observed TD (file or dir).")
    p.add_argument("--obs-value", default="Value", help="Observed value column name.")
    p.add_argument("--label-a", default="v1")
    p.add_argument("--label-b", default="v2")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory for PNG plots.")
    p.add_argument("--md-out", required=True, type=Path, help="Markdown report output.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    obs = load_observed(args.obs, value_col=args.obs_value)
    results = [score(load_dataset(args.pred_a, label=args.label_a), obs, label=args.label_a)]
    if args.pred_b is not None:
        results.append(score(load_dataset(args.pred_b, label=args.label_b), obs, label=args.label_b))

    write_markdown(results, args.md_out)
    make_plots(results, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
