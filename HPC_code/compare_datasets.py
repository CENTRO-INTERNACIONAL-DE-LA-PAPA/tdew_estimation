#!/usr/bin/env python3
"""
HPC_code.compare_datasets

D5 — compare two TDEW datasets produced by running the *same* pipeline on two PISCOt
input versions (e.g. v1 vs v2). The pipeline is run twice into separate ``--results``
roots that differ only in the input variables (``--base`` / ``--td-var`` / ``--tmin-var``);
this tool then loads the two coefficient datasets (and, optionally, the two prediction
datasets), aligns them on their natural keys, and reports the distribution differences
that fill the PRAM "Datasets: v1 vs v2" subsection.

What it reports
---------------
Coefficients (aligned on ``(ID, doy)``):
  * coverage — fits present in A only / B only / both,
  * per-coefficient summary stats (mean/std/min/max) for each dataset,
  * paired Δ = B − A on the common fits (mean, std, mean|Δ|, max|Δ|, Pearson r),
    one row per coefficient incl. ``r_squared_anom``.
Predictions (aligned on ``(ID, FECHA)``, optional):
  * coverage,
  * agreement of ``TD_predicted`` — RMSE, MAE, bias (mean B−A), Pearson r,
  * monthly mean |Δ| (seasonality of the disagreement).

Outputs: a markdown report (``--md-out``) and PNG plots (``--out-dir``) — Δ histograms
per coefficient, an R² scatter, and (with predictions) a TD_predicted Δ histogram and a
monthly mean-|Δ| bar chart.

Inputs are layout-agnostic: each ``--coeffs-*`` / ``--pred-*`` argument may be a single
combined parquet file or a directory (e.g. a bucketed root
``id_bucket=XXXX/coeffs.parquet`` / ``.../pred.parquet``); directories are read by
recursively concatenating every ``*.parquet`` under them.

Example::

    python HPC_code/compare_datasets.py \
        --coeffs-a /data/run_v1/results/llr_coeffs_anomaly_dataset \
        --coeffs-b /data/run_v2/results/llr_coeffs_anomaly_dataset \
        --pred-a   /data/run_v1/results/predictions \
        --pred-b   /data/run_v2/results/predictions \
        --label-a v1 --label-b v2 \
        --out-dir results/compare_plots --md-out results/compare_v1_v2.md
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

log = logging.getLogger("compare_datasets")

# Coefficient columns compared per (ID, doy). Matches the training output schema
# (tdew_estimation.anomaly_train.fit_anomaly_coeffs_for_prepared_id).
COEFF_METRICS = [
    "const_anom",
    "TMIN_anom_coeff",
    "TD_anom_lag1",
    "TD_anom_lag2",
    "TMIN_anom_lag1",
    "r_squared_anom",
]
COEFF_KEYS = ["ID", "doy"]

PRED_VALUE = "TD_predicted"
PRED_KEYS = ["ID", "FECHA"]


# ---------------------------------------------------------------------------------------
# Loading (layout-agnostic: single parquet file or a directory tree of parquet shards).
# ---------------------------------------------------------------------------------------
def load_dataset(path: Path, *, label: str) -> pd.DataFrame:
    """Load a dataset from a single parquet file or a directory of parquet shards."""
    path = Path(path)
    if path.is_file():
        df = pd.read_parquet(path)
    elif path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"[{label}] no *.parquet found under {path}")
        df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    else:
        raise FileNotFoundError(f"[{label}] path not found: {path}")
    log.info("[%s] loaded %s rows from %s", label, len(df), path)
    return df


# ---------------------------------------------------------------------------------------
# Coefficient comparison.
# ---------------------------------------------------------------------------------------
def compare_coeffs(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
) -> dict:
    """Compare two coefficient datasets aligned on ``(ID, doy)``.

    Returns a dict with ``coverage`` (counts), ``summary`` (per-dataset per-coeff stats),
    ``delta`` (paired B−A stats per coeff), and ``merged`` (the inner-joined frame for
    plotting).
    """
    for name, df in ((label_a, a), (label_b, b)):
        missing = [c for c in COEFF_KEYS + COEFF_METRICS if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] coeffs missing columns: {missing}")

    a = a[COEFF_KEYS + COEFF_METRICS].drop_duplicates(COEFF_KEYS)
    b = b[COEFF_KEYS + COEFF_METRICS].drop_duplicates(COEFF_KEYS)

    keys_a = set(map(tuple, a[COEFF_KEYS].to_numpy()))
    keys_b = set(map(tuple, b[COEFF_KEYS].to_numpy()))
    coverage = {
        f"{label_a}_total": len(a),
        f"{label_b}_total": len(b),
        "common": len(keys_a & keys_b),
        f"{label_a}_only": len(keys_a - keys_b),
        f"{label_b}_only": len(keys_b - keys_a),
    }

    summary_rows = []
    for lbl, df in ((label_a, a), (label_b, b)):
        for col in COEFF_METRICS:
            s = df[col].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "dataset": lbl,
                    "coeff": col,
                    "mean": np.nanmean(s),
                    "std": np.nanstd(s),
                    "min": np.nanmin(s),
                    "max": np.nanmax(s),
                }
            )
    summary = pd.DataFrame(summary_rows)

    merged = a.merge(b, on=COEFF_KEYS, suffixes=(f"_{label_a}", f"_{label_b}"))
    delta_rows = []
    for col in COEFF_METRICS:
        va = merged[f"{col}_{label_a}"].to_numpy(dtype=float)
        vb = merged[f"{col}_{label_b}"].to_numpy(dtype=float)
        d = vb - va
        delta_rows.append(
            {
                "coeff": col,
                "mean_delta": np.nanmean(d),
                "std_delta": np.nanstd(d),
                "mean_abs_delta": np.nanmean(np.abs(d)),
                "max_abs_delta": np.nanmax(np.abs(d)),
                "pearson_r": _safe_corr(va, vb),
            }
        )
    delta = pd.DataFrame(delta_rows)

    return {"coverage": coverage, "summary": summary, "delta": delta, "merged": merged}


# ---------------------------------------------------------------------------------------
# Prediction comparison.
# ---------------------------------------------------------------------------------------
def compare_predictions(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
) -> dict:
    """Compare two prediction datasets aligned on ``(ID, FECHA)`` for ``TD_predicted``."""
    for name, df in ((label_a, a), (label_b, b)):
        missing = [c for c in PRED_KEYS + [PRED_VALUE] if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] predictions missing columns: {missing}")

    a = a[PRED_KEYS + [PRED_VALUE]].copy()
    b = b[PRED_KEYS + [PRED_VALUE]].copy()
    a["FECHA"] = pd.to_datetime(a["FECHA"])
    b["FECHA"] = pd.to_datetime(b["FECHA"])
    a = a.drop_duplicates(PRED_KEYS)
    b = b.drop_duplicates(PRED_KEYS)

    keys_a = set(map(tuple, a[PRED_KEYS].to_numpy()))
    keys_b = set(map(tuple, b[PRED_KEYS].to_numpy()))
    coverage = {
        f"{label_a}_total": len(a),
        f"{label_b}_total": len(b),
        "common": len(keys_a & keys_b),
        f"{label_a}_only": len(keys_a - keys_b),
        f"{label_b}_only": len(keys_b - keys_a),
    }

    merged = a.merge(b, on=PRED_KEYS, suffixes=(f"_{label_a}", f"_{label_b}"))
    va = merged[f"{PRED_VALUE}_{label_a}"].to_numpy(dtype=float)
    vb = merged[f"{PRED_VALUE}_{label_b}"].to_numpy(dtype=float)
    d = vb - va
    mask = ~np.isnan(d)
    agreement = {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.nanmean(d**2))) if mask.any() else float("nan"),
        "mae": float(np.nanmean(np.abs(d))) if mask.any() else float("nan"),
        "bias": float(np.nanmean(d)) if mask.any() else float("nan"),
        "pearson_r": _safe_corr(va, vb),
    }

    merged["abs_delta"] = np.abs(d)
    merged["month"] = merged["FECHA"].dt.month
    monthly = (
        merged.groupby("month", as_index=False)["abs_delta"]
        .mean()
        .rename(columns={"abs_delta": "mean_abs_delta"})
        .sort_values("month")
    )

    return {"coverage": coverage, "agreement": agreement, "monthly": monthly, "merged": merged}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r over finite, non-degenerate pairs; NaN otherwise."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    xs, ys = x[m], y[m]
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


# ---------------------------------------------------------------------------------------
# Reporting: markdown + plots.
# ---------------------------------------------------------------------------------------
def write_markdown(
    coeff_res: dict,
    pred_res: Optional[dict],
    md_out: Path,
    *,
    label_a: str,
    label_b: str,
) -> None:
    lines: list[str] = [f"# Dataset comparison — {label_a} vs {label_b}", ""]

    cov = coeff_res["coverage"]
    lines += ["## Coefficients", "", "### Coverage (fits keyed by (ID, doy))", ""]
    lines.append("| metric | count |")
    lines.append("|---|---:|")
    for k, v in cov.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines += ["### Per-coefficient summary", "", "| dataset | coeff | mean | std | min | max |"]
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in coeff_res["summary"].iterrows():
        lines.append(
            f"| {r['dataset']} | {r['coeff']} | {r['mean']:.4g} | {r['std']:.4g} | "
            f"{r['min']:.4g} | {r['max']:.4g} |"
        )
    lines.append("")

    lines += [
        f"### Paired Δ = {label_b} − {label_a} (common fits)",
        "",
        "| coeff | mean Δ | std Δ | mean &#124;Δ&#124; | max &#124;Δ&#124; | Pearson r |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in coeff_res["delta"].iterrows():
        lines.append(
            f"| {r['coeff']} | {r['mean_delta']:.4g} | {r['std_delta']:.4g} | "
            f"{r['mean_abs_delta']:.4g} | {r['max_abs_delta']:.4g} | {r['pearson_r']:.4f} |"
        )
    lines.append("")

    if pred_res is not None:
        cov = pred_res["coverage"]
        ag = pred_res["agreement"]
        lines += ["## Predictions (TD_predicted)", "", "### Coverage (keyed by (ID, FECHA))", ""]
        lines.append("| metric | count |")
        lines.append("|---|---:|")
        for k, v in cov.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines += ["### Agreement", "", "| n | RMSE | MAE | bias (B−A) | Pearson r |"]
        lines.append("|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {ag['n']} | {ag['rmse']:.4g} | {ag['mae']:.4g} | {ag['bias']:.4g} | "
            f"{ag['pearson_r']:.4f} |"
        )
        lines.append("")
        lines += ["### Monthly mean &#124;Δ&#124;", "", "| month | mean &#124;Δ&#124; |", "|---:|---:|"]
        for _, r in pred_res["monthly"].iterrows():
            lines.append(f"| {int(r['month'])} | {r['mean_abs_delta']:.4g} |")
        lines.append("")

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines), encoding="utf-8")
    log.info("[compare] wrote report -> %s", md_out)


def make_plots(
    coeff_res: dict,
    pred_res: Optional[dict],
    out_dir: Path,
    *,
    label_a: str,
    label_b: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    merged = coeff_res["merged"]

    # Per-coefficient Δ histograms.
    for col in COEFF_METRICS:
        va = merged[f"{col}_{label_a}"].to_numpy(dtype=float)
        vb = merged[f"{col}_{label_b}"].to_numpy(dtype=float)
        d = vb - va
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(d, bins=60, color="steelblue", alpha=0.85)
        ax.axvline(0.0, linestyle="--", color="gray")
        ax.set_xlabel(f"Δ {col}  ({label_b} − {label_a})")
        ax.set_ylabel("count")
        ax.set_title(f"Δ distribution — {col}")
        ax.grid(True, alpha=0.3)
        f = out_dir / f"delta_hist_{col}.png"
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        written.append(f)

    # R² scatter (a vs b).
    ra = merged[f"r_squared_anom_{label_a}"].to_numpy(dtype=float)
    rb = merged[f"r_squared_anom_{label_b}"].to_numpy(dtype=float)
    m = np.isfinite(ra) & np.isfinite(rb)
    if m.any():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(ra[m], rb[m], s=4, alpha=0.3, color="darkorange")
        lo = float(min(ra[m].min(), rb[m].min()))
        hi = float(max(ra[m].max(), rb[m].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="y = x")
        ax.set_xlabel(f"r²  ({label_a})")
        ax.set_ylabel(f"r²  ({label_b})")
        ax.set_title("Anomaly r² agreement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        f = out_dir / "r2_scatter.png"
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        written.append(f)

    # Predictions: Δ histogram + monthly mean |Δ|.
    if pred_res is not None:
        pm = pred_res["merged"]
        d = (
            pm[f"{PRED_VALUE}_{label_b}"].to_numpy(dtype=float)
            - pm[f"{PRED_VALUE}_{label_a}"].to_numpy(dtype=float)
        )
        d = d[np.isfinite(d)]
        if d.size:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(d, bins=60, color="seagreen", alpha=0.85)
            ax.axvline(0.0, linestyle="--", color="gray")
            ax.set_xlabel(f"Δ TD_predicted  ({label_b} − {label_a})")
            ax.set_ylabel("count")
            ax.set_title("Δ distribution — TD_predicted")
            ax.grid(True, alpha=0.3)
            f = out_dir / "delta_hist_TD_predicted.png"
            fig.tight_layout()
            fig.savefig(f, dpi=120)
            plt.close(fig)
            written.append(f)

        monthly = pred_res["monthly"]
        if not monthly.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(monthly["month"], monthly["mean_abs_delta"], color="seagreen", alpha=0.85)
            ax.set_xlabel("month")
            ax.set_ylabel("mean |Δ| TD_predicted")
            ax.set_title("Monthly disagreement")
            ax.set_xticks(range(1, 13))
            ax.grid(True, alpha=0.3)
            f = out_dir / "pred_monthly_abs_delta.png"
            fig.tight_layout()
            fig.savefig(f, dpi=120)
            plt.close(fig)
            written.append(f)

    log.info("[compare] wrote %s plot(s) -> %s", len(written), out_dir)
    return written


# ---------------------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two TDEW coefficient (and optional prediction) datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coeffs-a", required=True, type=Path, help="Dataset A coeffs (file or dir).")
    p.add_argument("--coeffs-b", required=True, type=Path, help="Dataset B coeffs (file or dir).")
    p.add_argument("--pred-a", type=Path, default=None, help="Dataset A predictions (file or dir).")
    p.add_argument("--pred-b", type=Path, default=None, help="Dataset B predictions (file or dir).")
    p.add_argument("--label-a", default="v1", help="Label for dataset A.")
    p.add_argument("--label-b", default="v2", help="Label for dataset B.")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory for PNG plots.")
    p.add_argument("--md-out", required=True, type=Path, help="Markdown report output.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    if (args.pred_a is None) != (args.pred_b is None):
        raise SystemExit("--pred-a and --pred-b must be given together (or neither).")

    coeffs_a = load_dataset(args.coeffs_a, label=args.label_a)
    coeffs_b = load_dataset(args.coeffs_b, label=args.label_b)
    coeff_res = compare_coeffs(coeffs_a, coeffs_b, label_a=args.label_a, label_b=args.label_b)

    pred_res = None
    if args.pred_a is not None:
        pred_a = load_dataset(args.pred_a, label=f"{args.label_a}-pred")
        pred_b = load_dataset(args.pred_b, label=f"{args.label_b}-pred")
        pred_res = compare_predictions(
            pred_a, pred_b, label_a=args.label_a, label_b=args.label_b
        )

    write_markdown(coeff_res, pred_res, args.md_out, label_a=args.label_a, label_b=args.label_b)
    make_plots(coeff_res, pred_res, args.out_dir, label_a=args.label_a, label_b=args.label_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
