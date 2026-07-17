"""Phase-A temporal forward backtest + baseline ladder (plan §5, tasks A1/A2).

This is the paper's core evidence: an **honest, autoregressive** forward backtest that
mimics the real 2017-2020 gap. Train on early years, forecast the held-out recent years
with NO observed Td inside the block (predicted Td feeds its own lags, via
``forecast_zoned``), and score against observed v1.1 Td. Everything is scored on the
identical holdout for a fair ladder:

    climatology  ->  Td = Tmin  ->  zone OLS Td~Tmin  ->  tuned (refit on train-years)

Design decisions (locked, see PAPER_PLAN §5/§9):

* **Leave-last-N holdout.** ll4 = train 1981-2012 / predict 2013-2016 (the headline,
  mimics the 4-year gap); ll1 and ll2 give the lead-time error-growth curve.
* **Train-only climatology.** Climatology is recomputed on the *train* years for each
  split, so neither the climatology baseline nor the model's anomaly anchor can see the
  holdout (no leakage). Climatology is a plain per-(ID,doy) mean, matching
  ``tdew_estimation.climatology``.
* **Variant (a) refit-only** here: keep the existing recipe manifest, refit coefficients
  on the train years (reusing the GPU ``train_bucket_zoned`` so coeffs are method-identical
  to production), then forecast. Variant (b) full re-select is a separate, slower run
  (task A5).
* **Bucket-subset sample.** Because ``id_bucket = ID % 8192`` and IDs are dense in grid
  (row-major) order, every bucket is a regular nationwide lattice touching all 41 zones.
  Reading a handful of whole buckets is therefore a representative, all-zone spatial sample
  at bounded I/O -- far cheaper than scanning all 8192 shards for scattered sampled IDs.

Outputs (under ``$RES/tuning/backtest/<split>/``): ``predictions.parquet`` (one row per
held-out ID x day with every method's prediction + the observed Td), ``metrics.csv`` (the
ladder table), and ``zone_ols.csv`` (fitted per-zone OLS). The effect-size / bootstrap /
null-test module (A3) consumes ``predictions.parquet``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from tdew_estimation.anomaly_train import AnomalyTrainingConfig  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402
from tdew_estimation.parquet_io import as_path, read_parquet_any  # noqa: E402

from HPC_code_tunning import zones  # noqa: E402
from HPC_code_tunning.feature_spec import DEFAULT_CANDIDATE_POOL, TuningConfig  # noqa: E402
from HPC_code_tunning.forecast_zoned import forecast_bucket_zoned  # noqa: E402
from HPC_code_tunning.manifest import ZoneManifest, read_manifest  # noqa: E402

logger = logging.getLogger("backtest")

# Splits: (name, train_end_year, holdout_years). Train always starts 1981.
SPLITS: Dict[str, Tuple[int, Tuple[int, int]]] = {
    "ll1": (2015, (2016, 2016)),
    "ll2": (2014, (2015, 2016)),
    "ll4": (2012, (2013, 2016)),
}
TRAIN_START = 1981


# ---------------------------------------------------------------------------------------
# Train-only climatology (plain per-(ID,doy) mean over train years)
# ---------------------------------------------------------------------------------------
def train_only_climatology(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(ID,doy) mean Td/Tmin over the rows given (already restricted to train years)."""
    g = (
        train_df.groupby(["ID", "doy"], sort=False)
        .agg(TD_clim=("TD", "mean"), TMIN_clim=("TMIN", "mean"))
        .reset_index()
    )
    return g


# ---------------------------------------------------------------------------------------
# Zone OLS Td ~ Tmin: streamed sufficient statistics, fit once globally per zone
# ---------------------------------------------------------------------------------------
@dataclass
class _OLSAccum:
    n: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.sxy += float((x * y).sum())

    def fit(self) -> Tuple[float, float]:
        """Return (intercept, slope) for y ~ a + b x; falls back to (mean, 0) if singular."""
        denom = self.n * self.sxx - self.sx * self.sx
        if self.n < 2 or abs(denom) < 1e-9:
            return (self.sy / self.n if self.n else 0.0, 0.0)
        b = (self.n * self.sxy - self.sx * self.sy) / denom
        a = (self.sy - b * self.sx) / self.n
        return (a, b)


# ---------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------
def _tuning_config(base_path: Path, train_end_year: int, candidates, h_grid, min_samples) -> TuningConfig:
    base = AnomalyTrainingConfig(
        base_path=base_path,
        td_var="td",
        tmin_var="tmin_v12",
        train_year_range=(TRAIN_START, train_end_year),
        kernel="Tricube",
        min_samples=min_samples,
    )
    return TuningConfig(base=base, candidate_pool=tuple(candidates), h_grid=tuple(h_grid))


# ---------------------------------------------------------------------------------------
# One split
# ---------------------------------------------------------------------------------------
def _bucket_split(
    shard: pd.DataFrame,
    split_name: str,
    manifest: ZoneManifest,
    id_to_zone: Dict[int, int],
    tuning: TuningConfig,
    ols: Dict[int, _OLSAccum],
    train_bucket_zoned,
) -> Optional[pd.DataFrame]:
    """Refit + autoregressive forecast for one split on one already-loaded bucket shard.

    Updates ``ols`` in place with this bucket's train-year sufficient stats; returns the
    per-cell-day prediction frame (or ``None`` if nothing was produced).
    """
    train_end_year, holdout_years = SPLITS[split_name]
    y0, y1 = holdout_years
    ids_here = sorted(set(shard["ID"].astype(int).unique()) & set(id_to_zone))
    if not ids_here:
        return None

    train_df = shard[shard["year"] <= train_end_year][["ID", "FECHA", "TD", "TMIN", "doy"]].copy()
    clim_df = train_only_climatology(train_df)

    # Zone-OLS sufficient stats (train years, finite Td & Tmin).
    tv = train_df.dropna(subset=["TD", "TMIN"])
    if not tv.empty:
        zid = tv["ID"].map(id_to_zone).to_numpy()
        xall = tv["TMIN"].to_numpy(dtype=float)
        yall = tv["TD"].to_numpy(dtype=float)
        for z in np.unique(zid):
            m = zid == z
            ols.setdefault(int(z), _OLSAccum()).add(xall[m], yall[m])

    # Refit coeffs on train years (reuse the tested GPU zoned trainer) + forecast.
    bucket_zone = {i: id_to_zone[i] for i in ids_here}
    coeffs = train_bucket_zoned(train_df, clim_df, bucket_zone, manifest, tuning)
    if coeffs.empty:
        return None
    hist_df = shard[shard["year"] == train_end_year][["ID", "FECHA", "TD", "TMIN"]]
    fut_df = shard[(shard["year"] >= y0) & (shard["year"] <= y1)][["ID", "FECHA", "TMIN"]]
    pred = forecast_bucket_zoned(coeffs, clim_df, hist_df, fut_df, prediction_years=holdout_years)
    if pred.empty:
        return None

    obs = shard[(shard["year"] >= y0) & (shard["year"] <= y1)][["ID", "FECHA", "TD", "TMIN", "doy"]]
    obs = obs.rename(columns={"TD": "TD_obs", "TMIN": "Tmin_hold"})
    df = pred.merge(obs, on=["ID", "FECHA"], how="inner")
    df = df.merge(clim_df.rename(columns={"TD_clim": "pred_clim"})[["ID", "doy", "pred_clim"]],
                  on=["ID", "doy"], how="left")
    df["zone_id"] = df["ID"].map(id_to_zone).astype("Int64")
    return df[["ID", "zone_id", "FECHA", "doy", "TD_obs", "TD_predicted", "pred_clim", "Tmin_hold"]]


def run_backtest(
    *,
    prepared_root: Path,
    manifest: ZoneManifest,
    id_to_zone: Dict[int, int],
    tunings: Dict[str, TuningConfig],
    split_names: Sequence[str],
    bucket_ids: Sequence[int],
    out_root: Path,
) -> pd.DataFrame:
    """Read each bucket shard once, run every requested split from it, finalize + score all."""
    from HPC_code_tunning.train_zoned import train_bucket_zoned  # GPU import, kept lazy

    frames: Dict[str, List[pd.DataFrame]] = {s: [] for s in split_names}
    ols: Dict[str, Dict[int, _OLSAccum]] = {s: {} for s in split_names}
    for k, bid in enumerate(bucket_ids):
        shard = read_parquet_any(bucket_dir(prepared_root, bid))
        shard["FECHA"] = pd.to_datetime(shard["FECHA"])
        shard["year"] = shard["FECHA"].dt.year
        ids_here = sorted(set(shard["ID"].astype(int).unique()) & set(id_to_zone))
        if not ids_here:
            continue
        shard = shard[shard["ID"].isin(ids_here)]
        for s in split_names:
            df = _bucket_split(shard, s, manifest, id_to_zone, tunings[s], ols[s], train_bucket_zoned)
            if df is not None:
                frames[s].append(df)
        if (k + 1) % 10 == 0:
            logger.info("%d/%d buckets processed", k + 1, len(bucket_ids))

    all_metrics = []
    for s in split_names:
        m = _finalize_split(s, frames[s], ols[s], out_root / s)
        all_metrics.append(m)
        logger.info("[%s] metrics:\n%s", s, m.to_string(index=False))
    combined = pd.concat(all_metrics, ignore_index=True)
    out_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_root / "metrics_all.csv", index=False)
    return combined


def _finalize_split(split_name: str, frames: List[pd.DataFrame],
                    ols: Dict[int, _OLSAccum], out_dir: Path) -> pd.DataFrame:
    """Concatenate a split's per-bucket frames, add global zone-OLS, score, write outputs."""
    train_end_year, holdout_years = SPLITS[split_name]
    if not frames:
        raise RuntimeError(f"[{split_name}] no predictions produced")
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.rename(columns={"TD_predicted": "pred_tuned"})
    all_df["pred_tmin"] = all_df["Tmin_hold"]

    # Global zone-OLS predictions.
    coefs = {z: acc.fit() for z, acc in ols.items()}
    a = all_df["zone_id"].map(lambda z: coefs.get(int(z), (np.nan, np.nan))[0])
    b = all_df["zone_id"].map(lambda z: coefs.get(int(z), (np.nan, np.nan))[1])
    all_df["pred_ols"] = a.to_numpy() + b.to_numpy() * all_df["Tmin_hold"].to_numpy()

    # Only score rows with a finite observation.
    all_df = all_df.dropna(subset=["TD_obs"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out_dir / "predictions.parquet", engine="pyarrow", index=False)
    pd.DataFrame(
        [{"zone_id": z, "intercept": v[0], "slope": v[1]} for z, v in sorted(coefs.items())]
    ).to_csv(out_dir / "zone_ols.csv", index=False)

    metrics = _score_all(all_df, split_name, train_end_year, holdout_years)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    logger.info("[%s] wrote %s (%d rows)", split_name, out_dir, len(all_df))
    return metrics


# ---------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------
def _metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    err = pred - obs
    finite = np.isfinite(err)
    err = err[finite]
    n = err.size
    if n == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r": np.nan}
    p, o = pred[finite], obs[finite]
    r = float(np.corrcoef(p, o)[0, 1]) if n > 1 and np.std(p) > 0 and np.std(o) > 0 else np.nan
    return {
        "n": int(n),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r": r,
    }


def _score_all(df: pd.DataFrame, split_name: str, train_end_year: int,
               holdout_years: Tuple[int, int]) -> pd.DataFrame:
    obs = df["TD_obs"].to_numpy(dtype=float)
    methods = {
        "climatology": "pred_clim",
        "td_eq_tmin": "pred_tmin",
        "zone_ols": "pred_ols",
        "tuned": "pred_tuned",
    }
    clim_rmse = _metrics(df["pred_clim"].to_numpy(dtype=float), obs)["rmse"]
    rows = []
    for name, col in methods.items():
        m = _metrics(df[col].to_numpy(dtype=float), obs)
        m["method"] = name
        m["split"] = split_name
        m["train_end"] = train_end_year
        m["holdout"] = f"{holdout_years[0]}-{holdout_years[1]}"
        # Skill vs climatology (1 - RMSE/RMSE_clim); >0 means better than climatology.
        m["skill_vs_clim"] = (1.0 - m["rmse"] / clim_rmse) if clim_rmse and np.isfinite(clim_rmse) else np.nan
        rows.append(m)
    cols = ["split", "method", "n", "rmse", "mae", "bias", "r", "skill_vs_clim", "train_end", "holdout"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------
def choose_buckets(prepared_root: Path, n_buckets: int, seed: int = 0) -> List[int]:
    """Pick ``n_buckets`` bucket ids (each a nationwide lattice) deterministically."""
    all_b = discover_bucket_ids(prepared_root)
    if n_buckets >= len(all_b):
        return all_b
    rng = np.random.default_rng(seed)
    return sorted(int(b) for b in rng.choice(all_b, size=n_buckets, replace=False))


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, type=Path, help="$RES results base")
    p.add_argument("--prepared-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None, help="recipe manifest (default {base}/tuning/manifest.parquet)")
    p.add_argument("--zone-table", type=Path, default=None, help="default {base}/zone_table.parquet")
    p.add_argument("--out-root", type=Path, default=None, help="default {base}/tuning/backtest")
    p.add_argument("--splits", type=str, default="ll1,ll2,ll4", help="comma of ll1,ll2,ll4")
    p.add_argument("--n-buckets", type=int, default=64, help="whole buckets to sample (nationwide lattice each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATE_POOL))
    p.add_argument("--h-grid", type=str, default="7,11,15,21", help="must cover the manifest's h values")
    p.add_argument("--min-samples", type=int, default=15)
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    base = as_path(args.base)
    prepared_root = as_path(args.prepared_root or base / "bucketed_training_data")
    manifest_path = as_path(args.manifest or base / "tuning" / "manifest.parquet")
    zone_table_path = as_path(args.zone_table or base / "zone_table.parquet")
    out_root = as_path(args.out_root or base / "tuning" / "backtest")

    zt = pd.read_parquet(zone_table_path)
    id_to_zone = {int(i): int(z) for i, z in zip(zt["ID"], zt["zone_id"]) if int(z) != zones.UNASSIGNED}
    logger.info("zone table: %d assigned IDs, %d zones", len(id_to_zone), len(set(id_to_zone.values())))

    manifest = ZoneManifest(read_manifest(manifest_path))
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    h_grid = [int(h) for h in args.h_grid.split(",")]
    buckets = choose_buckets(prepared_root, args.n_buckets, seed=args.seed)
    logger.info("sampling %d buckets: %s%s", len(buckets), buckets[:8], " ..." if len(buckets) > 8 else "")

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    tunings = {
        s: _tuning_config(base, SPLITS[s][0], candidates, h_grid, args.min_samples)
        for s in split_names
    }
    combined = run_backtest(
        prepared_root=prepared_root, manifest=manifest, id_to_zone=id_to_zone,
        tunings=tunings, split_names=split_names, bucket_ids=buckets, out_root=out_root,
    )
    logger.info("ALL METRICS:\n%s", combined.to_string(index=False))
    logger.info("wrote %s", out_root / "metrics_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
