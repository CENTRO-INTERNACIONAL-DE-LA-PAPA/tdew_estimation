"""Phase-A task A5: honest **re-selection** on train-years only (backtest variant b).

Variant (a) (``backtest.py``) refits coefficients on the train years but keeps the recipe
``manifest.parquet`` that was selected over **all** of 1981-2016 -- so the per-zone x doy
feature/``h`` choice already saw the held-out years. This module re-runs the LOYOCV
backward-stepwise / h-grid **selection using only the train years**, producing a train-only
manifest. Feeding that manifest back through the same backtest gives variant (b); the gap
``skill(a) − skill(b)`` is the **selection-bias estimate** (§5).

Why a dedicated driver (not ``run_tuning_hpc --stage select --train-years "1981 2012"``):
``selection.load_training_for_ids`` reads *all* shard years and ``assemble_from_feature_frame``
bins them by ``searchsorted(year_values, ...)``. If the frame carries years outside
``year_values`` (the holdout), their year index is out of range and corrupts the scatter.
So we must hand ``select_zone`` frames that are **pre-filtered** to the train years, with a
**train-only climatology** (recomputed here, matching ``backtest.train_only_climatology``),
so neither the LOYOCV objective nor the anomaly anchor sees the holdout.

Output: ``manifest_trainonly_<split>.parquet`` under the backtest root. Then run, e.g.::

    $PY -m HPC_code_tunning.backtest --base $RES --splits ll4 --n-buckets 64 \
        --manifest $RES/tuning/backtest/manifest_trainonly_ll4.parquet \
        --out-root $RES/tuning/backtest_reselect
    $PY -m HPC_code_tunning.effect_sizes --backtest-root $RES/tuning/backtest_reselect --splits ll4

and compare ``skill_vs_clim`` to the variant-(a) run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from tdew_estimation.anomaly_train import AnomalyTrainingConfig  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir  # noqa: E402
from tdew_estimation.parquet_io import as_path, read_parquet_any  # noqa: E402

from HPC_code_tunning import zones  # noqa: E402
from HPC_code_tunning.backtest import SPLITS, TRAIN_START, choose_buckets, train_only_climatology  # noqa: E402
from HPC_code_tunning.feature_spec import DEFAULT_CANDIDATE_POOL, TuningConfig, build_feature_frame  # noqa: E402
from HPC_code_tunning.manifest import write_manifest  # noqa: E402
from HPC_code_tunning.selection import select_zone  # noqa: E402

logger = logging.getLogger("select_backtest")


def _tuning(base: Path, train_end_year: int, candidates, h_grid, min_samples, per_zone_n, id_chunk) -> TuningConfig:
    b = AnomalyTrainingConfig(
        base_path=base, td_var="td", tmin_var="tmin_v12",
        train_year_range=(TRAIN_START, train_end_year), kernel="Tricube", min_samples=min_samples,
    )
    return TuningConfig(base=b, candidate_pool=tuple(candidates), h_grid=tuple(h_grid),
                        per_zone_n=per_zone_n, id_chunk=id_chunk, granularity="doy")


def reselect_train_only(
    *, prepared_root: Path, zone_table: pd.DataFrame, bucket_ids: List[int],
    id_to_zone: Dict[int, int], per_zone_n: int, tuning: TuningConfig,
    train_end_year: int, seed: int,
) -> pd.DataFrame:
    """Re-run per-zone selection on train-only frames, reading the chosen buckets ONCE.

    The N chosen buckets (each a nationwide, all-zone lattice) are read once and held in
    memory (train years only, assigned IDs only); selection then serves each zone from that
    in-memory pool. This bounds I/O to N reads instead of ``zones x buckets`` re-reads.
    """
    registry = tuning.registry()
    labels = (
        zone_table.drop_duplicates("zone_id").set_index("zone_id")["zone_label"].to_dict()
        if "zone_label" in zone_table.columns else {}
    )
    parts_in: List[pd.DataFrame] = []
    for bid in bucket_ids:
        sh = read_parquet_any(bucket_dir(prepared_root, bid))
        sh["FECHA"] = pd.to_datetime(sh["FECHA"])
        sh = sh[(sh["FECHA"].dt.year <= train_end_year) & (sh["ID"].isin(id_to_zone))]
        if not sh.empty:
            parts_in.append(sh[["ID", "FECHA", "TD", "TMIN", "doy"]])
    if not parts_in:
        raise RuntimeError("no training rows loaded for re-selection")
    train_all = pd.concat(parts_in, ignore_index=True)
    train_all["zone_id"] = train_all["ID"].map(id_to_zone).astype(int)
    logger.info("loaded %d train rows across %d buckets, %d cells",
                len(train_all), len(bucket_ids), train_all["ID"].nunique())

    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for zone_id, gz in train_all.groupby("zone_id", sort=True):
        avail = np.sort(gz["ID"].unique())
        pick = avail if len(avail) <= per_zone_n else rng.choice(avail, per_zone_n, replace=False)
        sub = gz[gz["ID"].isin(set(pick.tolist()))]
        clim_sub = train_only_climatology(sub[["ID", "doy", "TD", "TMIN"]])
        frame = build_feature_frame(sub[["ID", "FECHA", "TD", "TMIN", "doy"]], clim_sub, registry)
        if frame.empty:
            continue
        assert int(frame["year"].max()) <= train_end_year, "holdout year leaked into selection frame"
        uniq = np.sort(frame["ID"].unique())
        frames = [
            frame[frame["ID"].isin(set(uniq[s:s + tuning.id_chunk].tolist()))].reset_index(drop=True)
            for s in range(0, len(uniq), tuning.id_chunk)
        ]
        part = select_zone(frames, registry, tuning, zone_id=int(zone_id),
                           zone_label=str(labels.get(int(zone_id), "")))
        parts.append(part)
        logger.info("zone %s: reselected %d units from %d/%d IDs",
                    zone_id, len(part), len(pick), len(avail))
    if not parts:
        raise RuntimeError("re-selection produced no recipes")
    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--prepared-root", type=Path, default=None)
    p.add_argument("--clim-root", type=Path, default=None)
    p.add_argument("--zone-table", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None, help="default {base}/tuning/backtest")
    p.add_argument("--split", type=str, default="ll4", help="which leave-last-N split to re-select for")
    p.add_argument("--n-buckets", type=int, default=64, help="restrict selection IDs to these buckets (bounded I/O)")
    p.add_argument("--per-zone-n", type=int, default=500)
    p.add_argument("--id-chunk", type=int, default=96)
    p.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATE_POOL))
    p.add_argument("--h-grid", type=str, default="7,11,15,21", help="match variant (a) to isolate selection bias")
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    base = as_path(args.base)
    prepared_root = as_path(args.prepared_root or base / "bucketed_training_data")
    clim_root = as_path(args.clim_root or base / "climatology_by_bucket")
    zone_table_path = as_path(args.zone_table or base / "zone_table.parquet")
    out_root = as_path(args.out_root or base / "tuning" / "backtest")
    out_root.mkdir(parents=True, exist_ok=True)

    split = args.split
    train_end_year, _holdout = SPLITS[split]
    zt = pd.read_parquet(zone_table_path)
    id_to_zone = {int(i): int(z) for i, z in zip(zt["ID"], zt["zone_id"]) if int(z) != zones.UNASSIGNED}

    buckets = choose_buckets(prepared_root, args.n_buckets, seed=args.seed)
    tuning = _tuning(base, train_end_year, [c.strip() for c in args.candidates.split(",") if c.strip()],
                     [int(h) for h in args.h_grid.split(",")], args.min_samples, args.per_zone_n, args.id_chunk)
    logger.info("[%s] re-select train 1981-%d over %d buckets (per_zone_n=%d, h-grid=%s)",
                split, train_end_year, len(buckets), tuning.per_zone_n, tuning.h_grid)

    manifest = reselect_train_only(
        prepared_root=prepared_root, zone_table=zt, bucket_ids=buckets,
        id_to_zone=id_to_zone, per_zone_n=args.per_zone_n, tuning=tuning,
        train_end_year=train_end_year, seed=args.seed,
    )
    out_path = out_root / f"manifest_trainonly_{split}.parquet"
    write_manifest(manifest, out_path)
    logger.info("wrote %s (%d rows). Mean skill_uplift vs fixed-5: %.4f (median %.4f)",
                out_path, len(manifest),
                float(manifest["skill_uplift"].mean(skipna=True)),
                float(manifest["skill_uplift"].median(skipna=True)))
    logger.info("distinct h: %s", sorted(manifest["h"].unique().tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
