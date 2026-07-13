#!/usr/bin/env python3
"""CLI entrypoint: zones -> feature/h selection -> zoned full-grid training.

Mirrors ``HPC_code/run_training_hpc.py`` in spirit. Three stages (``--stage``):

* ``select`` — build the ID->zone table, draw a stratified per-zone ID sample, run the
  LOYOCV backward-stepwise / h-grid search, and write the recipe ``manifest.parquet``.
* ``train``  — read an existing manifest and train the full grid in one zoned pass,
  writing tidy/long coeffs per bucket.
* ``all``    — both, in sequence (default).

Selection runs on the single visible GPU in-process. Training can fan out over GPUs with
``--cluster cuda`` (``hpc.make_local_cuda_cluster``).

Example (tiny end-to-end on a dev subset)::

    python -m HPC_code_tunning.run_tuning_hpc \
        --base /media/.../subset --coords /media/.../grid_index.parquet \
        --zones-zip ~/Downloads/clasif_clima_peru.zip \
        --per-zone-n 200 --h-grid 7,11,15 --stage all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sibling packages importable regardless of cwd (mirror gpu_train entrypoint).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from tdew_estimation.anomaly_train import AnomalyTrainingConfig  # noqa: E402

from HPC_code_tunning import zones  # noqa: E402
from HPC_code_tunning.feature_spec import DEFAULT_CANDIDATE_POOL, TuningConfig  # noqa: E402
from HPC_code_tunning.manifest import ZoneManifest, read_manifest, write_manifest  # noqa: E402
from HPC_code_tunning.selection import run_selection  # noqa: E402
from HPC_code_tunning.train_zoned import run_zoned_training  # noqa: E402

logger = logging.getLogger("run_tuning_hpc")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, type=Path, help="results base (holds bucketed data)")
    p.add_argument("--prepared-root", type=Path, default=None,
                   help="bucketed training root (default {base}/bucketed_training_data)")
    p.add_argument("--clim-root", type=Path, default=None,
                   help="bucketed climatology root (default {base}/climatology_by_bucket)")
    # zones
    p.add_argument("--coords", type=Path, default=None,
                   help="grid_index.parquet or point .shp (required unless --zone-table exists)")
    p.add_argument("--zones-zip", type=Path, default=None, help="SENAMHI clasif_clima_peru.zip")
    p.add_argument("--zone-table", type=Path, default=None,
                   help="cache parquet for the ID->zone table (default {base}/zone_table.parquet)")
    # tuning config
    p.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATE_POOL),
                   help="comma feature list (must start with const)")
    p.add_argument("--h-grid", type=str, default="7,11,15,21")
    p.add_argument("--granularity", choices=["doy", "zone"], default="doy")
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--per-zone-n", type=int, default=2000)
    p.add_argument("--id-chunk", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    # base anomaly config
    p.add_argument("--td-var", type=str, default="td")
    p.add_argument("--tmin-var", type=str, default="tmin_v12")
    p.add_argument("--train-years", type=str, default="1981 2016", help="'start end'")
    p.add_argument("--kernel", type=str, default="Tricube")
    p.add_argument("--min-samples", type=int, default=15)
    # outputs / control
    p.add_argument("--manifest", type=Path, default=None,
                   help="manifest parquet (default {base}/tuning/manifest.parquet)")
    p.add_argument("--coeffs-root", type=Path, default=None,
                   help="tidy zoned coeffs root (default {base}/tuning/zoned_coeffs)")
    p.add_argument("--stage", choices=["select", "train", "all"], default="all")
    p.add_argument("--cluster", choices=["none", "cuda"], default="none")
    p.add_argument("--n-gpus", type=int, default=None, help="GPU workers for --cluster cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def _tuning_config(args) -> TuningConfig:
    years = tuple(int(x) for x in str(args.train_years).split())
    base = AnomalyTrainingConfig(
        base_path=Path(args.base),
        td_var=args.td_var,
        tmin_var=args.tmin_var,
        train_year_range=(years[0], years[1]),
        kernel=args.kernel,
        min_samples=args.min_samples,
    )
    return TuningConfig(
        base=base,
        candidate_pool=tuple(c.strip() for c in args.candidates.split(",") if c.strip()),
        h_grid=tuple(int(h) for h in args.h_grid.split(",")),
        tol=args.tol,
        granularity=args.granularity,
        per_zone_n=args.per_zone_n,
        id_chunk=args.id_chunk,
        seed=args.seed,
    )


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    base = Path(args.base)
    prepared_root = args.prepared_root or base / "bucketed_training_data"
    clim_root = args.clim_root or base / "climatology_by_bucket"
    zone_table_path = args.zone_table or base / "zone_table.parquet"
    manifest_path = args.manifest or base / "tuning" / "manifest.parquet"
    coeffs_root = args.coeffs_root or base / "tuning" / "zoned_coeffs"
    tuning = _tuning_config(args)

    # ---- ID -> zone table (build once, cache) --------------------------------------
    if zone_table_path.exists():
        import pandas as pd
        zone_table = pd.read_parquet(zone_table_path)
        logger.info("loaded zone table %s (%d rows)", zone_table_path, len(zone_table))
    else:
        if args.coords is None or args.zones_zip is None:
            raise SystemExit("--coords and --zones-zip are required to build the zone table")
        logger.info("building zone table -> %s", zone_table_path)
        zone_table = zones.build_zone_table(args.coords, args.zones_zip, zone_table_path)
    counts = zones.zone_counts(zone_table)
    logger.info("zones: %d, unassigned=%d",
                counts["zone_id"].nunique(), int((zone_table["zone_id"] == zones.UNASSIGNED).sum()))

    # ---- Stage: select --------------------------------------------------------------
    if args.stage in ("select", "all"):
        sample = zones.stratified_sample(
            zone_table, present_ids=None, per_zone_n=tuning.per_zone_n, seed=tuning.seed
        )
        n_ids = sum(len(v) for v in sample.values())
        logger.info("sampled %d IDs across %d zones", n_ids, len(sample))
        manifest_df = run_selection(zone_table, sample, prepared_root, clim_root, tuning)
        write_manifest(manifest_df, manifest_path)
        logger.info("wrote manifest %s (%d rows)", manifest_path, len(manifest_df))
        if not manifest_df.empty:
            # sanity: which features survived most often (TMIN_anom should dominate)?
            from collections import Counter
            feats = Counter()
            for fl in manifest_df["feature_list"]:
                feats.update(fl.split(","))
            logger.info("feature retention: %s", dict(feats.most_common()))

    # ---- Stage: train ---------------------------------------------------------------
    if args.stage in ("train", "all"):
        manifest = ZoneManifest(read_manifest(manifest_path))
        client = None
        if args.cluster == "cuda":
            from HPC_code.hpc import make_local_cuda_cluster
            client = make_local_cuda_cluster(n_workers=args.n_gpus)
            logger.info("using LocalCUDACluster: %s", client)
        try:
            summary = run_zoned_training(
                prepared_training_root=prepared_root,
                bucketed_climatology_root=clim_root,
                coeffs_output_root=coeffs_root,
                zone_table=zone_table,
                manifest=manifest,
                tuning=tuning,
                overwrite=args.overwrite,
                client=client,
            )
        finally:
            if client is not None:
                cluster = getattr(client, "cluster", None)
                client.close()
                if cluster is not None:
                    cluster.close()
        total = int(summary["rows"].sum()) if not summary.empty else 0
        logger.info("zoned training: %d buckets, %d tidy coeff rows -> %s",
                    len(summary), total, coeffs_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
