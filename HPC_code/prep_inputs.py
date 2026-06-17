#!/usr/bin/env python3
"""
HPC_code.prep_inputs

Phase-0 PREP ONLY — build the reusable bucketed inputs that every downstream compute job
(`run_training_hpc.py`, `benchmark_scaling.py`, the sbatch files) assumes already exist
under ``--results``. This is Algorithm 1 lines 1–3 (+ 4b) of the PRAM doc, with NO training
or forecasting. Run it once per dataset version; the scaling sweeps and the training run
then reuse the same buckets.

Produces under ``--results``:
  * ``daily_climatology.parquet``        — per-(ID, doy) seasonal normals
  * ``bucketed_training_data/``          — bucket-year merged TD+TMIN (the expensive join)
  * ``climatology_by_bucket/``           — climatology sharded to the same buckets
  * ``future_tmin_by_bucket/``           — future TMIN over the forecast horizon
                                           (skipped with ``--no-future``)

The potato-only vs whole-PISCO choice is NOT made here — it is a property of ``--base``
(set when the data was extracted by ``sbatch/download_data.sh``: potato points by default,
the full ~2M-point grid with ``PERU_POTATO=0`` into a separate base). This script just reads
whatever ``{base}/{var}/Outputs`` contains.

Example::

    python HPC_code/prep_inputs.py \
        --base /media/.../henry_simcast_peru --results results_v11 \
        --td-var td --tmin-var tmin_v11 \
        --train-start 1981 --train-end 2016 --pred-start 2017 --pred-end 2020 \
        --num-buckets 1024 --n-workers 32

``--n-workers`` parallelises the dominant **bucket-year build** (one process per training
year); climatology and the shard steps stay sequential. Results are identical to a sequential
run — only the wall time changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from tdew_estimation.bucketed_data import (  # noqa: E402
    build_bucketed_training_dataset,
    shard_climatology_by_bucket,
    shard_future_tmin_by_bucket,
)
from tdew_estimation.climatology import calculate_and_save_climatology_chunked  # noqa: E402

log = logging.getLogger("prep_inputs")


# --------------------------------------------------------------------------------------- #
# Parallel bucket-year build. The per-year build is embarrassingly parallel and safe to run
# concurrently into the SAME output_dir: each year uses its own ``.tmp_year_{y}`` scratch,
# writes ``id_bucket=XXXX/train_{y}.parquet`` (year in the filename) and a ``.done_{y}``
# marker, so different years never touch the same files. We fan one task per year across a
# process pool (processes, not threads: the merge/sort holds the GIL).
# --------------------------------------------------------------------------------------- #
def _build_one_year(payload: dict) -> int:
    """Worker: build the bucket shards for a single year. Top-level so it is picklable."""
    from tdew_estimation.bucketed_data import build_bucketed_training_dataset as _build

    y = payload["year"]
    _build(
        year_range=(y, y),
        base_path=payload["base"],
        output_dir=payload["output_dir"],
        td_var=payload["td_var"],
        tmin_var=payload["tmin_var"],
        outputs_subdir=payload["outputs_subdir"],
        num_buckets=payload["num_buckets"],
        overwrite=payload["overwrite"],
    )
    return y


def _parallel_bucket_build(
    *,
    base: Path,
    output_dir: Path,
    td_var: str,
    tmin_var: str,
    outputs_subdir: str,
    num_buckets: int,
    overwrite: bool,
    train_year_range: tuple[int, int],
    n_workers: int,
) -> None:
    years = list(range(train_year_range[0], train_year_range[1] + 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "year": y,
            "base": str(base),
            "output_dir": str(output_dir),
            "td_var": td_var,
            "tmin_var": tmin_var,
            "outputs_subdir": outputs_subdir,
            "num_buckets": num_buckets,
            "overwrite": overwrite,
        }
        for y in years
    ]
    workers = min(n_workers, len(years))
    log.info("[prep] 2/4 bucketed training data (B=%d) — %d years over %d workers",
             num_buckets, len(years), workers)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_build_one_year, p): p["year"] for p in payloads}
        for fut in as_completed(futures):
            y = fut.result()  # re-raises any worker exception
            done += 1
            log.info("[prep]   year %d done (%d/%d)", y, done, len(years))


def prep(
    *,
    base: Path,
    results: Path,
    td_var: str,
    tmin_var: str,
    train_year_range: tuple[int, int],
    num_buckets: int,
    outputs_subdir: str = "Outputs",
    pred_year_range: tuple[int, int] | None = None,
    future_tmin_var: str | None = None,
    overwrite: bool = False,
    n_workers: int = 1,
) -> None:
    results.mkdir(parents=True, exist_ok=True)
    clim_path = results / "daily_climatology.parquet"
    bucket_dir_out = results / "bucketed_training_data"

    log.info("[prep] 1/4 climatology %s (%s)", train_year_range, tmin_var)
    calculate_and_save_climatology_chunked(
        train_year_range, base, clim_path,
        td_var=td_var, tmin_var=tmin_var, outputs_subdir=outputs_subdir,
    )

    if n_workers and n_workers > 1:
        _parallel_bucket_build(
            base=base, output_dir=bucket_dir_out,
            td_var=td_var, tmin_var=tmin_var, outputs_subdir=outputs_subdir,
            num_buckets=num_buckets, overwrite=overwrite,
            train_year_range=train_year_range, n_workers=n_workers,
        )
    else:
        log.info("[prep] 2/4 bucketed training data (B=%d) — sequential", num_buckets)
        build_bucketed_training_dataset(
            year_range=train_year_range, base_path=base,
            output_dir=bucket_dir_out,
            td_var=td_var, tmin_var=tmin_var, outputs_subdir=outputs_subdir,
            num_buckets=num_buckets, overwrite=overwrite,
        )

    log.info("[prep] 3/4 shard climatology by bucket")
    shard_climatology_by_bucket(
        climatology_path=clim_path,
        output_dir=results / "climatology_by_bucket",
        num_buckets=num_buckets, overwrite=overwrite,
    )

    if pred_year_range is not None:
        fvar = future_tmin_var or tmin_var
        log.info("[prep] 4/4 shard future TMIN %s (%s)", pred_year_range, fvar)
        shard_future_tmin_by_bucket(
            prediction_years=pred_year_range, base_path=base,
            output_dir=results / "future_tmin_by_bucket",
            future_tmin_var=fvar, outputs_subdir=outputs_subdir,
            num_buckets=num_buckets, overwrite=overwrite,
        )
    else:
        log.info("[prep] 4/4 future TMIN skipped (--no-future)")

    log.info("[prep] done -> %s", results)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-0: build bucketed inputs (climatology, bucket-year, shards) only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base", required=True, type=Path, help="Dataset base ({var}/Outputs/...).")
    p.add_argument("--results", required=True, type=Path, help="Output root for the bucketed inputs.")
    p.add_argument("--td-var", default="td")
    p.add_argument("--tmin-var", default="tmin_v12", help="TMIN folder: tmin_v11 (v1.1) | tmin_v12 (v1.2).")
    p.add_argument("--train-start", type=int, default=1981)
    p.add_argument("--train-end", type=int, default=2016)
    p.add_argument("--pred-start", type=int, default=None, help="Forecast-horizon start year.")
    p.add_argument("--pred-end", type=int, default=None, help="Forecast-horizon end year.")
    p.add_argument("--future-tmin-var", default=None, help="Defaults to --tmin-var.")
    p.add_argument("--no-future", action="store_true", help="Skip future-TMIN sharding.")
    p.add_argument("--num-buckets", type=int, default=1024, help="B; use B >= 4*p_max for scaling.")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Processes for the per-year bucket build (1 = sequential). Each worker "
                        "holds ~1 month of data, so peak RAM scales with this.")
    p.add_argument("--outputs-subdir", default="Outputs")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    pred_range = None
    if not args.no_future and args.pred_start is not None and args.pred_end is not None:
        pred_range = (args.pred_start, args.pred_end)

    prep(
        base=args.base,
        results=args.results,
        td_var=args.td_var,
        tmin_var=args.tmin_var,
        train_year_range=(args.train_start, args.train_end),
        num_buckets=args.num_buckets,
        outputs_subdir=args.outputs_subdir,
        pred_year_range=pred_range,
        future_tmin_var=args.future_tmin_var,
        overwrite=args.overwrite,
        n_workers=args.n_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
