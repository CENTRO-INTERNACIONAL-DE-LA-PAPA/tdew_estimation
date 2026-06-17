#!/usr/bin/env python3
"""
HPC_code.make_subset

Build a small ID-subset of the per-point parquet data, for quick local tests and for the CPU
scaling benchmark (which characterises scaling on a representative subset rather than the full
~300k IDs — the scaling ratio is size-independent, and a full p=1 baseline on 300k is ~80 h).

It filters each ``{base}/{var}/Outputs/{var}_daily_YYYY_MM.parquet`` to the first ``--n-ids``
IDs (``ID < n``) over an optional year range, writing the SAME filenames into
``{out}/{var}/Outputs/`` — so the subset is a drop-in ``--base`` for ``prep_inputs.py``,
``run_training_hpc.py``, etc.

Example::

    python HPC_code/make_subset.py \
        --base /media/.../henry_simcast_peru --out /tmp/sub4k \
        --n-ids 4000 --vars td,tmin_v12 --year-range 1981,2016
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq

log = logging.getLogger("make_subset")


def make_subset(
    *,
    base: Path,
    out: Path,
    n_ids: int,
    variables: list[str],
    year_range: tuple[int, int] | None = None,
    outputs_subdir: str = "Outputs",
    columns: tuple[str, ...] = ("ID", "FECHA", "Value"),
) -> dict[str, int]:
    """Filter ``base`` to the first ``n_ids`` IDs into ``out``. Returns rows written per var."""
    keep = list(range(int(n_ids)))
    written: dict[str, int] = {}
    for var in variables:
        src_dir = base / var / outputs_subdir
        if not src_dir.is_dir():
            raise FileNotFoundError(f"missing {src_dir}")
        dst_dir = out / var / outputs_subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        rows = 0
        for src in sorted(src_dir.glob(f"{var}_daily_*.parquet")):
            if year_range is not None:
                # filename is {var}_daily_YYYY_MM.parquet -> pull the year.
                try:
                    year = int(src.stem.split("_daily_")[1].split("_")[0])
                except (IndexError, ValueError):
                    year = None
                if year is not None and not (year_range[0] <= year <= year_range[1]):
                    continue
            table = pq.read_table(src, columns=list(columns), filters=[("ID", "in", keep)])
            pq.write_table(table, dst_dir / src.name)
            rows += table.num_rows
        written[var] = rows
        log.info("[subset] %s: %d rows -> %s", var, rows, dst_dir)
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter the per-point parquet data to the first N IDs (a drop-in --base subset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base", required=True, type=Path, help="Full dataset base ({var}/Outputs/...).")
    p.add_argument("--out", required=True, type=Path, help="Output subset base.")
    p.add_argument("--n-ids", required=True, type=int, help="Keep IDs 0..n-1.")
    p.add_argument("--vars", default="td,tmin_v12", help="Comma list of variable folders to subset.")
    p.add_argument("--year-range", default=None, help="Inclusive 'A,B' year filter (default: all).")
    p.add_argument("--outputs-subdir", default="Outputs")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    yr = None
    if args.year_range:
        a, b = (int(x) for x in args.year_range.split(","))
        yr = (a, b)
    variables = [v.strip() for v in args.vars.split(",") if v.strip()]
    written = make_subset(
        base=args.base, out=args.out, n_ids=args.n_ids,
        variables=variables, year_range=yr, outputs_subdir=args.outputs_subdir,
    )
    log.info("[subset] done -> %s (%s)", args.out,
             ", ".join(f"{v}={n}" for v, n in written.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
