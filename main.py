"""
tdew_estimation.main

Example "main" showing a PATH-AGNOSTIC workflow to:

1) compute EXPECTED_COUNT from a grid file (e.g., PotatoZonning DBF)
2) detect incomplete DOYs in the *combined anomaly coefficient* parquet
   (Option A: actual_count != expected_count)
3) retrain anomaly coefficients ONLY for DOYs_TO_FIX (producing a patch parquet)
4) patch the combined coefficients parquet by replacing those DOYs
5) (optional) re-check completeness after patching

This is an example usage script, not a CLI. It avoids hard-coded paths by using
arguments and clearly marked placeholders.

Assumed artifacts (from your pipeline)
--------------------------------------
- Grid file containing IDs (e.g., CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf)
- Daily climatology parquet:
    daily_climatology.parquet
  with columns: ID, doy, TD_clim, TMIN_clim
- Combined anomaly coefficients parquet (the one you check/patch):
    llr_coeffs_anomaly_final_direct.parquet
  with columns including at least: ID, doy, const_anom, TMIN_anom_coeff, TD_anom_lag1, ...

Notes
-----
- This script uses the "anomaly" model only (no GPU per-DOY model files).
- Retraining by DOY means: fit the same anomaly regression, but restrict doy_target
  loop to DOYs_TO_FIX for every ID.
- The retrain step is implemented here using the Dask-only runner in
  tdew_estimation.anomaly_dask (so this example assumes Dask is available).

Example usage
-------------
python tdew_estimation/main.py \
  --base-path "/media/ppalacios/Data1/henry_simcast_peru" \
  --grid-file "/media/ppalacios/Data1/henry_simcast_peru/PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf" \
  --results-dir "/media/ppalacios/Data1/henry_simcast_peru/results"

If you already know DOYs_TO_FIX and want to rerun directly:
python tdew_estimation/main.py \
  --base-path "/path/to/base" \
  --grid-file "/path/to/grid.dbf" \
  --results-dir "/path/to/results" \
  --doys-to-fix 13,54,65,82
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from tdew_estimation.anomaly_dask import rerun_failed_doys_with_dask
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.checks import detect_incomplete_anomaly_doys
from tdew_estimation.grid import expected_count_from_grid, load_grid_ids
from tdew_estimation.patch_coeffs import patch_anomaly_coeffs_inplace


def _parse_csv_ints(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Example workflow: detect incomplete anomaly DOYs and rerun+patch coefficients."
        )
    )

    p.add_argument(
        "--base-path",
        required=True,
        help="Root directory containing variable folders (e.g., td/Outputs, tmin_v1/Outputs).",
    )
    p.add_argument(
        "--grid-file",
        required=True,
        help="Path to grid file defining expected IDs (e.g., .dbf/.shp/.gpkg).",
    )
    p.add_argument(
        "--results-dir",
        required=True,
        help="Directory for results artifacts (coeffs/climatology/chunks).",
    )

    p.add_argument(
        "--coeffs-file",
        default="llr_coeffs_anomaly_final_direct.parquet",
        help="Combined anomaly coefficients parquet filename within results-dir.",
    )
    p.add_argument(
        "--climatology-file",
        default="daily_climatology.parquet",
        help="Daily climatology parquet filename within results-dir.",
    )

    p.add_argument(
        "--doys-to-fix",
        default="",
        help=(
            "Optional comma-separated DOY list to rerun. "
            "If omitted, script will detect incomplete DOYs automatically."
        ),
    )

    p.add_argument(
        "--id-column",
        default="ID",
        help="ID column name in grid/coeffs/climatology (default: ID).",
    )
    p.add_argument(
        "--doy-column",
        default="doy",
        help="DOY column name in coeffs/climatology (default: doy).",
    )

    # Training hyperparameters (match notebook defaults)
    p.add_argument("--train-start-year", type=int, default=1981)
    p.add_argument("--train-end-year", type=int, default=2016)
    p.add_argument("--h", type=int, default=11)
    p.add_argument(
        "--kernel",
        default="Tricube",
        choices=["Tricube", "Gaussian"],
        help="Kernel used for DOY neighborhood weighting.",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="Minimum neighborhood samples required to fit a DOY regression.",
    )

    p.add_argument(
        "--patch-file",
        default="llr_coeffs_anomaly_patch_doys.parquet",
        help="Filename for the patch parquet to write within results-dir.",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    base_path = Path(args.base_path).expanduser().resolve()
    grid_file = Path(args.grid_file).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()

    coeffs_path = results_dir / args.coeffs_file
    climatology_path = results_dir / args.climatology_file
    patch_path = results_dir / args.patch_file

    if not base_path.exists():
        print(f"ERROR: base-path not found: {base_path}", file=sys.stderr)
        return 2
    if not grid_file.exists():
        print(f"ERROR: grid-file not found: {grid_file}", file=sys.stderr)
        return 2
    if not results_dir.exists():
        print(f"ERROR: results-dir not found: {results_dir}", file=sys.stderr)
        return 2
    if not coeffs_path.exists():
        print(f"ERROR: coeffs file not found: {coeffs_path}", file=sys.stderr)
        return 2
    if not climatology_path.exists():
        print(f"ERROR: climatology file not found: {climatology_path}", file=sys.stderr)
        return 2

    # 1) expected_count from grid
    expected_count = expected_count_from_grid(grid_file, id_column=args.id_column)
    print(f"Expected ID count from grid: {expected_count}")

    # 2) Determine DOYs to fix
    doys_to_fix = _parse_csv_ints(args.doys_to_fix)
    if not doys_to_fix:
        print("\nDetecting incomplete DOYs from combined anomaly coefficients...")
        report = detect_incomplete_anomaly_doys(
            coeffs_path,
            expected_count=expected_count,
            id_col=args.id_column,
            doy_col=args.doy_column,
            strict_inequality=True,  # Option A: actual_count != expected_count
            include_missing=True,
            verbose=True,
        )
        doys_to_fix = report.to_fix_list()

    if not doys_to_fix:
        print("\n✅ No incomplete DOYs detected. Nothing to rerun.")
        return 0

    print("\nDOYs to retrain:", doys_to_fix)

    # 3) Load all IDs (you can later optimize this by reading only needed IDs)
    ids = load_grid_ids(grid_file, id_column=args.id_column)
    print(f"Loaded {len(ids)} IDs from grid.")

    # 4) Retrain ONLY those DOYs -> patch parquet
    cfg = AnomalyTrainingConfig(
        base_path=base_path,
        td_var="td",
        tmin_var="tmin_v1",
        train_year_range=(args.train_start_year, args.train_end_year),
        h=args.h,
        kernel=args.kernel,
        min_samples=args.min_samples,
    )

    print("\nRetraining anomaly coefficients for failed DOYs only (Dask)...")
    print(f"- Writing patch parquet to: {patch_path}")
    rerun_failed_doys_with_dask(
        ids=ids,
        failed_doys=doys_to_fix,
        config=cfg,
        climatology_path=climatology_path,
        patch_output_path=patch_path,
    )

    # 5) Patch combined coefficients inplace
    print("\nPatching combined coefficient parquet inplace...")
    print(f"- Base coeffs:  {coeffs_path}")
    print(f"- Patch coeffs: {patch_path}")
    summary = patch_anomaly_coeffs_inplace(
        base_coeffs_path=coeffs_path,
        patch_coeffs_path=patch_path,
        doys_to_patch=doys_to_fix,
        id_col=args.id_column,
        doy_col=args.doy_column,
    )
    print("\nPatch summary:")
    print(f"- Base rows before:   {summary.base_rows_before}")
    print(f"- Base rows removed:  {summary.base_rows_removed}")
    print(f"- Patch rows loaded:  {summary.patch_rows_loaded}")
    print(f"- Patch rows used:    {summary.patch_rows_used}")
    print(f"- Output rows:        {summary.output_rows}")

    # 6) Re-check (optional but useful)
    print("\nRe-checking completeness after patch...")
    report2 = detect_incomplete_anomaly_doys(
        coeffs_path,
        expected_count=expected_count,
        id_col=args.id_column,
        doy_col=args.doy_column,
        strict_inequality=True,
        include_missing=True,
        verbose=True,
    )
    if report2.ok:
        print("\n✅ Coefficients now complete for all DOYs in range.")
        return 0

    print("\n⚠️ Some DOYs are still incomplete after patch.")
    print("You may need to rerun again or investigate those IDs/DOYs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
