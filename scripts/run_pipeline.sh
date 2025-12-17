#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline runner for tdew_estimation anomaly model
#
# Steps:
#  1) Compute daily climatology (TD_clim, TMIN_clim)
#  2) Train anomaly coefficients at scale with Dask (chunks + combined parquet)
#  3) Check for incomplete DOYs (actual ID count != expected)
#  4) If needed: rerun only failed DOYs with Dask -> patch parquet
#  5) Patch combined coefficients parquet inplace and re-check
#
# This script is intentionally "bash-first" and uses small Python one-liners to call
# the library functions. It assumes you run it from the repository root.
#
# Requirements:
#  - python environment with: dask[distributed], pandas, pyarrow, statsmodels, tqdm
#  - input monthly parquet files under:
#      ${BASE}/td/Outputs/td_daily_YYYY_MM.parquet
#      ${BASE}/tmin_v1/Outputs/tmin_daily_YYYY_MM.parquet (legacy naming supported)
#
# Example:
#   bash tdew_estimation/scripts/run_pipeline.sh \
#     --base /path/to/base \
#     --results /path/to/results \
#     --grid /path/to/grid.dbf \
#     --n-workers 8 --threads 4 --mem 16GB --batch-size 1000
#
# Tip:
#   Use DRY_RUN=1 to print resolved settings without running heavy steps.

usage() {
  cat <<'USAGE'
Usage:
  run_pipeline.sh --base BASE --results RESULTS --grid GRID [options]

Required:
  --base PATH        Base directory containing td/Outputs and tmin_v1/Outputs
  --results PATH     Results directory for outputs (climatology, chunks, coeffs, patch)
  --grid PATH        Grid file with ID column (dbf/shp/gpkg/geojson/csv/parquet)

Options (training years / hyperparams):
  --train-start YEAR   (default: 1981)
  --train-end YEAR     (default: 2016)
  --h INT              (default: 11)
  --kernel NAME        (default: Tricube)  one of: Tricube, Gaussian
  --min-samples INT    (default: 15)

Options (Dask local cluster settings):
  --n-workers N        (default: 8)
  --threads N          (default: 4)
  --mem STR            (default: 16GB)
  --batch-size N       (default: 1000)

Options (paths / filenames):
  --id-col NAME        (default: ID)
  --doy-col NAME       (default: doy)
  --td-var NAME        (default: td)
  --tmin-var NAME      (default: tmin_v1)
  --outputs-subdir STR (default: Outputs)
  --climatology FILE   (default: daily_climatology.parquet)
  --chunks-dir DIR     (default: anomaly_coeffs_chunks)
  --coeffs FILE        (default: llr_coeffs_anomaly_final_direct.parquet)
  --patch FILE         (default: llr_coeffs_anomaly_patch_doys.parquet)

Flags:
  --overwrite-chunks   Overwrite existing chunk files (default: false)
  --skip-climatology   Do not compute climatology even if missing (default: false)
  --skip-train         Do not train coefficients (default: false)
  --skip-patch         Do not patch even if failures exist (default: false)
  -h, --help           Show help

Environment:
  DRY_RUN=1            Print config and exit

USAGE
}

# Defaults
BASE=""
RESULTS=""
GRID=""

TRAIN_START=1981
TRAIN_END=2016
H=11
KERNEL="Tricube"
MIN_SAMPLES=15

N_WORKERS=8
THREADS=4
MEM="16GB"
BATCH_SIZE=1000

ID_COL="ID"
DOY_COL="doy"
TD_VAR="td"
TMIN_VAR="tmin_v1"
OUTPUTS_SUBDIR="Outputs"

CLIMATOLOGY_FILE="daily_climatology.parquet"
CHUNKS_DIR="anomaly_coeffs_chunks"
COEFFS_FILE="llr_coeffs_anomaly_final_direct.parquet"
PATCH_FILE="llr_coeffs_anomaly_patch_doys.parquet"

OVERWRITE_CHUNKS=0
SKIP_CLIMATOLOGY=0
SKIP_TRAIN=0
SKIP_PATCH=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="${2:-}"; shift 2;;
    --results) RESULTS="${2:-}"; shift 2;;
    --grid) GRID="${2:-}"; shift 2;;

    --train-start) TRAIN_START="${2:-}"; shift 2;;
    --train-end) TRAIN_END="${2:-}"; shift 2;;
    --h) H="${2:-}"; shift 2;;
    --kernel) KERNEL="${2:-}"; shift 2;;
    --min-samples) MIN_SAMPLES="${2:-}"; shift 2;;

    --n-workers) N_WORKERS="${2:-}"; shift 2;;
    --threads) THREADS="${2:-}"; shift 2;;
    --mem) MEM="${2:-}"; shift 2;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2;;

    --id-col) ID_COL="${2:-}"; shift 2;;
    --doy-col) DOY_COL="${2:-}"; shift 2;;
    --td-var) TD_VAR="${2:-}"; shift 2;;
    --tmin-var) TMIN_VAR="${2:-}"; shift 2;;
    --outputs-subdir) OUTPUTS_SUBDIR="${2:-}"; shift 2;;

    --climatology) CLIMATOLOGY_FILE="${2:-}"; shift 2;;
    --chunks-dir) CHUNKS_DIR="${2:-}"; shift 2;;
    --coeffs) COEFFS_FILE="${2:-}"; shift 2;;
    --patch) PATCH_FILE="${2:-}"; shift 2;;

    --overwrite-chunks) OVERWRITE_CHUNKS=1; shift 1;;
    --skip-climatology) SKIP_CLIMATOLOGY=1; shift 1;;
    --skip-train) SKIP_TRAIN=1; shift 1;;
    --skip-patch) SKIP_PATCH=1; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${BASE}" || -z "${RESULTS}" || -z "${GRID}" ]]; then
  echo "ERROR: --base, --results, and --grid are required." >&2
  usage
  exit 2
fi

CLIM="${RESULTS%/}/${CLIMATOLOGY_FILE}"
CHUNKS="${RESULTS%/}/${CHUNKS_DIR}"
COEFFS="${RESULTS%/}/${COEFFS_FILE}"
PATCH="${RESULTS%/}/${PATCH_FILE}"

echo "=== tdew_estimation pipeline ==="
echo "BASE:    ${BASE}"
echo "RESULTS: ${RESULTS}"
echo "GRID:    ${GRID}"
echo
echo "Outputs:"
echo "  CLIM:   ${CLIM}"
echo "  CHUNKS: ${CHUNKS}"
echo "  COEFFS: ${COEFFS}"
echo "  PATCH:  ${PATCH}"
echo
echo "Training:"
echo "  years: ${TRAIN_START}-${TRAIN_END}"
echo "  h: ${H}"
echo "  kernel: ${KERNEL}"
echo "  min_samples: ${MIN_SAMPLES}"
echo
echo "Dask(local):"
echo "  workers: ${N_WORKERS}"
echo "  threads/worker: ${THREADS}"
echo "  memory_limit: ${MEM}"
echo "  batch_size: ${BATCH_SIZE}"
echo
echo "Flags:"
echo "  overwrite_chunks: ${OVERWRITE_CHUNKS}"
echo "  skip_climatology: ${SKIP_CLIMATOLOGY}"
echo "  skip_train:       ${SKIP_TRAIN}"
echo "  skip_patch:       ${SKIP_PATCH}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; exiting."
  exit 0
fi

mkdir -p "${RESULTS}"

# 1) Climatology
if [[ "${SKIP_CLIMATOLOGY}" == "1" ]]; then
  echo "[1/5] Skipping climatology (per flag)."
else
  if [[ -f "${CLIM}" ]]; then
    echo "[1/5] Climatology already exists: ${CLIM}"
  else
    echo "[1/5] Computing climatology -> ${CLIM}"
    python -c '
from pathlib import Path
from tdew_estimation.climatology import calculate_and_save_climatology_chunked

calculate_and_save_climatology_chunked(
    year_range=('"${TRAIN_START}"', '"${TRAIN_END}"'),
    base_path=Path("'"${BASE}"'"),
    output_file=Path("'"${CLIM}"'"),
    td_var="'"${TD_VAR}"'",
    tmin_var="'"${TMIN_VAR}"'",
    outputs_subdir="'"${OUTPUTS_SUBDIR}"'",
    engine="pyarrow",
    cleanup=True,
)
print("OK: wrote climatology:", "'"${CLIM}"'")
'
  fi
fi

# 2) Train anomaly coefficients (Dask)
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[2/5] Skipping training (per flag)."
else
  echo "[2/5] Training anomaly coefficients with Dask -> ${COEFFS}"
  python -c '
from pathlib import Path
from tdew_estimation.grid import load_grid_ids
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import run_anomaly_training_dask, DaskAnomalyConfig

ids = load_grid_ids(Path("'"${GRID}"'"), id_column="'"${ID_COL}"'")

cfg = AnomalyTrainingConfig(
    base_path=Path("'"${BASE}"'"),
    td_var="'"${TD_VAR}"'",
    tmin_var="'"${TMIN_VAR}"'",
    train_year_range=('"${TRAIN_START}"', '"${TRAIN_END}"'),
    h=int("'"${H}"'"),
    kernel="'"${KERNEL}"'",
    min_samples=int("'"${MIN_SAMPLES}"'"),
)

dc = DaskAnomalyConfig(
    n_workers=int("'"${N_WORKERS}"'"),
    threads_per_worker=int("'"${THREADS}"'"),
    memory_limit="'"${MEM}"'",
    batch_size=int("'"${BATCH_SIZE}"'"),
)

run_anomaly_training_dask(
    ids=ids,
    config=cfg,
    climatology_path=Path("'"${CLIM}"'"),
    chunk_dir=Path("'"${CHUNKS}"'"),
    combine_output_path=Path("'"${COEFFS}"'"),
    dask_config=dc,
    overwrite_chunks=bool(int("'"${OVERWRITE_CHUNKS}"'")),
)
print("OK: wrote coefficients:", "'"${COEFFS}"'")
'
fi

# 3) Check failures
echo "[3/5] Checking for incomplete DOYs in coefficients..."
DOYS_TO_FIX="$(
python -c '
from pathlib import Path
from tdew_estimation.grid import expected_count_from_grid
from tdew_estimation.checks import detect_incomplete_anomaly_doys

expected = expected_count_from_grid(Path("'"${GRID}"'"), id_column="'"${ID_COL}"'")
report = detect_incomplete_anomaly_doys(
    Path("'"${COEFFS}"'"),
    expected_count=expected,
    id_col="'"${ID_COL}"'",
    doy_col="'"${DOY_COL}"'",
    strict_inequality=True,
    include_missing=True,
    verbose=True,
)
print(",".join(str(d) for d in report.incomplete_doys))
'
)"
if [[ -z "${DOYS_TO_FIX}" ]]; then
  echo "✅ No incomplete DOYs detected."
  echo "Pipeline complete."
  exit 0
fi
echo "❌ DOYs to fix: ${DOYS_TO_FIX}"

# 4) Retrain failures -> patch parquet
if [[ "${SKIP_PATCH}" == "1" ]]; then
  echo "[4/5] Skipping rerun/patch creation (per flag)."
else
  echo "[4/5] Retraining failed DOYs with Dask -> patch: ${PATCH}"
  python -c '
from pathlib import Path
from tdew_estimation.grid import load_grid_ids
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import rerun_failed_doys_with_dask, DaskAnomalyConfig

doys = [int(x) for x in "'"${DOYS_TO_FIX}"'".split(",") if x.strip()]
ids = load_grid_ids(Path("'"${GRID}"'"), id_column="'"${ID_COL}"'")

cfg = AnomalyTrainingConfig(
    base_path=Path("'"${BASE}"'"),
    td_var="'"${TD_VAR}"'",
    tmin_var="'"${TMIN_VAR}"'",
    train_year_range=('"${TRAIN_START}"', '"${TRAIN_END}"'),
    h=int("'"${H}"'"),
    kernel="'"${KERNEL}"'",
    min_samples=int("'"${MIN_SAMPLES}"'"),
)

dc = DaskAnomalyConfig(
    n_workers=int("'"${N_WORKERS}"'"),
    threads_per_worker=int("'"${THREADS}"'"),
    memory_limit="'"${MEM}"'",
    batch_size=int("'"${BATCH_SIZE}"'"),
)

rerun_failed_doys_with_dask(
    ids=ids,
    failed_doys=doys,
    config=cfg,
    climatology_path=Path("'"${CLIM}"'"),
    patch_output_path=Path("'"${PATCH}"'"),
    dask_config=dc,
)
print("OK: wrote patch:", "'"${PATCH}"'")
'
fi

# 5) Patch + re-check
if [[ "${SKIP_PATCH}" == "1" ]]; then
  echo "[5/5] Skipping patching/re-check (per flag)."
  echo "Pipeline complete (without patching)."
  exit 0
fi

echo "[5/5] Patching coefficients inplace and re-checking..."
python -c '
from pathlib import Path
from tdew_estimation.grid import expected_count_from_grid
from tdew_estimation.checks import detect_incomplete_anomaly_doys
from tdew_estimation.patch_coeffs import patch_anomaly_coeffs_inplace

grid = Path("'"${GRID}"'")
coeffs = Path("'"${COEFFS}"'")
patch = Path("'"${PATCH}"'")

expected = expected_count_from_grid(grid, id_column="'"${ID_COL}"'")
report = detect_incomplete_anomaly_doys(coeffs, expected_count=expected, verbose=False)
doys = report.incomplete_doys
print("DOYs to patch:", doys)

if not doys:
    print("Nothing to patch; already complete.")
else:
    summary = patch_anomaly_coeffs_inplace(
        base_coeffs_path=coeffs,
        patch_coeffs_path=patch,
        doys_to_patch=doys,
        id_col="'"${ID_COL}"'",
        doy_col="'"${DOY_COL}"'",
    )
    print("Patch summary:", summary)

report2 = detect_incomplete_anomaly_doys(
    coeffs,
    expected_count=expected,
    id_col="'"${ID_COL}"'",
    doy_col="'"${DOY_COL}"'",
    strict_inequality=True,
    include_missing=True,
    verbose=True,
)
print("OK:", report2.ok)
'
echo "Pipeline complete."
