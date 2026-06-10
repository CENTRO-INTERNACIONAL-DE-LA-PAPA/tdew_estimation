#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_pipeline.sh --base BASE --results RESULTS --grid GRID [options]

Required:
  --base PATH        Base directory containing td/Outputs and tmin_v1/Outputs
  --results PATH     Results directory for outputs
  --grid PATH        Grid file with ID column (dbf/shp/gpkg/geojson/csv/parquet)

Options (training years / hyperparams):
  --train-start YEAR   (default: 1981)
  --train-end YEAR     (default: 2016)
  --h INT              (default: 11)
  --kernel NAME        (default: Tricube)  one of: Tricube, Gaussian
  --min-samples INT    (default: 15)

Options (forecast horizon):
  --pred-start YEAR    (default: 2017)
  --pred-end YEAR      (default: 2020)
  --history-end YEAR   (default: --train-end) year used to seed TD/TMIN lags
  --future-tmin-var NM (default: tmin)   variable folder for forecast-horizon TMIN

Options (Dask local cluster settings):
  --n-workers N        (default: $SLURM_CPUS_PER_TASK or nproc) one process per core
  --threads N          (default: 1)    threads/worker; keep 1 for BLAS-heavy fits
  --mem STR            (default: auto) per-worker memory; "auto" splits total RAM
  --batch-size N       (default: 64)   max bucket tasks in flight (sliding window)
  --local-dir DIR      (default: $TMPDIR/dask-tdew-$USER) worker scratch/spill dir

Options (bucketed dataset layout):
  --num-buckets N      (default: 1024)
  --prepared-dir DIR   (default: bucketed_training_data)
  --clim-buckets DIR   (default: climatology_by_bucket)
  --coeffs-dir DIR     (default: llr_coeffs_anomaly_dataset)
  --patch-dir DIR      (default: llr_coeffs_anomaly_patch_dataset)
  --failures-dir DIR   (default: anomaly_failures)
  --patch-failures DIR (default: anomaly_patch_failures)
  --future-tmin-dir DIR  (default: future_tmin_by_bucket)
  --predictions-dir DIR  (default: predictions)        bucketed prediction shards
  --predictions-file NM  (default: td_predictions.parquet) combined output

Options (paths / filenames):
  --id-col NAME        (default: ID)
  --doy-col NAME       (default: doy)
  --td-var NAME        (default: td)
  --tmin-var NAME      (default: tmin_v1)
  --outputs-subdir STR (default: Outputs)
  --climatology FILE   (default: daily_climatology.parquet)

Flags:
  --overwrite-prepared       Rebuild monthly bucketed training shards
  --overwrite-clim-buckets   Rebuild climatology bucket shards
  --overwrite-train-output   Overwrite coefficient dataset output
  --overwrite-forecast       Overwrite future-TMIN shards and prediction shards
  --skip-climatology         Skip climatology computation
  --skip-prepare            Skip building bucketed training inputs
  --skip-train              Skip bucketed model fitting
  --skip-patch              Skip rerun/patch workflow
  --skip-forecast           Skip forecasting (pipeline ends at coefficients)
  -h, --help                Show help

Environment:
  DRY_RUN=1                 Print config and exit

USAGE
}

BASE=""
RESULTS=""
GRID=""

TRAIN_START=1981
TRAIN_END=2016
H=11
KERNEL="Tricube"
MIN_SAMPLES=15

# Forecast horizon
PRED_START=2017
PRED_END=2020
HISTORY_END=""          # defaults to TRAIN_END when empty
FUTURE_TMIN_VAR="tmin"

N_WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
THREADS=1
MEM="auto"
BATCH_SIZE=64
LOCAL_DIR="${TMPDIR:-/tmp}/dask-tdew-${USER:-$(id -un)}"

NUM_BUCKETS=1024

ID_COL="ID"
DOY_COL="doy"
TD_VAR="td"
TMIN_VAR="tmin_v1"
OUTPUTS_SUBDIR="Outputs"

CLIMATOLOGY_FILE="daily_climatology.parquet"
PREPARED_DIR="bucketed_training_data"
CLIM_BUCKETS_DIR="climatology_by_bucket"
COEFFS_DIR="llr_coeffs_anomaly_dataset"
PATCH_DIR="llr_coeffs_anomaly_patch_dataset"
FAILURES_DIR="anomaly_failures"
PATCH_FAILURES_DIR="anomaly_patch_failures"
FUTURE_TMIN_DIR="future_tmin_by_bucket"
PREDICTIONS_DIR="predictions"
PREDICTIONS_FILE="td_predictions.parquet"

OVERWRITE_PREPARED=0
OVERWRITE_CLIM_BUCKETS=0
OVERWRITE_TRAIN_OUTPUT=0
OVERWRITE_FORECAST=0
SKIP_CLIMATOLOGY=0
SKIP_PREPARE=0
SKIP_TRAIN=0
SKIP_PATCH=0
SKIP_FORECAST=0

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

    --pred-start) PRED_START="${2:-}"; shift 2;;
    --pred-end) PRED_END="${2:-}"; shift 2;;
    --history-end) HISTORY_END="${2:-}"; shift 2;;
    --future-tmin-var) FUTURE_TMIN_VAR="${2:-}"; shift 2;;

    --n-workers) N_WORKERS="${2:-}"; shift 2;;
    --threads) THREADS="${2:-}"; shift 2;;
    --mem) MEM="${2:-}"; shift 2;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2;;
    --local-dir) LOCAL_DIR="${2:-}"; shift 2;;

    --num-buckets) NUM_BUCKETS="${2:-}"; shift 2;;
    --prepared-dir) PREPARED_DIR="${2:-}"; shift 2;;
    --clim-buckets) CLIM_BUCKETS_DIR="${2:-}"; shift 2;;
    --coeffs-dir) COEFFS_DIR="${2:-}"; shift 2;;
    --patch-dir) PATCH_DIR="${2:-}"; shift 2;;
    --failures-dir) FAILURES_DIR="${2:-}"; shift 2;;
    --patch-failures) PATCH_FAILURES_DIR="${2:-}"; shift 2;;
    --future-tmin-dir) FUTURE_TMIN_DIR="${2:-}"; shift 2;;
    --predictions-dir) PREDICTIONS_DIR="${2:-}"; shift 2;;
    --predictions-file) PREDICTIONS_FILE="${2:-}"; shift 2;;

    --id-col) ID_COL="${2:-}"; shift 2;;
    --doy-col) DOY_COL="${2:-}"; shift 2;;
    --td-var) TD_VAR="${2:-}"; shift 2;;
    --tmin-var) TMIN_VAR="${2:-}"; shift 2;;
    --outputs-subdir) OUTPUTS_SUBDIR="${2:-}"; shift 2;;
    --climatology) CLIMATOLOGY_FILE="${2:-}"; shift 2;;

    --overwrite-prepared) OVERWRITE_PREPARED=1; shift 1;;
    --overwrite-clim-buckets) OVERWRITE_CLIM_BUCKETS=1; shift 1;;
    --overwrite-train-output) OVERWRITE_TRAIN_OUTPUT=1; shift 1;;
    --overwrite-forecast) OVERWRITE_FORECAST=1; shift 1;;
    --skip-climatology) SKIP_CLIMATOLOGY=1; shift 1;;
    --skip-prepare) SKIP_PREPARE=1; shift 1;;
    --skip-train) SKIP_TRAIN=1; shift 1;;
    --skip-patch) SKIP_PATCH=1; shift 1;;
    --skip-forecast) SKIP_FORECAST=1; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${BASE}" || -z "${RESULTS}" || -z "${GRID}" ]]; then
  echo "ERROR: --base, --results, and --grid are required." >&2
  usage
  exit 2
fi

# Pin BLAS/OpenMP to one thread per process. Dask runs one single-threaded worker
# process per core; nested BLAS threads would otherwise oversubscribe the CPU.
# These are exported so the spawned Dask worker processes inherit them.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

HISTORY_END="${HISTORY_END:-$TRAIN_END}"

CLIM="${RESULTS%/}/${CLIMATOLOGY_FILE}"
PREPARED="${RESULTS%/}/${PREPARED_DIR}"
CLIM_BUCKETS="${RESULTS%/}/${CLIM_BUCKETS_DIR}"
COEFFS="${RESULTS%/}/${COEFFS_DIR}"
PATCH="${RESULTS%/}/${PATCH_DIR}"
FAILURES="${RESULTS%/}/${FAILURES_DIR}"
PATCH_FAILURES="${RESULTS%/}/${PATCH_FAILURES_DIR}"
FUTURE_TMIN="${RESULTS%/}/${FUTURE_TMIN_DIR}"
PREDICTIONS="${RESULTS%/}/${PREDICTIONS_DIR}"
PREDICTIONS_OUT="${RESULTS%/}/${PREDICTIONS_FILE}"

echo "=== tdew_estimation pipeline ==="
echo "BASE:    ${BASE}"
echo "RESULTS: ${RESULTS}"
echo "GRID:    ${GRID}"
echo
echo "Outputs:"
echo "  CLIM:           ${CLIM}"
echo "  PREPARED:       ${PREPARED}"
echo "  CLIM_BUCKETS:   ${CLIM_BUCKETS}"
echo "  COEFFS:         ${COEFFS}"
echo "  PATCH:          ${PATCH}"
echo "  FAILURES:       ${FAILURES}"
echo "  PATCH_FAILURES: ${PATCH_FAILURES}"
echo "  FUTURE_TMIN:    ${FUTURE_TMIN}"
echo "  PREDICTIONS:    ${PREDICTIONS}"
echo "  PREDICTIONS_OUT:${PREDICTIONS_OUT}"
echo
echo "Training:"
echo "  years: ${TRAIN_START}-${TRAIN_END}"
echo "  h: ${H}"
echo "  kernel: ${KERNEL}"
echo "  min_samples: ${MIN_SAMPLES}"
echo "  num_buckets: ${NUM_BUCKETS}"
echo
echo "Forecast:"
echo "  pred years: ${PRED_START}-${PRED_END}"
echo "  history_end: ${HISTORY_END}"
echo "  future_tmin_var: ${FUTURE_TMIN_VAR}"
echo
echo "Dask(local):"
echo "  workers: ${N_WORKERS}"
echo "  threads/worker: ${THREADS}"
echo "  memory_limit: ${MEM}"
echo "  in_flight_window: ${BATCH_SIZE}"
echo "  local_directory: ${LOCAL_DIR}"
echo
echo "Flags:"
echo "  overwrite_prepared:     ${OVERWRITE_PREPARED}"
echo "  overwrite_clim_buckets: ${OVERWRITE_CLIM_BUCKETS}"
echo "  overwrite_train_output: ${OVERWRITE_TRAIN_OUTPUT}"
echo "  overwrite_forecast:     ${OVERWRITE_FORECAST}"
echo "  skip_climatology:       ${SKIP_CLIMATOLOGY}"
echo "  skip_prepare:           ${SKIP_PREPARE}"
echo "  skip_train:             ${SKIP_TRAIN}"
echo "  skip_patch:             ${SKIP_PATCH}"
echo "  skip_forecast:          ${SKIP_FORECAST}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; exiting."
  exit 0
fi

mkdir -p "${RESULTS}"
mkdir -p "${LOCAL_DIR}"

if [[ "${SKIP_CLIMATOLOGY}" == "1" ]]; then
  echo "[1/8] Skipping climatology (per flag)."
else
  if [[ -f "${CLIM}" ]]; then
    echo "[1/8] Climatology already exists: ${CLIM}"
  else
    echo "[1/8] Computing climatology -> ${CLIM}"
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

if [[ "${SKIP_PREPARE}" == "1" ]]; then
  echo "[2/8] Skipping bucketed input preparation (per flag)."
else
  echo "[2/8] Building bucketed training inputs and climatology shards..."
  python -c '
from pathlib import Path
from tdew_estimation.bucketed_data import (
    build_bucketed_training_dataset,
    shard_climatology_by_bucket,
)

build_result = build_bucketed_training_dataset(
    year_range=('"${TRAIN_START}"', '"${TRAIN_END}"'),
    base_path=Path("'"${BASE}"'"),
    output_dir=Path("'"${PREPARED}"'"),
    td_var="'"${TD_VAR}"'",
    tmin_var="'"${TMIN_VAR}"'",
    outputs_subdir="'"${OUTPUTS_SUBDIR}"'",
    num_buckets=int("'"${NUM_BUCKETS}"'"),
    overwrite=bool(int("'"${OVERWRITE_PREPARED}"'")),
)
print("Prepared bucketed training data:", build_result)

clim_result = shard_climatology_by_bucket(
    climatology_path=Path("'"${CLIM}"'"),
    output_dir=Path("'"${CLIM_BUCKETS}"'"),
    num_buckets=int("'"${NUM_BUCKETS}"'"),
    overwrite=bool(int("'"${OVERWRITE_CLIM_BUCKETS}"'")),
)
print("Prepared climatology bucket shards:", clim_result)
'
fi

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[3/8] Skipping bucketed anomaly training (per flag)."
else
  echo "[3/8] Training anomaly coefficients by bucket -> ${COEFFS}"
  python -c '
from pathlib import Path
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import DaskAnomalyConfig, run_bucketed_anomaly_training_dask

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
    local_directory="'"${LOCAL_DIR}"'",
)

summaries = run_bucketed_anomaly_training_dask(
    prepared_training_root=Path("'"${PREPARED}"'"),
    bucketed_climatology_root=Path("'"${CLIM_BUCKETS}"'"),
    coeffs_output_root=Path("'"${COEFFS}"'"),
    failure_output_root=Path("'"${FAILURES}"'"),
    config=cfg,
    dask_config=dc,
    overwrite=bool(int("'"${OVERWRITE_TRAIN_OUTPUT}"'")),
)
print("Bucket training summaries:", len(summaries))
print("Buckets with rows:", sum(1 for s in summaries if s.coeff_rows > 0))
print("Failure rows:", sum(s.failure_rows for s in summaries))
'
fi

echo "[4/8] Checking for incomplete DOYs in coefficients..."
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
DO_PATCH=1
if [[ -z "${DOYS_TO_FIX}" ]]; then
  echo "✅ No incomplete DOYs detected; skipping rerun/patch."
  DO_PATCH=0
else
  echo "❌ DOYs to fix: ${DOYS_TO_FIX}"
fi
if [[ "${SKIP_PATCH}" == "1" ]]; then
  echo "[5/8] Skipping rerun/patch (per flag)."
  DO_PATCH=0
fi

if [[ "${DO_PATCH}" == "1" ]]; then
  echo "[5/8] Retraining failed DOYs by bucket -> ${PATCH}"
  python -c '
from pathlib import Path
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import DaskAnomalyConfig, rerun_failed_doys_with_bucketed_dask

doys = [int(x) for x in "'"${DOYS_TO_FIX}"'".split(",") if x.strip()]

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
    local_directory="'"${LOCAL_DIR}"'",
)

summaries = rerun_failed_doys_with_bucketed_dask(
    prepared_training_root=Path("'"${PREPARED}"'"),
    bucketed_climatology_root=Path("'"${CLIM_BUCKETS}"'"),
    patch_output_root=Path("'"${PATCH}"'"),
    failure_output_root=Path("'"${PATCH_FAILURES}"'"),
    failed_doys=doys,
    config=cfg,
    dask_config=dc,
)
print("Patch bucket summaries:", len(summaries))
print("Patch buckets with rows:", sum(1 for s in summaries if s.coeff_rows > 0))
print("Patch failure rows:", sum(s.failure_rows for s in summaries))
'
fi

if [[ "${DO_PATCH}" == "1" ]]; then
  echo "[6/8] Patching coefficient dataset in place and re-checking..."
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
fi

# ---------------------------------------------------------------------------
# Forecast (bucketed): shard forecast-horizon TMIN, forecast by bucket, combine
# ---------------------------------------------------------------------------
if [[ "${SKIP_FORECAST}" == "1" ]]; then
  echo "[7/8] Skipping future-TMIN sharding (per flag)."
  echo "[8/8] Skipping forecast (per flag)."
  echo "Pipeline complete (coefficients only)."
  exit 0
fi

echo "[7/8] Sharding forecast-horizon TMIN by bucket -> ${FUTURE_TMIN}"
python -c '
from pathlib import Path
from tdew_estimation.bucketed_data import shard_future_tmin_by_bucket

res = shard_future_tmin_by_bucket(
    prediction_years=('"${PRED_START}"', '"${PRED_END}"'),
    base_path=Path("'"${BASE}"'"),
    output_dir=Path("'"${FUTURE_TMIN}"'"),
    future_tmin_var="'"${FUTURE_TMIN_VAR}"'",
    outputs_subdir="'"${OUTPUTS_SUBDIR}"'",
    num_buckets=int("'"${NUM_BUCKETS}"'"),
    overwrite=bool(int("'"${OVERWRITE_FORECAST}"'")),
)
print("Future TMIN shards:", res)
'

echo "[8/8] Forecasting TD by bucket -> ${PREDICTIONS_OUT}"
python -c '
from pathlib import Path
from tdew_estimation.forecast import (
    DaskForecastConfig,
    run_bucketed_forecast_dask,
    combine_bucketed_predictions,
)

dc = DaskForecastConfig(
    n_workers=int("'"${N_WORKERS}"'"),
    threads_per_worker=int("'"${THREADS}"'"),
    memory_limit="'"${MEM}"'",
    batch_size=int("'"${BATCH_SIZE}"'"),
    local_directory="'"${LOCAL_DIR}"'",
)

summaries = run_bucketed_forecast_dask(
    coeffs_root=Path("'"${COEFFS}"'"),
    climatology_root=Path("'"${CLIM_BUCKETS}"'"),
    prepared_training_root=Path("'"${PREPARED}"'"),
    future_tmin_root=Path("'"${FUTURE_TMIN}"'"),
    predictions_output_root=Path("'"${PREDICTIONS}"'"),
    prediction_years=('"${PRED_START}"', '"${PRED_END}"'),
    history_end_year=int("'"${HISTORY_END}"'"),
    dask_config=dc,
    overwrite=bool(int("'"${OVERWRITE_FORECAST}"'")),
)
ok = sum(1 for s in summaries if s.status == "ok")
rows = sum(s.pred_rows for s in summaries)
print("Forecast bucket summaries:", len(summaries), "| ok:", ok, "| pred rows:", rows)

out = combine_bucketed_predictions(Path("'"${PREDICTIONS}"'"), output_file=Path("'"${PREDICTIONS_OUT}"'"))
print("Wrote combined predictions:", out)
'

echo "Pipeline complete."
