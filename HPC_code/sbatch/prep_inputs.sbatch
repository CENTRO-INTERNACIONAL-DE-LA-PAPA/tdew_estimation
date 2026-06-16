#!/usr/bin/env bash
#SBATCH --job-name=tdew-prep
#SBATCH --partition=standard             # KHIPU CPU partition (sinfo: standard -> n[003-006])
#SBATCH --account=postgrado              # KHIPU default account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32               # account limit cpu=32; prep fans 1 process per year
#SBATCH --mem=64G
#SBATCH --time=04:00:00                  # KHIPU account max is 08:00:00
#SBATCH --output=logs/tdew-prep-%j.out
#SBATCH --error=logs/tdew-prep-%j.err
#
# Phase 0 (PREP ONLY) — build the reusable bucketed inputs for ONE dataset version, with the
# bucket-year build parallelised across training years (one process per year). No training.
# Run once per version (tmin_v11, then tmin_v12), pointing RESULTS at a per-version dir.
#
# The dominant bucket-year build is parallelised (--n-workers = $SLURM_CPUS_PER_TASK); the
# climatology and shard steps are sequential. Output is identical to a sequential run.
#
# Examples:
#   BASE=/media/.../henry_simcast_peru RESULTS=results_v11 TMIN_VAR=tmin_v11 \
#       sbatch HPC_code/sbatch/prep_inputs.sbatch
#   BASE=/media/.../henry_simcast_peru RESULTS=results_v12 TMIN_VAR=tmin_v12 \
#       sbatch HPC_code/sbatch/prep_inputs.sbatch
#   # whole-PISCO base + a quick local test (override partition; fewer workers):
#   CLUSTER_TEST=1 BASE=/tmp/synth RESULTS=/tmp/prep N_WORKERS=4 NUM_BUCKETS=8 \
#       TRAIN_START=2001 TRAIN_END=2002 NO_FUTURE=1 sbatch -p debug HPC_code/sbatch/prep_inputs.sbatch
#
# IMPORTANT: run `mkdir -p logs` in the submit dir FIRST (SLURM opens --output before the body).
#
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# SLURM runs the batch script from a spool copy, so resolve paths from the SUBMIT directory
# (its default CWD) like the other sbatch files — relative HPC_code/... paths then work.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# --- TODO (KHIPU): environment activation if not using $PYTHON directly ---
# module load python/3.13 ; source /path/to/.venv/bin/activate
PYTHON="${PYTHON:-python}"

BASE="${BASE:?set BASE=/path/to/base}"
RESULTS="${RESULTS:?set RESULTS=/path/to/results}"
TD_VAR="${TD_VAR:-td}"
TMIN_VAR="${TMIN_VAR:-tmin_v12}"          # tmin_v11 (v1.1) | tmin_v12 (v1.2)
TRAIN_START="${TRAIN_START:-1981}"
TRAIN_END="${TRAIN_END:-2016}"
PRED_START="${PRED_START:-2017}"
PRED_END="${PRED_END:-2020}"
NUM_BUCKETS="${NUM_BUCKETS:-1024}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

extra=()
[ -n "${NO_FUTURE:-}" ] && extra+=(--no-future)
[ -n "${OVERWRITE:-}" ] && extra+=(--overwrite)
[ -z "${NO_FUTURE:-}" ] && extra+=(--pred-start "$PRED_START" --pred-end "$PRED_END")

"$PYTHON" HPC_code/prep_inputs.py \
    --base "$BASE" --results "$RESULTS" \
    --td-var "$TD_VAR" --tmin-var "$TMIN_VAR" \
    --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
    --num-buckets "$NUM_BUCKETS" --n-workers "$N_WORKERS" \
    "${extra[@]}"

echo "[sbatch] prep done -> $RESULTS (bucketed_training_data/, climatology_by_bucket/, future_tmin_by_bucket/)"
