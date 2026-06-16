#!/usr/bin/env bash
#SBATCH --job-name=tdew-train-cpu
#SBATCH --partition=standard             # KHIPU CPU partition (sinfo: standard -> n[003-006])
#SBATCH --account=postgrado              # KHIPU default account (myaccount: postgrado)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32               # account limit is cpu=32 (so P_LIST tops out at 32)
#SBATCH --mem=64G
#SBATCH --time=04:00:00                  # KHIPU account max is 08:00:00
#SBATCH --output=logs/tdew-train-cpu-%j.out
#SBATCH --error=logs/tdew-train-cpu-%j.err
#
# CPU driver job — two MODEs (set MODE=train | benchmark; default train):
#   MODE=train      one production training run at P workers (run_training_hpc.py).
#   MODE=benchmark  a strong/weak scaling SWEEP over P_LIST, writing OUT_CSV
#                   (benchmark_scaling.py -> feed analyze_scaling.py for S(p)/E(p)).
#
# This job runs the DRIVER (scheduler + python process). The actual worker fleet is
# requested by dask-jobqueue (SLURMCluster) as *separate* SLURM jobs (--cluster slurm),
# sized to P (train) or each P in P_LIST (benchmark). For a single-node run use
# --cluster local and size workers to --cpus-per-task.
#
# Examples:
#   KHIPU:  BASE=… RESULTS=… P=32 FORECAST=1 sbatch HPC_code/sbatch/train_cpu.sbatch
#   KHIPU:  MODE=benchmark BASE=… RESULTS=… P_LIST=1,2,4,8,16,32 TRIALS=3 NUM_BUCKETS=256 \
#               sbatch HPC_code/sbatch/train_cpu.sbatch
#   LOCAL TEST (single node, synthetic data, override partition; CLUSTER=local):
#     MODE=benchmark CLUSTER=local SYNTH=1 BASE=/tmp/synth RESULTS=/tmp/cpu_bench \
#         P_LIST=1,2,4 TRIALS=1 NUM_BUCKETS=32 sbatch -p debug HPC_code/sbatch/train_cpu.sbatch
#
# IMPORTANT: run `mkdir -p logs` in the submit dir FIRST — SLURM opens --output/--error
# (logs/...) before the script body runs, so a missing logs/ makes the job fail with no output.
#
# Pin BLAS to one thread per worker (workers inherit this via the job prologue).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

# --- TODO: environment activation (module load / venv / conda) ---
# module load python/3.13
# source /path/to/.venv/bin/activate
PYTHON="${PYTHON:-python}"

BASE="${BASE:?set BASE=/path/to/base}"
RESULTS="${RESULTS:?set RESULTS=/path/to/results}"
MODE="${MODE:-train}"                      # train | benchmark
P="${P:-32}"                               # MODE=train: worker processes = P
SCRATCH="${TMPDIR:-/tmp}/dask-tdew-$USER"
CLUSTER="${CLUSTER:-slurm}"                # slurm = multi-node fleet (KHIPU); local = this node only (testing)
QUEUE="${SLURM_JOB_PARTITION:-standard}"
ACCOUNT="${SLURM_JOB_ACCOUNT:-postgrado}"

# dask-jobqueue worker-fleet args, only meaningful for --cluster slurm.
SLURM_ARGS=()
if [[ "$CLUSTER" == "slurm" ]]; then
  SLURM_ARGS=(--slurm-queue "$QUEUE" --slurm-account "$ACCOUNT"
              --slurm-cores 16 --slurm-memory 64GB --slurm-walltime 02:00:00)
fi

if [[ "$MODE" == "benchmark" ]]; then
  # Scaling SWEEP: build a fresh SLURM fleet per P in P_LIST, time each, write CSV.
  P_LIST="${P_LIST:-1,2,4,8,16,32}"
  TRIALS="${TRIALS:-3}"
  NUM_BUCKETS="${NUM_BUCKETS:-256}"        # B for strong mode (ideally >= 4*max(P_LIST))
  BENCH_MODE="${BENCH_MODE:-strong}"       # strong | weak
  N0="${N0:-4}"                            # weak-scaling base buckets per processor
  PHASES="${PHASES:-train}"                # train[,forecast]
  DATASET_LABEL="${DATASET_LABEL:-v1}"
  OUT_CSV="${OUT_CSV:-$RESULTS/scaling_cpu_${DATASET_LABEL}.csv}"

  "$PYTHON" HPC_code/benchmark_scaling.py \
    --base "$BASE" --results "$RESULTS" \
    --hw cpu --cluster "$CLUSTER" \
    --mode "$BENCH_MODE" --p-list "$P_LIST" --trials "$TRIALS" \
    --num-buckets "$NUM_BUCKETS" --n0 "$N0" \
    --phases "$PHASES" --dataset-label "$DATASET_LABEL" --out-csv "$OUT_CSV" \
    "${SLURM_ARGS[@]}" \
    --batch-size 64 --local-dir "$SCRATCH" ${SYNTH:+--synth}
  echo "[sbatch] benchmark CSV -> $OUT_CSV (analyze with HPC_code/analyze_scaling.py)"
else
  # --cluster slurm => multi-node fleet via dask-jobqueue; --cluster local => this node only.
  "$PYTHON" HPC_code/run_training_hpc.py \
    --base "$BASE" --results "$RESULTS" \
    --cluster "$CLUSTER" \
    "${SLURM_ARGS[@]}" \
    --n-workers "$P" --threads 1 --batch-size 64 \
    --local-dir "$SCRATCH" \
    ${BUCKET_IDS:+--bucket-ids "$BUCKET_IDS"} \
    ${FORECAST:+--forecast}
fi
