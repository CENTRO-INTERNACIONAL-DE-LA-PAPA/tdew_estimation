#!/usr/bin/env bash
#SBATCH --job-name=tdew-train-cpu
#SBATCH --partition=CPU_PARTITION        # <-- TODO: set your CPU partition
#SBATCH --account=MY_ACCOUNT             # <-- TODO: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32               # cores for the scheduler/driver node
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
#   BASE=… RESULTS=… P=32 FORECAST=1 sbatch HPC_code/sbatch/train_cpu.sbatch
#   MODE=benchmark BASE=… RESULTS=… P_LIST=1,2,4,8,16,32 TRIALS=3 NUM_BUCKETS=256 \
#       sbatch HPC_code/sbatch/train_cpu.sbatch
#
# Pin BLAS to one thread per worker (workers inherit this via the job prologue).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

# --- TODO: environment activation (module load / venv / conda) ---
# module load python/3.13
# source /path/to/.venv/bin/activate

BASE="${BASE:?set BASE=/path/to/base}"
RESULTS="${RESULTS:?set RESULTS=/path/to/results}"
MODE="${MODE:-train}"                      # train | benchmark
P="${P:-32}"                               # MODE=train: worker processes = P
SCRATCH="${TMPDIR:-/tmp}/dask-tdew-$USER"
QUEUE="${SLURM_JOB_PARTITION:-CPU_PARTITION}"
ACCOUNT="${SLURM_JOB_ACCOUNT:-MY_ACCOUNT}"

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

  python HPC_code/benchmark_scaling.py \
    --base "$BASE" --results "$RESULTS" \
    --hw cpu --cluster slurm \
    --mode "$BENCH_MODE" --p-list "$P_LIST" --trials "$TRIALS" \
    --num-buckets "$NUM_BUCKETS" --n0 "$N0" \
    --phases "$PHASES" --dataset-label "$DATASET_LABEL" --out-csv "$OUT_CSV" \
    --slurm-queue "$QUEUE" --slurm-account "$ACCOUNT" \
    --slurm-cores 16 --slurm-memory 64GB --slurm-walltime 02:00:00 \
    --batch-size 64 --local-dir "$SCRATCH"
  echo "[sbatch] benchmark CSV -> $OUT_CSV (analyze with HPC_code/analyze_scaling.py)"
else
  # Multi-node fleet via dask-jobqueue (recommended for true multi-node scaling):
  python HPC_code/run_training_hpc.py \
    --base "$BASE" --results "$RESULTS" \
    --cluster slurm \
    --slurm-queue "$QUEUE" \
    --slurm-account "$ACCOUNT" \
    --slurm-cores 16 --slurm-memory 64GB --slurm-walltime 02:00:00 \
    --n-workers "$P" --threads 1 --batch-size 64 \
    --local-dir "$SCRATCH" \
    ${BUCKET_IDS:+--bucket-ids "$BUCKET_IDS"} \
    ${FORECAST:+--forecast}
  # Single-node alternative (one node, P = cpus-per-task):
  #   --cluster local --n-workers "$SLURM_CPUS_PER_TASK"
fi
