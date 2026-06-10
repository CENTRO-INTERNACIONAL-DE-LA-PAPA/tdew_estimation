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
# CPU strong/weak scaling driver job.
#
# This job runs the DRIVER (scheduler + the run_training_hpc.py process). The actual
# worker fleet is requested by dask-jobqueue (SLURMCluster) as *separate* SLURM jobs,
# so set --cluster slurm below and size the fleet with --n-workers (= P) and the
# per-job --slurm-* resources. For a single-node run, use --cluster local instead and
# size --n-workers to --cpus-per-task.
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
P="${P:-32}"                              # number of worker processes = parallel processors
SCRATCH="${TMPDIR:-/tmp}/dask-tdew-$USER"

# Multi-node fleet via dask-jobqueue (recommended for true multi-node scaling):
python HPC_code/run_training_hpc.py \
  --base "$BASE" --results "$RESULTS" \
  --cluster slurm \
  --slurm-queue "${SLURM_JOB_PARTITION:-CPU_PARTITION}" \
  --slurm-account "${SLURM_JOB_ACCOUNT:-MY_ACCOUNT}" \
  --slurm-cores 16 --slurm-memory 64GB --slurm-walltime 02:00:00 \
  --n-workers "$P" --threads 1 --batch-size 64 \
  --local-dir "$SCRATCH" \
  ${BUCKET_IDS:+--bucket-ids "$BUCKET_IDS"} \
  ${FORECAST:+--forecast}

# Single-node alternative (one node, P = cpus-per-task):
#   --cluster local --n-workers "$SLURM_CPUS_PER_TASK"
