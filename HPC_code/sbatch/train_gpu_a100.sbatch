#!/usr/bin/env bash
#SBATCH --job-name=tdew-train-a100
#SBATCH --partition=GPU_PARTITION        # <-- TODO: set your A100 partition
#SBATCH --account=MY_ACCOUNT             # <-- TODO: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4                 # <-- P = number of A100s for this run
#SBATCH --time=04:00:00
#SBATCH --output=logs/tdew-train-a100-%j.out
#SBATCH --error=logs/tdew-train-a100-%j.err
#
# A100 GPU strong-scaling driver job. P = number of GPUs (--gres above and --n-workers
# below must match). dask-cuda starts one worker per visible GPU on this node.
#
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

# --- TODO: environment activation + GPU deps (see HPC_code/requirements-gpu.txt) ---
# module load cuda/12
# source /path/to/.venv-gpu/bin/activate

BASE="${BASE:?set BASE=/path/to/base}"
RESULTS="${RESULTS:?set RESULTS=/path/to/results}"
P="${P:-4}"                               # number of A100 GPUs = parallel processors
SCRATCH="${TMPDIR:-/tmp}/dask-tdew-$USER"

python HPC_code/run_training_hpc.py \
  --base "$BASE" --results "$RESULTS" \
  --cluster cuda \
  --n-workers "$P" \
  --rmm-pool-size "${RMM_POOL_SIZE:-35GB}" \
  --device-memory-limit "${DEVICE_MEMORY_LIMIT:-38GB}" \
  --batch-size 64 \
  --local-dir "$SCRATCH" \
  ${BUCKET_IDS:+--bucket-ids "$BUCKET_IDS"} \
  ${FORECAST:+--forecast}
