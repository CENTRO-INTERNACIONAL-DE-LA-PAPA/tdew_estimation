#!/usr/bin/env bash
#SBATCH --job-name=tdew-gpu
#SBATCH --partition=gpu                   # KHIPU GPU partition (sinfo: gpu -> ag001,ds001,g00x)
#SBATCH --account=postgrado               # KHIPU default account (myaccount: postgrado)
#SBATCH --nodelist=ag001                  # the A100 node from the prof's example (remove to let SLURM pick)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1         # ONE A100 MIG slice (3g.20gb). Account limit: gres/gpu=1.
#SBATCH --time=02:00:00                   # KHIPU account max is 08:00:00
#SBATCH --output=logs/tdew-gpu-%j.out
#SBATCH --error=logs/tdew-gpu-%j.err
#
# Single-GPU job for KHIPU (the account is limited to gres/gpu=1, and the device is an A100
# MIG slice a100_3g.20gb ~ 20 GB / ~3-of-7 of a full A100). Two MODEs (MODE=train|benchmark):
#
#   MODE=train      one training run on the GPU
#                   (run_training_hpc.py --cluster local --gpu-train; client=None, sequential).
#   MODE=benchmark  the GPU "scaling" story for a SINGLE GPU:
#                     (1) benchmark_gpu_kernel.py  -> threads-per-block sweep + throughput-vs-M
#                     (2) benchmark_scaling.py --hw gpu --p-list 1  -> the single GPU end-to-end
#                         point (pair with a CPU sweep from train_cpu.sbatch for CPU-vs-GPU).
#                   Analyse both CSVs with HPC_code/analyze_gpu.py.
#
# Single GPU => NO dask-cuda / RAPIDS. Env = plain venv with cupy-cuda12x + the project.
# (The multi-GPU dask-cuda path "--cluster cuda --n-workers P" is kept commented for the future.)
#
# Local-SLURM test (single dev box, no MIG): override the GPU/account directives, e.g.
#   MODE=benchmark SYNTH=1 sbatch -p debug -A "" --gres=gpu:1 --nodelist=$(hostname -s) \
#       HPC_code/sbatch/train_gpu_a100.sbatch
#   (local SLURM with GresTypes unset can't satisfy --gres at all; run the benchmark commands
#    below directly on a GPU box instead. KHIPU has the MIG gres, so the directive works there.)
#
# IMPORTANT: run `mkdir -p logs` in the submit dir FIRST — SLURM opens --output/--error
# (logs/...) before the script body runs, so a missing logs/ makes the job fail with no output.
#
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

# --- TODO (KHIPU): environment activation (cupy venv) ---
# module load cuda/12
# source /path/to/.venv/bin/activate
PYTHON="${PYTHON:-python}"

BASE="${BASE:?set BASE=/path/to/base}"
RESULTS="${RESULTS:?set RESULTS=/path/to/results}"
MODE="${MODE:-train}"                       # train | benchmark
GPU_BLOCK="${GPU_BLOCK:-128}"               # threads-per-block (validated default 128)
SCRATCH="${TMPDIR:-/tmp}/dask-tdew-$USER"
mkdir -p "$RESULTS"

if [[ "$MODE" == "benchmark" ]]; then
  # (1) kernel micro-benchmark: block sweep at fixed M + throughput sweep at the best block.
  KERNEL_CSV="${KERNEL_CSV:-$RESULTS/gpu_kernel.csv}"
  BLOCKS="${BLOCKS:-32,64,128,256,512}"
  KM="${KM:-200000}"                        # #fits for the block sweep
  M_LIST="${M_LIST:-50000,100000,200000,400000,800000}"
  REPS="${REPS:-30}"
  "$PYTHON" HPC_code/benchmark_gpu_kernel.py \
    --m "$KM" --blocks "$BLOCKS" --m-list "$M_LIST" \
    --best-block "$GPU_BLOCK" --reps "$REPS" --out-csv "$KERNEL_CSV"

  # (2) single-GPU end-to-end point (same CSV schema as the CPU sweep -> CPU-vs-GPU overlay).
  SCALING_CSV="${SCALING_CSV:-$RESULTS/scaling_gpu.csv}"
  NUM_BUCKETS="${NUM_BUCKETS:-256}"
  TRIALS="${TRIALS:-3}"
  DATASET_LABEL="${DATASET_LABEL:-v12}"
  "$PYTHON" HPC_code/benchmark_scaling.py \
    --base "$BASE" --results "$RESULTS" \
    --hw gpu --cluster local --gpu-train --gpu-block "$GPU_BLOCK" \
    --mode strong --p-list 1 --trials "$TRIALS" --num-buckets "$NUM_BUCKETS" \
    --phases train --dataset-label "$DATASET_LABEL" --out-csv "$SCALING_CSV" \
    --batch-size 64 --local-dir "$SCRATCH" ${SYNTH:+--synth}
  echo "[sbatch] GPU CSVs -> $KERNEL_CSV , $SCALING_CSV (analyse with HPC_code/analyze_gpu.py)"
else
  "$PYTHON" HPC_code/run_training_hpc.py \
    --base "$BASE" --results "$RESULTS" \
    --cluster local --gpu-train --gpu-block "$GPU_BLOCK" \
    --n-workers 1 --batch-size "${BATCH_SIZE:-64}" \
    --local-dir "$SCRATCH" \
    ${BUCKET_IDS:+--bucket-ids "$BUCKET_IDS"} \
    ${FORECAST:+--forecast}
  # Future multi-GPU (needs dask-cuda / RAPIDS conda, P>1 GPUs):
  #   --cluster cuda --n-workers "$P" --rmm-pool-size 35GB --device-memory-limit 38GB
fi
