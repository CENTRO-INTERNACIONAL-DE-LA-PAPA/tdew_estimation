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

# --- Environment activation (KHIPU-tuned; guarded so it no-ops on other machines) ---
set +e
source /etc/profile.d/lmod.sh 2>/dev/null
module load python3/3.11.11 2>/dev/null            # KHIPU OpenHPC module on shared /opt/ohpc
module load cuda/12.6 2>/dev/null                  # cupy RawKernel JIT-compiles the GPU solve → needs a CUDA toolkit
for _v in "${SLURM_SUBMIT_DIR:-$PWD}/.venv" "$HOME/tdew_estimation/.venv" ".venv"; do
  [ -f "$_v/bin/activate" ] && { source "$_v/bin/activate"; break; }
done
set -e
_pylib=/opt/ohpc/pub/libs/gnu12/python3/3.11.11/lib
[ -d "$_pylib" ] && export LD_LIBRARY_PATH="$_pylib:${LD_LIBRARY_PATH:-}"
# cupy compiles the GPU solve kernel at runtime (NVRTC) → it needs CUDA toolkit *headers*, found via
# CUDA_PATH. The `module load cuda` above only puts nvcc on PATH; point CUDA_PATH at the toolkit root
# (its include/ symlink holds cuda_runtime.h). Guarded so it no-ops off KHIPU. WITHOUT this, every GPU
# bucket fails with "Failed to find CUDA headers" — and MODE=train still exits 0 with 0 valid rows.
_cuda=/opt/ohpc/pub/apps/cuda/12.6
[ -d "$_cuda/include" ] && { export CUDA_HOME="$_cuda" CUDA_PATH="$_cuda"; \
  export LD_LIBRARY_PATH="$_cuda/lib64:${LD_LIBRARY_PATH:-}"; }
PYTHON="${PYTHON:-python}"

BASE="${BASE:?set BASE=/path/to/base}"
RESULTS="${RESULTS:?set RESULTS=/path/to/results}"
MODE="${MODE:-train}"                       # train | benchmark
GPU_BLOCK="${GPU_BLOCK:-128}"               # threads-per-block (validated default 128)
SCRATCH="${TMPDIR:-/tmp}/dask-tdew-$USER"
mkdir -p "$RESULTS"

if [[ "$MODE" == "benchmark" ]]; then
  # (1) solve-kernel micro-benchmark: block sweep at fixed M + throughput sweep at best block.
  KERNEL_CSV="${KERNEL_CSV:-$RESULTS/gpu_kernel.csv}"
  BLOCKS="${BLOCKS:-32,64,128,256,512}"
  KM="${KM:-200000}"                        # #fits for the block sweep
  M_LIST="${M_LIST:-50000,100000,200000,400000,800000}"
  REPS="${REPS:-30}"
  "$PYTHON" HPC_code/benchmark_gpu_kernel.py \
    --m "$KM" --blocks "$BLOCKS" --m-list "$M_LIST" \
    --best-block "$GPU_BLOCK" --reps "$REPS" --out-csv "$KERNEL_CSV"

  # (1b) full-pipeline roofline: per-stage (assemble/convolve/solve) GFLOPS + AI, curves per
  #      N=IDs/bucket over the M=#fits axis (analyze with analyze_gpu.py --pipeline-csv).
  PIPELINE_CSV="${PIPELINE_CSV:-$RESULTS/gpu_pipeline.csv}"
  N_IDS_LIST="${N_IDS_LIST:-64,128,293,512,1024}"
  "$PYTHON" HPC_code/benchmark_gpu_pipeline.py \
    --n-ids-list "$N_IDS_LIST" --years "${YEARS:-40}" --block "$GPU_BLOCK" \
    --reps "${PIPE_REPS:-10}" --out-csv "$PIPELINE_CSV"

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
  echo "[sbatch] GPU CSVs -> $KERNEL_CSV , $PIPELINE_CSV , $SCALING_CSV"
  echo "[sbatch] analyse: HPC_code/analyze_gpu.py --kernel-csv $KERNEL_CSV \\"
  echo "             --pipeline-csv $PIPELINE_CSV --scaling-csv $SCALING_CSV \\"
  echo "             --peak-fp64-gflops 4200 --out-dir $RESULTS/gpu_plots --md-out $RESULTS/gpu_report.md"
else
  "$PYTHON" HPC_code/run_training_hpc.py \
    --base "$BASE" --results "$RESULTS" \
    --cluster local --gpu-train --gpu-block "$GPU_BLOCK" \
    --n-workers 1 --batch-size "${BATCH_SIZE:-64}" \
    --local-dir "$SCRATCH" \
    ${TRAIN_START:+--train-start "$TRAIN_START"} ${TRAIN_END:+--train-end "$TRAIN_END"} \
    ${PRED_START:+--pred-start "$PRED_START"} ${PRED_END:+--pred-end "$PRED_END"} \
    ${HISTORY_END:+--history-end "$HISTORY_END"} ${TMIN_VAR:+--tmin-var "$TMIN_VAR"} \
    ${BUCKET_IDS:+--bucket-ids "$BUCKET_IDS"} \
    ${FORECAST:+--forecast}
  # Future multi-GPU (needs dask-cuda / RAPIDS conda, P>1 GPUs):
  #   --cluster cuda --n-workers "$P" --rmm-pool-size 35GB --device-memory-limit 38GB
fi
