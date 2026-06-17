#!/usr/bin/env bash
#SBATCH --job-name=tdew-bench-family
#SBATCH --partition=standard             # KHIPU CPU partition (sinfo: standard -> n[003-006])
#SBATCH --account=postgrado              # KHIPU default account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32               # account limit cpu=32 -> P_LIST tops out at 32
#SBATCH --mem=64G
#SBATCH --time=08:00:00                  # KHIPU account max
#SBATCH --output=logs/tdew-bench-family-%j.out
#SBATCH --error=logs/tdew-bench-family-%j.err
#
# CPU scaling FAMILY in one job: for each problem size N in N_LIST (= --num-buckets values, a
# subset of the prepped buckets), run a STRONG p-sweep over P_LIST, then a WEAK sweep — all
# appended to ONE CSV. Analyse with `analyze_scaling.py --by-size` to get Time/Speedup/
# Efficiency vs p with one curve per N, and the weak diagonals (constant N/p).
#
# Inputs must be prebuilt under RESULTS (prep_inputs.py) at a bucket count >= max(N_LIST).
# Reuses benchmark_scaling.py::select_buckets (strong = first num_buckets; weak = n0*p).
#
# Examples:
#   KHIPU: RESULTS=$SCRATCH/res_cpu_v12 N_LIST=64,128,256 P_LIST=1,2,4,8,16,32 \
#              sbatch HPC_code/sbatch/bench_cpu_family.sbatch
#   LOCAL test: CLUSTER=local RESULTS=/media/.../res_cpu_v12 N_LIST=8,16 P_LIST=1,2,4 \
#              sbatch -p debug HPC_code/sbatch/bench_cpu_family.sbatch
#
# IMPORTANT: run `mkdir -p logs` in the submit dir FIRST (SLURM opens --output before the body).
#
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# --- TODO (KHIPU): environment activation if not using $PYTHON directly ---
# module load python/3.13 ; source /path/to/.venv/bin/activate
PYTHON="${PYTHON:-python}"

RESULTS="${RESULTS:?set RESULTS=/path/to/prepped/results}"
BASE="${BASE:-$RESULTS}"                   # --base is required by argparse; inputs read from RESULTS
DATASET_LABEL="${DATASET_LABEL:-v12}"
N_LIST="${N_LIST:-64,128,256}"             # problem sizes = num-buckets values (subset of prepped B)
P_LIST="${P_LIST:-1,2,4,8,16,32}"
TRIALS="${TRIALS:-1}"
N0="${N0:-4}"                              # weak base buckets per worker
PHASES="${PHASES:-train}"
CLUSTER="${CLUSTER:-local}"                # standard node = local cluster sized to p
OUT_CSV="${OUT_CSV:-$RESULTS/scaling_cpu_${DATASET_LABEL}.csv}"
SCRATCH="${TMPDIR:-/tmp}/dask-tdew-$USER"
rm -f "$OUT_CSV"

SLURM_ARGS=()
if [[ "$CLUSTER" == "slurm" ]]; then
  SLURM_ARGS=(--slurm-queue "${SLURM_JOB_PARTITION:-standard}" --slurm-account "${SLURM_JOB_ACCOUNT:-postgrado}"
              --slurm-cores 16 --slurm-memory 64GB --slurm-walltime 02:00:00)
fi

for NB in ${N_LIST//,/ }; do
  echo "===== N = $NB buckets : STRONG sweep ====="
  "$PYTHON" HPC_code/benchmark_scaling.py --base "$BASE" --results "$RESULTS" \
    --hw cpu --cluster "$CLUSTER" --mode strong --p-list "$P_LIST" --trials "$TRIALS" \
    --num-buckets "$NB" --phases "$PHASES" --dataset-label "$DATASET_LABEL" \
    --out-csv "$OUT_CSV" "${SLURM_ARGS[@]}" --batch-size 64 --local-dir "$SCRATCH"
done

echo "===== WEAK sweep (n0=$N0) ====="
"$PYTHON" HPC_code/benchmark_scaling.py --base "$BASE" --results "$RESULTS" \
  --hw cpu --cluster "$CLUSTER" --mode weak --p-list "$P_LIST" --trials "$TRIALS" \
  --n0 "$N0" --phases "$PHASES" --dataset-label "$DATASET_LABEL" \
  --out-csv "$OUT_CSV" "${SLURM_ARGS[@]}" --batch-size 64 --local-dir "$SCRATCH"

echo "[sbatch] family CSV -> $OUT_CSV"
echo "[sbatch] analyse: HPC_code/analyze_scaling.py --by-size --csv $OUT_CSV \\"
echo "             --out-dir $RESULTS/cpu_family_plots --md-out $RESULTS/cpu_family_tables.md"
