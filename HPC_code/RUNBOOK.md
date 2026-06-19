# TDEW HPC runbook — as actually run (prep → Job A / B / C)

Canonical, reproducible record of the full benchmark + comparison pipeline on KHIPU
(OpenHPC + Slurm). This is what was *actually run*, with every gotcha baked in.

> **Prerequisites — the GPU sbatch (`sbatch/train_gpu_a100.sbatch`) must contain two fixes**, or the
> GPU jobs fail / silently produce garbage:
> 1. **CUDA toolkit** — `module load cuda/12.6` + `export CUDA_PATH=/opt/ohpc/pub/apps/cuda/12.6`.
>    cupy `RawKernel` JIT-compiles the solve kernel and needs CUDA headers. Without it the benchmark
>    crashes (`Failed to find CUDA headers`) and `MODE=train` exits 0 with **0 valid coefficients**.
> 2. **Year-range passthrough** — the train branch forwards
>    `TRAIN_START/TRAIN_END/PRED_START/PRED_END/HISTORY_END/TMIN_VAR` from env. Without it the forecast
>    uses defaults (`--pred-start 2017 --pred-end 2020`, `--history-end 2016`) → **empty predictions**
>    for any other window (this is what bit Job C the first time).
>
> Verify: `grep -E 'cuda/12.6|CUDA_PATH|--pred-start' HPC_code/sbatch/train_gpu_a100.sbatch`
>
> **KHIPU gotchas:** there is **no `$SCRATCH`** (use `/home`), and env must be **exported then
> `sbatch --export=ALL`** — the inline `VAR=val sbatch …` form is **not** propagated.

## 1. Local — build + prep the three datasets (outputs under `$FULL`)

```bash
cd ~/Documents/tdew_estimation
export PY=$PWD/.venv/bin/python
export FULL=/media/ppalacios/Data/henry_simcast_peru

# 1. CPU-benchmark dataset → 4000 IDs, v1.2, B=256
$PY HPC_code/make_subset.py --base "$FULL" --out "$FULL/sub4k" --n-ids 4000 \
    --vars td,tmin_v12 --year-range 1981,2016
$PY HPC_code/prep_inputs.py --base "$FULL/sub4k" --results "$FULL/res_cpu_v12" \
    --tmin-var tmin_v12 --train-start 1981 --train-end 2016 --no-future \
    --num-buckets 256 --n-workers 24

# 2. GPU-benchmark + 300k model dataset → full 300k, v1.2, B=1024
$PY HPC_code/prep_inputs.py --base "$FULL" --results "$FULL/res_v12_300k" \
    --tmin-var tmin_v12 --train-start 1981 --train-end 2016 \
    --pred-start 2017 --pred-end 2018 --num-buckets 1024 --n-workers 24

# 3. Comparison dataset → 20k IDs, both versions, held-out, B=512
$PY HPC_code/make_subset.py --base "$FULL" --out "$FULL/cmp20k" --n-ids 20000 \
    --vars td,tmin_v11,tmin_v12 --year-range 1981,2015
for V in v11 v12; do
  $PY HPC_code/prep_inputs.py --base "$FULL/cmp20k" --results "$FULL/cmp_$V" \
      --tmin-var tmin_$V --train-start 1981 --train-end 2014 \
      --pred-start 2015 --pred-end 2015 --num-buckets 512 --n-workers 24
done
```

## 2. Transfer to KHIPU (rsync; **no `$SCRATCH`**)

```bash
KH=piero.palacios@khipu.utec.edu.pe ; RUN=/home/piero.palacios/tdew_run
ssh "$KH" "mkdir -p $RUN"
for d in res_cpu_v12 res_v12_300k cmp_v11 cmp_v12 cmp20k; do
  rsync -a --info=progress2 "$FULL/$d" "$KH:$RUN/"     # cmp20k carries observed td for accuracy
done
```

## 3. KHIPU — session header (every shell)

```bash
cd ~/tdew_estimation
source /etc/profile.d/lmod.sh; module load python3/3.11.11; source .venv/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu12/python3/3.11.11/lib:${LD_LIBRARY_PATH:-}
export RUN=/home/piero.palacios/tdew_run; mkdir -p logs
```

## 4. Job A — CPU benchmark (standard partition, ~4.5 h)

`N_LIST=32,64,128` buckets = 512/1024/2048 IDs. **Not** `64,128,256` — that times out at the 8 h cap.

```bash
export RESULTS=$RUN/res_cpu_v12 N_LIST=32,64,128 P_LIST=1,2,4,8,16,32 CLUSTER=local
sbatch --export=ALL HPC_code/sbatch/bench_cpu_family.sbatch
# when COMPLETED:
python HPC_code/analyze_scaling.py --by-size --csv $RUN/res_cpu_v12/scaling_cpu_v12.csv \
    --out-dir $RUN/res_cpu_v12/cpu_family_plots --md-out $RUN/res_cpu_v12/cpu_family_tables.md
```

## 5. Job B — GPU benchmark + 300k model (`BASE` required; forecast **dropped**)

The autoregressive forecast is sequential and the GPU does not accelerate it, so the 300k forecast is
**dropped** — Job B keeps the GPU benchmark + the trained 300k coefficients only.

```bash
export RESULTS=$RUN/res_v12_300k BASE=$RUN/res_v12_300k
export MODE=benchmark ;                  sbatch --export=ALL HPC_code/sbatch/train_gpu_a100.sbatch
export MODE=train     ; unset FORECAST ; sbatch --export=ALL HPC_code/sbatch/train_gpu_a100.sbatch
# when both COMPLETED (verify train: grep 'failure_rows=0' logs/tdew-gpu-*.out):
python HPC_code/analyze_gpu.py --kernel-csv $RUN/res_v12_300k/gpu_kernel.csv \
    --pipeline-csv $RUN/res_v12_300k/gpu_pipeline.csv --scaling-csv $RUN/res_v12_300k/scaling_gpu.csv \
    --peak-fp64-gflops 4200 --out-dir $RUN/res_v12_300k/gpu_plots --md-out $RUN/res_v12_300k/gpu_report.md
```

## 6. Job C — v1.1 vs v1.2 comparison (with forecast; 2015 window; `--time=07:59:00`)

The forecast takes ~1 h 55 m per version, so override the 2 h default walltime.

```bash
export MODE=train FORECAST=1 PRED_START=2015 PRED_END=2015 HISTORY_END=2014 TRAIN_START=1981 TRAIN_END=2014
for V in v11 v12; do
  export RESULTS=$RUN/cmp_$V BASE=$RUN/cmp_$V TMIN_VAR=tmin_$V
  sbatch --time=07:59:00 --export=ALL HPC_code/sbatch/train_gpu_a100.sbatch
done
# VERIFY predictions actually landed (must be 512, not 0):
for V in v11 v12; do find $RUN/cmp_$V/predictions -name '*.parquet' | wc -l ; done
python HPC_code/evaluate_accuracy.py --pred-a $RUN/cmp_v11/predictions --pred-b $RUN/cmp_v12/predictions \
    --obs $RUN/cmp20k/td/Outputs --label-a v1.1 --label-b v1.2 \
    --out-dir $RUN/cmp/plots --md-out $RUN/cmp/accuracy_v11_v12.md
python HPC_code/compare_datasets.py \
    --coeffs-a $RUN/cmp_v11/llr_coeffs_anomaly_dataset --coeffs-b $RUN/cmp_v12/llr_coeffs_anomaly_dataset \
    --pred-a $RUN/cmp_v11/predictions --pred-b $RUN/cmp_v12/predictions \
    --label-a v1.1 --label-b v1.2 --out-dir $RUN/cmp/plots --md-out $RUN/cmp/compare_v11_v12.md
```

## Appendix — Job A matched-N on the workstation (apples-to-apples CPU vs KHIPU)

Same N as KHIPU, run directly (no Slurm). ~2 h on an i9-14900K. Writes a separate CSV.

```bash
cd ~/Documents/tdew_estimation; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=$FULL/res_cpu_v12
for NB in 32 64 128; do
  python HPC_code/benchmark_scaling.py --base "$R" --results "$R" --hw cpu --cluster local \
    --mode strong --p-list 1,2,4,8,16,32 --trials 1 --num-buckets $NB --phases train \
    --dataset-label v12 --out-csv "$R/scaling_cpu_v12_workstation.csv" --batch-size 64 --local-dir /tmp/dask-ws
done
python HPC_code/benchmark_scaling.py --base "$R" --results "$R" --hw cpu --cluster local \
  --mode weak --p-list 1,2,4,8,16,32 --trials 1 --n0 4 --phases train \
  --dataset-label v12 --out-csv "$R/scaling_cpu_v12_workstation.csv" --batch-size 64 --local-dir /tmp/dask-ws
```
