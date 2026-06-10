# HPC_code — SLURM / GPU scaling for TDEW estimation

This folder holds everything specific to running the TDEW pipeline on the university
HPC (SLURM, CPU multi-node) and on A100 GPUs, plus the benchmark harness for the
scaling experiments described in `tdew_estimation_pram.qmd`. Local single-machine
execution lives in [`../Local/`](../Local/); the shared algorithm/library lives in the
`tdew_estimation` package and is used unchanged by both.

The key design hook: the bucketed runners
(`run_bucketed_anomaly_training_dask`, `run_bucketed_forecast_dask`) accept an injected
Dask `client`. This folder only supplies different **clusters** — the work `W` and the
outputs are identical; only the parallel-processor count `P` changes.

## Layout

| File | Purpose |
|------|---------|
| `hpc.py` | Cluster builders: `make_slurm_cluster` (dask-jobqueue), `make_local_cuda_cluster` (dask-cuda), `make_local_cpu_cluster` (baseline / `P=1`). Each returns a live `Client`. |
| `run_training_hpc.py` | Entrypoint: build `local\|slurm\|cuda` cluster → inject client → run TRAIN-COEFFICIENTS (+ optional FORECAST). Reuses the package runners. |
| `sbatch/train_cpu.sbatch` | SLURM driver job for CPU strong/weak scaling (placeholders for partition/account). |
| `sbatch/train_gpu_a100.sbatch` | SLURM driver job for A100 runs (`--gres=gpu:a100:N`, `P` = #GPUs). |
| `requirements-gpu.txt` | Optional GPU/HPC deps (cu12 wheels + dask-jobqueue). |

## Prerequisite: bucketed inputs

`run_training_hpc.py` runs only the parallel **compute** phases. Build the bucketed
inputs once (locally or in a prep job) with `../Local/run_pipeline.sh`, so that under
`--results` you have `bucketed_training_data/`, `climatology_by_bucket/`, and — for
forecasting — `future_tmin_by_bucket/`.

## Install

```bash
pip install -e .[hpc]                                    # base + dask-jobqueue (SLURM)
pip install --extra-index-url=https://pypi.nvidia.com \
    -r HPC_code/requirements-gpu.txt                     # + GPU/RAPIDS (Linux, CUDA 12.x)
```

## Run

```bash
# Baseline P=1 (any machine)
python HPC_code/run_training_hpc.py --base "$BASE" --results "$RESULTS" \
    --cluster local --n-workers 1

# CPU SLURM, P=32, train+forecast
BASE=... RESULTS=... P=32 FORECAST=1 sbatch HPC_code/sbatch/train_cpu.sbatch

# A100, P=4 GPUs
BASE=... RESULTS=... P=4 sbatch HPC_code/sbatch/train_gpu_a100.sbatch
```

> **Needed before SLURM submission:** the cluster's **account**, **CPU partition**, and
> **A100 partition** names — fill the `TODO` placeholders in the `sbatch/` files.

`--bucket-ids start:end` restricts the run to a deterministic subset of buckets — used
for weak scaling (`n = n0·p`) and for smoke tests.
