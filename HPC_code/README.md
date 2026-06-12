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
| `benchmark_scaling.py` | D4 driver: fresh cluster per `(p, trial)` → time the injected-client runners → append timing rows to a CSV. |
| `analyze_scaling.py` | D4 analysis: median over trials → speedup `S(p)`/efficiency `E(p)` → markdown tables + PNG plots. |
| `_synth.py` | Tiny synthetic raw dataset generator (runs the real prep pipeline) for smoke tests. |
| `tests/test_benchmark_smoke.py` | RAPIDS-free pytest smoke for the benchmark + analysis flow. |

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

## Benchmarking (D4)

`benchmark_scaling.py` produces the scaling CSVs and `analyze_scaling.py` turns them
into the tables/plots embedded in `tdew_estimation_pram.qmd`. CPU-only for now:
`--hw gpu` is reserved for D6 (needs the D3 GPU training path + the conda/RAPIDS env)
and currently raises a clear error; the CSV's `hw` column already reserves the slot so
GPU rows append cleanly later.

Install the (lightweight) plotting extra:

```bash
pip install -e .[benchmark]    # matplotlib only
```

End-to-end synthetic smoke (no real data, no RAPIDS, no SLURM):

```bash
# Strong scaling on a tiny synthetic dataset built by the REAL prep pipeline.
python HPC_code/benchmark_scaling.py \
    --base /tmp/tdew_synth/base --results /tmp/tdew_synth/results \
    --hw cpu --mode strong --p-list 1,2,4 --trials 1 --num-buckets 8 \
    --phases train --dataset-label synth --synth \
    --out-csv /tmp/tdew_synth/scaling.csv

python HPC_code/analyze_scaling.py --csv /tmp/tdew_synth/scaling.csv \
    --out-dir /tmp/tdew_synth/plots --md-out /tmp/tdew_synth/tables.md
```

`run_training_hpc.py` and `benchmark_scaling.py` are **different jobs**:
`run_training_hpc.py` does *one* run at a fixed `P` (production training or a single
scaling point); `benchmark_scaling.py` *sweeps* `P` over `--p-list`, repeats `--trials`,
and writes the scaling CSV. They share one SLURM file (`sbatch/train_cpu.sbatch`) via a
`MODE` switch.

The benchmark cluster is chosen with `--cluster`:

* `--cluster local` (default) — single-node `LocalCluster` sized to `p` (dev box / one
  fat node).
* `--cluster slurm` — multi-node `dask-jobqueue` fleet of `p` worker processes (the D6
  path). Requires `--slurm-queue` (+ usually `--slurm-account`). The harness calls
  `client.wait_for_workers(p)` before timing, so SLURM queue/startup latency is excluded
  from `wall_s`. One fleet is built per `p` and reused across that `p`'s trials.

Real CPU scaling run on SLURM (inputs prebuilt under `--results`), via the sbatch:

```bash
MODE=benchmark BASE="$BASE" RESULTS="$RESULTS" \
    P_LIST=1,2,4,8,16,32 TRIALS=3 NUM_BUCKETS=256 PHASES=train,forecast \
    sbatch HPC_code/sbatch/train_cpu.sbatch
# -> writes $RESULTS/scaling_cpu_<label>.csv

python HPC_code/analyze_scaling.py --csv "$RESULTS/scaling_cpu_v1.csv" \
    --out-dir "$RESULTS/scaling_plots" --md-out "$RESULTS/scaling_tables.md"
```

Or invoke the driver directly (e.g. on one node with `--cluster local`):

```bash
python HPC_code/benchmark_scaling.py \
    --base "$BASE" --results "$RESULTS" \
    --hw cpu --cluster slurm --slurm-queue "$PARTITION" --slurm-account "$ACCT" \
    --mode strong --p-list 1,2,4,8,16,32 --trials 3 \
    --num-buckets 256 --phases train,forecast --dataset-label v1 \
    --out-csv "$RESULTS/scaling_cpu_v1.csv"
```

Modes: **strong** holds a fixed bucket set (`--num-buckets B`, ideally `B ≥ 4·max(p)`)
across all `p`; **weak** grows the set as `n(p) = n0·p` (`--n0`). The CSV schema is
`dataset, mode, hw, p, n_ids, B, phase, trial, wall_s, timestamp`, appended
incrementally so an interrupted run still yields a usable file.

Run the smoke test with `pytest HPC_code/tests/test_benchmark_smoke.py -q`.
