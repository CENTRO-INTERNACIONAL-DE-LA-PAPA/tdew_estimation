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
| `nc_to_point_parquet.py` | Extract PISCOt `.nc` rasters → per-point monthly parquet (Python/xarray port of the R/`terra` step). Potato-points or full-grid via `--peru-potato`. |
| `prep_inputs.py` | **Phase 0**: build the reusable bucketed inputs (climatology, bucket-year dataset, climatology/future-TMIN shards) — *no training*. Run once per dataset version. |
| `benchmark_gpu_pipeline.py` | Single-GPU **roofline** of the full bucket path (assemble/convolve/solve): per-stage GFLOPS + arithmetic intensity, swept over N=IDs/bucket. |
| `sbatch/download_data.sh` | Download the 3 figshare PISCOt products (`.nc`) and run the extractor per variable. Resume-safe, idempotent, `BASE`/`VARS`-overridable. |
| `_synth.py` | Tiny synthetic raw dataset generator (runs the real prep pipeline) for smoke tests. |
| `tests/test_benchmark_smoke.py` | RAPIDS-free pytest smoke for the benchmark + analysis flow. |

## Data preparation: PISCO `.nc` → point parquet

The raw climate inputs are PISCOt `.nc` rasters; the pipeline consumes per-point monthly
parquet (`{base}/{var}/Outputs/{var}_daily_YYYY_MM.parquet`, columns `ID,FECHA,Value`).
`nc_to_point_parquet.py` is the headless Python/xarray port of the legacy R/`terra`
extraction — verified to reproduce the existing parquet **bit-for-bit** (`max|Δ|=0`).

```bash
pip install -e .[netcdf]                  # xarray + netCDF4 (geopandas already in core)

# Download all three products and extract at the potato-zoning centroids (~302k points):
bash HPC_code/sbatch/download_data.sh                 # BASE/VARS/PERU_POTATO/PURGE_RAW overridable

# Full PISCO grid (~2M points, heavier benchmark workload):
PERU_POTATO=0 bash HPC_code/sbatch/download_data.sh

# Safety gate — diff a fresh extraction against existing data before overwriting (no writes):
python HPC_code/nc_to_point_parquet.py --var tmin_v11 --nc-dir <dir-with-one-.nc> \
    --base "$BASE" --shp "$BASE/PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.shp" \
    --peru-potato --verify-against-existing
```

`--peru-potato` (default) samples each daily layer at the CENAGRO potato centroids (the
science subset, `ID` = shapefile feature order); `--no-peru-potato` keeps the full grid
(`ID` = row-major `(lat,lon)`, plus a `grid_index.parquet`).

Version-explicit folders (the older local `tmin`/`tmin_v1` folders were labeled *opposite*
to their real PISCOt version — confirmed bit-for-bit against figshare). Source→variable map:
`tmin_v11`←PISCOt v1.1 TMIN (16372509, 1981–2016), `tmin_v12`←PISCOt v1.2 TMIN (20533715 v2,
1981–2020), `td`←PISCOt v1.1 TDEW (16305341, 1981–2016). The v11-vs-v12 comparison overlaps
1981–2016.

## Dataset choice: potato-only vs whole PISCO

Every command below operates on whatever lives under `--base`. **Which points** are in the
data is fixed when it is *extracted* (`sbatch/download_data.sh`), not later:

```bash
# Potato planting zones only (~302k points) — the default, already extracted:
export BASE=/media/ppalacios/Data/henry_simcast_peru

# OR the whole PISCO grid (~2M points) — extract ONCE into a SEPARATE base (so it does not
# overwrite the potato data), then point everything at it:
PERU_POTATO=0 BASE=/media/ppalacios/Data/pisco_full bash HPC_code/sbatch/download_data.sh
export BASE=/media/ppalacios/Data/pisco_full
```

The whole-grid run is ~6.6× the work (heavier scaling workload); the potato subset is the
science target. Pick `BASE` accordingly — nothing else changes.

## Phase 0: prep the bucketed inputs (run once per version)

`run_training_hpc.py` / `benchmark_scaling.py` run only the **compute** phases; they assume
the bucketed inputs already exist under `--results`. Build them once per dataset version with
`prep_inputs.py` (no training):

```bash
mkdir -p logs
# PISCOt v1.1:
python HPC_code/prep_inputs.py --base "$BASE" --results results_v11 \
    --td-var td --tmin-var tmin_v11 --train-start 1981 --train-end 2016 \
    --pred-start 2017 --pred-end 2020 --num-buckets 1024
# PISCOt v1.2:
python HPC_code/prep_inputs.py --base "$BASE" --results results_v12 \
    --td-var td --tmin-var tmin_v12 --train-start 1981 --train-end 2016 \
    --pred-start 2017 --pred-end 2020 --num-buckets 1024
```

This writes under each `results_*/`: `daily_climatology.parquet`, `bucketed_training_data/`,
`climatology_by_bucket/`, `future_tmin_by_bucket/`. **The dataset version is baked in here**
(via `--tmin-var`); downstream jobs just point at `--results`. On KHIPU wrap it on the CPU
partition, e.g. `srun -p standard -A postgrado -c 16 python HPC_code/prep_inputs.py …`.
Use `--num-buckets 1024` so `B ≥ 4·p_max` for the CPU scaling sweep.

## Runbook A — prep → benchmark → results

Measure scaling/throughput (re-uses the Phase-0 `results_v11`; swap to `results_v12` for v1.2):

```bash
# CPU strong + weak scaling (KHIPU: standard/postgrado preset in the sbatch).
MODE=benchmark BASE="$BASE" RESULTS=results_v11 DATASET_LABEL=v11 \
    P_LIST=1,2,4,8,16,32 TRIALS=3 NUM_BUCKETS=1024 \
    sbatch HPC_code/sbatch/train_cpu.sbatch                       # -> results_v11/scaling_cpu_v11.csv
MODE=benchmark BENCH_MODE=weak BASE="$BASE" RESULTS=results_v11 DATASET_LABEL=v11 \
    P_LIST=1,2,4,8,16,32 N0=4 sbatch HPC_code/sbatch/train_cpu.sbatch   # appends weak rows

# GPU roofline (kernel + full-pipeline) + single-GPU end-to-end point (KHIPU: gpu/ag001/MIG).
MODE=benchmark BASE="$BASE" RESULTS=results_v11 DATASET_LABEL=v11 \
    sbatch HPC_code/sbatch/train_gpu_a100.sbatch
    # -> results_v11/{gpu_kernel.csv, gpu_pipeline.csv, scaling_gpu.csv}

# Turn the CSVs into tables + plots:
python HPC_code/analyze_scaling.py --csv results_v11/scaling_cpu_v11.csv \
    --out-dir results_v11/cpu_plots --md-out results_v11/cpu_tables.md
python HPC_code/analyze_gpu.py --kernel-csv results_v11/gpu_kernel.csv \
    --pipeline-csv results_v11/gpu_pipeline.csv --scaling-csv results_v11/scaling_gpu.csv \
    --peak-fp64-gflops 4200 --out-dir results_v11/gpu_plots --md-out results_v11/gpu_report.md
```

**Where the results land** (under `results_v11/`):
- CSVs: `scaling_cpu_v11.csv` (CPU strong+weak), `gpu_kernel.csv`, `gpu_pipeline.csv`, `scaling_gpu.csv`.
- CPU tables/plots: `cpu_tables.md` + `cpu_plots/` (`speedup_*`, `efficiency_*`, `time_*`.png).
- GPU report/plots: `gpu_report.md` + `gpu_plots/` (`gpu_roofline.png`, `gpu_pipeline_*_vs_M.png`,
  `gpu_block_tuning.png`, `gpu_throughput.png`, `cpu_vs_gpu.png`).

(For the GPU roofline ceiling, confirm the MIG slice's FP64 peak / HBM BW on `ag001`;
`benchmark_gpu_pipeline.py` also measures bandwidth empirically.)

## Runbook B — prep → train on the data

Produce the actual coefficients (+ optional forecast), not a benchmark:

```bash
# CPU multi-node training at P=32, with forecast:
BASE="$BASE" RESULTS=results_v11 P=32 FORECAST=1 \
    sbatch HPC_code/sbatch/train_cpu.sbatch

# OR train on the single GPU (client=None, no dask-cuda):
BASE="$BASE" RESULTS=results_v11 FORECAST=1 \
    sbatch HPC_code/sbatch/train_gpu_a100.sbatch
```

**Outputs** (under `results_v11/`): `llr_coeffs_anomaly_dataset/` (fitted coefficients per
bucket) and, with `FORECAST=1`, `predictions/` (+ combined `td_predictions.parquet`). Swap
`results_v11`→`results_v12` to train on v1.2. For a single dev box, add `CLUSTER=local`.

## Install

```bash
pip install -e .[hpc]                                    # base + dask-jobqueue (SLURM)
pip install --extra-index-url=https://pypi.nvidia.com \
    -r HPC_code/requirements-gpu.txt                     # + GPU/RAPIDS (Linux, CUDA 12.x)
```

## Run

```bash
mkdir -p logs    # REQUIRED first: SLURM opens logs/*.out before the job body runs

# Baseline P=1 (any machine)
python HPC_code/run_training_hpc.py --base "$BASE" --results "$RESULTS" \
    --cluster local --n-workers 1

# CPU multi-node fleet on SLURM (KHIPU: partition standard, account postgrado), P=32:
BASE=... RESULTS=... P=32 FORECAST=1 sbatch HPC_code/sbatch/train_cpu.sbatch

# Single GPU (KHIPU: one A100 MIG slice, partition gpu/ag001), client=None — no dask-cuda:
BASE=... RESULTS=... sbatch HPC_code/sbatch/train_gpu_a100.sbatch
```

The `sbatch/` files are preset for **KHIPU** (`account=postgrado`; CPU `standard`; GPU `gpu`
node `ag001`, `--gres=gpu:a100_3g.20gb:1`; `cpu=32` / `08:00:00` account limits). Override per
site on the command line, e.g. `sbatch -p debug ...`. For a single dev box with a local SLURM,
set `CLUSTER=local` so the CPU job runs on the one node instead of spawning a `dask-jobqueue`
fleet.

`--bucket-ids start:end` restricts the run to a deterministic subset of buckets — used
for weak scaling (`n = n0·p`) and for smoke tests.

## Benchmarking (D4)

`benchmark_scaling.py` produces the scaling CSVs and `analyze_scaling.py` / `analyze_gpu.py`
turn them into standalone tables + PNG plots. `--hw cpu` sweeps CPU workers; `--hw gpu
--gpu-train` runs the D3 GPU batched-WLS trainer (single GPU with `--cluster local`, `p-list 1`;
multi-GPU with `--cluster cuda` later). The CSV's `hw` column separates CPU and GPU rows so
`analyze_gpu.py` can build the CPU-vs-GPU overlay.

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
