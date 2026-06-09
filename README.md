# tdew_estimation

This repository contains the code used to estimate daily dewpoint temperature (`td`) needed to run Simcast workflows when `td` is not directly available.

PRAM pseudocode and Big-O complexity are documented in:
- `tdew_estimation_pram.qmd` (Quarto source, Typst format)
- `tdew_estimation_pram.pdf` (rendered PDF)

The core approach implemented here is an **anomaly-based local linear regression model** (“anomaly model”) that:
- builds a **daily climatology** for each grid cell (`ID`) and day-of-year (`doy`)
- models deviations from climatology (`TD_anom`) as a function of temperature anomalies and lagged anomalies
- produces **per-(ID, doy)** regression coefficients that can later be used for forecasting / filling `td`

Inspiration / related work:
- **Improving Subseasonal Forecasting in the Western U.S. with Machine Learning**  
  Jessica Hwang, Paulo Orenstein, Judah Cohen, Karl Pfeiffer, Lester Mackey  
  arXiv:1809.07394 — https://arxiv.org/abs/1809.07394  
  DOI: https://doi.org/10.48550/arXiv.1809.07394  
  We used this as inspiration for the weighted local linear regression framing and feature engineering ideas in our anomaly model.

> This README is focused on what the anomaly model is, what it produces, and how to run the full pipeline end-to-end (including Dask). A section near the end explains how to detect and repair missing/incomplete DOYs.

---

## Quickstart (bash: full pipeline)

You have two ways to run the pipeline:

- **Recommended**: run the one-shot helper script:
  - `scripts/run_pipeline.sh`
- **Manual**: call the Python modules directly if you want to inspect or customize each phase

### Environment (recommended)
```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# install the project and its runtime deps
pip install -e .

# or install the main runtime deps manually
pip install "dask[distributed]" geopandas numpy pandas pyarrow statsmodels tqdm
```

### Option A (recommended): run the helper script

The helper script runs:
1) compute climatology
2) build a **bucketed merged training dataset** (`TD`, `TMIN`, `doy`) on disk
3) shard climatology by the same ID buckets
4) train anomaly coefficients **by bucket** with Dask
5) DOY completeness check
6) rerun only failed DOYs into a patch dataset
7) patch + re-check

```bash
bash scripts/run_pipeline.sh \
  --base "/path/to/base" \
  --results "/path/to/results" \
  --grid "/path/to/grid.dbf" \
  --n-workers 8 \
  --threads 4 \
  --mem 16GB \
  --batch-size 64 \
  --num-buckets 1024
```

Outputs written under `--results`:
- `daily_climatology.parquet`
- `bucketed_training_data/`
- `climatology_by_bucket/`
- `llr_coeffs_anomaly_dataset/`
- `anomaly_failures/`
- `llr_coeffs_anomaly_patch_dataset/` (only if failures exist)
- `anomaly_patch_failures/` (only if failures exist)

Useful flags:
```bash
# print resolved settings, don’t run heavy steps
DRY_RUN=1 bash scripts/run_pipeline.sh --base "$BASE" --results "$RESULTS" --grid "$GRID"

# if you already prepared the bucketed inputs and only want model fitting
bash scripts/run_pipeline.sh --base "$BASE" --results "$RESULTS" --grid "$GRID" --skip-climatology --skip-prepare --skip-patch

# rebuild bucketed training shards and overwrite coefficient outputs
bash scripts/run_pipeline.sh --base "$BASE" --results "$RESULTS" --grid "$GRID" --overwrite-prepared --overwrite-clim-buckets --overwrite-train-output
```

### Option B (manual): run each step

These examples show the full workflow:
1) compute climatology
2) build a bucketed TD/TMIN training dataset
3) shard climatology by the same buckets
4) train coefficients by bucket with Dask
5) check which DOYs are incomplete
6) rerun only those DOYs (Dask) into a patch dataset
7) patch the coefficient dataset and re-check

There is no installed console script; you run Python modules/scripts directly.

#### 0) Define paths once (bash variables)
```bash
BASE="/path/to/base"                 # contains td/Outputs and tmin_v1/Outputs
RESULTS="/path/to/results"           # where you will write outputs
GRID="/path/to/grid.dbf"             # grid file defining IDs (dbf/shp/gpkg/geojson/csv/parquet)

CLIM="${RESULTS}/daily_climatology.parquet"
PREPARED="${RESULTS}/bucketed_training_data"
CLIM_BUCKETS="${RESULTS}/climatology_by_bucket"
COEFFS="${RESULTS}/llr_coeffs_anomaly_dataset"
FAILURES="${RESULTS}/anomaly_failures"
PATCH="${RESULTS}/llr_coeffs_anomaly_patch_dataset"
PATCH_FAILURES="${RESULTS}/anomaly_patch_failures"
```

#### 1) Compute daily climatology (TD_clim and TMIN_clim)
This step reads monthly TD/TMIN parquet inputs for the training years and writes a single climatology parquet.

```bash
python -c '
from pathlib import Path
from tdew_estimation.climatology import calculate_and_save_climatology_chunked

calculate_and_save_climatology_chunked(
    year_range=(1981, 2016),
    base_path=Path("'"${BASE}"'"),
    output_file=Path("'"${CLIM}"'"),
    td_var="td",
    tmin_var="tmin_v1",
    outputs_subdir="Outputs",
    engine="pyarrow",
    cleanup=True,
)
print("Wrote:", "'"${CLIM}"'")
'
```

Expected output:
- `daily_climatology.parquet` with columns: `ID`, `doy`, `TD_clim`, `TMIN_clim`

#### 2) Build bucketed training shards and bucketed climatology
This step pays the expensive TD/TMIN merge once, writes merged monthly shards partitioned by `id_bucket`, and writes matching climatology shards.

```bash
python -c '
from pathlib import Path
from tdew_estimation.bucketed_data import (
    build_bucketed_training_dataset,
    shard_climatology_by_bucket,
)

build_bucketed_training_dataset(
    year_range=(1981, 2016),
    base_path=Path("'"${BASE}"'"),
    output_dir=Path("'"${PREPARED}"'"),
    td_var="td",
    tmin_var="tmin_v1",
    outputs_subdir="Outputs",
    num_buckets=1024,
    overwrite=False,
)

shard_climatology_by_bucket(
    climatology_path=Path("'"${CLIM}"'"),
    output_dir=Path("'"${CLIM_BUCKETS}"'"),
    num_buckets=1024,
    overwrite=False,
)
print("Prepared:", "'"${PREPARED}"'")
print("Climatology shards:", "'"${CLIM_BUCKETS}"'")
'
```

Expected outputs:
- yearly merged bucket shards under `bucketed_training_data/id_bucket=*/train_YYYY.parquet`
- one climatology parquet per bucket under `climatology_by_bucket/id_bucket=*/climatology.parquet`

#### 3) Train anomaly coefficients by bucket with Dask
This runs one Dask task per bucket, writes one coefficient parquet per bucket, and writes structured failure logs per bucket.

```bash
python -c '
from pathlib import Path
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import DaskAnomalyConfig, run_bucketed_anomaly_training_dask

cfg = AnomalyTrainingConfig(
    base_path=Path("'"${BASE}"'"),
    td_var="td",
    tmin_var="tmin_v1",
    train_year_range=(1981, 2016),
    h=11,
    kernel="Tricube",
    min_samples=15,
)

dc = DaskAnomalyConfig(
    n_workers=8,
    threads_per_worker=4,
    memory_limit="16GB",
    batch_size=64,
)

summaries = run_bucketed_anomaly_training_dask(
    prepared_training_root=Path("'"${PREPARED}"'"),
    bucketed_climatology_root=Path("'"${CLIM_BUCKETS}"'"),
    coeffs_output_root=Path("'"${COEFFS}"'"),
    failure_output_root=Path("'"${FAILURES}"'"),
    config=cfg,
    dask_config=dc,
    overwrite=False,
)
print("Buckets processed:", len(summaries))
'
```

Expected outputs:
- coefficient dataset under `llr_coeffs_anomaly_dataset/id_bucket=*/coeffs.parquet`
- failure logs under `anomaly_failures/id_bucket=*/failures.parquet`

#### 4) Check which DOYs are incomplete (failures)
This checks, for each DOY, how many distinct IDs exist in the coefficient dataset and compares that to the expected ID count from the grid file.

```bash
python -c '
from pathlib import Path
from tdew_estimation.grid import expected_count_from_grid
from tdew_estimation.checks import detect_incomplete_anomaly_doys

expected = expected_count_from_grid(Path("'"${GRID}"'"), id_column="ID")
report = detect_incomplete_anomaly_doys(
    Path("'"${COEFFS}"'"),
    expected_count=expected,
    id_col="ID",
    doy_col="doy",
    strict_inequality=True,
    include_missing=True,
    verbose=True,
)
print("DOYS_TO_FIX =", report.incomplete_doys)
'
```

If `DOYS_TO_FIX` is empty, you’re done.

#### 5) Retrain only the failed DOYs with Dask (create a patch dataset)
This reruns training restricted to `DOYS_TO_FIX` for all buckets and writes a patch coefficient dataset.

```bash
python -c '
from pathlib import Path
from tdew_estimation.grid import expected_count_from_grid
from tdew_estimation.checks import detect_incomplete_anomaly_doys
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import rerun_failed_doys_with_bucketed_dask, DaskAnomalyConfig

grid = Path("'"${GRID}"'")
coeffs = Path("'"${COEFFS}"'")
patch = Path("'"${PATCH}"'")

expected = expected_count_from_grid(grid, id_column="ID")
report = detect_incomplete_anomaly_doys(coeffs, expected_count=expected, verbose=False)
doys_to_fix = report.incomplete_doys
print("DOYS_TO_FIX =", doys_to_fix)

cfg = AnomalyTrainingConfig(
    base_path=Path("'"${BASE}"'"),
    td_var="td",
    tmin_var="tmin_v1",
    train_year_range=(1981, 2016),
    h=11,
    kernel="Tricube",
    min_samples=15,
)

dc = DaskAnomalyConfig(n_workers=8, threads_per_worker=4, memory_limit="16GB", batch_size=64)

if doys_to_fix:
    rerun_failed_doys_with_bucketed_dask(
        prepared_training_root=Path("'"${PREPARED}"'"),
        bucketed_climatology_root=Path("'"${CLIM_BUCKETS}"'"),
        patch_output_root=patch,
        failure_output_root=Path("'"${PATCH_FAILURES}"'"),
        failed_doys=doys_to_fix,
        config=cfg,
        dask_config=dc,
    )
    print("Wrote patch dataset:", patch)
else:
    print("No DOYs to fix; skipping patch creation.")
'
```

#### 6) Patch the coefficient dataset and re-check completeness
This replaces the failed DOY rows in the base coefficient dataset with the rerun results from the patch dataset.

```bash
python -c '
from pathlib import Path
from tdew_estimation.grid import expected_count_from_grid
from tdew_estimation.checks import detect_incomplete_anomaly_doys
from tdew_estimation.patch_coeffs import patch_anomaly_coeffs_inplace

grid = Path("'"${GRID}"'")
coeffs = Path("'"${COEFFS}"'")
patch = Path("'"${PATCH}"'")

expected = expected_count_from_grid(grid, id_column="ID")
report = detect_incomplete_anomaly_doys(coeffs, expected_count=expected, verbose=False)
doys_to_fix = report.incomplete_doys

if not doys_to_fix:
    print("No DOYs to patch; coefficients already complete.")
else:
    summary = patch_anomaly_coeffs_inplace(
        base_coeffs_path=coeffs,
        patch_coeffs_path=patch,
        doys_to_patch=doys_to_fix,
        id_col="ID",
        doy_col="doy",
    )
    print("Patch summary:", summary)

    report2 = detect_incomplete_anomaly_doys(coeffs, expected_count=expected, verbose=True)
    print("OK:", report2.ok)
'
```

---

## Running Dask across multiple computers (design guide; not implemented here)

The current code starts a *local* Dask cluster inside `run_bucketed_anomaly_training_dask(...)` by calling `Client(n_workers=..., threads_per_worker=..., ...)`.

To run across multiple machines, you keep the **training logic unchanged** and only change **how the Dask `Client` is created**:
- start a scheduler + workers outside Python (on multiple computers)
- connect the driver script to the remote scheduler address

### High-level architecture

You will run:
- 1 machine as the **scheduler** (coordinates tasks)
- N machines as **workers** (execute bucket training tasks)
- 1 machine as the **driver** (your script that submits tasks; can be the scheduler machine, but doesn’t have to be)

### Network and security prerequisites (typical)

Before you try multi-host Dask, confirm with your IT/security team:

1) **Network reachability**
   - Workers must be able to reach the scheduler host (by DNS name or fixed IP).
   - The driver must be able to reach the scheduler as well.

2) **Firewall / ports (concrete examples)**
   - Dask distributed requires TCP connectivity between scheduler, workers, and your driver.
   - Common/default ports you’ll see in practice:
     - **Scheduler port**: `8786/tcp` (clients + workers connect here)
     - **Dashboard port**: `8787/tcp` (optional web UI)
     - **Worker ports**: can be **dynamic/ephemeral** by default unless you pin them
   - In many organizations, inbound ports are blocked by default. The minimum you typically need is:
     - Allow **workers → scheduler** to reach `SCHEDULER_HOST:8786/tcp`
     - Allow **driver → scheduler** to reach `SCHEDULER_HOST:8786/tcp`
     - (Optional) Allow **your browser → scheduler** to reach `SCHEDULER_HOST:8787/tcp` for the dashboard
   - If security policy requires a fixed port range for workers, plan to pin worker ports to a known range (example: `9000–9100/tcp`) and open that range as needed.

3) **VPN / routing**
   - If the machines are on different networks/subnets, you may need:
     - VPN access, or
     - a routed network path between subnets, or
     - a bastion/jump host strategy.
   - Make sure the scheduler host is reachable from all worker nodes over the VPN.
   - In many orgs, you’ll need explicit permission to run compute traffic over the VPN.

4) **Authentication / authorization**
   - In secured environments, running a scheduler that accepts arbitrary task submissions may require:
     - running inside a trusted network segment,
     - access controls at the network layer, and/or
     - TLS and authenticated connections (recommended if crossing trust boundaries).
   - If your org requires it, use Dask’s TLS support and distribute certs/keys securely (do not commit secrets).

5) **Data access**
   - All workers must be able to read the same training parquet inputs (TD/TMIN), bucketed training shards, and coefficient / patch datasets.
   - Common patterns:
     - **Shared filesystem** (NFS/SMB/Lustre): simplest; paths must be valid on every machine.
     - **Object storage** (S3/GCS/Azure): requires adapting I/O paths and credentials; not implemented here.
   - Avoid “local-only” paths that exist only on the driver machine.

### Operational requirements

- **Consistent environment**: all nodes should run the same Python version and package versions (including `dask[distributed]`, `pandas`, parquet engine).
- **Resource sizing**:
  - Workers should have enough RAM to load one bucket shard of merged TD/TMIN rows and the matching climatology shard.
  - Tune number of workers and threads based on CPU/RAM and parquet I/O throughput.
- **Logging / observability**:
  - For a cluster, plan where logs go (local logs, central logging, etc.).
  - Consider monitoring task failures and re-running subsets (your DOY patch workflow helps).

### What exact code you must add/change (where and what)

#### File: `tdew_estimation/anomaly_dask.py`
In `run_bucketed_anomaly_training_dask(...)`, change the function signature to accept a remote scheduler address:

1) Add this parameter to `run_bucketed_anomaly_training_dask(...)`:
- `scheduler_address: Optional[str] = None`

2) Replace the current “Start Dask client (local cluster via distributed defaults)” block with a conditional:

- If `scheduler_address` is provided:
  - connect to the existing multi-host scheduler:
    - `client = Client(scheduler_address, timeout=...)`
  - example values (match your scheduler host/port):
    - `scheduler_address="tcp://SCHEDULER_HOST:8786"`
    - `scheduler_address="tls://SCHEDULER_HOST:8786"` (if you enable TLS)
- Else (existing behavior):
  - start a local cluster as today:
    - `client = Client(n_workers=..., threads_per_worker=..., memory_limit=..., timeout=...)`

That’s the only code change required to support multi-computer execution.

#### File: `tdew_estimation/anomaly_dask.py`
In `rerun_failed_doys_with_bucketed_dask(...)`, forward the scheduler address:
- Add `scheduler_address: Optional[str] = None` to the signature
- Pass it through when calling `run_bucketed_anomaly_training_dask(...)`

#### What you must start outside Python (operational, not implemented here)
- Start scheduler on a reachable host. Typical/default:
  - scheduler listens on `8786/tcp`
  - dashboard on `8787/tcp` (optional)
- Start workers on each machine and point them to:
  - `tcp://SCHEDULER_HOST:8786`
- Ensure your network/firewall rules allow at least:
  - driver → `SCHEDULER_HOST:8786/tcp`
  - workers → `SCHEDULER_HOST:8786/tcp`
  - (optional) your browser → `SCHEDULER_HOST:8787/tcp`

### Minimal usage pattern (conceptual)

This repository does not ship cluster bootstrap scripts. Conceptually, you would:
1) Start a scheduler on a reachable host (e.g., the machine with IP/DNS `SCHEDULER_HOST`).
2) Start workers on each worker machine, pointing them at the scheduler address.
3) Run your driver script (the Python `-c` blocks in this README) but connect to the scheduler by passing:
   - `scheduler_address="tcp://SCHEDULER_HOST:8786"` (example)

The `run_bucketed_anomaly_training_dask(...)` function is designed so that the only refactor is the `Client(...)` creation; the bucketed anomaly training logic remains the same.

---

## Conceptual overview (what the anomaly model does)

### Inputs (time series)
For each grid cell `ID` and date `FECHA`, the pipeline expects daily parquet inputs for:
- `td`  (dewpoint) stored as `Value` → renamed to `TD`
- `tmin_v1` (minimum temperature) stored as `Value` → renamed to `TMIN`

Required columns in the parquet inputs:
- `ID` (grid cell / location identifier)
- `FECHA` (date)
- `Value` (numeric value)

Typical monthly file convention used by the original pipeline:
- `{base_path}/td/Outputs/td_daily_YYYY_MM.parquet`
- `{base_path}/tmin_v1/Outputs/tmin_daily_YYYY_MM.parquet` (legacy naming)
  - also supports `{variable}_daily_YYYY_MM.parquet`

### Step 1 — Daily climatology per (ID, doy)
For each `ID` and each day-of-year `doy ∈ [1..366]`, compute:
- `TD_clim(ID, doy)` = mean daily TD across training years on that DOY
- `TMIN_clim(ID, doy)` = mean daily TMIN across training years on that DOY

The climatology is stored as:
- `daily_climatology.parquet` with columns: `ID`, `doy`, `TD_clim`, `TMIN_clim`

### Step 2 — Convert to anomalies
For each observation:
- `TD_anom = TD - TD_clim(ID, doy)`
- `TMIN_anom = TMIN - TMIN_clim(ID, doy)`

Create lag features (time-lagged anomalies):
- `TD_anom_lag1 = TD_anom(t-1)`
- `TD_anom_lag2 = TD_anom(t-2)`
- `TMIN_anom_lag1 = TMIN_anom(t-1)`

### Step 3 — Local linear regression per DOY (per grid cell)
For each grid cell `ID` and each target DOY `doy_target`, fit a weighted regression in a DOY neighborhood:

Model form (per `ID`, `doy_target`):
- `TD_anom ≈ const_anom
            + TMIN_anom * TMIN_anom_coeff
            + TD_anom_lag1 * TD_anom_lag1
            + TD_anom_lag2 * TD_anom_lag2
            + TMIN_anom_lag1 * TMIN_anom_lag1`

Neighborhood selection:
- training samples are taken from a circular DOY window around `doy_target` of half-width `h`
- circular distance is used: `dist = min(|doy - doy_target|, 366 - |doy - doy_target|)`

Weights:
- default: **Tricube kernel**, matching the notebook implementation:
  - `w = (1 - |dist/h|^3)^3` for `dist ≤ h`, else 0
- optionally: Gaussian kernel

Fitting:
- weighted least squares (WLS)

Output coefficients (per `ID`, `doy`):
- `const_anom`
- `TMIN_anom_coeff`
- `TD_anom_lag1`
- `TD_anom_lag2`
- `TMIN_anom_lag1`
- plus diagnostics such as `r_squared_anom` (when available)

These coefficients are stored in:
- `llr_coeffs_anomaly_dataset/id_bucket=*/coeffs.parquet`

---

## Running at scale (Dask-oriented execution)

The original implementation was designed to run for *many* IDs and *all* DOYs, which is expensive. The practical approach is:

1. **Precompute climatology once** for the training period.
2. Build a **bucketed merged training dataset** once from monthly TD/TMIN inputs.
3. Shard climatology by the same ID buckets.
4. Run anomaly training in parallel across buckets.

Typical pattern:
- Use a Dask scheduler (`dask.distributed.Client`)
- Submit one task per bucket
- Each task reads one bucket shard, fits all IDs in that bucket, and writes `coeffs.parquet`
- Store structured worker failures alongside the coefficient dataset

Why Dask helps:
- parallelizes across spatial buckets
- isolates failures (a single ID failure should not kill the full run)
- supports batching to control scheduler pressure

Key knobs you typically tune:
- number of workers / threads per worker
- number of ID buckets
- bucket submit batch size (to manage scheduler overhead)
- neighborhood half-width `h`
- minimum samples per DOY regression

> This repository uses **Dask-only orchestration** for bucket-based training runs. The core regression math remains in `tdew_estimation.anomaly_train`, while bucket preparation and orchestration live in `tdew_estimation.bucketed_data` and `tdew_estimation.anomaly_dask`.

---

## Outputs / artifacts (what you should expect on disk)

Within your `results_dir` you typically have:

### 1) Climatology
- `daily_climatology.parquet`
  - columns: `ID`, `doy`, `TD_clim`, `TMIN_clim`

### 2) Anomaly coefficients
- `llr_coeffs_anomaly_dataset/`
  - one parquet file per bucket: `id_bucket=*/coeffs.parquet`
  - key columns: `ID`, `doy`
  - coefficient columns: `const_anom`, `TMIN_anom_coeff`, `TD_anom_lag1`, `TD_anom_lag2`, `TMIN_anom_lag1`
  - diagnostics: e.g., `r_squared_anom`

### 3) Prepared training shards
- `bucketed_training_data/id_bucket=*/train_YYYY.parquet`
  - merged yearly TD/TMIN training rows partitioned by `id_bucket`

### 4) Climatology shards
- `climatology_by_bucket/id_bucket=*/climatology.parquet`
  - one climatology parquet per bucket

### 5) Failure logs
- `anomaly_failures/id_bucket=*/failures.parquet`
  - structured bucket / ID / DOY failures recorded during training

---

## Code organization (extracted modules)

The extracted, path-agnostic modules live under `tdew_estimation/`:

- `grid.py`
  - loads grid IDs from a grid file (DBF/SHP/GPKG/GeoJSON/CSV/Parquet)
  - computes `EXPECTED_COUNT` (number of unique IDs)

- `checks.py`
  - detects incomplete DOYs in a coefficient parquet file or a bucketed coefficient dataset using “Option A”
  - produces a copy-paste-friendly `DOYS_TO_FIX` list

- `anomaly_train.py`
  - core anomaly coefficient training logic
  - supports fitting from already prepared TD/TMIN frames and restricting to a subset of DOYs

- `patch_coeffs.py`
  - patches a single coefficient parquet or a bucketed coefficient dataset by replacing only the DOYs being repaired

- `anomaly_dask.py`
  - Dask-based orchestration for anomaly training at scale
  - runs one task per bucket and writes bucket-local coefficient outputs
  - records structured failure logs
  - includes a helper to rerun only failed/incomplete DOYs into a patch dataset

- `bucketed_data.py`
  - builds the bucketed merged TD/TMIN training dataset
  - shards climatology by the same bucket layout

- `main.py`
  - a lightweight CLI/example entrypoint wiring: detect → retrain DOYs (bucketed Dask) → patch → re-check

---

## Small but practical: repairing failed / incomplete DOYs

Large distributed runs can occasionally produce partial results for some DOYs. The repository supports a lightweight repair strategy:

**Definition (Option A, as used in your workflow):**
For each `doy`, count distinct IDs present in the coefficient table.
If `actual_count != EXPECTED_COUNT`, then that DOY is incomplete and should be retrained.

Repair workflow:
1. Compute `EXPECTED_COUNT` from the grid file.
2. Detect `DOYS_TO_FIX` from `llr_coeffs_anomaly_dataset/`.
3. Retrain anomaly coefficients **only** for those DOYs into a bucketed patch dataset.
4. Patch the base coefficient dataset by replacing those DOYs.
5. Re-check completeness.

This keeps the repair step small and avoids re-running the full 366-DOY training.

---

## Notes / future improvements (not implemented here yet)

- Remote-scheduler support for multi-host Dask runs
- Direct NumPy weighted least-squares to reduce `statsmodels` overhead
- Forecast refactor to use the same bucketed data layout as training
- A dedicated CLI entrypoint (thin layer over the extracted modules)
- Rolling-origin, time-blocked cross-validation on a representative subset of IDs for tuning `h`, kernels, and version-shift corrections
- Testing whether a single global `h` is sufficient or whether broad macro-regions in Peru should use different `h` values, adopting regional bandwidths only if validation shows a clear gain
