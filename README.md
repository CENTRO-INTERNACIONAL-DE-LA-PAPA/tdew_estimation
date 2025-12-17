# tdew_estimation

This repository contains the code used to estimate daily dewpoint temperature (`td`) needed to run Simcast workflows when `td` is not directly available.

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
  - `tdew_estimation/scripts/run_pipeline.sh`
- **Manual**: run each step as a `python -c '...'` block (kept below for transparency and easy modification)

### Environment (recommended)
```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# install runtime deps (adjust to your environment)
pip install "dask[distributed]" pandas pyarrow statsmodels tqdm
```

### Option A (recommended): run the helper script

The helper script runs:
1) compute climatology
2) Dask training (chunks + combined parquet)
3) DOY completeness check
4) Dask rerun failures (patch parquet)
5) patch + re-check

```bash
bash tdew_estimation/scripts/run_pipeline.sh \
  --base "/path/to/base" \
  --results "/path/to/results" \
  --grid "/path/to/grid.dbf" \
  --n-workers 8 \
  --threads 4 \
  --mem 16GB \
  --batch-size 1000
```

Outputs written under `--results`:
- `daily_climatology.parquet`
- `anomaly_coeffs_chunks/` (batch chunk parquets)
- `llr_coeffs_anomaly_final_direct.parquet` (combined coefficients)
- `llr_coeffs_anomaly_patch_doys.parquet` (only if failures exist)

Useful flags:
```bash
# print resolved settings, don’t run heavy steps
DRY_RUN=1 bash tdew_estimation/scripts/run_pipeline.sh --base "$BASE" --results "$RESULTS" --grid "$GRID"

# if you want to re-run only the Dask training step
bash tdew_estimation/scripts/run_pipeline.sh --base "$BASE" --results "$RESULTS" --grid "$GRID" --skip-climatology --skip-patch

# overwrite existing chunk files (careful)
bash tdew_estimation/scripts/run_pipeline.sh --base "$BASE" --results "$RESULTS" --grid "$GRID" --overwrite-chunks
```

### Option B (manual): run each step

These examples show the full workflow:
1) compute climatology
2) train anomaly coefficients at scale (Dask) into chunk files and combine
3) check which DOYs are incomplete
4) rerun only those DOYs (Dask) to create a patch parquet
5) patch the combined coefficients and re-check

There is no installed console script; you run Python modules/scripts directly.

#### 0) Define paths once (bash variables)
```bash
BASE="/path/to/base"                 # contains td/Outputs and tmin_v1/Outputs
RESULTS="/path/to/results"           # where you will write outputs
GRID="/path/to/grid.dbf"             # grid file defining IDs (dbf/shp/gpkg/geojson/csv/parquet)

CLIM="${RESULTS}/daily_climatology.parquet"
CHUNKS="${RESULTS}/anomaly_coeffs_chunks"
COEFFS="${RESULTS}/llr_coeffs_anomaly_final_direct.parquet"

PATCH="${RESULTS}/llr_coeffs_anomaly_patch_doys.parquet"
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

#### 2) Train anomaly coefficients at scale (Dask) and combine chunks
This runs per-ID training tasks with Dask, writes chunk parquets, and optionally combines them into a final coefficient parquet.

```bash
python -c '
from pathlib import Path
from tdew_estimation.grid import load_grid_ids
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import run_anomaly_training_dask, DaskAnomalyConfig

ids = load_grid_ids(Path("'"${GRID}"'"), id_column="ID")

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
    batch_size=1000,
)

run_anomaly_training_dask(
    ids=ids,
    config=cfg,
    climatology_path=Path("'"${CLIM}"'"),
    chunk_dir=Path("'"${CHUNKS}"'"),
    combine_output_path=Path("'"${COEFFS}"'"),
    dask_config=dc,
    overwrite_chunks=False,
)
print("Wrote:", "'"${COEFFS}"'")
'
```

Expected outputs:
- chunk files under `results/anomaly_coeffs_chunks/batch_*.parquet`
- combined coefficients parquet:
  - `llr_coeffs_anomaly_final_direct.parquet`

#### 3) Check which DOYs are incomplete (failures)
This checks, for each DOY, how many distinct IDs exist in the combined coefficient parquet and compares that to the expected ID count from the grid file.

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

#### 4) Retrain only the failed DOYs with Dask (create a patch parquet)
This reruns training restricted to `DOYS_TO_FIX` for all IDs and writes a patch parquet containing only those DOY rows.

```bash
python -c '
from pathlib import Path
from tdew_estimation.grid import load_grid_ids, expected_count_from_grid
from tdew_estimation.checks import detect_incomplete_anomaly_doys
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import rerun_failed_doys_with_dask, DaskAnomalyConfig

grid = Path("'"${GRID}"'")
coeffs = Path("'"${COEFFS}"'")
clim = Path("'"${CLIM}"'")
patch = Path("'"${PATCH}"'")

expected = expected_count_from_grid(grid, id_column="ID")
report = detect_incomplete_anomaly_doys(coeffs, expected_count=expected, verbose=False)
doys_to_fix = report.incomplete_doys
print("DOYS_TO_FIX =", doys_to_fix)

ids = load_grid_ids(grid, id_column="ID")

cfg = AnomalyTrainingConfig(
    base_path=Path("'"${BASE}"'"),
    td_var="td",
    tmin_var="tmin_v1",
    train_year_range=(1981, 2016),
    h=11,
    kernel="Tricube",
    min_samples=15,
)

dc = DaskAnomalyConfig(n_workers=8, threads_per_worker=4, memory_limit="16GB", batch_size=1000)

if doys_to_fix:
    rerun_failed_doys_with_dask(
        ids=ids,
        failed_doys=doys_to_fix,
        config=cfg,
        climatology_path=clim,
        patch_output_path=patch,
        dask_config=dc,
    )
    print("Wrote patch:", patch)
else:
    print("No DOYs to fix; skipping patch creation.")
'
```

#### 5) Patch the combined coefficients and re-check completeness
This replaces the failed DOY rows in the combined coefficients parquet with the rerun results from the patch parquet.

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

The current code starts a *local* Dask cluster inside `run_anomaly_training_dask(...)` by calling `Client(n_workers=..., threads_per_worker=..., ...)`.

To run across multiple machines, you keep the **training logic unchanged** and only change **how the Dask `Client` is created**:
- start a scheduler + workers outside Python (on multiple computers)
- connect the driver script to the remote scheduler address

### High-level architecture

You will run:
- 1 machine as the **scheduler** (coordinates tasks)
- N machines as **workers** (execute `train_anomaly_coeffs_for_one_id` tasks)
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
   - All workers must be able to read the same training parquet inputs (TD/TMIN) and write outputs (chunk/patch parquet).
   - Common patterns:
     - **Shared filesystem** (NFS/SMB/Lustre): simplest; paths must be valid on every machine.
     - **Object storage** (S3/GCS/Azure): requires adapting I/O paths and credentials; not implemented here.
   - Avoid “local-only” paths that exist only on the driver machine.

### Operational requirements

- **Consistent environment**: all nodes should run the same Python version and package versions (including `dask[distributed]`, `pandas`, parquet engine).
- **Resource sizing**:
  - Workers should have enough RAM to load per-ID time series and the climatology broadcast.
  - Tune number of workers and threads based on CPU/RAM and parquet I/O throughput.
- **Logging / observability**:
  - For a cluster, plan where logs go (local logs, central logging, etc.).
  - Consider monitoring task failures and re-running subsets (your DOY patch workflow helps).

### What exact code you must add/change (where and what)

#### File: `tdew_estimation/tdew_estimation/anomaly_dask.py`
In `run_anomaly_training_dask(...)`, change the function signature to accept a remote scheduler address:

1) Add this parameter to `run_anomaly_training_dask(...)`:
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

#### File: `tdew_estimation/tdew_estimation/anomaly_dask.py`
In `rerun_failed_doys_with_dask(...)`, forward the scheduler address:
- Add `scheduler_address: Optional[str] = None` to the signature
- Pass it through when calling `run_anomaly_training_dask(...)`

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

The `run_anomaly_training_dask(...)` function is designed so that the only refactor is the `Client(...)` creation; the anomaly training logic remains the same.

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
- `llr_coeffs_anomaly_final_direct.parquet` (combined output)

---

## Running at scale (Dask-oriented execution)

The original implementation was designed to run for *many* IDs and *all* DOYs, which is expensive. The practical approach is:

1. **Precompute climatology once** for the training period.
2. Run anomaly training in parallel across IDs (and/or in batches of IDs).
3. Write intermediate chunks to disk and then combine.

Typical pattern:
- Use a Dask scheduler (`dask.distributed.Client`)
- Scatter heavy shared data (e.g., climatology table) to workers once (broadcast)
- Submit per-ID (or per-batch) tasks
- Collect results and write chunk parquet files to `results/anomaly_coeffs_chunks/`
- Combine chunk files into the final coefficient parquet

Why Dask helps:
- parallelizes across spatial IDs
- isolates failures (a single ID failure should not kill the full run)
- supports batching to control memory pressure

Key knobs you typically tune:
- number of workers / threads per worker
- ID batch size (to manage memory and reduce scheduler overhead)
- neighborhood half-width `h`
- minimum samples per DOY regression

> This repository uses **Dask-only orchestration** for multi-ID training runs. The core per-ID fitting function remains in `tdew_estimation.anomaly_train`, while orchestration (batching, chunk writes, combining, and DOY reruns) lives in `tdew_estimation.anomaly_dask`.

---

## Outputs / artifacts (what you should expect on disk)

Within your `results_dir` you typically have:

### 1) Climatology
- `daily_climatology.parquet`
  - columns: `ID`, `doy`, `TD_clim`, `TMIN_clim`

### 2) Anomaly coefficients (combined)
- `llr_coeffs_anomaly_final_direct.parquet`
  - key columns: `ID`, `doy`
  - coefficient columns: `const_anom`, `TMIN_anom_coeff`, `TD_anom_lag1`, `TD_anom_lag2`, `TMIN_anom_lag1`
  - diagnostics: e.g., `r_squared_anom`

### 3) Intermediate chunks (optional)
- `anomaly_coeffs_chunks/batch_*.parquet`
  - per-batch outputs written during distributed execution before combining

---

## Code organization (extracted modules)

The extracted, path-agnostic modules live under `tdew_estimation/tdew_estimation/`:

- `grid.py`
  - loads grid IDs from a grid file (DBF/SHP/GPKG/GeoJSON/CSV/Parquet)
  - computes `EXPECTED_COUNT` (number of unique IDs)

- `checks.py`
  - detects incomplete DOYs in the combined anomaly coefficient parquet using “Option A”
  - produces a copy-paste-friendly `DOYS_TO_FIX` list

- `anomaly_train.py`
  - core anomaly coefficient training logic (per-ID)
  - supports restricting training to a subset of DOYs (used for repair runs submitted via Dask)

- `patch_coeffs.py`
  - patches the combined coefficient parquet by replacing only the DOYs being repaired

- `anomaly_dask.py`
  - Dask-based orchestration for anomaly training at scale
  - runs per-ID training tasks (`train_anomaly_coeffs_for_one_id`) in batches
  - writes chunk parquet files and optionally combines them into a final coefficients parquet
  - includes a helper to rerun only failed/incomplete DOYs and write a patch parquet

- `main.py`
  - an example usage script wiring: detect → retrain DOYs (Dask) → patch → re-check
  - not a CLI; uses arguments only as an example of path-agnostic configuration

---

## Small but practical: repairing failed / incomplete DOYs

Large distributed runs can occasionally produce partial results for some DOYs. The repository supports a lightweight repair strategy:

**Definition (Option A, as used in your workflow):**
For each `doy`, count distinct IDs present in the coefficient table.
If `actual_count != EXPECTED_COUNT`, then that DOY is incomplete and should be retrained.

Repair workflow:
1. Compute `EXPECTED_COUNT` from the grid file.
2. Detect `DOYS_TO_FIX` from `llr_coeffs_anomaly_final_direct.parquet`.
3. Retrain anomaly coefficients **only** for those DOYs (for all IDs) into a “patch” parquet.
4. Patch the combined coefficient parquet by replacing those DOYs.
5. Re-check completeness.

This keeps the repair step small and avoids re-running the full 366-DOY training.

---

## Notes / future improvements (not implemented here yet)

- Better I/O patterns for training (reduce repeated parquet reads per ID)
- Smarter caching strategies for per-ID time series
- More efficient partitioning strategies (per DOY or per ID shards)
- A dedicated CLI entrypoint (thin layer over the extracted modules)
