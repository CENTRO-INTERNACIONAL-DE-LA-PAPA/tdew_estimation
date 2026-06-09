"""
tdew_estimation.forecast

Recursive forecast utilities for the TD anomaly model, extracted and refactored from a
Colab-derived workflow (see `tdew_estimation_arimax.py`).

This module provides:
- A *single-ID* recursive forecast function that:
  - seeds TD lags from a historical period
  - uses known future TMIN inputs (exogenous)
  - applies per-(ID, doy) anomaly coefficients + climatology to generate TD day-by-day
- A Dask-based driver to scale the forecast across many IDs in parallel
- Utilities to combine chunk outputs and split a combined prediction parquet into monthly files

Design goals
------------
- Path-agnostic: no hard-coded paths.
- Functional: the functions are runnable as long as you provide correct inputs (paths/config).
- Conservative dependencies: pandas, numpy, dask/distributed, geopandas optional (only if you want grid reading here).

Data model expectations
-----------------------
Input parquets are expected to use these columns (as in the original pipeline):
- ID: int-like
- FECHA: datetime-like (or parseable)
- Value: float-like

Variables:
- td      -> contains dewpoint values ("Value") to seed history and optionally evaluate
- tmin_v1 -> used as exogenous in history window to seed TMIN lag
- tmin    -> used as exogenous for the forecast horizon (future period)

Climatology parquet:
- columns: ID, doy, TD_clim, TMIN_clim

Anomaly coefficients parquet:
- columns: ID, doy, const_anom, TMIN_anom_coeff, TD_anom_lag1, TD_anom_lag2, TMIN_anom_lag1
  (and optionally other diagnostics; ignored)

Output (predictions):
- columns: FECHA, ID, TD_predicted

Notes on correctness
--------------------
- The recursion uses the predicted TD value as TD lag inputs for subsequent days.
- TMIN uses *observed* (known) future TMIN for each forecast day.
- Anomalies are computed using climatology for the relevant DOYs.
- DOY is computed with pandas dayofyear; leap-day behavior:
  - pandas dayofyear will produce 366 in leap years.
  - coefficient/climatology tables are assumed to have doy 1..366.
  - If your coefficient/climatology were trained with 1..366, this matches.
  - If you need leap-day handling (e.g., mapping 366 -> 365), add that explicitly.

"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .bucket_layout import bucket_dir, discover_bucket_ids
from .parquet_io import as_path, read_parquet_any

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Parquet discovery helpers (monthly files)
# --------------------------------------------------------------------------------------


def find_monthly_parquet_files(
    base_path: PathLike,
    variable: str,
    date_range: Tuple[Union[str, pd.Timestamp], Union[str, pd.Timestamp]],
    *,
    outputs_subdir: str = "Outputs",
    tmin_v1_legacy_name: bool = True,
) -> List[Path]:
    """
    Find monthly parquet files for a variable across a date range.

    Expected naming convention:
      {base_path}/{variable}/Outputs/{variable}_daily_YYYY_MM.parquet

    Legacy support:
      If variable == "tmin_v1" and tmin_v1_legacy_name=True, also accept:
        {base_path}/tmin_v1/Outputs/tmin_daily_YYYY_MM.parquet
    """
    base = Path(base_path).expanduser().resolve()
    start_date, end_date = [pd.Timestamp(d) for d in date_range]
    months = pd.date_range(start_date, end_date, freq="MS").strftime("%Y_%m").unique()

    out_dir = base / variable / outputs_subdir
    files: List[Path] = []
    for ym in months:
        if variable == "tmin_v1" and tmin_v1_legacy_name:
            legacy = out_dir / f"tmin_daily_{ym}.parquet"
            if legacy.exists():
                files.append(legacy)
                continue
        candidate = out_dir / f"{variable}_daily_{ym}.parquet"
        if candidate.exists():
            files.append(candidate)
    return sorted(files)


def read_variable_for_id(
    files: Sequence[Path],
    *,
    location_id: int,
    value_name: str,
    required_cols: Sequence[str] = ("ID", "FECHA", "Value"),
) -> pd.DataFrame:
    """
    Read variable parquet files filtered to a single ID and return a concatenated DataFrame.

    Returns columns:
      ID, FECHA, <value_name>
    """
    dfs: List[pd.DataFrame] = []
    for p in files:
        df_any: object
        try:
            df_any = pd.read_parquet(p, filters=[("ID", "==", location_id)], columns=list(required_cols))
        except TypeError:
            # fallback: read all then filter
            df_any = pd.read_parquet(p)

        df = df_any if isinstance(df_any, pd.DataFrame) else pd.DataFrame(df_any)
        df = pd.DataFrame(df)
        if df.empty:
            continue

        if "ID" in df.columns:
            df = pd.DataFrame(df[df["ID"] == location_id])
        if df.empty:
            continue

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            continue

        df = df[list(required_cols)].copy()
        df = df.rename(columns={"Value": value_name})
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["ID", "FECHA", value_name])

    out = pd.concat(dfs, ignore_index=True)
    out["FECHA"] = pd.to_datetime(out["FECHA"])
    return out


# --------------------------------------------------------------------------------------
# Single-ID recursive forecast
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastConfig:
    """
    Configuration for recursive forecast.

    Variables:
    - history_td_var: td variable name for history seeding (default 'td')
    - history_tmin_var: tmin variable name for history seeding (default 'tmin_v1')
    - future_tmin_var: tmin variable name for future exogenous (default 'tmin')
    """
    base_path: Path
    history_td_var: str = "td"
    history_tmin_var: str = "tmin_v1"
    future_tmin_var: str = "tmin"
    outputs_subdir: str = "Outputs"
    tmin_v1_legacy_name: bool = True


def generate_recursive_forecast_for_prepared_id(
    location_id: int,
    *,
    coeffs_id_df: pd.DataFrame,
    clim_id_df: pd.DataFrame,
    history_df: pd.DataFrame,
    future_tmin_df: pd.DataFrame,
    prediction_years: Tuple[int, int],
) -> Optional[pd.DataFrame]:
    """
    Recursive TD forecast for one ID from already-loaded per-ID frames.

    This is the in-memory core shared by the per-ID disk path
    (``generate_recursive_forecast_for_one_id``) and the bucketed Dask path
    (``run_bucketed_forecast_dask``). It contains no I/O.

    Parameters
    ----------
    coeffs_id_df:
        Anomaly coefficients for this ID. Must contain: doy, const_anom,
        TMIN_anom_coeff, TD_anom_lag1, TD_anom_lag2, TMIN_anom_lag1.
    clim_id_df:
        Climatology for this ID. Must contain: doy, TD_clim, TMIN_clim.
    history_df:
        Historical TD/TMIN used only to seed the two TD/TMIN lags. Must contain
        FECHA, TD, TMIN with at least two rows (the last two by date are used).
    future_tmin_df:
        Exogenous TMIN over the forecast horizon. Must contain FECHA, TMIN.
    prediction_years:
        (start_year, end_year) inclusive forecast horizon.

    Returns
    -------
    DataFrame with columns FECHA, ID, TD_predicted; an empty frame with those
    columns if no day could be predicted; or None if required inputs are missing.
    """
    if coeffs_id_df is None or coeffs_id_df.empty or clim_id_df is None or clim_id_df.empty:
        return None

    required_coeff_cols = {"doy", "const_anom", "TMIN_anom_coeff", "TD_anom_lag1", "TD_anom_lag2", "TMIN_anom_lag1"}
    if not required_coeff_cols.issubset(set(coeffs_id_df.columns)):
        return None
    required_clim_cols = {"doy", "TD_clim", "TMIN_clim"}
    if not required_clim_cols.issubset(set(clim_id_df.columns)):
        return None

    if history_df is None or len(history_df) < 2:
        return None
    if future_tmin_df is None or future_tmin_df.empty:
        return None

    history_df = history_df.sort_values("FECHA")
    last_two_days = history_df.tail(2)[["FECHA", "TD", "TMIN"]].copy().reset_index(drop=True)

    # Index climatology and coeffs by DOY for fast lookup
    clim_id = clim_id_df.copy()
    clim_id["doy"] = pd.to_numeric(clim_id["doy"], errors="coerce").astype("Int64")
    clim_id = clim_id.dropna(subset=["doy"]).copy()
    clim_id["doy"] = clim_id["doy"].astype(int)
    clim_id = clim_id.set_index("doy")

    coeffs_id = coeffs_id_df.copy()
    coeffs_id["doy"] = pd.to_numeric(coeffs_id["doy"], errors="coerce").astype("Int64")
    coeffs_id = coeffs_id.dropna(subset=["doy"]).copy()
    coeffs_id["doy"] = coeffs_id["doy"].astype(int)
    coeffs_id = coeffs_id.set_index("doy")

    future_tmin = future_tmin_df.copy()
    future_tmin["FECHA"] = pd.to_datetime(future_tmin["FECHA"])
    future_tmin = future_tmin.set_index("FECHA")

    pred_start_year, pred_end_year = prediction_years
    pred_range = (f"{pred_start_year}-01-01", f"{pred_end_year}-12-31")
    prediction_dates = pd.date_range(start=pred_range[0], end=pred_range[1], freq="D")
    predictions: List[Dict[str, object]] = []

    def _doy(dt: pd.Timestamp) -> int:
        return int(dt.dayofyear)

    for current_day in prediction_dates:
        doy = _doy(current_day)

        # Require: coefficient and climatology for this DOY and future TMIN for this day
        if doy not in clim_id.index or doy not in coeffs_id.index or current_day not in future_tmin.index:
            continue

        tmin_today = float(future_tmin.loc[current_day, "TMIN"])
        td_lag1 = float(last_two_days.iloc[-1]["TD"])
        td_lag2 = float(last_two_days.iloc[-2]["TD"])
        tmin_lag1 = float(last_two_days.iloc[-1]["TMIN"])

        # Climatology lookups (today, lag1, lag2)
        clim_today = clim_id.loc[doy]

        doy_lag1 = _doy(current_day - pd.Timedelta(days=1))
        doy_lag2 = _doy(current_day - pd.Timedelta(days=2))
        if doy_lag1 not in clim_id.index or doy_lag2 not in clim_id.index:
            continue

        clim_lag1 = clim_id.loc[doy_lag1]
        clim_lag2 = clim_id.loc[doy_lag2]

        # Anomalies
        tmin_anom = tmin_today - float(clim_today["TMIN_clim"])
        td_anom_lag1 = td_lag1 - float(clim_lag1["TD_clim"])
        td_anom_lag2 = td_lag2 - float(clim_lag2["TD_clim"])
        tmin_anom_lag1 = tmin_lag1 - float(clim_lag1["TMIN_clim"])

        coeffs = coeffs_id.loc[doy]

        predicted_anomaly = (
            float(coeffs["const_anom"])
            + (tmin_anom * float(coeffs["TMIN_anom_coeff"]))
            + (td_anom_lag1 * float(coeffs["TD_anom_lag1"]))
            + (td_anom_lag2 * float(coeffs["TD_anom_lag2"]))
            + (tmin_anom_lag1 * float(coeffs["TMIN_anom_lag1"]))
        )

        predicted_td = predicted_anomaly + float(clim_today["TD_clim"])

        predictions.append(
            {"FECHA": current_day, "ID": int(location_id), "TD_predicted": float(predicted_td)}
        )

        # Update rolling window (keep last two days)
        new_row = pd.DataFrame([{"FECHA": current_day, "TD": predicted_td, "TMIN": tmin_today}])
        last_two_days = pd.concat([last_two_days.iloc[1:], new_row], ignore_index=True)

    if not predictions:
        return pd.DataFrame(columns=["FECHA", "ID", "TD_predicted"])

    return pd.DataFrame(predictions)


def generate_recursive_forecast_for_one_id(
    location_id: int,
    *,
    prediction_years: Tuple[int, int],
    history_end_year: int,
    coeffs_path: PathLike,
    climatology_path: PathLike,
    config: ForecastConfig,
) -> Optional[pd.DataFrame]:
    """
    Generate daily TD predictions for a single ID using anomaly coefficients and climatology.

    Parameters
    ----------
    location_id:
        Spatial ID to forecast.
    prediction_years:
        (start_year, end_year) inclusive for forecast horizon.
    history_end_year:
        Year used to seed the lag window (the function uses that entire year).
    coeffs_path:
        Parquet file with anomaly coefficients (combined), filtered to the ID.
    climatology_path:
        Parquet file with climatology, filtered to the ID.
    config:
        ForecastConfig controlling base_path and variable names.

    Returns
    -------
    DataFrame with columns: FECHA, ID, TD_predicted
    or None if required inputs are missing.
    """
    coeffs_p = Path(coeffs_path).expanduser().resolve()
    clim_p = Path(climatology_path).expanduser().resolve()

    # Load coeffs/climatology for ID only if possible (parquet filters)
    try:
        coeffs_df = pd.read_parquet(coeffs_p, filters=[("ID", "==", location_id)])
    except TypeError:
        coeffs_df = pd.read_parquet(coeffs_p)
        coeffs_df = coeffs_df[coeffs_df["ID"] == location_id]

    try:
        clim_df = pd.read_parquet(clim_p, filters=[("ID", "==", location_id)])
    except TypeError:
        clim_df = pd.read_parquet(clim_p)
        clim_df = clim_df[clim_df["ID"] == location_id]

    if coeffs_df.empty or clim_df.empty:
        return None

    # Required coefficient columns (as used in the original fixed code)
    required_coeff_cols = {"doy", "const_anom", "TMIN_anom_coeff", "TD_anom_lag1", "TD_anom_lag2", "TMIN_anom_lag1"}
    if not required_coeff_cols.issubset(set(coeffs_df.columns)):
        return None

    required_clim_cols = {"doy", "TD_clim", "TMIN_clim"}
    if not required_clim_cols.issubset(set(clim_df.columns)):
        return None

    # Load history: last year TD + TMIN to seed lags
    hist_range = (f"{history_end_year}-01-01", f"{history_end_year}-12-31")
    td_files = find_monthly_parquet_files(
        config.base_path,
        config.history_td_var,
        hist_range,
        outputs_subdir=config.outputs_subdir,
        tmin_v1_legacy_name=config.tmin_v1_legacy_name,
    )
    tmin_hist_files = find_monthly_parquet_files(
        config.base_path,
        config.history_tmin_var,
        hist_range,
        outputs_subdir=config.outputs_subdir,
        tmin_v1_legacy_name=config.tmin_v1_legacy_name,
    )

    if not td_files or not tmin_hist_files:
        return None

    hist_td = read_variable_for_id(td_files, location_id=location_id, value_name="TD")
    hist_tmin = read_variable_for_id(tmin_hist_files, location_id=location_id, value_name="TMIN")
    if hist_td.empty or hist_tmin.empty:
        return None

    history_df = pd.merge(hist_td, hist_tmin, on=["ID", "FECHA"], how="inner").sort_values("FECHA")
    if len(history_df) < 2:
        return None

    # Load future tmin for forecast horizon
    pred_start_year, pred_end_year = prediction_years
    pred_range = (f"{pred_start_year}-01-01", f"{pred_end_year}-12-31")
    future_tmin_files = find_monthly_parquet_files(
        config.base_path,
        config.future_tmin_var,
        pred_range,
        outputs_subdir=config.outputs_subdir,
        tmin_v1_legacy_name=False,
    )
    if not future_tmin_files:
        return None

    future_tmin = read_variable_for_id(future_tmin_files, location_id=location_id, value_name="TMIN")
    if future_tmin.empty:
        return None

    # Delegate the (I/O-free) recursion to the shared in-memory core.
    return generate_recursive_forecast_for_prepared_id(
        location_id,
        coeffs_id_df=coeffs_df,
        clim_id_df=clim_df,
        history_df=history_df,
        future_tmin_df=future_tmin,
        prediction_years=prediction_years,
    )


# --------------------------------------------------------------------------------------
# Dask scaling
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DaskForecastConfig:
    """
    Dask configuration for distributed forecasting.

    Parameters
    ----------
    n_workers:
        Number of Dask worker processes (set to physical cores for CPU-bound work).
    threads_per_worker:
        Threads per worker. Default 1: forecasting is pure-Python + NumPy/BLAS, so
        extra threads contend on the GIL and oversubscribe BLAS. See
        ``_configure_blas_threads``.
    memory_limit:
        Per-worker memory limit, "auto", or None. "auto" splits the machine's total
        memory across workers instead of hard-coding a value that can oversubscribe RAM.
    batch_size:
        Number of IDs per batch (chunk file). Bounds driver memory because each batch's
        per-ID results are concatenated before being written.
    local_directory:
        Worker scratch/spill directory; on HPC point this at fast node-local storage.
    dashboard_address:
        Address for the Dask dashboard, or None to disable.
    task_retries:
        Number of times the scheduler retries a task that fails on a worker
        (used by the bucketed path). 0 disables retries.
    """
    n_workers: int = 8
    threads_per_worker: int = 1
    memory_limit: Optional[str] = "auto"
    batch_size: int = 500  # number of IDs per batch (chunk file); also the bucket in-flight window
    local_directory: Optional[str] = None
    dashboard_address: Optional[str] = ":8787"
    task_retries: int = 2


def _configure_blas_threads() -> None:
    """
    Pin BLAS/OpenMP thread pools to one thread per worker process so nested BLAS
    threads do not oversubscribe the CPU. ``setdefault`` lets an explicit shell env
    var win; must run before the Client (and its workers) are created.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _make_forecast_client(dc: DaskForecastConfig):
    """Create a local Dask distributed Client from a DaskForecastConfig."""
    from dask.distributed import Client

    kwargs = dict(
        n_workers=dc.n_workers,
        threads_per_worker=dc.threads_per_worker,
        memory_limit=dc.memory_limit,
    )
    if dc.local_directory:
        kwargs["local_directory"] = dc.local_directory
    if dc.dashboard_address is not None:
        kwargs["dashboard_address"] = dc.dashboard_address
    return Client(**kwargs)


def forecast_with_dask(
    ids: Sequence[int],
    *,
    prediction_years: Tuple[int, int],
    history_end_year: int,
    coeffs_path: PathLike,
    climatology_path: PathLike,
    config: ForecastConfig,
    dask_config: Optional[DaskForecastConfig] = None,
    chunk_dir: PathLike,
    chunk_prefix: str = "pred_batch_",
    client: Optional[Any] = None,
) -> List[Path]:
    """
    Run recursive forecasts across many IDs using Dask and write chunk parquet files.

    If ``client`` is provided it is reused (and left open); otherwise a local cluster
    is created from ``dask_config`` and closed on exit.

    Returns list of chunk file paths written.
    """
    from dask.distributed import as_completed  # imported lazily

    dc = dask_config or DaskForecastConfig()
    chunk_dir_p = Path(chunk_dir).expanduser().resolve()
    chunk_dir_p.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    _configure_blas_threads()
    owns_client = client is None
    if client is None:
        client = _make_forecast_client(dc)

    try:
        # Process IDs in batches to keep memory stable
        ids_list = list(ids)
        for i in range(0, len(ids_list), dc.batch_size):
            batch = ids_list[i : i + dc.batch_size]

            futures = [
                client.submit(
                    generate_recursive_forecast_for_one_id,
                    int(_id),
                    prediction_years=prediction_years,
                    history_end_year=history_end_year,
                    coeffs_path=coeffs_path,
                    climatology_path=climatology_path,
                    config=config,
                )
                for _id in batch
            ]

            results: List[pd.DataFrame] = []
            for fut in as_completed(futures):
                try:
                    df = fut.result()
                except Exception:
                    continue
                if df is not None and not df.empty:
                    results.append(df)

            if results:
                batch_df = pd.concat(results, ignore_index=True)
                out_file = chunk_dir_p / f"{chunk_prefix}{i}.parquet"
                batch_df.to_parquet(out_file, engine="pyarrow", index=False)
                written.append(out_file)

    finally:
        if owns_client:
            client.close()

    return written


# --------------------------------------------------------------------------------------
# Bucketed Dask scaling (reads each coeffs/climatology/history/future-TMIN shard once)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastBucketSummary:
    bucket_id: int
    id_count: int
    pred_rows: int
    status: str
    predictions_path: Path


def _error_forecast_summary(
    bucket_id: int,
    predictions_output_root: Path,
    exc: BaseException,
) -> ForecastBucketSummary:
    """Summary for a bucket whose task raised on the driver side (e.g. KilledWorker)."""
    if bucket_id is not None and bucket_id >= 0:
        pred_file = bucket_dir(predictions_output_root, bucket_id) / "pred.parquet"
    else:
        pred_file = predictions_output_root
    return ForecastBucketSummary(
        bucket_id=int(bucket_id),
        id_count=0,
        pred_rows=0,
        status=f"error:{type(exc).__name__}",
        predictions_path=pred_file,
    )


def _groups_by_id(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group a frame by integer ID into a dict, tolerant of missing/odd ID values."""
    if df is None or df.empty or "ID" not in df.columns:
        return {}
    out = df.copy()
    out["ID"] = pd.to_numeric(out["ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["ID"])
    out["ID"] = out["ID"].astype(int)
    return {int(k): v for k, v in out.groupby("ID", sort=True)}


def _forecast_bucket_task(
    *,
    bucket_id: int,
    coeffs_root: PathLike,
    climatology_root: PathLike,
    prepared_training_root: PathLike,
    future_tmin_root: PathLike,
    predictions_output_root: PathLike,
    prediction_years: Tuple[int, int],
    history_end_year: int,
    overwrite: bool = False,
) -> ForecastBucketSummary:
    """
    Worker task: forecast every ID in one bucket and write its predictions directly.

    Each input shard for the bucket is read exactly once:
    - coeffs:       ``coeffs_root/id_bucket=XXXX/coeffs.parquet`` (from training)
    - climatology:  ``climatology_root/id_bucket=XXXX/climatology.parquet``
    - history:      ``prepared_training_root/id_bucket=XXXX/`` filtered to history_end_year
    - future TMIN:  ``future_tmin_root/id_bucket=XXXX/`` (forecast-horizon exogenous)
    """
    pred_dir = bucket_dir(predictions_output_root, bucket_id)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "pred.parquet"

    if pred_file.exists() and not overwrite:
        return ForecastBucketSummary(
            bucket_id=int(bucket_id), id_count=0, pred_rows=0,
            status="skipped", predictions_path=pred_file,
        )

    coeffs_file = bucket_dir(coeffs_root, bucket_id) / "coeffs.parquet"
    clim_file = bucket_dir(climatology_root, bucket_id) / "climatology.parquet"
    if not coeffs_file.exists() or not clim_file.exists():
        return ForecastBucketSummary(
            bucket_id=int(bucket_id), id_count=0, pred_rows=0,
            status="missing_inputs", predictions_path=pred_file,
        )

    coeffs_df = pd.read_parquet(coeffs_file)
    clim_df = pd.read_parquet(clim_file)

    # History: read the prepared training shard, keep only history_end_year.
    history_df = read_parquet_any(bucket_dir(prepared_training_root, bucket_id))
    history_df["FECHA"] = pd.to_datetime(history_df["FECHA"])
    history_df = history_df[history_df["FECHA"].dt.year == int(history_end_year)]

    # Future exogenous TMIN over the forecast horizon.
    future_tmin_df = read_parquet_any(bucket_dir(future_tmin_root, bucket_id))
    future_tmin_df["FECHA"] = pd.to_datetime(future_tmin_df["FECHA"])

    coeffs_by_id = _groups_by_id(coeffs_df)
    clim_by_id = _groups_by_id(clim_df)
    hist_by_id = _groups_by_id(history_df)
    ftmin_by_id = _groups_by_id(future_tmin_df)

    frames: List[pd.DataFrame] = []
    id_count = 0
    for location_id, coeffs_id_df in coeffs_by_id.items():
        id_count += 1
        df = generate_recursive_forecast_for_prepared_id(
            location_id,
            coeffs_id_df=coeffs_id_df,
            clim_id_df=clim_by_id.get(location_id, pd.DataFrame()),
            history_df=hist_by_id.get(location_id, pd.DataFrame()),
            future_tmin_df=ftmin_by_id.get(location_id, pd.DataFrame()),
            prediction_years=prediction_years,
        )
        if df is not None and not df.empty:
            frames.append(df)

    if frames:
        pred = pd.concat(frames, ignore_index=True).sort_values(["ID", "FECHA"]).reset_index(drop=True)
        pred.to_parquet(pred_file, engine="pyarrow", index=False)
        status = "ok"
        pred_rows = len(pred)
    else:
        if overwrite and pred_file.exists():
            pred_file.unlink()
        status = "empty"
        pred_rows = 0

    return ForecastBucketSummary(
        bucket_id=int(bucket_id),
        id_count=int(id_count),
        pred_rows=int(pred_rows),
        status=status,
        predictions_path=pred_file,
    )


def run_bucketed_forecast_dask(
    *,
    coeffs_root: PathLike,
    climatology_root: PathLike,
    prepared_training_root: PathLike,
    future_tmin_root: PathLike,
    predictions_output_root: PathLike,
    prediction_years: Tuple[int, int],
    history_end_year: int,
    bucket_ids: Optional[Sequence[int]] = None,
    dask_config: Optional[DaskForecastConfig] = None,
    overwrite: bool = False,
    client: Optional[Any] = None,
) -> List[ForecastBucketSummary]:
    """
    Recursive forecast by bucket. Mirrors ``run_bucketed_anomaly_training_dask``: each
    worker task forecasts all IDs in one bucket, reading each input shard once.

    Bucket tasks use a sliding window (``dask_config.batch_size`` in flight) consumed
    with ``as_completed``; a bucket that fails on a worker is recorded as an error
    summary and skipped rather than aborting the run. If ``client`` is provided it is
    reused (and left open); otherwise a local cluster is created and closed on exit.
    """
    from dask.distributed import as_completed  # imported lazily

    dc = dask_config or DaskForecastConfig()
    coeffs_root_p = as_path(coeffs_root)
    climatology_root_p = as_path(climatology_root)
    prepared_root_p = as_path(prepared_training_root)
    future_tmin_root_p = as_path(future_tmin_root)
    predictions_root_p = as_path(predictions_output_root)
    predictions_root_p.mkdir(parents=True, exist_ok=True)

    if bucket_ids is None:
        buckets_to_run = discover_bucket_ids(coeffs_root_p)
    else:
        buckets_to_run = sorted({int(b) for b in bucket_ids})
    if not buckets_to_run:
        raise ValueError(f"No buckets found under {coeffs_root_p}")

    _configure_blas_threads()
    owns_client = client is None
    if client is None:
        client = _make_forecast_client(dc)

    window = dc.batch_size if dc.batch_size and dc.batch_size > 0 else len(buckets_to_run)
    max_in_flight = max(window, dc.n_workers)

    def _submit(bucket_id: int):
        return client.submit(
            _forecast_bucket_task,
            bucket_id=int(bucket_id),
            coeffs_root=coeffs_root_p,
            climatology_root=climatology_root_p,
            prepared_training_root=prepared_root_p,
            future_tmin_root=future_tmin_root_p,
            predictions_output_root=predictions_root_p,
            prediction_years=prediction_years,
            history_end_year=history_end_year,
            overwrite=overwrite,
            pure=False,
            retries=dc.task_retries,
        )

    summaries: List[ForecastBucketSummary] = []
    pending: Dict[Any, int] = {}
    next_idx = 0
    try:
        ac = as_completed()
        while next_idx < len(buckets_to_run) and len(pending) < max_in_flight:
            bid = buckets_to_run[next_idx]
            fut = _submit(bid)
            pending[fut] = bid
            ac.add(fut)
            next_idx += 1

        for fut in ac:
            bid = pending.pop(fut, -1)
            try:
                summaries.append(fut.result())
            except Exception as exc:
                logger.warning("Forecast bucket %s failed: %s: %s", bid, type(exc).__name__, exc)
                summaries.append(_error_forecast_summary(bid, predictions_root_p, exc))

            if next_idx < len(buckets_to_run):
                bid = buckets_to_run[next_idx]
                fut = _submit(bid)
                pending[fut] = bid
                ac.add(fut)
                next_idx += 1
    finally:
        if owns_client:
            client.close()

    return sorted(summaries, key=lambda item: item.bucket_id)


def combine_bucketed_predictions(
    predictions_root: PathLike,
    *,
    output_file: PathLike,
) -> Path:
    """
    Combine per-bucket prediction shards (``id_bucket=XXXX/pred.parquet``) into a
    single parquet with columns FECHA, ID, TD_predicted.
    """
    root = as_path(predictions_root)
    out = Path(output_file).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    final_df = read_parquet_any(root)
    # read_parquet_any infers a hive "id_bucket" column from the shard paths; drop it
    # so predictions keep their canonical schema (FECHA, ID, TD_predicted).
    if "id_bucket" in final_df.columns:
        final_df = final_df.drop(columns=["id_bucket"])
    final_df = final_df.sort_values(["ID", "FECHA"]).reset_index(drop=True)
    final_df.to_parquet(out, engine="pyarrow", index=False)
    return out


def combine_prediction_chunks(
    chunk_files: Sequence[PathLike],
    *,
    output_file: PathLike,
) -> Path:
    """
    Combine chunk parquet files produced by `forecast_with_dask` into a single parquet.
    """
    files = [Path(p).expanduser().resolve() for p in chunk_files]
    if not files:
        raise ValueError("No chunk files provided.")
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"Chunk file not found: {p}")

    out = Path(output_file).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    final_df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    final_df.to_parquet(out, engine="pyarrow", index=False)
    return out


# --------------------------------------------------------------------------------------
# Monthly export (split combined predictions into monthly td_daily_YYYY_MM.parquet)
# --------------------------------------------------------------------------------------


def split_predictions_to_monthly_parquet(
    input_file: PathLike,
    output_dir: PathLike,
    *,
    date_col: str = "FECHA",
    id_col: str = "ID",
    value_col: str = "TD_predicted",
    output_value_col: str = "Value",
) -> List[Path]:
    """
    Split a combined predictions parquet into monthly parquet files.

    Output files are named:
      td_daily_YYYY_MM.parquet

    Output schema:
      ID, FECHA, Value
    """
    inp = Path(input_file).expanduser().resolve()
    if not inp.exists():
        raise FileNotFoundError(f"Input prediction file not found: {inp}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    if df.empty:
        return []

    df = df[[id_col, date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_month"] = df[date_col].dt.strftime("%Y_%m")

    outputs: List[Path] = []
    for ym, g in df.groupby("year_month"):
        out_df = g[[id_col, date_col, value_col]].rename(columns={value_col: output_value_col})
        out_path = out_dir / f"td_daily_{ym}.parquet"
        out_df.to_parquet(out_path, engine="pyarrow", index=False)
        outputs.append(out_path)

    return outputs
