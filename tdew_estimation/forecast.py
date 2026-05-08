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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


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

    last_two_days = history_df.tail(2).copy().reset_index(drop=True)

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

    # Index climatology and coeffs by DOY for fast lookup
    clim_id = clim_df.copy()
    clim_id["doy"] = pd.to_numeric(clim_id["doy"], errors="coerce").astype("Int64")
    clim_id = clim_id.dropna(subset=["doy"]).copy()
    clim_id["doy"] = clim_id["doy"].astype(int)
    clim_id = clim_id.set_index("doy")

    coeffs_id = coeffs_df.copy()
    coeffs_id["doy"] = pd.to_numeric(coeffs_id["doy"], errors="coerce").astype("Int64")
    coeffs_id = coeffs_id.dropna(subset=["doy"]).copy()
    coeffs_id["doy"] = coeffs_id["doy"].astype(int)
    coeffs_id = coeffs_id.set_index("doy")

    future_tmin = future_tmin.copy()
    future_tmin["FECHA"] = pd.to_datetime(future_tmin["FECHA"])
    future_tmin = future_tmin.set_index("FECHA")

    prediction_dates = pd.date_range(start=pred_range[0], end=pred_range[1], freq="D")
    predictions: List[Dict[str, object]] = []

    # Precompute DOY function
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
        new_row = pd.DataFrame([{"FECHA": current_day, "ID": location_id, "TD": predicted_td, "TMIN": tmin_today}])
        last_two_days = pd.concat([last_two_days.iloc[1:], new_row], ignore_index=True)

    if not predictions:
        return pd.DataFrame(columns=["FECHA", "ID", "TD_predicted"])

    return pd.DataFrame(predictions)


# --------------------------------------------------------------------------------------
# Dask scaling
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DaskForecastConfig:
    """
    Dask configuration for distributed forecasting.
    """
    n_workers: int = 8
    threads_per_worker: int = 4
    memory_limit: Optional[str] = "16GB"
    batch_size: int = 500  # number of IDs per batch (chunk file)


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
) -> List[Path]:
    """
    Run recursive forecasts across many IDs using Dask and write chunk parquet files.

    Returns list of chunk file paths written.
    """
    from dask.distributed import Client, as_completed  # imported lazily

    dc = dask_config or DaskForecastConfig()
    chunk_dir_p = Path(chunk_dir).expanduser().resolve()
    chunk_dir_p.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    client = Client(n_workers=dc.n_workers, threads_per_worker=dc.threads_per_worker, memory_limit=dc.memory_limit)

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
        client.close()

    return written


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
