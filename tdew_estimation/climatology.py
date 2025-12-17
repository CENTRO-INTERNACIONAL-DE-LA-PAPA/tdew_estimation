"""
tdew_estimation.climatology

Chunked daily climatology computation for TD (dewpoint temperature) and TMIN (minimum temperature).

This module is extracted from a Colab-derived workflow and refactored to be:
- path-agnostic (no hard-coded base paths)
- memory-stable (processes by year and aggregates from disk)
- consistent with the anomaly model pipeline (produces TD_clim and TMIN_clim per (ID, doy))

Default variable names
----------------------
- TD variable folder:  "td"
- TMIN variable folder: "tmin_v1"  (default as requested)

Expected input data layout
--------------------------
The original pipeline expects monthly parquet files stored under:

    {base_path}/{variable}/Outputs/{variable}_daily_YYYY_MM.parquet

For `tmin_v1`, legacy naming may also exist:

    {base_path}/tmin_v1/Outputs/tmin_daily_YYYY_MM.parquet

Each parquet should contain at least:
- ID (int-like)
- FECHA (datetime-like)
- Value (float-like)

Output
------
A parquet file with columns:
- ID
- doy
- TD_clim
- TMIN_clim

Method
------
For each year in the training range:
1) load all monthly TD and TMIN files for that year
2) merge TD and TMIN on (FECHA, ID)
3) compute day-of-year (doy)
4) aggregate sum(TD), sum(TMIN), count(N) per (ID, doy)
5) write yearly aggregates as temporary parquet

Then:
6) iteratively read yearly aggregate parquets and sum them
7) compute climatology means:
      TD_clim = TD_sum / N
      TMIN_clim = TMIN_sum / N
8) write final climatology parquet and optionally remove temp dir

Notes
-----
- This implementation prioritizes fidelity to the original approach.
- It is intentionally single-process; parallelization can be layered later if needed.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, cast

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ClimatologyResult:
    """Structured result of climatology computation."""

    output_file: Path
    temp_dir: Path
    year_range: Tuple[int, int]
    td_var: str
    tmin_var: str
    created_chunks: int
    used_chunks: int


def find_parquet_files(
    base_path: PathLike,
    variable: str,
    date_range: Tuple[str, str],
    *,
    outputs_subdir: str = "Outputs",
    tmin_v1_legacy_name: bool = True,
) -> List[Path]:
    """
    Find parquet files for a variable over a date range (YYYY-MM-DD strings).

    Parameters
    ----------
    base_path:
        Root directory containing variable folders.
    variable:
        Variable folder name (e.g., 'td', 'tmin_v1').
    date_range:
        Tuple of (start_date, end_date) as strings or values parseable by pandas.Timestamp.
    outputs_subdir:
        Subdirectory containing outputs (default: 'Outputs').
    tmin_v1_legacy_name:
        If True and variable == 'tmin_v1', also check legacy filename pattern
        'tmin_daily_YYYY_MM.parquet' inside tmin_v1/Outputs.

    Returns
    -------
    List[Path]
        Sorted list of matching parquet files.
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


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def calculate_and_save_climatology_chunked(
    year_range: Tuple[int, int],
    base_path: PathLike,
    output_file: PathLike,
    *,
    td_var: str = "td",
    tmin_var: str = "tmin_v1",
    outputs_subdir: str = "Outputs",
    temp_dir_name: str = "climatology_temp_chunks",
    engine: Literal["auto", "pyarrow", "fastparquet"] = "pyarrow",
    cleanup: bool = True,
    logger: Optional[logging.Logger] = None,
) -> ClimatologyResult:
    """
    Calculate daily climatology in yearly chunks (memory stable) and save to parquet.

    Parameters
    ----------
    year_range:
        Inclusive year range, e.g. (1981, 2016).
    base_path:
        Root data path containing variable folders.
    output_file:
        Destination parquet path for the climatology output.
    td_var:
        TD variable folder name (default: 'td').
    tmin_var:
        TMIN variable folder name (default: 'tmin_v1').
    outputs_subdir:
        Subdirectory under each variable folder containing parquet files.
    temp_dir_name:
        Temporary directory name created alongside output_file to store yearly aggregates.
    engine:
        Parquet engine used for final output (default: 'pyarrow').
    cleanup:
        If True, remove temp_dir after successfully writing output.
    logger:
        Optional logger; if not provided, uses module logger.

    Returns
    -------
    ClimatologyResult
        Metadata about the produced climatology and chunk usage.

    Raises
    ------
    FileNotFoundError
        If no climatology chunk files are created.
    """
    log = logger or logging.getLogger(__name__)

    start_year, end_year = year_range
    if start_year > end_year:
        raise ValueError(f"Invalid year_range={year_range}: start_year > end_year")

    out_path = Path(output_file).expanduser().resolve()
    _ensure_parent_dir(out_path)

    temp_dir = out_path.parent / temp_dir_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_years = range(start_year, end_year + 1)

    log.info("Calculating daily climatology in yearly chunks to conserve memory...")
    created_chunks = 0

    for year in tqdm(all_years, desc="Processing Climatology by Year"):
        chunk_file = temp_dir / f"clim_chunk_{year}.parquet"
        if chunk_file.exists():
            continue

        date_range_chunk = (f"{year}-01-01", f"{year}-12-31")

        td_files = find_parquet_files(
            base_path, td_var, date_range_chunk, outputs_subdir=outputs_subdir
        )
        tmin_files = find_parquet_files(
            base_path,
            tmin_var,
            date_range_chunk,
            outputs_subdir=outputs_subdir,
            tmin_v1_legacy_name=True,
        )

        if not td_files or not tmin_files:
            continue

        # Load full year for each variable. This matches the original approach; it is
        # why chunking by year is important for memory stability.
        df_td_list = [pd.read_parquet(p) for p in td_files]
        df_tmin_list = [pd.read_parquet(p) for p in tmin_files]
        if not df_td_list or not df_tmin_list:
            continue

        df_td = pd.concat(df_td_list, ignore_index=True)
        df_td = df_td.rename(columns={"Value": "TD"})
        df_tmin = pd.concat(df_tmin_list, ignore_index=True)
        df_tmin = df_tmin.rename(columns={"Value": "TMIN"})

        # Merge on (FECHA, ID), same as the notebook
        merged = cast(
            DataFrame,
            pd.merge(
                cast(DataFrame, df_td),
                cast(DataFrame, df_tmin),
                on=["FECHA", "ID"],
                how="inner",
            ),
        )
        if merged.empty:
            continue

        merged["FECHA"] = pd.to_datetime(merged["FECHA"])
        merged["doy"] = merged["FECHA"].dt.dayofyear

        agg_sum = cast(DataFrame, merged.groupby(["ID", "doy"])[["TD", "TMIN"]].sum())
        agg_count = cast(DataFrame, merged.groupby(["ID", "doy"])[["TD"]].count())
        agg_count = cast(DataFrame, agg_count.rename(columns={"TD": "N"}))

        # Merge on MultiIndex (ID, doy) to get TD, TMIN, N columns
        yearly_stats_df = cast(
            DataFrame,
            pd.merge(
                agg_sum,
                agg_count,
                left_index=True,
                right_index=True,
            ).reset_index(),
        )
        yearly_stats_df.to_parquet(chunk_file, engine="pyarrow", index=False)
        created_chunks += 1

    log.info("Combining yearly statistics from disk iteratively...")

    chunk_files = sorted(temp_dir.glob("clim_chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No climatology chunk files were created under {temp_dir}."
        )

    final_agg_df: DataFrame = DataFrame()
    used_chunks = 0

    for f in tqdm(chunk_files, desc="Aggregating Chunks"):
        yearly_stats_df = cast(DataFrame, pd.read_parquet(f))
        current_agg = cast(
            DataFrame,
            yearly_stats_df.groupby(["ID", "doy"])[["TD", "TMIN", "N"]].sum(),
        )
        if final_agg_df.empty:
            final_agg_df = current_agg
        else:
            final_agg_df = cast(DataFrame, final_agg_df.add(current_agg, fill_value=0))
        used_chunks += 1

    final_agg_df["TD_clim"] = final_agg_df["TD"] / final_agg_df["N"]
    final_agg_df["TMIN_clim"] = final_agg_df["TMIN"] / final_agg_df["N"]
    climatology_df = final_agg_df[["TD_clim", "TMIN_clim"]].reset_index()

    climatology_df.to_parquet(out_path, engine=engine, index=False)
    log.info("✅ Climatology saved to %s", out_path)

    if cleanup:
        try:
            shutil.rmtree(temp_dir)
            log.info("Cleaned up temporary directory: %s", temp_dir)
        except Exception as exc:
            log.warning("Could not remove temporary directory %s: %s", temp_dir, exc)

    return ClimatologyResult(
        output_file=out_path,
        temp_dir=temp_dir,
        year_range=year_range,
        td_var=td_var,
        tmin_var=tmin_var,
        created_chunks=created_chunks,
        used_chunks=used_chunks,
    )
