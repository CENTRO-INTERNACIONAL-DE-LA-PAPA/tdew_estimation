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


def _global_max_id(files, *, logger: Optional[logging.Logger] = None) -> int:
    """Largest ID across the given parquet files.

    Uses parquet footer statistics (near-instant, no data read) with a fallback to
    reading only the ``ID`` column. IDs are assumed dense 0..N-1 (grid feature order),
    so ``max_id + 1`` is the size of the (ID, doy) accumulator.
    """
    import pyarrow.parquet as _pq  # noqa: PLC0415

    log = logger or logging.getLogger(__name__)
    max_id = -1
    for p in files:
        pf = _pq.ParquetFile(p)
        names = list(pf.schema_arrow.names)
        ci = names.index("ID") if "ID" in names else -1
        got: Optional[int] = None
        if ci >= 0:
            try:
                for rg in range(pf.metadata.num_row_groups):
                    st = pf.metadata.row_group(rg).column(ci).statistics
                    if st is not None and st.has_min_max and st.max is not None:
                        got = int(st.max) if got is None else max(got, int(st.max))
            except Exception:  # pragma: no cover - stats may be absent
                got = None
        if got is None:
            got = int(_pq.read_table(p, columns=["ID"]).column("ID").to_numpy().max())
        max_id = max(max_id, got)
    if max_id < 0:
        raise ValueError("Could not determine max ID from the provided parquet files.")
    log.info("Global max ID across %d files: %d", len(list(files)), max_id)
    return max_id


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

    all_years = list(range(start_year, end_year + 1))

    # Locate every (year, month)'s TD/TMIN monthly files up front.
    month_jobs: list = []
    for year in all_years:
        for month in range(1, 13):
            month_start = pd.Timestamp(year=year, month=month, day=1)
            next_month = (
                pd.Timestamp(year=year + 1, month=1, day=1)
                if month == 12
                else pd.Timestamp(year=year, month=month + 1, day=1)
            )
            month_end = next_month - pd.Timedelta(days=1)
            month_range = (
                month_start.strftime("%Y-%m-%d"),
                month_end.strftime("%Y-%m-%d"),
            )
            td_files = find_parquet_files(
                base_path, td_var, month_range, outputs_subdir=outputs_subdir
            )
            tmin_files = find_parquet_files(
                base_path,
                tmin_var,
                month_range,
                outputs_subdir=outputs_subdir,
                tmin_v1_legacy_name=True,
            )
            if td_files and tmin_files:
                month_jobs.append((td_files, tmin_files))

    if not month_jobs:
        raise FileNotFoundError(
            f"No TD/TMIN monthly files found under {base_path} for {year_range}."
        )

    # Memory-stable aggregation via a FIXED dense accumulator keyed by a flat
    # (ID, doy) index = ID * 366 + (doy - 1). Each month's per-slot sums/counts are
    # scattered in with np.bincount — we never build or realign a billion-row pandas
    # frame (that .add(fill_value=0) realign is what OOM-killed the previous approach).
    # Peak memory is the fixed accumulator (~25 GB for the ~2M-point grid) plus one
    # month of raw rows and one transient bincount vector.
    import numpy as np  # noqa: PLC0415

    DOY = 366
    log.info("Scanning TD file footers for the ID range...")
    n_ids = (
        _global_max_id(
            [p for td_files, _ in month_jobs for p in td_files], logger=log
        )
        + 1
    )
    n_slots = n_ids * DOY
    log.info(
        "Climatology accumulator: n_ids=%d, slots=%d (~%.1f GB fixed)",
        n_ids,
        n_slots,
        n_slots * 24 / 1e9,
    )

    sum_td = np.zeros(n_slots, dtype=np.float64)
    sum_tmin = np.zeros(n_slots, dtype=np.float64)
    cnt = np.zeros(n_slots, dtype=np.int64)

    months_processed = 0
    for td_files, tmin_files in tqdm(
        month_jobs, desc="Processing Climatology by Month"
    ):
        df_td = pd.concat(
            [pd.read_parquet(p, columns=["ID", "FECHA", "Value"]) for p in td_files],
            ignore_index=True,
        ).rename(columns={"Value": "TD"})
        df_tmin = pd.concat(
            [pd.read_parquet(p, columns=["ID", "FECHA", "Value"]) for p in tmin_files],
            ignore_index=True,
        ).rename(columns={"Value": "TMIN"})
        merged = pd.merge(df_td, df_tmin, on=["FECHA", "ID"], how="inner")
        del df_td, df_tmin
        if merged.empty:
            continue

        doy = pd.to_datetime(merged["FECHA"]).dt.dayofyear.to_numpy()
        flat = merged["ID"].to_numpy(dtype=np.int64) * DOY + (doy - 1)
        # Sequential bincount adds keep only one ~n_slots transient alive at a time.
        sum_td += np.bincount(
            flat, weights=merged["TD"].to_numpy(dtype=np.float64), minlength=n_slots
        )
        sum_tmin += np.bincount(
            flat, weights=merged["TMIN"].to_numpy(dtype=np.float64), minlength=n_slots
        )
        cnt += np.bincount(flat, minlength=n_slots)
        del merged, doy, flat
        months_processed += 1

    # Stream the ~1B-row result out in ID blocks so it is never one giant pandas frame.
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as _pq  # noqa: PLC0415

    log.info("Writing climatology (streamed by ID block) to %s", out_path)
    out_schema = pa.schema(
        [
            ("ID", pa.int64()),
            ("doy", pa.int16()),
            ("TD_clim", pa.float64()),
            ("TMIN_clim", pa.float64()),
        ]
    )
    writer = _pq.ParquetWriter(str(out_path), out_schema)
    used_chunks = 0
    id_block = 200_000
    try:
        for id_start in range(0, n_ids, id_block):
            id_end = min(id_start + id_block, n_ids)
            s, e = id_start * DOY, id_end * DOY
            c = cnt[s:e]
            mask = c > 0
            if not bool(mask.any()):
                continue
            local = np.nonzero(mask)[0]
            ids_b = (id_start + local // DOY).astype(np.int64)
            doy_b = (local % DOY + 1).astype(np.int16)
            c_m = c[mask]
            td_b = sum_td[s:e][mask] / c_m
            tmin_b = sum_tmin[s:e][mask] / c_m
            writer.write_table(
                pa.table(
                    {"ID": ids_b, "doy": doy_b, "TD_clim": td_b, "TMIN_clim": tmin_b},
                    schema=out_schema,
                )
            )
            used_chunks += 1
    finally:
        writer.close()
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
        created_chunks=months_processed,
        used_chunks=used_chunks,
    )
