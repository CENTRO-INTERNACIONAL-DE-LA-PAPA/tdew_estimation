"""
tdew_estimation.bucketed_data

Preprocessing helpers for bucket-based anomaly training.

The goal is to pay the TD/TMIN merge cost once, write bucketed training shards
to disk, and then run model fitting by bucket instead of by ID.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from .bucket_layout import bucket_dir, bucket_for_id, discover_bucket_ids
from .climatology import find_parquet_files
from .parquet_io import as_path, read_parquet_any

PathLike = Union[str, Path]


@dataclass(frozen=True)
class BucketedTrainingBuildResult:
    output_dir: Path
    year_range: Tuple[int, int]
    num_buckets: int
    years_processed: int
    bucket_files_written: int


@dataclass(frozen=True)
class BucketedClimatologyResult:
    output_dir: Path
    num_buckets: int
    shards_written: int


def _month_range_strings(month_start: pd.Timestamp) -> tuple[str, str]:
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
    return (month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))


def _merge_monthly_training_inputs(
    *,
    base_path: PathLike,
    td_var: str,
    tmin_var: str,
    outputs_subdir: str,
    month_start: pd.Timestamp,
) -> Optional[DataFrame]:
    date_range = _month_range_strings(month_start)
    td_files = find_parquet_files(
        base_path,
        td_var,
        date_range,
        outputs_subdir=outputs_subdir,
        tmin_v1_legacy_name=True,
    )
    tmin_files = find_parquet_files(
        base_path,
        tmin_var,
        date_range,
        outputs_subdir=outputs_subdir,
        tmin_v1_legacy_name=True,
    )
    if not td_files or not tmin_files:
        return None

    # Project only the needed columns — never load the fat `source_file` string column.
    df_td_list = [pd.read_parquet(p, columns=["ID", "FECHA", "Value"]) for p in td_files]
    df_tmin_list = [
        pd.read_parquet(p, columns=["ID", "FECHA", "Value"]) for p in tmin_files
    ]
    if not df_td_list or not df_tmin_list:
        return None

    df_td = pd.concat(df_td_list, ignore_index=True).rename(columns={"Value": "TD"})
    df_tmin = pd.concat(df_tmin_list, ignore_index=True).rename(columns={"Value": "TMIN"})
    if df_td.empty or df_tmin.empty:
        return None

    merged = pd.merge(df_td, df_tmin, on=["FECHA", "ID"], how="inner")
    if merged.empty:
        return None

    merged["FECHA"] = pd.to_datetime(merged["FECHA"])
    merged["doy"] = merged["FECHA"].dt.dayofyear.astype("int16")
    return merged[["ID", "FECHA", "TD", "TMIN", "doy"]].copy()


def build_bucketed_training_dataset(
    *,
    year_range: Tuple[int, int],
    base_path: PathLike,
    output_dir: PathLike,
    td_var: str = "td",
    tmin_var: str = "tmin_v12",
    outputs_subdir: str = "Outputs",
    num_buckets: int = 1024,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> BucketedTrainingBuildResult:
    """
    Build merged training shards under ``output_dir/id_bucket=XXXX/train_YYYY.parquet``.
    """
    log = logger or logging.getLogger(__name__)
    start_year, end_year = year_range
    if start_year > end_year:
        raise ValueError(f"Invalid year_range={year_range}: start_year > end_year")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")

    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years_processed = 0
    bucket_files_written = 0

    for year in tqdm(range(start_year, end_year + 1), desc="Building Bucketed Training Data"):
        year_marker = out_dir / f".done_{year}"
        if year_marker.exists() and not overwrite:
            continue

        temp_year_dir = out_dir / f".tmp_year_{year}"
        if temp_year_dir.exists():
            shutil.rmtree(temp_year_dir)
        temp_year_dir.mkdir(parents=True, exist_ok=True)

        month_starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS")
        for month_start in month_starts:
            ym = month_start.strftime("%Y_%m")
            merged = _merge_monthly_training_inputs(
                base_path=base_path,
                td_var=td_var,
                tmin_var=tmin_var,
                outputs_subdir=outputs_subdir,
                month_start=month_start,
            )
            if merged is None or merged.empty:
                continue

            merged["ID"] = pd.to_numeric(merged["ID"], errors="coerce").astype("Int64")
            merged = merged.dropna(subset=["ID"]).copy()
            merged["ID"] = merged["ID"].astype(int)
            merged["id_bucket"] = merged["ID"].map(
                lambda value: bucket_for_id(int(value), num_buckets=num_buckets)
            )

            for bucket_id, bucket_df in merged.groupby("id_bucket", sort=True):
                temp_bucket_dir = bucket_dir(temp_year_dir, int(bucket_id))
                temp_bucket_dir.mkdir(parents=True, exist_ok=True)
                temp_file = temp_bucket_dir / f"part_{ym}.parquet"
                to_write = bucket_df.drop(columns=["id_bucket"]).sort_values(
                    ["ID", "FECHA"]
                )
                to_write.to_parquet(temp_file, engine="pyarrow", index=False)

        bucket_ids_in_year = discover_bucket_ids(temp_year_dir)
        if not bucket_ids_in_year:
            shutil.rmtree(temp_year_dir)
            continue

        for bucket_id in bucket_ids_in_year:
            bucket_out_dir = bucket_dir(out_dir, int(bucket_id))
            bucket_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = bucket_out_dir / f"train_{year}.parquet"
            if out_file.exists() and not overwrite:
                continue

            yearly_bucket_df = read_parquet_any(bucket_dir(temp_year_dir, int(bucket_id)))
            if "id_bucket" in yearly_bucket_df.columns:
                yearly_bucket_df = yearly_bucket_df.drop(columns=["id_bucket"])
            yearly_bucket_df = yearly_bucket_df.sort_values(["ID", "FECHA"]).reset_index(
                drop=True
            )
            yearly_bucket_df.to_parquet(out_file, engine="pyarrow", index=False)
            bucket_files_written += 1

        shutil.rmtree(temp_year_dir)
        year_marker.write_text("ok\n", encoding="ascii")
        years_processed += 1

    log.info(
        "Built bucketed training dataset at %s with %s year(s) and %s file(s).",
        out_dir,
        years_processed,
        bucket_files_written,
    )
    return BucketedTrainingBuildResult(
        output_dir=out_dir,
        year_range=year_range,
        num_buckets=num_buckets,
        years_processed=years_processed,
        bucket_files_written=bucket_files_written,
    )


def shard_climatology_by_bucket(
    *,
    climatology_path: PathLike,
    output_dir: PathLike,
    num_buckets: int = 1024,
    overwrite: bool = False,
) -> BucketedClimatologyResult:
    """
    Write one climatology shard per bucket under ``output_dir/id_bucket=XXXX``.
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")

    clim_path = as_path(climatology_path)
    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shard the (possibly ~1B-row) climatology WITHOUT pandas. Two earlier versions
    # OOM-killed on the ~2.8M-ID national grid:
    #   * pd.read_parquet + .map + sorted groupby  -> several full-frame copies;
    #   * lean read + groupby(sort=False) ITERATION -> pandas' group iterator takes a
    #     full sorted COPY of the whole frame (~123 GB observed on a 125 GB box).
    # Pure-NumPy plan: pull the four columns as flat arrays, stable-argsort by bucket
    # (= ID % B), reorder each column ONE AT A TIME, then write each bucket's slice
    # directly with Arrow. Peak ~= columns (~26 GB) + sort index (~8 GB) + one
    # column-reorder transient (~8 GB) — bounded and grid-size-proportional.
    import numpy as np  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as _pq  # noqa: PLC0415

    _table = _pq.read_table(clim_path, columns=["ID", "doy", "TD_clim", "TMIN_clim"])
    if _table.num_rows == 0:
        raise ValueError(f"Climatology is empty: {clim_path}")
    ids = _table.column("ID").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    doy = _table.column("doy").to_numpy(zero_copy_only=False)
    td_clim = _table.column("TD_clim").to_numpy(zero_copy_only=False)
    tmin_clim = _table.column("TMIN_clim").to_numpy(zero_copy_only=False)
    del _table

    # bucket_for_id(id, B) == id % B (vectorised).
    bucket_codes = (ids % num_buckets).astype(np.int32)
    order = np.argsort(bucket_codes, kind="stable")
    counts = np.bincount(bucket_codes, minlength=num_buckets)
    offsets = np.zeros(num_buckets + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    del bucket_codes

    # Reorder one column at a time so only ONE full-size transient exists at once.
    ids = ids[order]
    doy = doy[order]
    td_clim = td_clim[order]
    tmin_clim = tmin_clim[order]
    del order

    shards_written = 0
    for bucket_id in range(num_buckets):
        s, e = int(offsets[bucket_id]), int(offsets[bucket_id + 1])
        if s == e:
            continue
        bucket_out_dir = bucket_dir(out_dir, int(bucket_id))
        bucket_out_dir.mkdir(parents=True, exist_ok=True)
        out_file = bucket_out_dir / "climatology.parquet"
        if out_file.exists() and not overwrite:
            continue

        # Within-bucket (ID, doy) order: cheap per-bucket lexsort on the small slice
        # (robust even if the source parquet was not globally ID-sorted).
        sub = np.lexsort((doy[s:e], ids[s:e]))
        _pq.write_table(
            pa.table(
                {
                    "ID": ids[s:e][sub],
                    "doy": doy[s:e][sub],
                    "TD_clim": td_clim[s:e][sub],
                    "TMIN_clim": tmin_clim[s:e][sub],
                }
            ),
            out_file,
        )
        shards_written += 1

    return BucketedClimatologyResult(
        output_dir=out_dir,
        num_buckets=num_buckets,
        shards_written=shards_written,
    )


@dataclass(frozen=True)
class BucketedForecastTminResult:
    output_dir: Path
    year_range: Tuple[int, int]
    num_buckets: int
    years_processed: int
    bucket_files_written: int


def _read_monthly_future_tmin(
    *,
    base_path: PathLike,
    future_tmin_var: str,
    outputs_subdir: str,
    month_start: pd.Timestamp,
) -> Optional[DataFrame]:
    date_range = _month_range_strings(month_start)
    files = find_parquet_files(
        base_path,
        future_tmin_var,
        date_range,
        outputs_subdir=outputs_subdir,
        tmin_v1_legacy_name=False,
    )
    if not files:
        return None
    frames = [pd.read_parquet(p) for p in files]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).rename(columns={"Value": "TMIN"})
    if df.empty or "ID" not in df.columns or "FECHA" not in df.columns:
        return None
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    return df[["ID", "FECHA", "TMIN"]].copy()


def shard_future_tmin_by_bucket(
    *,
    prediction_years: Tuple[int, int],
    base_path: PathLike,
    output_dir: PathLike,
    future_tmin_var: str = "tmin",
    outputs_subdir: str = "Outputs",
    num_buckets: int = 1024,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> BucketedForecastTminResult:
    """
    Shard forecast-horizon TMIN into ``output_dir/id_bucket=XXXX/future_tmin_YYYY.parquet``.

    Mirrors ``build_bucketed_training_dataset`` but for the single exogenous future-TMIN
    variable over the prediction horizon, so the bucketed forecast task can read its
    exogenous inputs once per bucket instead of once per ID.
    """
    log = logger or logging.getLogger(__name__)
    start_year, end_year = prediction_years
    if start_year > end_year:
        raise ValueError(f"Invalid prediction_years={prediction_years}: start_year > end_year")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")

    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years_processed = 0
    bucket_files_written = 0

    for year in tqdm(range(start_year, end_year + 1), desc="Sharding Future TMIN"):
        year_marker = out_dir / f".done_{year}"
        if year_marker.exists() and not overwrite:
            continue

        temp_year_dir = out_dir / f".tmp_year_{year}"
        if temp_year_dir.exists():
            shutil.rmtree(temp_year_dir)
        temp_year_dir.mkdir(parents=True, exist_ok=True)

        month_starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS")
        for month_start in month_starts:
            ym = month_start.strftime("%Y_%m")
            monthly = _read_monthly_future_tmin(
                base_path=base_path,
                future_tmin_var=future_tmin_var,
                outputs_subdir=outputs_subdir,
                month_start=month_start,
            )
            if monthly is None or monthly.empty:
                continue

            monthly["ID"] = pd.to_numeric(monthly["ID"], errors="coerce").astype("Int64")
            monthly = monthly.dropna(subset=["ID"]).copy()
            monthly["ID"] = monthly["ID"].astype(int)
            monthly["id_bucket"] = monthly["ID"].map(
                lambda value: bucket_for_id(int(value), num_buckets=num_buckets)
            )

            for bucket_id, bucket_df in monthly.groupby("id_bucket", sort=True):
                temp_bucket_dir = bucket_dir(temp_year_dir, int(bucket_id))
                temp_bucket_dir.mkdir(parents=True, exist_ok=True)
                temp_file = temp_bucket_dir / f"part_{ym}.parquet"
                to_write = bucket_df.drop(columns=["id_bucket"]).sort_values(["ID", "FECHA"])
                to_write.to_parquet(temp_file, engine="pyarrow", index=False)

        bucket_ids_in_year = discover_bucket_ids(temp_year_dir)
        if not bucket_ids_in_year:
            shutil.rmtree(temp_year_dir)
            continue

        for bucket_id in bucket_ids_in_year:
            bucket_out_dir = bucket_dir(out_dir, int(bucket_id))
            bucket_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = bucket_out_dir / f"future_tmin_{year}.parquet"
            if out_file.exists() and not overwrite:
                continue

            yearly_bucket_df = read_parquet_any(bucket_dir(temp_year_dir, int(bucket_id)))
            if "id_bucket" in yearly_bucket_df.columns:
                yearly_bucket_df = yearly_bucket_df.drop(columns=["id_bucket"])
            yearly_bucket_df = yearly_bucket_df.sort_values(["ID", "FECHA"]).reset_index(drop=True)
            yearly_bucket_df.to_parquet(out_file, engine="pyarrow", index=False)
            bucket_files_written += 1

        shutil.rmtree(temp_year_dir)
        year_marker.write_text("ok\n", encoding="ascii")
        years_processed += 1

    log.info(
        "Sharded future TMIN at %s with %s year(s) and %s file(s).",
        out_dir, years_processed, bucket_files_written,
    )
    return BucketedForecastTminResult(
        output_dir=out_dir,
        year_range=prediction_years,
        num_buckets=num_buckets,
        years_processed=years_processed,
        bucket_files_written=bucket_files_written,
    )
