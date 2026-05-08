"""
tdew_estimation.anomaly_dask

Dask-based anomaly training driver for TD (dewpoint) estimation.

This module provides a *functional*, path-agnostic orchestration layer to run the
anomaly model at scale, following the pattern used in the original Colab notebook:

- compute/load daily climatology (outside this module; see tdew_estimation.climatology)
- create a Dask distributed Client
- scatter/broadcast climatology to workers
- submit per-ID training tasks (train_anomaly_coeffs_for_one_id)
- collect results and write batch chunk files
- optionally combine chunks into a single coefficients parquet

It also supports targeted reruns for a subset of DOYs (e.g., DOYS_TO_FIX) by passing
a `doys` set into the per-ID trainer.

Important: this module uses the anomaly model only (no GPU per-DOY model files).

Dependencies
-----------
- dask[distributed]
- pandas
- statsmodels (used by core trainer)
- pyarrow or fastparquet for parquet I/O

See also
--------
- tdew_estimation.anomaly_train: core training logic (per-ID)
- tdew_estimation.climatology: climatology computation
- tdew_estimation.checks: detect incomplete DOYs in combined coefficient parquet
- tdew_estimation.patch_coeffs: patch combined coefficients with rerun results

Example usage (illustrative; adjust paths)
------------------------------------------
from pathlib import Path
from tdew_estimation.anomaly_train import AnomalyTrainingConfig
from tdew_estimation.anomaly_dask import run_anomaly_training_dask

cfg = AnomalyTrainingConfig(
    base_path=Path("/path/to/base"),
    td_var="td",
    tmin_var="tmin_v1",
    train_year_range=(1981, 2016),
    h=11,
    kernel="Tricube",
    min_samples=15,
)

chunk_dir = Path("/path/to/results/anomaly_coeffs_chunks")
out = Path("/path/to/results/llr_coeffs_anomaly_final_direct.parquet")

run_anomaly_training_dask(
    ids=[1,2,3],
    config=cfg,
    climatology_path=Path("/path/to/results/daily_climatology.parquet"),
    chunk_dir=chunk_dir,
    combine_output_path=out,
    n_workers=8,
    threads_per_worker=4,
    memory_limit="16GB",
)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Union

import pandas as pd

from .anomaly_train import (
    AnomalyTrainingConfig,
    fit_anomaly_coeffs_for_prepared_id,
    train_anomaly_coeffs_for_one_id,
)
from .bucket_layout import bucket_dir, discover_bucket_ids
from .parquet_io import as_path, read_parquet_any

PathLike = Union[str, Path]


def _as_path(p: PathLike) -> Path:
    return as_path(p)


@dataclass(frozen=True)
class DaskAnomalyConfig:
    """
    Dask execution settings for anomaly training.

    Parameters
    ----------
    n_workers:
        Number of Dask workers.
    threads_per_worker:
        Threads per worker.
    memory_limit:
        Per-worker memory limit string (e.g., "16GB") or None.
    batch_size:
        Submission batch size. In the legacy per-ID path it means IDs per output chunk;
        in the bucketed path it means bucket tasks submitted at once.
    scheduler_timeout_s:
        Timeout for workers to connect in seconds.
    """

    n_workers: int = 8
    threads_per_worker: int = 4
    memory_limit: Optional[str] = "16GB"
    batch_size: int = 1000
    scheduler_timeout_s: int = 120

def _normalize_doys(doys: Optional[Sequence[int]]) -> Optional[Set[int]]:
    if doys is None:
        return None
    out: Set[int] = set()
    for d in doys:
        di = int(d)
        if 1 <= di <= 366:
            out.add(di)
    return out or None


def _write_chunk(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def _combine_chunks(chunk_files: Sequence[Path], out_path: Path) -> Path:
    if not chunk_files:
        raise ValueError("No chunk files provided to combine.")
    for p in chunk_files:
        if not p.exists():
            raise FileNotFoundError(f"Chunk file not found: {p}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df = pd.concat([pd.read_parquet(p) for p in chunk_files], ignore_index=True)
    final_df.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


@dataclass(frozen=True)
class BucketTrainingSummary:
    bucket_id: int
    id_count: int
    coeff_rows: int
    failure_rows: int
    status: str
    coeffs_path: Path
    failures_path: Optional[Path]


def _failure_row(
    *,
    phase: str,
    bucket_id: int,
    message: str,
    location_id: Optional[int] = None,
    doy: Optional[int] = None,
    exception_type: str = "",
) -> dict[str, Any]:
    return {
        "phase": phase,
        "bucket_id": int(bucket_id),
        "ID": None if location_id is None else int(location_id),
        "doy": None if doy is None else int(doy),
        "exception_type": exception_type,
        "message": str(message),
    }


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [df for df in frames if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


def _train_anomaly_bucket_task(
    *,
    bucket_id: int,
    prepared_training_root: PathLike,
    bucketed_climatology_root: PathLike,
    coeffs_output_root: PathLike,
    config: AnomalyTrainingConfig,
    doys: Optional[Sequence[int]] = None,
    failure_output_root: Optional[PathLike] = None,
    overwrite: bool = False,
) -> BucketTrainingSummary:
    """
    Worker task: fit all IDs in one bucket and write results directly to disk.
    """
    bucket_training_dir = bucket_dir(prepared_training_root, bucket_id)
    bucket_clim_dir = bucket_dir(bucketed_climatology_root, bucket_id)
    coeffs_dir = bucket_dir(coeffs_output_root, bucket_id)
    coeffs_dir.mkdir(parents=True, exist_ok=True)
    coeffs_file = coeffs_dir / "coeffs.parquet"

    failures_file: Optional[Path] = None
    if failure_output_root is not None:
        failures_dir = bucket_dir(failure_output_root, bucket_id)
        failures_dir.mkdir(parents=True, exist_ok=True)
        failures_file = failures_dir / "failures.parquet"

    if coeffs_file.exists() and not overwrite:
        return BucketTrainingSummary(
            bucket_id=int(bucket_id),
            id_count=0,
            coeff_rows=0,
            failure_rows=0,
            status="skipped",
            coeffs_path=coeffs_file,
            failures_path=failures_file if failures_file and failures_file.exists() else None,
        )

    failures: List[dict[str, Any]] = []
    coeff_frames: List[pd.DataFrame] = []
    id_count = 0

    try:
        train_df = read_parquet_any(bucket_training_dir)
    except Exception as exc:
        failures.append(
            _failure_row(
                phase="read_training_bucket",
                bucket_id=bucket_id,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
        )
        train_df = pd.DataFrame()

    clim_file = bucket_clim_dir / "climatology.parquet"
    if clim_file.exists():
        clim_df = pd.read_parquet(clim_file)
    else:
        failures.append(
            _failure_row(
                phase="read_climatology_bucket",
                bucket_id=bucket_id,
                message=f"Missing climatology shard: {clim_file}",
            )
        )
        clim_df = pd.DataFrame()

    if not train_df.empty and not clim_df.empty:
        train_df["ID"] = pd.to_numeric(train_df["ID"], errors="coerce").astype("Int64")
        train_df = train_df.dropna(subset=["ID"]).copy()
        train_df["ID"] = train_df["ID"].astype(int)
        clim_df["ID"] = pd.to_numeric(clim_df["ID"], errors="coerce").astype("Int64")
        clim_df = clim_df.dropna(subset=["ID"]).copy()
        clim_df["ID"] = clim_df["ID"].astype(int)

        doys_set = _normalize_doys(doys)
        for location_id, id_df in train_df.groupby("ID", sort=True):
            id_count += 1
            try:
                coeffs_df, id_failures = fit_anomaly_coeffs_for_prepared_id(
                    int(location_id),
                    prepared_df=id_df,
                    climatology_df=clim_df[clim_df["ID"] == int(location_id)].copy(),
                    config=config,
                    doys=doys_set,
                )
            except Exception as exc:
                failures.append(
                    _failure_row(
                        phase="fit_id",
                        bucket_id=bucket_id,
                        location_id=int(location_id),
                        exception_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            if coeffs_df is not None and not coeffs_df.empty:
                coeff_frames.append(coeffs_df)
            if id_failures is not None and not id_failures.empty:
                id_failures = id_failures.copy()
                id_failures["bucket_id"] = int(bucket_id)
                failures.extend(id_failures.to_dict(orient="records"))

    coeffs_df = _concat_frames(coeff_frames)
    failures_df = pd.DataFrame(failures)

    if not coeffs_df.empty:
        coeffs_df = coeffs_df.sort_values(["ID", "doy"]).reset_index(drop=True)
        coeffs_df.to_parquet(coeffs_file, engine="pyarrow", index=False)
    elif overwrite and coeffs_file.exists():
        coeffs_file.unlink()

    if failures_file is not None:
        if not failures_df.empty:
            failures_df = failures_df.sort_values(
                ["bucket_id", "ID", "doy"], na_position="last"
            ).reset_index(drop=True)
            failures_df.to_parquet(failures_file, engine="pyarrow", index=False)
        elif overwrite and failures_file.exists():
            failures_file.unlink()

    status = "ok" if not coeffs_df.empty else "empty"
    return BucketTrainingSummary(
        bucket_id=int(bucket_id),
        id_count=int(id_count),
        coeff_rows=int(len(coeffs_df)),
        failure_rows=int(len(failures_df)),
        status=status,
        coeffs_path=coeffs_file,
        failures_path=failures_file if failures_file and failures_file.exists() else None,
    )


def run_anomaly_training_dask(
    *,
    ids: Sequence[int],
    config: AnomalyTrainingConfig,
    climatology_path: PathLike,
    chunk_dir: PathLike,
    doys: Optional[Sequence[int]] = None,
    combine_output_path: Optional[PathLike] = None,
    dask_config: Optional[DaskAnomalyConfig] = None,
    chunk_prefix: str = "batch_",
    overwrite_chunks: bool = False,
    persist_climatology_on_workers: bool = True,
) -> List[Path]:
    """
    Run anomaly training across many IDs using Dask and write chunk parquet outputs.

    Parameters
    ----------
    ids:
        IDs to train for (typically all grid IDs).
    config:
        AnomalyTrainingConfig (includes base_path, year range, h, kernel, etc.).
    climatology_path:
        Path to climatology parquet (daily_climatology.parquet).
    chunk_dir:
        Directory to write chunk parquet files (e.g., results/anomaly_coeffs_chunks).
    doys:
        Optional subset of DOYs to train (e.g., DOYS_TO_FIX). If omitted, trains all 1..366.
    combine_output_path:
        If provided, combines chunk files into this final parquet and returns written chunk list.
    dask_config:
        Optional DaskAnomalyConfig.
    chunk_prefix:
        Prefix for chunk files. Default "batch_" (files will be batch_{offset}.parquet).
    overwrite_chunks:
        If False, existing chunk files are skipped.
    persist_climatology_on_workers:
        If True, scatters climatology to workers once (broadcast) and passes a future to tasks.

    Returns
    -------
    List[Path]
        Paths of chunk files written (or already present if skipped).
    """
    from dask.distributed import Client, as_completed  # imported lazily

    dc = dask_config or DaskAnomalyConfig()
    chunk_dir_p = _as_path(chunk_dir)
    chunk_dir_p.mkdir(parents=True, exist_ok=True)

    clim_path = _as_path(climatology_path)
    if not clim_path.exists():
        raise FileNotFoundError(f"Climatology file not found: {clim_path}")

    # Load climatology once on driver
    climatology_df = pd.read_parquet(clim_path)
    doys_set = _normalize_doys(doys)

    # Start Dask client (local cluster via distributed defaults)
    client = Client(
        n_workers=dc.n_workers,
        threads_per_worker=dc.threads_per_worker,
        memory_limit=dc.memory_limit,
        timeout=f"{dc.scheduler_timeout_s}s",
    )

    chunk_paths: List[Path] = []

    try:
        # Scatter climatology to all workers once, to reduce repeated transfer
        clim_ref = (
            client.scatter(climatology_df, broadcast=True)
            if persist_climatology_on_workers
            else climatology_df
        )

        ids_list = [int(i) for i in ids]
        total = len(ids_list)

        for offset in range(0, total, dc.batch_size):
            batch_ids = ids_list[offset : offset + dc.batch_size]
            batch_file = chunk_dir_p / f"{chunk_prefix}{offset}.parquet"

            if batch_file.exists() and not overwrite_chunks:
                chunk_paths.append(batch_file)
                continue

            futures = [
                client.submit(
                    train_anomaly_coeffs_for_one_id,
                    _id,
                    config=config,
                    climatology_df=clim_ref,  # can be DataFrame or a Future
                    doys=doys_set,
                    pure=False,  # avoid accidental caching across retries
                )
                for _id in batch_ids
            ]

            results: List[pd.DataFrame] = []
            for fut in as_completed(futures):
                try:
                    df = fut.result()
                except Exception:
                    # A single ID failure should not abort the batch
                    continue
                if df is not None and not df.empty:
                    results.append(df)

            if results:
                batch_df = pd.concat(results, ignore_index=True)
                _write_chunk(batch_df, batch_file)
                chunk_paths.append(batch_file)
            else:
                # Still create an empty marker? Keep behavior simple: don't write.
                # The caller can detect missing batch files.
                pass

        if combine_output_path is not None:
            out_p = _as_path(combine_output_path)
            _combine_chunks(chunk_paths, out_p)

    finally:
        client.close()

    return chunk_paths


def run_bucketed_anomaly_training_dask(
    *,
    prepared_training_root: PathLike,
    bucketed_climatology_root: PathLike,
    coeffs_output_root: PathLike,
    config: AnomalyTrainingConfig,
    bucket_ids: Optional[Sequence[int]] = None,
    doys: Optional[Sequence[int]] = None,
    dask_config: Optional[DaskAnomalyConfig] = None,
    failure_output_root: Optional[PathLike] = None,
    overwrite: bool = False,
) -> List[BucketTrainingSummary]:
    """
    Run anomaly training by bucket instead of by individual ID.

    Each worker task reads one bucket shard of the prepared training dataset, fits
    all IDs inside that bucket, and writes its own parquet output directly.
    """
    from dask.distributed import Client, as_completed  # imported lazily

    dc = dask_config or DaskAnomalyConfig()
    prepared_root_p = as_path(prepared_training_root)
    bucketed_clim_root_p = as_path(bucketed_climatology_root)
    coeffs_root_p = as_path(coeffs_output_root)
    coeffs_root_p.mkdir(parents=True, exist_ok=True)
    failure_root_p = as_path(failure_output_root) if failure_output_root is not None else None
    if failure_root_p is not None:
        failure_root_p.mkdir(parents=True, exist_ok=True)

    if bucket_ids is None:
        buckets_to_run = discover_bucket_ids(prepared_root_p)
    else:
        buckets_to_run = sorted({int(b) for b in bucket_ids})
    if not buckets_to_run:
        raise ValueError(f"No buckets found under {prepared_root_p}")

    client = Client(
        n_workers=dc.n_workers,
        threads_per_worker=dc.threads_per_worker,
        memory_limit=dc.memory_limit,
        timeout=f"{dc.scheduler_timeout_s}s",
    )

    summaries: List[BucketTrainingSummary] = []
    try:
        for offset in range(0, len(buckets_to_run), dc.batch_size):
            bucket_batch = buckets_to_run[offset : offset + dc.batch_size]
            futures = [
                client.submit(
                    _train_anomaly_bucket_task,
                    bucket_id=int(bucket_id),
                    prepared_training_root=prepared_root_p,
                    bucketed_climatology_root=bucketed_clim_root_p,
                    coeffs_output_root=coeffs_root_p,
                    config=config,
                    doys=doys,
                    failure_output_root=failure_root_p,
                    overwrite=overwrite,
                    pure=False,
                )
                for bucket_id in bucket_batch
            ]
            for fut in as_completed(futures):
                summaries.append(fut.result())
    finally:
        client.close()

    return sorted(summaries, key=lambda item: item.bucket_id)


def rerun_failed_doys_with_bucketed_dask(
    *,
    prepared_training_root: PathLike,
    bucketed_climatology_root: PathLike,
    patch_output_root: PathLike,
    failed_doys: Sequence[int],
    config: AnomalyTrainingConfig,
    dask_config: Optional[DaskAnomalyConfig] = None,
    failure_output_root: Optional[PathLike] = None,
    bucket_ids: Optional[Sequence[int]] = None,
) -> List[BucketTrainingSummary]:
    """
    Retrain only selected DOYs using the bucket-based execution path.
    """
    return run_bucketed_anomaly_training_dask(
        prepared_training_root=prepared_training_root,
        bucketed_climatology_root=bucketed_climatology_root,
        coeffs_output_root=patch_output_root,
        config=config,
        bucket_ids=bucket_ids,
        doys=failed_doys,
        dask_config=dask_config,
        failure_output_root=failure_output_root,
        overwrite=True,
    )


def rerun_failed_doys_with_dask(
    *,
    ids: Sequence[int],
    failed_doys: Sequence[int],
    config: AnomalyTrainingConfig,
    climatology_path: PathLike,
    patch_output_path: PathLike,
    dask_config: Optional[DaskAnomalyConfig] = None,
    temp_chunk_dir: Optional[PathLike] = None,
) -> Path:
    """
    Convenience wrapper: retrain anomaly coefficients ONLY for failed DOYs and write a single patch parquet.

    This function:
    1) runs Dask training restricted to `failed_doys`, writing chunk files
    2) combines those chunk files into `patch_output_path`

    Parameters
    ----------
    ids:
        IDs to retrain for (typically all grid IDs).
    failed_doys:
        DOYs to retrain (DOYS_TO_FIX).
    config:
        AnomalyTrainingConfig.
    climatology_path:
        Path to daily climatology parquet.
    patch_output_path:
        Output parquet containing only the retrained DOY rows for all IDs.
    dask_config:
        Dask settings.
    temp_chunk_dir:
        Optional directory for intermediate chunk files. If omitted, uses a sibling directory
        next to patch_output_path called "<patch_stem>_chunks".

    Returns
    -------
    Path
        Path to written patch parquet.
    """
    patch_p = _as_path(patch_output_path)
    if temp_chunk_dir is None:
        chunk_dir = patch_p.parent / f"{patch_p.stem}_chunks"
    else:
        chunk_dir = _as_path(temp_chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = run_anomaly_training_dask(
        ids=ids,
        config=config,
        climatology_path=climatology_path,
        chunk_dir=chunk_dir,
        doys=list(failed_doys),
        combine_output_path=patch_p,
        dask_config=dask_config,
        chunk_prefix="patch_batch_",
        overwrite_chunks=True,
    )
    # combine_output_path already wrote patch_p
    if not patch_p.exists():
        raise RuntimeError("Patch output was not created.")
    return patch_p
