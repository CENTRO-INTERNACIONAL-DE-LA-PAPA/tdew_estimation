"""
tdew_estimation.parquet_io

Small helpers for reading parquet outputs that may be a single file or a
directory containing many parquet files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]


def as_path(p: PathLike) -> Path:
    return Path(p).expanduser().resolve()


def list_parquet_files(path: PathLike) -> list[Path]:
    """
    Return all parquet files under a file-or-directory path.
    """
    p = as_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet path not found: {p}")
    if p.is_file():
        return [p]
    files = sorted(fp for fp in p.rglob("*.parquet") if fp.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {p}")
    return files


def read_parquet_any(
    path: PathLike,
    *,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[object] = None,
    engine: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a parquet file or a directory tree of parquet files into pandas.
    """
    files = list_parquet_files(path)
    read_kwargs: dict[str, object] = {}
    if columns is not None:
        read_kwargs["columns"] = list(columns)
    if filters is not None:
        read_kwargs["filters"] = filters
    if engine is not None:
        read_kwargs["engine"] = engine
    return pd.read_parquet([str(p) for p in files], **read_kwargs)
