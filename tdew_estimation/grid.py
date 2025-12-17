"""
tdew_estimation.grid

Small utilities for working with the spatial "grid" that defines the set of IDs
we expect models for.

This module exists to make downstream checks path-agnostic and reproducible:
- load the grid file (e.g., a .dbf/.shp/.gpkg) and extract the ID column
- compute the EXPECTED_COUNT used to verify completeness of per-DOY results

The user reported the grid source as:
    POTATO_GRID_FILE = ".../PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf"

Design goals
------------
- Keep dependencies minimal: we prefer pandas for DBF/CSV/Parquet where possible.
- Support geopandas when available for general geospatial formats.
- Fail loudly with actionable errors.

Notes
-----
- In your anomaly coefficient outputs, "expected_count" should correspond to the number
  of unique spatial IDs. When validating per-DOY completeness, you typically compare
  the number of rows (or unique IDs) for each DOY against this expected count.

Typical usage
-------------
from pathlib import Path
from tdew_estimation.grid import load_grid_ids, expected_count_from_grid

grid_path = Path(".../CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf")
ids = load_grid_ids(grid_path, id_column="ID")
expected = expected_count_from_grid(grid_path, id_column="ID")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]


def load_grid_ids(
    grid_path: PathLike,
    *,
    id_column: str = "ID",
    fallback_to_index: bool = True,
) -> List[int]:
    """
    Load unique grid IDs from a grid file.

    Parameters
    ----------
    grid_path:
        Path to the grid file. Supported:
        - .dbf (via geopandas if available, otherwise pandas.read_fwf is not reliable)
        - .shp/.gpkg/.geojson (via geopandas)
        - .parquet (via pandas)
        - .csv (via pandas)
    id_column:
        Column name that contains the grid ID.
    fallback_to_index:
        If True and id_column does not exist, fall back to using the row index as IDs.

    Returns
    -------
    List[int]
        Sorted list of unique IDs (as ints when possible).
    """
    p = Path(grid_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Grid file not found: {p}")

    ext = p.suffix.lower()

    df: pd.DataFrame
    if ext in {".csv"}:
        df = pd.read_csv(p)
    elif ext in {".parquet"}:
        df = pd.read_parquet(p)
    else:
        # Geospatial / DBF: prefer geopandas
        try:
            import geopandas as gpd  # type: ignore

            df = gpd.read_file(p)  # returns GeoDataFrame, compatible with pandas ops
        except Exception as exc:
            raise RuntimeError(
                "Failed to read grid file. For DBF/SHP/GPKG/GeoJSON, install geopandas "
                "and its drivers (fiona/pyogrio). Original error: "
                f"{exc}"
            ) from exc

    if id_column in df.columns:
        raw_ids = df[id_column].tolist()
    elif fallback_to_index:
        raw_ids = list(df.index)
    else:
        raise KeyError(
            f"ID column '{id_column}' not found in {p}. "
            f"Available columns: {list(df.columns)}"
        )

    # Normalize IDs: attempt int conversion, drop nulls
    ids: List[int] = []
    for v in raw_ids:
        if pd.isna(v):
            continue
        try:
            ids.append(int(v))
        except Exception:
            # If conversion fails, keep stable hash by enumerating unique strings
            # but still return ints for downstream expected_count math.
            # Map each distinct string to a deterministic integer by sorted order.
            pass

    if len(ids) == 0:
        # Fallback for non-numeric ids: map unique string values deterministically
        cleaned: List[str] = []
        for v in raw_ids:
            if pd.isna(v):
                continue
            cleaned.append(str(v))
        uniq = sorted(set(cleaned))
        mapping = {val: i for i, val in enumerate(uniq)}
        ids = [mapping[str(v)] for v in cleaned if not pd.isna(v)]

    # Unique + sorted for stable behavior
    return sorted(set(ids))


def expected_count_from_grid(
    grid_path: PathLike,
    *,
    id_column: str = "ID",
    fallback_to_index: bool = True,
) -> int:
    """
    Compute EXPECTED_COUNT from a grid file.

    This is typically the number of unique IDs that should appear for each DOY
    in your anomaly coefficient outputs (per-DOY completeness checks).

    Returns
    -------
    int
    """
    return len(
        load_grid_ids(
            grid_path, id_column=id_column, fallback_to_index=fallback_to_index
        )
    )


def expected_count_from_ids(ids: Sequence[Union[int, str]]) -> int:
    """
    Compute expected count from an in-memory list/sequence of IDs.

    Parameters
    ----------
    ids:
        IDs (ints or strings).

    Returns
    -------
    int
    """
    return len(set(ids))
