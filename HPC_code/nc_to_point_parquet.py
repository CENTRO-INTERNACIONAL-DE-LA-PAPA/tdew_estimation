#!/usr/bin/env python3
"""
HPC_code.nc_to_point_parquet

D6 T8 — extract PISCOt ``.nc`` rasters to the per-point monthly parquet files the rest of
the pipeline consumes (``{base}/{var}/Outputs/{var}_daily_YYYY_MM.parquet`` with columns
``ID, FECHA, Value, source_file``).

This is the Python/xarray port of the R/``terra`` step
(``Extraccion_PotatoPoints_PISCO_SENAMHI.R``: ``terra::extract(raster, potato_grid)``),
which samples each daily raster layer at a set of points — nearest grid cell, no
interpolation — and stacks the result to long form.

Two sampling modes, chosen by ``--peru-potato`` (default True):

* **potato=True**  — sample at the CENAGRO potato-zoning centroids (the ``--shp``). This is
  the science subset; on the full PISCO record it reproduces the existing 302449-point data
  (``ID`` = shapefile feature order, ``0..N-1``).
* **potato=False** — keep the *full* PISCO grid (~2e6 cells), ``ID`` = row-major
  ``(lat, lon)`` index of a fixed grid template (the first month). A one-time
  ``grid_index.parquet`` (``ID, lat, lon``) is written alongside the Outputs. Full-grid IDs
  are per-variable (each product's own grid) — only comparable within a variable.

A ``--verify-against-existing`` mode extracts a single month and diffs it against the
current parquet (identical ``(ID, FECHA)`` set + ``max|Δ Value|``) so the feature-order /
CRS assumption can be checked *before* overwriting good data.

Example::

    python HPC_code/nc_to_point_parquet.py \
        --var tmin_v11 --nc-dir /data/_raw/tmin_v11 \
        --base /media/ppalacios/Data/henry_simcast_peru \
        --shp  /media/.../PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.shp \
        --peru-potato

    # safety gate (no writes): compare one month to the existing data
    python HPC_code/nc_to_point_parquet.py --var tmin --nc-dir /data/_raw/tmin \
        --base /media/.../henry_simcast_peru --shp /data/.../CENAGRO...shp \
        --verify-against-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# Coordinate-name candidates seen across PISCO products / writeCDF outputs.
_LON_NAMES = ("longitude", "lon", "x", "X", "nlon")
_LAT_NAMES = ("latitude", "lat", "y", "Y", "nlat")
_TIME_NAMES = ("time", "TIME", "t", "valid_time")


# --------------------------------------------------------------------------- #
# Dataset / coordinate discovery
# --------------------------------------------------------------------------- #
def _pick(names: tuple[str, ...], available) -> str:
    for n in names:
        if n in available:
            return n
    raise KeyError(f"none of {names} found in {list(available)}")


def open_pisco(nc_dir: Path, *, chunk_time: int = 31):
    """Open every ``*.nc`` under ``nc_dir`` as one time-sorted dataset (chunked by time).

    Works for both a single multi-year file and a directory of monthly files.
    Returns ``(data_array, lon_name, lat_name, time_name)``.
    """
    import xarray as xr

    # Skip the long-term-mean climatology files (e.g. tmin_mean_1981-2010.nc) bundled with
    # each PISCO article — they have no daily time axis and would break the time grouping.
    files = sorted(
        p
        for p in nc_dir.rglob("*.nc")
        if p.is_file() and "mean" not in p.name.lower()
    )
    if not files:
        raise FileNotFoundError(f"no daily .nc files under {nc_dir}")
    if len(files) == 1:
        ds = xr.open_dataset(files[0], chunks={})
    else:
        ds = xr.open_mfdataset(
            files, combine="by_coords", chunks={}, parallel=False, decode_times=True
        )

    lon = _pick(_LON_NAMES, set(ds.coords) | set(ds.dims))
    lat = _pick(_LAT_NAMES, set(ds.coords) | set(ds.dims))
    tname = _pick(_TIME_NAMES, set(ds.coords) | set(ds.dims))

    # Pick the data variable: the (only) one carrying time+lat+lon.
    cand = [
        v
        for v in ds.data_vars
        if {tname, lat, lon}.issubset(set(ds[v].dims))
    ]
    if not cand:
        raise ValueError(f"no data var with dims ({tname},{lat},{lon}) in {list(ds.data_vars)}")
    da = ds[cand[0]].chunk({tname: chunk_time})
    return da, lon, lat, tname


def grid_crs(nc_dir: Path) -> str:
    """CRS of the raster grid. PISCO products are plain geographic WGS84."""
    return "EPSG:4326"


# --------------------------------------------------------------------------- #
# Points (potato mode)
# --------------------------------------------------------------------------- #
def load_points(shp: Path, target_crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Potato-grid centroids as ``(xs, ys)`` in *feature order* (ID = 0..N-1), reprojected
    to the raster CRS. Mirrors ``st_read`` + the implicit reprojection in ``terra::extract``.
    """
    import geopandas as gpd

    gdf = gpd.read_file(shp)
    if gdf.crs is not None and target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    geom = gdf.geometry
    # Polygons would arrive here only if the layer isn't points; use representative points.
    if not (geom.geom_type == "Point").all():
        geom = geom.representative_point()
    return geom.x.to_numpy(), geom.y.to_numpy()


# --------------------------------------------------------------------------- #
# Month iteration + sampling
# --------------------------------------------------------------------------- #
def month_groups(times) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(year, month, positional_indices)`` for each calendar month, sorted."""
    ti = pd.DatetimeIndex(pd.to_datetime(times))
    idx = pd.DataFrame({"pos": np.arange(len(ti)), "y": ti.year, "m": ti.month})
    for (y, m), g in idx.groupby(["y", "m"], sort=True):
        yield int(y), int(m), g["pos"].to_numpy()


def _long_frame(vals: np.ndarray, ids: np.ndarray, month_times, source_file: str) -> pd.DataFrame:
    """Stack a ``(T, P)`` value block to long ``ID, FECHA, Value, source_file``.

    Row order matches the legacy R output / existing parquet: within each date, IDs ascending.
    """
    T, P = vals.shape
    fecha = pd.DatetimeIndex(pd.to_datetime(month_times)).astype("datetime64[us]")
    df = pd.DataFrame(
        {
            "ID": np.tile(ids.astype(np.int64), T),
            "FECHA": np.repeat(fecha.values, P),
            "Value": vals.reshape(-1).astype(np.float32),
            "source_file": source_file,
        }
    )
    return df


def sample_potato(da_month, lon: str, lat: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Pointwise nearest-cell sample → ``(T, P)`` array. Equals terra cell-containment on a
    regular grid."""
    import xarray as xr

    xs_da = xr.DataArray(xs, dims="point")
    ys_da = xr.DataArray(ys, dims="point")
    sel = da_month.sel({lon: xs_da, lat: ys_da}, method="nearest")
    # Ensure (time, point) order.
    sel = sel.transpose(..., "point")
    return np.asarray(sel.values)


def sample_full(da_month, lon: str, lat: str) -> np.ndarray:
    """Full grid → ``(T, nlat*nlon)`` row-major over ``(lat, lon)``."""
    arr = np.asarray(da_month.transpose(da_month.dims[0], lat, lon).values)
    T = arr.shape[0]
    return arr.reshape(T, -1)


def full_grid_index(da, lon: str, lat: str) -> pd.DataFrame:
    """``ID, lat, lon`` for the full-grid template (row-major over (lat, lon))."""
    lat_v = np.asarray(da[lat].values)
    lon_v = np.asarray(da[lon].values)
    LAT, LON = np.meshgrid(lat_v, lon_v, indexing="ij")
    return pd.DataFrame(
        {"ID": np.arange(LAT.size, dtype=np.int64), "lat": LAT.reshape(-1), "lon": LON.reshape(-1)}
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _outputs_dir(base: Path, var: str, outputs_subdir: str) -> Path:
    return base / var / outputs_subdir


def extract(
    *,
    var: str,
    nc_dir: Path,
    base: Path,
    shp: Path | None,
    peru_potato: bool,
    outputs_subdir: str = "Outputs",
    year_range: tuple[int, int] | None = None,
    overwrite: bool = False,
    limit_months: int | None = None,
) -> int:
    """Extract every (or ``limit_months``) month for ``var`` into the Outputs tree."""
    da, lon, lat, tname = open_pisco(nc_dir)
    out_dir = _outputs_dir(base, var, outputs_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if peru_potato:
        if shp is None:
            raise SystemExit("--peru-potato requires --shp")
        xs, ys = load_points(shp, grid_crs(nc_dir))
        ids = np.arange(len(xs), dtype=np.int64)
        print(f"[{var}] potato mode: {len(ids)} points from {shp.name}")
    else:
        gi = full_grid_index(da, lon, lat)
        ids = gi["ID"].to_numpy()
        gi.to_parquet(out_dir / "grid_index.parquet", index=False)
        print(f"[{var}] full-grid mode: {len(ids)} cells; wrote grid_index.parquet")

    times = da[tname].values
    written = 0
    for y, m, pos in month_groups(times):
        if year_range and not (year_range[0] <= y <= year_range[1]):
            continue
        out_path = out_dir / f"{var}_daily_{y}_{m:02d}.parquet"
        if out_path.exists() and not overwrite:
            print(f"[{var}] {y}-{m:02d} exists, skip")
            continue
        da_m = da.isel({tname: pos})
        if peru_potato:
            vals = sample_potato(da_m, lon, lat, xs, ys)
        else:
            vals = sample_full(da_m, lon, lat)
        df = _long_frame(vals, ids, times[pos], f"{var}_daily_{y}_{m:02d}.nc")
        df.to_parquet(out_path, index=False)
        written += 1
        print(f"[{var}] wrote {out_path.name} ({len(df)} rows)")
        if limit_months and written >= limit_months:
            break
    print(f"[{var}] done: {written} month file(s) under {out_dir}")
    return 0


def verify_against_existing(
    *,
    var: str,
    nc_dir: Path,
    base: Path,
    shp: Path | None,
    peru_potato: bool,
    outputs_subdir: str = "Outputs",
) -> int:
    """Extract the first available month and diff it against the existing parquet."""
    da, lon, lat, tname = open_pisco(nc_dir)
    out_dir = _outputs_dir(base, var, outputs_subdir)

    if peru_potato:
        if shp is None:
            raise SystemExit("--peru-potato requires --shp")
        xs, ys = load_points(shp, grid_crs(nc_dir))
        ids = np.arange(len(xs), dtype=np.int64)
    else:
        ids = full_grid_index(da, lon, lat)["ID"].to_numpy()

    times = da[tname].values
    y, m, pos = next(month_groups(times))
    # Find an existing parquet to compare to (canonical or legacy tmin_v1 name).
    cands = [out_dir / f"{var}_daily_{y}_{m:02d}.parquet"]
    if var == "tmin_v1":
        cands.append(out_dir / f"tmin_daily_{y}_{m:02d}.parquet")
    existing = next((c for c in cands if c.exists()), None)
    if existing is None:
        raise SystemExit(f"no existing parquet for {var} {y}-{m:02d} under {out_dir} to verify against")

    da_m = da.isel({tname: pos})
    vals = sample_potato(da_m, lon, lat, xs, ys) if peru_potato else sample_full(da_m, lon, lat)
    new = _long_frame(vals, ids, times[pos], f"{var}_daily_{y}_{m:02d}.nc")
    ex = pd.read_parquet(existing, columns=["ID", "FECHA", "Value"])

    new_k = set(zip(new["ID"], pd.DatetimeIndex(new["FECHA"]).astype("datetime64[us]")))
    ex_k = set(zip(ex["ID"], pd.DatetimeIndex(ex["FECHA"]).astype("datetime64[us]")))
    same_keys = new_k == ex_k

    merged = new.merge(ex, on=["ID", "FECHA"], suffixes=("_new", "_old"))
    if len(merged):
        delta = (merged["Value_new"].astype(float) - merged["Value_old"].astype(float)).abs()
        max_abs = float(delta.max())
        mean_abs = float(delta.mean())
    else:
        max_abs = mean_abs = float("nan")

    print(f"=== verify {var} {y}-{m:02d} vs {existing.name} ===")
    print(f"  new rows={len(new)}  existing rows={len(ex)}  matched={len(merged)}")
    print(f"  ID range new={ids.min()}..{ids.max()}  identical (ID,FECHA) set={same_keys}")
    print(f"  max|Δ Value|={max_abs:.6g}  mean|Δ|={mean_abs:.6g}")
    ok = same_keys and len(merged) == len(ex) and (np.isnan(max_abs) or max_abs < 0.05)
    print(f"  VERDICT: {'PASS' if ok else 'REVIEW'}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract PISCO .nc rasters to per-point monthly parquet (potato or full grid).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--var", required=True, help="Output variable / folder (e.g. tmin, td, tmin_v1).")
    p.add_argument("--nc-dir", required=True, type=Path, help="Directory of source .nc file(s).")
    p.add_argument("--base", required=True, type=Path, help="Dataset base; writes {base}/{var}/Outputs.")
    p.add_argument("--shp", type=Path, default=None, help="Potato-zoning shapefile (required if --peru-potato).")
    p.add_argument(
        "--peru-potato",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample at potato points (True) or keep the full PISCO grid (--no-peru-potato).",
    )
    p.add_argument("--outputs-subdir", default="Outputs", help="Subfolder under {base}/{var}.")
    p.add_argument("--year-range", default=None, help="Inclusive 'A,B' year filter (default: all).")
    p.add_argument("--overwrite", action="store_true", help="Re-extract months whose parquet exists.")
    p.add_argument("--limit-months", type=int, default=None, help="Stop after N written months (testing).")
    p.add_argument(
        "--verify-against-existing",
        action="store_true",
        help="Extract one month and diff vs the existing parquet; write nothing.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    yr = None
    if args.year_range:
        a, b = (int(x) for x in args.year_range.split(","))
        yr = (a, b)
    if args.verify_against_existing:
        return verify_against_existing(
            var=args.var,
            nc_dir=args.nc_dir,
            base=args.base,
            shp=args.shp,
            peru_potato=args.peru_potato,
            outputs_subdir=args.outputs_subdir,
        )
    return extract(
        var=args.var,
        nc_dir=args.nc_dir,
        base=args.base,
        shp=args.shp,
        peru_potato=args.peru_potato,
        outputs_subdir=args.outputs_subdir,
        year_range=yr,
        overwrite=args.overwrite,
        limit_months=args.limit_months,
    )


if __name__ == "__main__":
    raise SystemExit(main())
