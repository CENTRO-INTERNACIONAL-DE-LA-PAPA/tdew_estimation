"""
Smoke + correctness test for the PISCO .nc -> point-parquet extractor (nc_to_point_parquet).

Builds a tiny synthetic monthly raster (known lat/lon grid, value = a deterministic function
of (lat, lon, day)) and a small point set sitting exactly on known cell centres, then checks:

* potato mode reproduces the known cell values at each point, in feature order (ID = 0..N-1),
  with the same schema/dtypes as the existing pipeline parquet (ID int64, FECHA us, Value f32);
* full-grid mode yields n_lat*n_lon IDs and a stable grid_index;
* the verify-against-existing mode reports an identical (ID, FECHA) set and ~0 max|Δ|.

Pure CPU; xarray/geopandas are importorskip-ed so the suite still runs without the netcdf extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

import nc_to_point_parquet as ncp  # noqa: E402

# A 3x4 lat/lon grid, 5 days in Jan 1990.
LATS = np.array([-10.0, -10.1, -10.2], dtype="float64")          # descending, like PISCO
LONS = np.array([-78.0, -77.9, -77.8, -77.7], dtype="float64")
DAYS = pd.date_range("1990-01-01", periods=5, freq="D")


def _cell_value(lat: float, lon: float, t: int) -> float:
    """Deterministic, unique per (lat, lon, day)."""
    return round(100 * lat + lon + 0.1 * t, 4)


def _make_nc(path: Path) -> None:
    data = np.empty((len(DAYS), len(LATS), len(LONS)), dtype="float32")
    for ti in range(len(DAYS)):
        for i, la in enumerate(LATS):
            for j, lo in enumerate(LONS):
                data[ti, i, j] = _cell_value(la, lo, ti)
    ds = xr.Dataset(
        {"tmin": (("time", "latitude", "longitude"), data)},
        coords={"time": DAYS.values, "latitude": LATS, "longitude": LONS},
    )
    ds.to_netcdf(path)


def _make_points(path: Path, picks: list[tuple[int, int]]) -> None:
    """Points exactly on the centres of the chosen (lat_idx, lon_idx) cells."""
    from shapely.geometry import Point

    geoms = [Point(LONS[j], LATS[i]) for (i, j) in picks]
    gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs="EPSG:4326").to_file(path)


def test_potato_mode_values_and_schema(tmp_path):
    nc_dir = tmp_path / "raw"
    nc_dir.mkdir()
    _make_nc(nc_dir / "tmin_daily_1990_01.nc")
    shp = tmp_path / "pts.shp"
    picks = [(0, 0), (2, 3), (1, 2)]  # feature order -> ID 0,1,2
    _make_points(shp, picks)

    base = tmp_path / "base"
    rc = ncp.extract(var="tmin", nc_dir=nc_dir, base=base, shp=shp, peru_potato=True)
    assert rc == 0

    out = base / "tmin" / "Outputs" / "tmin_daily_1990_01.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == ["ID", "FECHA", "Value", "source_file"]
    assert df["ID"].dtype == np.int64
    assert df["Value"].dtype == np.float32
    assert str(df["FECHA"].dtype) == "datetime64[us]"
    assert df["source_file"].iloc[0] == "tmin_daily_1990_01.nc"
    # 3 points x 5 days
    assert len(df) == 3 * 5
    assert sorted(df["ID"].unique()) == [0, 1, 2]

    # Each point must carry its known cell value on each day, in feature order.
    for pid, (i, j) in enumerate(picks):
        sub = df[df["ID"] == pid].sort_values("FECHA")
        expected = [_cell_value(LATS[i], LONS[j], t) for t in range(len(DAYS))]
        np.testing.assert_allclose(sub["Value"].to_numpy(), expected, atol=1e-3)


def test_full_grid_mode_and_index(tmp_path):
    nc_dir = tmp_path / "raw"
    nc_dir.mkdir()
    _make_nc(nc_dir / "tmin_daily_1990_01.nc")
    base = tmp_path / "base"

    rc = ncp.extract(var="tmin", nc_dir=nc_dir, base=base, shp=None, peru_potato=False)
    assert rc == 0

    outdir = base / "tmin" / "Outputs"
    gi = pd.read_parquet(outdir / "grid_index.parquet")
    assert len(gi) == len(LATS) * len(LONS)
    assert list(gi.columns) == ["ID", "lat", "lon"]

    df = pd.read_parquet(outdir / "tmin_daily_1990_01.parquet")
    assert len(df) == len(LATS) * len(LONS) * len(DAYS)
    # ID 0 is row-major (lat0, lon0); check its value on day 0.
    d0 = df[(df["ID"] == 0) & (df["FECHA"] == DAYS[0])]
    assert d0["Value"].iloc[0] == pytest.approx(_cell_value(LATS[0], LONS[0], 0), abs=1e-3)


def test_verify_against_existing_passes(tmp_path):
    nc_dir = tmp_path / "raw"
    nc_dir.mkdir()
    _make_nc(nc_dir / "tmin_daily_1990_01.nc")
    shp = tmp_path / "pts.shp"
    picks = [(0, 0), (2, 3), (1, 2)]
    _make_points(shp, picks)
    base = tmp_path / "base"

    # First produce the "existing" parquet, then verify a fresh extraction against it.
    assert ncp.extract(var="tmin", nc_dir=nc_dir, base=base, shp=shp, peru_potato=True) == 0
    rc = ncp.verify_against_existing(
        var="tmin", nc_dir=nc_dir, base=base, shp=shp, peru_potato=True
    )
    assert rc == 0
