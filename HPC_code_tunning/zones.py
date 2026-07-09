"""ID -> climatic-zone table + stratified sampler.

**Zone source:** the SENAMHI Thornthwaite climate classification, shipped as a shapefile
inside a zip (``clasif_clima_peru.zip`` -> ``clasif_clima_1981_2010.shp``; EPSG:4326
polygons; label field ``CODIGO``). Assignment to a grid point is a point-in-polygon join.

**Grid-point coordinates** come from one of:
  * the full-grid ``grid_index.parquet`` (columns ``ID, lat, lon``) produced by
    ``HPC_code/nc_to_point_parquet.py`` in ``--no-peru-potato`` mode -> the production path;
  * a point shapefile whose *feature order* is the ID (e.g. the potato
    ``CENAGRO_OnlyPotatoes_Pisco_Altitude.shp`` with ``longitude``/``latitude`` columns)
    -> the dev/validation substrate before the 2M grid is extracted.

Output table columns: ``ID, zone_id (int32), zone_label (str), lon, lat``.
"""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

# Field in the SENAMHI shapefile carrying the Thornthwaite zone label.
ZONE_LABEL_FIELD = "CODIGO"
UNASSIGNED = -1  # zone_id for points that match no polygon (should be ~none after nearest fallback)


# ---------------------------------------------------------------------------------------
# Coordinates loading (ID -> lon/lat)
# ---------------------------------------------------------------------------------------
def load_grid_coords(coords_source: str | Path) -> pd.DataFrame:
    """Return ``DataFrame[ID(int64), lon(float64), lat(float64)]`` from a coords source.

    Accepts a ``grid_index.parquet`` (``ID, lat, lon``) or a point shapefile whose feature
    order defines the ID (``longitude``/``latitude`` columns).
    """
    coords_source = Path(coords_source)
    suffix = coords_source.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(coords_source)
        cols = {c.lower(): c for c in df.columns}
        if "id" not in cols or "lon" not in cols or "lat" not in cols:
            raise ValueError(
                f"grid_index parquet must have ID/lon/lat columns; got {list(df.columns)}"
            )
        out = pd.DataFrame(
            {
                "ID": df[cols["id"]].to_numpy("int64"),
                "lon": df[cols["lon"]].to_numpy("float64"),
                "lat": df[cols["lat"]].to_numpy("float64"),
            }
        )
    elif suffix == ".shp":
        import geopandas as gpd

        gdf = gpd.read_file(coords_source)
        cols = {c.lower(): c for c in gdf.columns}
        lon_col = cols.get("longitude") or cols.get("lon") or cols.get("x")
        lat_col = cols.get("latitude") or cols.get("lat") or cols.get("y")
        if lon_col and lat_col:
            lon = gdf[lon_col].to_numpy("float64")
            lat = gdf[lat_col].to_numpy("float64")
        else:  # fall back to geometry centroids
            g = gdf.geometry
            lon = g.x.to_numpy("float64")
            lat = g.y.to_numpy("float64")
        # feature order IS the ID (matches HPC_code/nc_to_point_parquet.load_points potato mode)
        out = pd.DataFrame({"ID": np.arange(len(gdf), dtype="int64"), "lon": lon, "lat": lat})
    else:
        raise ValueError(f"unsupported coords source {coords_source!r} (want .parquet or .shp)")

    return out


# ---------------------------------------------------------------------------------------
# SENAMHI zone shapefile (inside the zip)
# ---------------------------------------------------------------------------------------
def extract_zone_shapefile(zone_zip: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Extract the ``.shp`` (with its sidecars) from ``zone_zip`` and return its path.

    Extracts to ``cache_dir`` (default: a temp dir); a shapefile needs its .dbf/.shx/.prj
    siblings so all members are extracted.
    """
    zone_zip = Path(zone_zip)
    cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="senamhi_zones_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zone_zip) as zf:
        members = zf.namelist()
        zf.extractall(cache_dir)
    shp = [m for m in members if m.lower().endswith(".shp")]
    if not shp:
        raise FileNotFoundError(f"no .shp inside {zone_zip}")
    return cache_dir / shp[0]


# ---------------------------------------------------------------------------------------
# ID -> zone table (point-in-polygon)
# ---------------------------------------------------------------------------------------
def build_zone_table(
    coords_source: str | Path,
    zone_zip: str | Path,
    out_parquet: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    nearest_fallback: bool = True,
) -> pd.DataFrame:
    """Assign each grid point (ID) to a SENAMHI Thornthwaite zone by point-in-polygon.

    Returns ``DataFrame[ID, zone_id, zone_label, lon, lat]`` and optionally writes it to
    ``out_parquet``. ``zone_id`` is a stable integer code (sorted unique labels).
    """
    import geopandas as gpd

    coords = load_grid_coords(coords_source)
    shp = extract_zone_shapefile(zone_zip, cache_dir)
    zones = gpd.read_file(shp)
    if zones.crs is None:
        zones = zones.set_crs(4326)
    else:
        zones = zones.to_crs(4326)

    label_field = ZONE_LABEL_FIELD if ZONE_LABEL_FIELD in zones.columns else zones.columns[0]
    zones = zones[[label_field, "geometry"]].rename(columns={label_field: "zone_label"})

    pts = gpd.GeoDataFrame(
        coords, geometry=gpd.points_from_xy(coords["lon"], coords["lat"]), crs=4326
    )

    joined = gpd.sjoin(pts, zones, predicate="within", how="left")
    # a point on a shared polygon edge can match >1 polygon -> keep the first match per point
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()
    label = joined["zone_label"]

    if nearest_fallback and label.isna().any():
        miss = label.isna().to_numpy()
        nn = gpd.sjoin_nearest(pts.iloc[miss], zones, how="left")
        nn = nn[~nn.index.duplicated(keep="first")].sort_index()
        label = label.copy()
        label.iloc[miss] = nn["zone_label"].to_numpy()

    label = label.astype("object")
    cats = sorted(pd.Series(label.dropna().unique()).astype(str))
    code = {c: i for i, c in enumerate(cats)}
    zone_id = np.array([code.get(str(v), UNASSIGNED) if pd.notna(v) else UNASSIGNED for v in label],
                       dtype="int32")

    out = pd.DataFrame(
        {
            "ID": coords["ID"].to_numpy("int64"),
            "zone_id": zone_id,
            "zone_label": [str(v) if pd.notna(v) else "" for v in label],
            "lon": coords["lon"].to_numpy("float64"),
            "lat": coords["lat"].to_numpy("float64"),
        }
    )
    if out_parquet is not None:
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_parquet, index=False)
    return out


# ---------------------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------------------
def stratified_sample(
    zone_table: pd.DataFrame,
    present_ids: Iterable[int] | None = None,
    *,
    per_zone_n: int = 2000,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Draw up to ``per_zone_n`` IDs from EACH zone (seeded), so small zones are not starved.

    ``present_ids`` restricts to IDs that actually have training data (e.g. from
    ``tdew_estimation.bucket_layout.discover_bucket_ids`` + shard reads). Returns
    ``{zone_id: sorted np.ndarray[ID]}``; skips the unassigned bucket (``zone_id == -1``).
    """
    rng = np.random.default_rng(seed)
    df = zone_table
    if present_ids is not None:
        df = df[df["ID"].isin(set(int(i) for i in present_ids))]
    out: dict[int, np.ndarray] = {}
    for zid, grp in df.groupby("zone_id"):
        if int(zid) == UNASSIGNED:
            continue
        ids = grp["ID"].to_numpy("int64")
        n = min(per_zone_n, len(ids))
        picked = rng.choice(ids, size=n, replace=False) if len(ids) else ids
        out[int(zid)] = np.sort(picked)
    return out


def zone_counts(zone_table: pd.DataFrame) -> pd.DataFrame:
    """Per-zone ID counts (for validation / deciding # of training passes)."""
    g = (
        zone_table.groupby(["zone_id", "zone_label"], dropna=False)
        .size()
        .reset_index(name="n_ids")
        .sort_values("n_ids", ascending=False, ignore_index=True)
    )
    return g


# ---------------------------------------------------------------------------------------
# CLI: build the zone table and report zone counts
# ---------------------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build ID->SENAMHI-zone table + report counts.")
    p.add_argument("--coords", required=True, help="grid_index.parquet OR a point .shp")
    p.add_argument("--zone-zip", required=True, help="clasif_clima_peru.zip")
    p.add_argument("--out", default=None, help="output parquet for the zone table")
    p.add_argument("--cache-dir", default=None, help="dir to extract the shapefile into")
    args = p.parse_args(argv)

    tbl = build_zone_table(args.coords, args.zone_zip, args.out, cache_dir=args.cache_dir)
    counts = zone_counts(tbl)
    n_unassigned = int((tbl["zone_id"] == UNASSIGNED).sum())
    print(f"[zones] {len(tbl)} points -> {counts['zone_id'].nunique()} zones "
          f"({n_unassigned} unassigned)")
    print(counts.to_string(index=False))
    if args.out:
        print(f"[zones] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
