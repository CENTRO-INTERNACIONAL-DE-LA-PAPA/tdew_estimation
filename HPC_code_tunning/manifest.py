"""Read/write and look up the per-``(zone x doy)`` tuning recipe manifest.

The manifest is a tidy parquet with one row per selected recipe:
``[zone_id, zone_label, doy, h, feature_list, n_features, skill]`` where ``feature_list``
is a comma-joined ordered feature-name list (always starting with ``const``) and ``doy`` is
``1..366`` for per-doy recipes or :data:`selection.ALL_DOYS` (``-1``) for a zone-wide recipe.

:class:`ZoneManifest` wraps it as a fast ``(zone_id, doy) -> (h, feature_names)`` lookup
with the per-zone fallback, which ``train_zoned`` uses to build each grid point's feature
mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .selection import ALL_DOYS

MANIFEST_COLUMNS = ["zone_id", "zone_label", "doy", "h", "feature_list", "n_features",
                    "skill", "skill_baseline", "skill_uplift"]


def write_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    """Write the manifest to ``path`` (parquet), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in MANIFEST_COLUMNS if c in manifest.columns]
    manifest[cols].to_parquet(path, engine="pyarrow", index=False)
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest parquet."""
    return pd.read_parquet(path)


class ZoneManifest:
    """Fast ``(zone_id, doy) -> (h, feature_names)`` lookup with a per-zone fallback."""

    def __init__(self, manifest: pd.DataFrame):
        self._exact: Dict[Tuple[int, int], Tuple[int, Tuple[str, ...]]] = {}
        self._zone: Dict[int, Tuple[int, Tuple[str, ...]]] = {}
        for row in manifest.itertuples(index=False):
            feats = tuple(str(row.feature_list).split(","))
            entry = (int(row.h), feats)
            zid, doy = int(row.zone_id), int(row.doy)
            if doy == ALL_DOYS:
                self._zone[zid] = entry
            else:
                self._exact[(zid, doy)] = entry

    @classmethod
    def from_path(cls, path: str | Path) -> "ZoneManifest":
        return cls(read_manifest(path))

    def lookup(self, zone_id: int, doy: int) -> Optional[Tuple[int, Tuple[str, ...]]]:
        """Return ``(h, feature_names)`` for a grid point, or ``None`` if unrecipe'd.

        Prefers the exact ``(zone, doy)`` recipe, then the zone-wide (``doy == -1``) one.
        """
        hit = self._exact.get((int(zone_id), int(doy)))
        if hit is not None:
            return hit
        return self._zone.get(int(zone_id))

    def distinct_h(self) -> List[int]:
        hs = {h for h, _ in self._exact.values()} | {h for h, _ in self._zone.values()}
        return sorted(hs)

    def zones(self) -> List[int]:
        return sorted({z for z, _ in self._exact} | set(self._zone))
