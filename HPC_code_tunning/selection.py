"""Multitask backward-stepwise feature selection + per-doy ``h`` grid search, per zone.

For each SENAMHI climate zone we assemble the Gram tensor for the *full* candidate
superset once (per ID chunk), then search for the best ``(h, feature_set)`` per
``zone x doy`` (or per ``zone`` when ``granularity == "zone"``) under the LOYOCV cosine
objective in :mod:`loyocv`:

* **h grid** rides the cheap DOY convolution only (the raw day-sums are assembled once).
* **backward-stepwise** starts from the full pool and drops the least-useful feature while
  the best available drop improves cosine skill by at least ``tol`` (``const`` is never
  dropped; at least one non-``const`` feature is kept). Features are chosen *jointly across
  grid points* (the cosine is over the spatial vector of all sampled IDs) but coefficients
  are fit per grid point — the paper's "multitask" scheme.

Memory is bounded by processing the zone's ID sample in chunks (``TuningConfig.id_chunk``);
the per-target cosine segment sums are additive across chunks so the result is exact. The
zone's feature frames are built once (the disk-bound climatology merge + lag construction)
and re-assembled on the GPU per ``h``/round.
"""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import cupy as cp

from tdew_estimation.bucket_layout import bucket_dir, bucket_for_id, discover_bucket_ids
from tdew_estimation.parquet_io import as_path, read_parquet_any

from .assemble_generic import DOY_AXIS, assemble_from_feature_frame
from .feature_spec import FeatureRegistry, TuningConfig, build_feature_frame
from .loyocv import (
    ConvolvedChunk,
    accumulate_segment_sums,
    cosine_from_segments,
    new_segment_buffers,
)

logger = logging.getLogger(__name__)

# doy value used in the manifest to mean "applies to every doy in the zone".
ALL_DOYS = -1


# ---------------------------------------------------------------------------------------
# Data loading (sampled IDs -> cached per-chunk feature frames)
# ---------------------------------------------------------------------------------------
def infer_num_buckets(prepared_root) -> int:
    """Infer the ``num_buckets`` used at prep time from the hive directories.

    Bucket assignment is ``ID % num_buckets`` and directories ``id_bucket=0000..K`` are
    written for non-empty buckets, so ``max(bucket_id) + 1`` recovers ``num_buckets`` as
    long as the last bucket is non-empty (true for any realistic ID population).
    """
    ids = discover_bucket_ids(prepared_root)
    if not ids:
        raise ValueError(f"no id_bucket=* directories under {prepared_root}")
    return int(max(ids)) + 1


def load_training_for_ids(
    prepared_root, clim_root, ids: Sequence[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read training + climatology rows for a set of IDs from the bucketed shards.

    Groups IDs by their bucket so each shard is read once, then filters to the requested
    IDs. Returns ``(train_df, clim_df)`` (possibly empty).
    """
    prepared_root = as_path(prepared_root)
    clim_root = as_path(clim_root)
    nb = infer_num_buckets(prepared_root)
    by_bucket: Dict[int, List[int]] = defaultdict(list)
    for i in ids:
        by_bucket[bucket_for_id(int(i), num_buckets=nb)].append(int(i))

    train_frames: List[pd.DataFrame] = []
    clim_frames: List[pd.DataFrame] = []
    for bid, bucket_ids in by_bucket.items():
        keep = set(bucket_ids)
        tdir = bucket_dir(prepared_root, bid)
        if tdir.exists():
            try:
                t = read_parquet_any(tdir)
                train_frames.append(t[t["ID"].isin(keep)])
            except Exception as exc:  # noqa: BLE001 - one bad shard must not abort tuning
                logger.warning("skip training bucket %s: %s", bid, exc)
        cfile = bucket_dir(clim_root, bid) / "climatology.parquet"
        if cfile.exists():
            c = pd.read_parquet(cfile)
            clim_frames.append(c[c["ID"].isin(keep)])

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    clim_df = pd.concat(clim_frames, ignore_index=True) if clim_frames else pd.DataFrame()
    return train_df, clim_df


def build_zone_frames(
    prepared_root,
    clim_root,
    ids: Sequence[int],
    registry: FeatureRegistry,
    *,
    id_chunk: int,
) -> List[pd.DataFrame]:
    """Build the feature frame for a zone's ID sample once, split into ID chunks.

    Chunking is by unique ID (chunks of ``id_chunk`` IDs) so a chunk's tensors stay within
    GPU memory. Returns a list of per-chunk feature frames (each from
    :func:`feature_spec.build_feature_frame`).
    """
    train_df, clim_df = load_training_for_ids(prepared_root, clim_root, ids)
    if train_df.empty or clim_df.empty:
        return []
    frame = build_feature_frame(train_df, clim_df, registry)
    if frame.empty:
        return []
    uniq = np.sort(frame["ID"].unique())
    frames: List[pd.DataFrame] = []
    for start in range(0, len(uniq), id_chunk):
        chunk_ids = set(uniq[start : start + id_chunk].tolist())
        frames.append(frame[frame["ID"].isin(chunk_ids)].reset_index(drop=True))
    return frames


# ---------------------------------------------------------------------------------------
# Subset evaluation: cosine skill per "unit" (doy or whole-zone) for a list of subsets.
# ---------------------------------------------------------------------------------------
def _evaluate_subsets(
    frames: Sequence[pd.DataFrame],
    registry: FeatureRegistry,
    subsets: Sequence[Tuple[int, ...]],
    *,
    h: int,
    kernel: str,
    min_samples: int,
    year_values: np.ndarray,
) -> Dict[Tuple[int, ...], cp.ndarray]:
    """Return ``{subset: cosine_matrix[366, Y]}`` for every subset, over all ID chunks.

    Convolution depends only on ``h`` (not the subset), so each chunk is assembled and
    convolved once and every subset's segment sums are accumulated from it.
    """
    n_year = int(year_values.shape[0])
    buffers = {s: new_segment_buffers(n_year) for s in subsets}
    subset_arrays = {s: cp.asarray(np.asarray(s, dtype=np.int64)) for s in subsets}

    for frame in frames:
        assembled = assemble_from_feature_frame(frame, registry, year_values=year_values)
        if assembled.n_id == 0:
            continue
        conv = ConvolvedChunk(assembled, h=h, kernel=kernel)
        for s in subsets:
            spy, sxx, syy = buffers[s]
            accumulate_segment_sums(
                assembled, conv, subset_arrays[s],
                min_samples=min_samples, spy=spy, sxx=sxx, syy=syy,
            )
        del conv, assembled
        cp._default_memory_pool.free_all_blocks()

    return {s: cosine_from_segments(*buffers[s], n_year) for s in subsets}


def _unit_skill(cos: cp.ndarray, granularity: str) -> np.ndarray:
    """Reduce a ``[366, Y]`` cosine matrix to per-unit skill.

    ``"doy"`` -> length-366 vector (mean over years per doy); ``"zone"`` -> length-1
    (mean over all target dates). NaNs are ignored.
    """
    arr = cp.asnumpy(cos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN, expected
        if granularity == "zone":
            m = np.nanmean(arr) if np.isfinite(arr).any() else np.nan
            return np.array([m], dtype=np.float64)
        skill = np.full(arr.shape[0], np.nan, dtype=np.float64)
        row_has_data = np.isfinite(arr).any(axis=1)
        skill[row_has_data] = np.nanmean(arr[row_has_data], axis=1)
        return skill  # per-doy


# ---------------------------------------------------------------------------------------
# Backward-stepwise for one h (all units share the chunk convolutions).
# ---------------------------------------------------------------------------------------
def _backward_stepwise_for_h(
    frames: Sequence[pd.DataFrame],
    registry: FeatureRegistry,
    *,
    h: int,
    kernel: str,
    min_samples: int,
    year_values: np.ndarray,
    tol: float,
    granularity: str,
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """Run multitask backward-stepwise at a fixed ``h``.

    Returns ``(sets, skills)`` where ``sets[u]`` is the selected feature-index tuple for
    unit ``u`` and ``skills[u]`` its LOYOCV cosine skill. Units are doys (``U=366``) or the
    whole zone (``U=1``) per ``granularity``.
    """
    n_units = 1 if granularity == "zone" else DOY_AXIS
    full = tuple(range(len(registry)))  # all feature indices; 0 == const

    def eval_sets(subsets: Sequence[Tuple[int, ...]]) -> Dict[Tuple[int, ...], np.ndarray]:
        cos = _evaluate_subsets(
            frames, registry, subsets,
            h=h, kernel=kernel, min_samples=min_samples, year_values=year_values,
        )
        return {s: _unit_skill(c, granularity) for s, c in cos.items()}

    base = eval_sets([full])[full]
    current: List[Tuple[int, ...]] = [full] * n_units
    skills = np.array(base, dtype=np.float64, copy=True)
    converged = ~np.isfinite(skills)  # units with no data are done immediately

    while not converged.all():
        # Gather the distinct candidate drops proposed by any still-active unit.
        proposals: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for u in range(n_units):
            if converged[u] or len(current[u]) <= 2:  # keep const + >=1 feature
                converged[u] = True
                continue
            for f in current[u]:
                if f == 0:  # never drop const
                    continue
                cand = tuple(x for x in current[u] if x != f)
                proposals[cand].append(u)
        if not proposals:
            break

        cand_skill = eval_sets(list(proposals.keys()))
        improved = np.zeros(n_units, dtype=bool)
        best_set: Dict[int, Tuple[int, ...]] = {}
        best_val = np.full(n_units, -np.inf)
        for cand, units in proposals.items():
            sk = cand_skill[cand]
            for u in units:
                v = sk[u]
                if np.isfinite(v) and v > best_val[u]:
                    best_val[u] = v
                    best_set[u] = cand
        for u in range(n_units):
            if u in best_set and best_val[u] - skills[u] >= tol:
                current[u] = best_set[u]
                skills[u] = best_val[u]
                improved[u] = True
            else:
                converged[u] = True
        if not improved.any():
            break

    return current, skills


# ---------------------------------------------------------------------------------------
# Per-zone selection over the h grid.
# ---------------------------------------------------------------------------------------
def select_zone(
    frames: Sequence[pd.DataFrame],
    registry: FeatureRegistry,
    tuning: TuningConfig,
    *,
    zone_id: int,
    zone_label: str = "",
) -> pd.DataFrame:
    """Select ``(h, feature_set)`` per unit for one zone across the whole ``h`` grid.

    Returns manifest rows ``[zone_id, zone_label, doy, h, feature_list, n_features, skill]``.
    ``doy`` is ``1..366`` for ``granularity == "doy"`` or :data:`ALL_DOYS` for ``"zone"``.
    """
    year_values = np.arange(
        tuning.base.train_year_range[0], tuning.base.train_year_range[1] + 1, dtype=np.int64
    )
    n_units = 1 if tuning.granularity == "zone" else DOY_AXIS

    best_h = np.zeros(n_units, dtype=np.int64)
    best_skill = np.full(n_units, -np.inf)
    best_sets: List[Tuple[int, ...]] = [(0,)] * n_units

    for h in tuning.h_grid:
        sets, skills = _backward_stepwise_for_h(
            frames, registry,
            h=int(h), kernel=tuning.base.kernel, min_samples=tuning.base.min_samples,
            year_values=year_values, tol=tuning.tol, granularity=tuning.granularity,
        )
        take = np.isfinite(skills) & (skills > best_skill)
        for u in np.where(take)[0]:
            best_skill[u] = skills[u]
            best_h[u] = int(h)
            best_sets[u] = sets[u]
        logger.info(
            "zone %s h=%s: median skill=%.4f (units with data=%d)",
            zone_id, h, float(np.nanmedian(np.where(np.isfinite(skills), skills, np.nan))),
            int(np.isfinite(skills).sum()),
        )

    rows = []
    for u in range(n_units):
        if not np.isfinite(best_skill[u]):
            continue
        feats = [registry.feature_cols[i] for i in sorted(best_sets[u])]
        rows.append(
            {
                "zone_id": int(zone_id),
                "zone_label": zone_label,
                "doy": (u + 1) if tuning.granularity != "zone" else ALL_DOYS,
                "h": int(best_h[u]),
                "feature_list": ",".join(feats),
                "n_features": len(feats),
                "skill": float(best_skill[u]),
            }
        )
    return pd.DataFrame(rows)


def run_selection(
    zone_table: pd.DataFrame,
    sample: Dict[int, np.ndarray],
    prepared_root,
    clim_root,
    tuning: TuningConfig,
) -> pd.DataFrame:
    """Run selection for every sampled zone and return the concatenated manifest.

    ``sample`` maps ``zone_id -> np.ndarray[ID]`` (from ``zones.stratified_sample``).
    """
    registry = tuning.registry()
    labels = (
        zone_table.drop_duplicates("zone_id").set_index("zone_id")["zone_label"].to_dict()
        if "zone_label" in zone_table.columns else {}
    )
    parts: List[pd.DataFrame] = []
    for zone_id, ids in sorted(sample.items()):
        frames = build_zone_frames(
            prepared_root, clim_root, ids, registry, id_chunk=tuning.id_chunk
        )
        if not frames:
            logger.warning("zone %s: no training data for %d sampled IDs", zone_id, len(ids))
            continue
        logger.info("zone %s: %d IDs -> %d chunk(s)", zone_id, len(ids), len(frames))
        part = select_zone(
            frames, registry, tuning,
            zone_id=int(zone_id), zone_label=str(labels.get(int(zone_id), "")),
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=["zone_id", "zone_label", "doy", "h", "feature_list", "n_features", "skill"]
        )
    return pd.concat(parts, ignore_index=True)
