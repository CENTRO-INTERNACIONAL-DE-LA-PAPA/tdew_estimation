"""Single-pass full-grid training that applies the per-``(zone x doy)`` recipe.

Zones are a per-grid-point *feature mask*, never a data partition (plan P4): the whole
grid is trained in **one** bucket-parallel pass. For each bucket:

1. assemble the day-sums for the full candidate superset once (no year axis);
2. convolve at each distinct ``h`` present in the manifest (``h`` rides the convolution);
3. for every ``(ID, doy)`` gather the normal equations from *its* zone's ``h`` and zero out
   the columns of features its recipe dropped — the "zero-column trick": set the dropped
   diagonal to 1 and its row/col + rhs to 0 so ``beta[j] = 0`` while the retained sub-block
   solves exactly. This keeps a single uniform ``F x F`` batched solve for the bucket
   (reusing ``gpu_train.solve_bucket_reference``), avoiding a ragged per-recipe solve.

Output is **tidy/long**: one row per ``(ID, doy, retained feature)``
``[ID, zone_id, doy, feature_name, coeff, r_squared_anom, h]``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import cupy as cp

from tdew_estimation.bucket_layout import bucket_dir, bucket_for_id, discover_bucket_ids
from tdew_estimation.parquet_io import as_path, read_parquet_any

from HPC_code.gpu_train import solve_bucket_reference  # reuse the array-level oracle solve

from .assemble_generic import DOY_AXIS, assemble_grams_noyear, convolve_doy
from .feature_spec import TuningConfig, build_feature_frame
from .manifest import ZoneManifest

logger = logging.getLogger(__name__)

TIDY_COLUMNS = ["ID", "zone_id", "doy", "feature_name", "coeff", "r_squared_anom", "h"]


def _zone_recipe_arrays(
    manifest: ZoneManifest,
    zone_id: int,
    feature_index: Dict[str, int],
    hslot: Dict[int, int],
    nf: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-doy ``(h-slot, keep-mask)`` for one zone. Slot ``-1`` == no recipe for that doy."""
    hh = np.full(DOY_AXIS, -1, dtype=np.int64)
    keep = np.zeros((DOY_AXIS, nf), dtype=bool)
    for d in range(1, DOY_AXIS + 1):
        rec = manifest.lookup(zone_id, d)
        if rec is None:
            continue
        h, feats = rec
        hh[d - 1] = hslot[h]
        for f in feats:
            if f in feature_index:
                keep[d - 1, feature_index[f]] = True
    return hh, keep


def train_bucket_zoned(
    train_df: pd.DataFrame,
    clim_df: pd.DataFrame,
    id_to_zone: Dict[int, int],
    manifest: ZoneManifest,
    tuning: TuningConfig,
) -> pd.DataFrame:
    """Train one bucket applying each grid point's zone recipe; return tidy coeffs."""
    if train_df.empty or clim_df.empty:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    registry = tuning.registry()
    nf = len(registry)
    feature_index = {c: i for i, c in enumerate(registry.feature_cols)}

    frame = build_feature_frame(train_df, clim_df, registry)
    S_xx, S_xy, S_yy, cnt, id_values = assemble_grams_noyear(frame, registry)
    n_id = int(id_values.shape[0])
    if n_id == 0:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    # Convolve once per distinct h; index into it per grid point below.
    distinct_h = manifest.distinct_h()
    if not distinct_h:
        return pd.DataFrame(columns=TIDY_COLUMNS)
    hslot = {h: i for i, h in enumerate(distinct_h)}
    A_by_slot, b_by_slot, syy_by_slot, nbr_by_slot = [], [], [], []
    for h in distinct_h:
        A_h, b_h, syy_h, nbr_h = convolve_doy(
            S_xx, S_xy, S_yy, cnt, h=int(h), kernel=tuning.base.kernel, axis=1
        )
        A_by_slot.append(A_h)
        b_by_slot.append(b_h)
        syy_by_slot.append(syy_h)
        nbr_by_slot.append(nbr_h)

    # Per-cell (N, 366) h-slot + keep mask from each ID's zone recipe.
    cell_hslot = np.full((n_id, DOY_AXIS), -1, dtype=np.int64)
    keep = np.zeros((n_id, DOY_AXIS, nf), dtype=bool)
    cell_zone = np.full((n_id, DOY_AXIS), -1, dtype=np.int64)
    zone_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, idv in enumerate(id_values.tolist()):
        z = id_to_zone.get(int(idv))
        if z is None:
            continue
        if z not in zone_cache:
            zone_cache[z] = _zone_recipe_arrays(manifest, int(z), feature_index, hslot, nf)
        hh, kk = zone_cache[z]
        cell_hslot[i] = hh
        keep[i] = kk
        cell_zone[i] = z

    valid_recipe = cell_hslot >= 0
    if not valid_recipe.any():
        return pd.DataFrame(columns=TIDY_COLUMNS)

    # Select A/b/syy/nbr from the matching h for every cell.
    A_sel = cp.zeros((n_id, DOY_AXIS, nf, nf))
    b_sel = cp.zeros((n_id, DOY_AXIS, nf))
    syy_sel = cp.zeros((n_id, DOY_AXIS))
    nbr_sel = cp.zeros((n_id, DOY_AXIS))
    hslot_g = cp.asarray(cell_hslot)
    for slot in range(len(distinct_h)):
        m = hslot_g == slot
        if not bool(m.any()):
            continue
        A_sel[m] = A_by_slot[slot][m]
        b_sel[m] = b_by_slot[slot][m]
        syy_sel[m] = syy_by_slot[slot][m]
        nbr_sel[m] = nbr_by_slot[slot][m]

    # Flatten cells and keep only those with a recipe and enough neighbors.
    cells = n_id * DOY_AXIS
    A_flat = A_sel.reshape(cells, nf, nf)
    b_flat = b_sel.reshape(cells, nf)
    syy_flat = syy_sel.reshape(cells)
    nbr_flat = nbr_sel.reshape(cells)
    keep_flat = cp.asarray(keep.reshape(cells, nf))
    valid = cp.asarray(valid_recipe.reshape(cells)) & (nbr_flat >= float(tuning.base.min_samples))
    vidx = cp.where(valid)[0]
    if int(vidx.shape[0]) == 0:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    A_v = A_flat[vidx]
    b_v = b_flat[vidx]
    syy_v = syy_flat[vidx]
    keep_v = keep_flat[vidx].astype(cp.float64)  # [M, F] 1.0 kept / 0.0 dropped

    # Zero-column trick: zero dropped rows/cols, set dropped diagonal to 1, zero dropped rhs.
    A_m = A_v * keep_v[:, :, None] * keep_v[:, None, :]
    drop = 1.0 - keep_v
    diag = cp.arange(nf)
    A_m[:, diag, diag] += drop
    b_m = b_v * keep_v

    beta, r2 = solve_bucket_reference(A_m, b_m, syy_v)

    # Emit tidy rows for retained features only.
    vidx_h = cp.asnumpy(vidx)
    beta_h = cp.asnumpy(beta)
    r2_h = cp.asnumpy(r2)
    keep_h = keep.reshape(cells, nf)[vidx_h]
    cell_i = (vidx_h // DOY_AXIS).astype(int)
    doy_h = (vidx_h % DOY_AXIS).astype(int) + 1
    ids_cell = id_values[cell_i].astype(int)
    zone_cell = cell_zone.reshape(cells)[vidx_h].astype(int)
    h_cell = np.asarray(distinct_h, dtype=int)[cell_hslot.reshape(cells)[vidx_h]]

    crow, kcol = np.nonzero(keep_h)  # (retained feature entries)
    if crow.size == 0:
        return pd.DataFrame(columns=TIDY_COLUMNS)
    feature_names = np.asarray(registry.feature_cols)
    out = pd.DataFrame(
        {
            "ID": ids_cell[crow],
            "zone_id": zone_cell[crow],
            "doy": doy_h[crow],
            "feature_name": feature_names[kcol],
            "coeff": beta_h[crow, kcol],
            "r_squared_anom": r2_h[crow],
            "h": h_cell[crow],
        }
    )
    return out.sort_values(["ID", "doy", "feature_name"]).reset_index(drop=True)[TIDY_COLUMNS]


def _train_bucket_task(
    *,
    bucket_id: int,
    prepared_training_root,
    bucketed_climatology_root,
    coeffs_output_root,
    id_to_zone: Dict[int, int],
    manifest: ZoneManifest,
    tuning: TuningConfig,
    overwrite: bool = False,
) -> dict:
    """Train one bucket and write ``coeffs.parquet`` (tidy). Returns a small summary."""
    tdir = bucket_dir(prepared_training_root, bucket_id)
    cdir = bucket_dir(bucketed_climatology_root, bucket_id)
    odir = bucket_dir(coeffs_output_root, bucket_id)
    odir.mkdir(parents=True, exist_ok=True)
    ofile = odir / "coeffs.parquet"
    if ofile.exists() and not overwrite:
        return {"bucket_id": int(bucket_id), "status": "skipped", "rows": 0}

    try:
        train_df = read_parquet_any(tdir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bucket %s: cannot read training shard: %s", bucket_id, exc)
        return {"bucket_id": int(bucket_id), "status": "error", "rows": 0}
    cfile = cdir / "climatology.parquet"
    clim_df = pd.read_parquet(cfile) if cfile.exists() else pd.DataFrame()

    coeffs = train_bucket_zoned(train_df, clim_df, id_to_zone, manifest, tuning)
    if not coeffs.empty:
        coeffs.to_parquet(ofile, engine="pyarrow", index=False)
    elif overwrite and ofile.exists():
        ofile.unlink()
    return {"bucket_id": int(bucket_id), "status": "ok", "rows": int(len(coeffs))}


def run_zoned_training(
    *,
    prepared_training_root,
    bucketed_climatology_root,
    coeffs_output_root,
    zone_table: pd.DataFrame,
    manifest: ZoneManifest,
    tuning: TuningConfig,
    bucket_ids: Optional[Sequence[int]] = None,
    overwrite: bool = False,
    client: Optional[Any] = None,
) -> pd.DataFrame:
    """Train the full grid in one bucket-parallel pass; return a per-bucket summary frame.

    ``client is None`` runs buckets sequentially in-process on the single visible GPU.
    A ``client`` (e.g. from ``hpc.make_local_cuda_cluster``) submits per-bucket tasks with a
    sliding window, mirroring ``gpu_train.run_bucketed_anomaly_training_gpu``.
    """
    prepared_root = as_path(prepared_training_root)
    clim_root = as_path(bucketed_climatology_root)
    coeffs_root = as_path(coeffs_output_root)
    coeffs_root.mkdir(parents=True, exist_ok=True)

    buckets = (
        discover_bucket_ids(prepared_root) if bucket_ids is None
        else sorted({int(b) for b in bucket_ids})
    )
    if not buckets:
        raise ValueError(f"No buckets found under {prepared_root}")

    # Pre-group the ID->zone map by bucket so each task gets only its slice.
    nb = int(max(discover_bucket_ids(prepared_root))) + 1
    by_bucket: Dict[int, Dict[int, int]] = {b: {} for b in buckets}
    zt = zone_table[["ID", "zone_id"]].to_numpy()
    for idv, zid in zt:
        b = bucket_for_id(int(idv), num_buckets=nb)
        if b in by_bucket:
            by_bucket[b][int(idv)] = int(zid)

    def _kwargs(b: int) -> dict:
        return dict(
            bucket_id=int(b),
            prepared_training_root=prepared_root,
            bucketed_climatology_root=clim_root,
            coeffs_output_root=coeffs_root,
            id_to_zone=by_bucket.get(int(b), {}),
            manifest=manifest,
            tuning=tuning,
            overwrite=overwrite,
        )

    summaries: List[dict] = []
    if client is None:
        for b in buckets:
            try:
                summaries.append(_train_bucket_task(**_kwargs(b)))
            except Exception as exc:  # noqa: BLE001 - one bad bucket must not abort the run
                logger.warning("bucket %s failed: %s: %s", b, type(exc).__name__, exc)
                summaries.append({"bucket_id": int(b), "status": "error", "rows": 0})
        return pd.DataFrame(summaries).sort_values("bucket_id").reset_index(drop=True)

    from distributed import as_completed  # lazy import

    n_workers = len(getattr(client, "scheduler_info", lambda: {})().get("workers", {})) or 1
    window = max(n_workers, 4)
    pending: Dict[Any, int] = {}
    nxt = 0
    ac = as_completed()
    while nxt < len(buckets) and len(pending) < window:
        fut = client.submit(_train_bucket_task, **_kwargs(buckets[nxt]), pure=False)
        pending[fut] = buckets[nxt]
        ac.add(fut)
        nxt += 1
    for fut in ac:
        b = pending.pop(fut, -1)
        try:
            summaries.append(fut.result())
        except Exception as exc:  # noqa: BLE001
            logger.warning("bucket %s failed: %s: %s", b, type(exc).__name__, exc)
            summaries.append({"bucket_id": int(b), "status": "error", "rows": 0})
        if nxt < len(buckets):
            fut = client.submit(_train_bucket_task, **_kwargs(buckets[nxt]), pure=False)
            pending[fut] = buckets[nxt]
            ac.add(fut)
            nxt += 1
    return pd.DataFrame(summaries).sort_values("bucket_id").reset_index(drop=True)
