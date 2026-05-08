"""
tdew_estimation.patch_coeffs

Patch (repair) anomaly coefficient outputs for specific DOYs (day-of-year).

Context
-------
In the anomaly-model pipeline, coefficient rows are keyed by (ID, doy). Runs can
partially fail for some DOYs. Instead of re-running everything, you can:
1) retrain only the failed DOYs and write a "patch" parquet containing rows for those DOYs
2) patch the combined coefficients parquet by replacing rows for those DOYs.

This module implements step (2) in a path-agnostic way.

Design
------
- Keys: (ID, doy) by default.
- Replacement strategy:
  - remove rows from the base coefficients where doy is in doys_to_patch
  - append patch rows (optionally filtered to only doys_to_patch)
  - de-duplicate by keys (keep the last occurrence => patch wins)
- Output: write a new parquet or overwrite an existing file.

Typical usage
-------------
from pathlib import Path
from tdew_estimation.patch_coeffs import patch_anomaly_coeffs

base = Path(".../results/llr_coeffs_anomaly_final_direct.parquet")
patch = Path(".../results/anomaly_coeffs_patch_doys.parquet")

patched = patch_anomaly_coeffs(
    base_coeffs_path=base,
    patch_coeffs_path=patch,
    doys_to_patch=[13, 54, 65],
    output_path=Path(".../results/llr_coeffs_anomaly_final_direct_patched.parquet"),
)

print("Wrote:", patched)

Notes
-----
- This is intended for the anomaly coefficient dataset (not GPU per-DOY model files).
- The patch file must have the same schema/columns as the base file, at least for
  the key columns. Extra columns will be preserved but must align for concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

from .bucket_layout import bucket_dir, discover_bucket_ids

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PatchSummary:
    """Structured summary of a patch operation."""

    base_coeffs_path: Path
    patch_coeffs_path: Path
    output_path: Path
    id_col: str
    doy_col: str
    doys_to_patch: List[int]
    base_rows_before: int
    base_rows_removed: int
    patch_rows_loaded: int
    patch_rows_used: int
    output_rows: int


@dataclass(frozen=True)
class DatasetPatchSummary:
    """Summary of patching a bucketed coefficient dataset in place."""

    base_coeffs_path: Path
    patch_coeffs_path: Path
    output_path: Path
    doys_to_patch: List[int]
    bucket_summaries: List[PatchSummary]


def _as_path(p: PathLike) -> Path:
    return Path(p).expanduser().resolve()


def _normalize_doys(doys: Iterable[int]) -> List[int]:
    out: List[int] = []
    for d in doys:
        try:
            di = int(d)
        except Exception as exc:
            raise ValueError(f"Invalid DOY value: {d!r}") from exc
        if di < 1 or di > 366:
            raise ValueError(f"DOY out of range (1..366): {di}")
        out.append(di)
    return sorted(set(out))


def _validate_key_columns(
    df: pd.DataFrame, id_col: str, doy_col: str, name: str
) -> None:
    missing = [c for c in (id_col, doy_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing required key columns: {missing}. Columns: {list(df.columns)}"
        )


def _patch_frames(
    *,
    base_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    doys: Sequence[int],
    id_col: str,
    doy_col: str,
    keep: str,
) -> tuple[pd.DataFrame, int, int, int, int, int]:
    _validate_key_columns(base_df, id_col, doy_col, "base_coeffs")
    _validate_key_columns(patch_df, id_col, doy_col, "patch_coeffs")

    base_df = base_df.copy()
    patch_df = patch_df.copy()
    base_df[doy_col] = pd.to_numeric(base_df[doy_col], errors="coerce").astype("Int64")
    patch_df[doy_col] = pd.to_numeric(patch_df[doy_col], errors="coerce").astype(
        "Int64"
    )

    base_rows_before = int(len(base_df))
    base_keep_df = base_df[~base_df[doy_col].isin(doys)].copy()
    base_rows_removed = base_rows_before - int(len(base_keep_df))

    patch_rows_loaded = int(len(patch_df))
    patch_use_df = patch_df[patch_df[doy_col].isin(doys)].copy()
    patch_rows_used = int(len(patch_use_df))

    combined = pd.concat([base_keep_df, patch_use_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=[id_col, doy_col], keep=keep)
    try:
        combined[doy_col] = combined[doy_col].astype(int)
    except Exception:
        pass
    combined = combined.sort_values([id_col, doy_col]).reset_index(drop=True)
    return (
        combined,
        base_rows_before,
        base_rows_removed,
        patch_rows_loaded,
        patch_rows_used,
        int(len(combined)),
    )


def patch_anomaly_coeffs(
    *,
    base_coeffs_path: PathLike,
    patch_coeffs_path: PathLike,
    doys_to_patch: Sequence[int],
    output_path: PathLike,
    id_col: str = "ID",
    doy_col: str = "doy",
    keep: str = "last",
    overwrite: bool = True,
    parquet_engine: Optional[str] = None,
) -> Path:
    """
    Patch anomaly coefficients by replacing rows for specified DOYs.

    Parameters
    ----------
    base_coeffs_path:
        Existing combined coefficient parquet (e.g., llr_coeffs_anomaly_final_direct.parquet).
    patch_coeffs_path:
        Parquet containing re-trained coefficient rows for DOYs in doys_to_patch.
    doys_to_patch:
        DOYs to replace in the base dataset.
    output_path:
        Where to write patched coefficients.
    id_col, doy_col:
        Key columns (default: "ID" and "doy").
    keep:
        De-duplication policy when concatenating (pandas drop_duplicates keep=...).
        Usually "last" so patch rows win.
    overwrite:
        If False, raises if output_path already exists.
    parquet_engine:
        Optional engine override for pandas.read_parquet / to_parquet.

    Returns
    -------
    Path
        Resolved path to the written output parquet.
    """
    base_p = _as_path(base_coeffs_path)
    patch_p = _as_path(patch_coeffs_path)
    out_p = _as_path(output_path)

    if not base_p.exists():
        raise FileNotFoundError(f"Base coefficients not found: {base_p}")
    if not patch_p.exists():
        raise FileNotFoundError(f"Patch coefficients not found: {patch_p}")
    if out_p.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=False: {out_p}")

    doys = _normalize_doys(doys_to_patch)
    if not doys:
        raise ValueError("doys_to_patch is empty; nothing to patch.")

    read_kwargs = {}
    write_kwargs = {}
    if parquet_engine:
        read_kwargs["engine"] = parquet_engine
        write_kwargs["engine"] = parquet_engine

    base_df = pd.read_parquet(base_p, **read_kwargs)
    patch_df = pd.read_parquet(patch_p, **read_kwargs)

    _validate_key_columns(base_df, id_col, doy_col, "base_coeffs")
    _validate_key_columns(patch_df, id_col, doy_col, "patch_coeffs")

    # Normalize DOY column types for stable filtering
    base_df = base_df.copy()
    patch_df = patch_df.copy()

    base_df[doy_col] = pd.to_numeric(base_df[doy_col], errors="coerce").astype("Int64")
    patch_df[doy_col] = pd.to_numeric(patch_df[doy_col], errors="coerce").astype(
        "Int64"
    )

    base_rows_before = int(len(base_df))

    # Remove target DOYs from base
    base_keep_df = base_df[~base_df[doy_col].isin(doys)].copy()
    base_rows_removed = base_rows_before - int(len(base_keep_df))

    # Keep only target DOYs from patch (defensive; allows patch file to contain more)
    patch_rows_loaded = int(len(patch_df))
    patch_use_df = patch_df[patch_df[doy_col].isin(doys)].copy()
    patch_rows_used = int(len(patch_use_df))

    # Concatenate and ensure patch rows win on duplicates
    combined = pd.concat([base_keep_df, patch_use_df], ignore_index=True, sort=False)

    # Drop duplicates on keys; keep patch values (keep="last")
    combined = combined.drop_duplicates(subset=[id_col, doy_col], keep=keep)

    # Optional: sort for readability
    try:
        combined[doy_col] = combined[doy_col].astype(int)
    except Exception:
        # best-effort; leave as is
        pass
    combined = combined.sort_values([id_col, doy_col]).reset_index(drop=True)

    # Write output
    out_p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_p, index=False, **write_kwargs)

    return out_p


def patch_anomaly_coeffs_inplace(
    *,
    base_coeffs_path: PathLike,
    patch_coeffs_path: PathLike,
    doys_to_patch: Sequence[int],
    id_col: str = "ID",
    doy_col: str = "doy",
    parquet_engine: Optional[str] = None,
) -> Union[PatchSummary, DatasetPatchSummary]:
    """
    In-place patch convenience wrapper.

    Overwrites base_coeffs_path with patched content.

    Returns a PatchSummary (counts/paths) for logging.
    """
    base_p = _as_path(base_coeffs_path)
    patch_p = _as_path(patch_coeffs_path)
    doys = _normalize_doys(doys_to_patch)

    if base_p.is_dir() or patch_p.is_dir():
        base_bucket_ids = set(discover_bucket_ids(base_p)) if base_p.is_dir() else set()
        patch_bucket_ids = set(discover_bucket_ids(patch_p)) if patch_p.is_dir() else set()
        bucket_ids = sorted(base_bucket_ids | patch_bucket_ids)
        summaries: List[PatchSummary] = []
        for bucket_id in bucket_ids:
            base_file = bucket_dir(base_p, bucket_id) / "coeffs.parquet"
            patch_file = bucket_dir(patch_p, bucket_id) / "coeffs.parquet"
            if not patch_file.exists():
                continue

            if base_file.exists():
                base_df = pd.read_parquet(base_file)
            else:
                patch_probe = pd.read_parquet(patch_file)
                base_df = pd.DataFrame(columns=patch_probe.columns)
            patch_df = pd.read_parquet(patch_file)

            (
                combined,
                base_rows_before,
                base_rows_removed,
                patch_rows_loaded,
                patch_rows_used,
                output_rows,
            ) = _patch_frames(
                base_df=base_df,
                patch_df=patch_df,
                doys=doys,
                id_col=id_col,
                doy_col=doy_col,
                keep="last",
            )

            base_file.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(base_file, index=False)
            summaries.append(
                PatchSummary(
                    base_coeffs_path=base_file,
                    patch_coeffs_path=patch_file,
                    output_path=base_file,
                    id_col=id_col,
                    doy_col=doy_col,
                    doys_to_patch=doys,
                    base_rows_before=base_rows_before,
                    base_rows_removed=base_rows_removed,
                    patch_rows_loaded=patch_rows_loaded,
                    patch_rows_used=patch_rows_used,
                    output_rows=output_rows,
                )
            )

        return DatasetPatchSummary(
            base_coeffs_path=base_p,
            patch_coeffs_path=patch_p,
            output_path=base_p,
            doys_to_patch=doys,
            bucket_summaries=summaries,
        )

    read_kwargs = {}
    write_kwargs = {}
    if parquet_engine:
        read_kwargs["engine"] = parquet_engine
        write_kwargs["engine"] = parquet_engine

    base_df = pd.read_parquet(base_p, **read_kwargs)
    patch_df = pd.read_parquet(patch_p, **read_kwargs)

    (
        combined,
        base_rows_before,
        base_rows_removed,
        patch_rows_loaded,
        patch_rows_used,
        output_rows,
    ) = _patch_frames(
        base_df=base_df,
        patch_df=patch_df,
        doys=doys,
        id_col=id_col,
        doy_col=doy_col,
        keep="last",
    )

    base_p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(base_p, index=False, **write_kwargs)

    return PatchSummary(
        base_coeffs_path=base_p,
        patch_coeffs_path=patch_p,
        output_path=base_p,
        id_col=id_col,
        doy_col=doy_col,
        doys_to_patch=doys,
        base_rows_before=base_rows_before,
        base_rows_removed=base_rows_removed,
        patch_rows_loaded=patch_rows_loaded,
        patch_rows_used=patch_rows_used,
        output_rows=output_rows,
    )
