"""
tdew_estimation.bucket_layout

Helpers for consistent bucketing of IDs into on-disk partitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]

BUCKET_DIR_PREFIX = "id_bucket="


def bucket_for_id(location_id: int, *, num_buckets: int) -> int:
    """
    Map an ID deterministically to a bucket in ``0..num_buckets-1``.
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")
    return int(location_id) % int(num_buckets)


def bucket_dir_name(bucket_id: int, *, width: int = 4) -> str:
    """
    Return a hive-style directory name for a bucket.
    """
    return f"{BUCKET_DIR_PREFIX}{int(bucket_id):0{width}d}"


def bucket_dir(root: PathLike, bucket_id: int, *, width: int = 4) -> Path:
    """
    Return the directory path for a bucket under a dataset root.
    """
    return Path(root).expanduser().resolve() / bucket_dir_name(bucket_id, width=width)


def discover_bucket_ids(root: PathLike) -> List[int]:
    """
    Discover bucket ids from hive-style directories under ``root``.
    """
    root_p = Path(root).expanduser().resolve()
    if not root_p.exists():
        raise FileNotFoundError(f"Bucket root not found: {root_p}")

    bucket_ids: List[int] = []
    for child in sorted(root_p.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith(BUCKET_DIR_PREFIX):
            continue
        try:
            bucket_ids.append(int(child.name.split("=", 1)[1]))
        except Exception:
            continue
    return sorted(set(bucket_ids))
