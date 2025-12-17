"""
tdew_estimation.incomplete_models

Utilities to verify completeness of per-DOY (day-of-year) model parquet files and
produce a list of missing/partial DOYs that should be retrained.

This module is extracted from the original Colab-derived script
`tdew_estimation_arimax.py` ("Checks" section) and made path-agnostic.

Assumptions
-----------
- You have per-DOY parquet files named like:
    gpu_models_doy_{doy}.parquet
  where doy is 1..366.
- Each file contains at least an "ID" column with one row per modeled location.

Notes
-----
- "Completeness" is defined as having exactly `expected_count` rows in the file.
  In the original notebook this value was hard-coded (302449). Here it's a required
  parameter so you can supply the correct count for your grid/version.
- We read only the "ID" column for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]


@dataclass(frozen=True)
class IncompleteModelReport:
    """Summary of verification results."""

    models_dir: Path
    pattern: str
    expected_count: int
    doy_range: Tuple[int, int]
    incomplete_days: List[int]
    missing_files: List[int]
    unreadable_files: List[int]
    wrong_count_files: List[Tuple[int, int]]  # (doy, actual_count)

    @property
    def ok(self) -> bool:
        return len(self.incomplete_days) == 0


def _iter_doys(doy_range: Tuple[int, int]) -> Iterable[int]:
    start, end = doy_range
    if start < 1 or end > 366 or start > end:
        raise ValueError(f"Invalid doy_range={doy_range}. Expected within 1..366.")
    return range(start, end + 1)


def detect_incomplete_model_files(
    models_dir: PathLike,
    expected_count: int,
    *,
    pattern: str = "gpu_models_doy_{doy}.parquet",
    doy_range: Tuple[int, int] = (1, 366),
    id_column: str = "ID",
    verbose: bool = True,
) -> IncompleteModelReport:
    """
    Check per-DOY model parquet files and return which DOYs are incomplete.

    A DOY is considered incomplete if:
    - the parquet file does not exist
    - the parquet file cannot be read
    - the file exists but row count != expected_count

    Parameters
    ----------
    models_dir:
        Directory containing per-DOY parquet files.
    expected_count:
        Expected number of modeled IDs (rows) per file.
    pattern:
        Filename pattern. Must include "{doy}" placeholder.
    doy_range:
        Inclusive range of DOYs to check (default 1..366).
    id_column:
        Column to read (only) from each parquet file.
    verbose:
        If True, prints a compact human-readable report.

    Returns
    -------
    IncompleteModelReport
    """
    models_dir_p = Path(models_dir).expanduser().resolve()
    if "{doy}" not in pattern:
        raise ValueError("pattern must include '{doy}' placeholder.")
    if expected_count <= 0:
        raise ValueError("expected_count must be a positive integer.")

    missing: List[int] = []
    unreadable: List[int] = []
    wrong_count: List[Tuple[int, int]] = []
    incomplete: List[int] = []

    for doy in _iter_doys(doy_range):
        file_path = models_dir_p / pattern.format(doy=doy)
        if not file_path.exists():
            missing.append(doy)
            incomplete.append(doy)
            continue

        try:
            df = pd.read_parquet(file_path, columns=[id_column])
            actual = int(len(df))
        except Exception:
            unreadable.append(doy)
            incomplete.append(doy)
            continue

        if actual != expected_count:
            wrong_count.append((doy, actual))
            incomplete.append(doy)

    report = IncompleteModelReport(
        models_dir=models_dir_p,
        pattern=pattern,
        expected_count=expected_count,
        doy_range=doy_range,
        incomplete_days=sorted(incomplete),
        missing_files=sorted(missing),
        unreadable_files=sorted(unreadable),
        wrong_count_files=sorted(wrong_count, key=lambda x: x[0]),
    )

    if verbose:
        _print_report(report)

    return report


def _print_report(report: IncompleteModelReport) -> None:
    print("\n--- Incomplete Model File Check ---")
    print(f"Models dir: {report.models_dir}")
    print(f"Pattern:    {report.pattern}")
    print(f"DOY range:  {report.doy_range[0]}..{report.doy_range[1]}")
    print(f"Expected rows per file: {report.expected_count}")

    if report.ok:
        print("✅ All checked DOY model files are complete.")
        return

    print(f"❌ Found {len(report.incomplete_days)} incomplete DOY(s).")

    if report.missing_files:
        print(f"- Missing files: {len(report.missing_files)}")
        print(f"  DOYs: {report.missing_files}")

    if report.unreadable_files:
        print(f"- Unreadable files: {len(report.unreadable_files)}")
        print(f"  DOYs: {report.unreadable_files}")

    if report.wrong_count_files:
        print(f"- Wrong row count: {len(report.wrong_count_files)}")
        # Print at most first 25 to keep output readable
        preview = report.wrong_count_files[:25]
        preview_str = ", ".join([f"{doy}:{cnt}" for doy, cnt in preview])
        more = (
            ""
            if len(report.wrong_count_files) <= 25
            else f" ... (+{len(report.wrong_count_files) - 25} more)"
        )
        print(f"  (doy:actual) {preview_str}{more}")

    print("\n# Copy-paste friendly list for retraining")
    print(f"DOYS_TO_FIX = {report.incomplete_days}")


def parse_expected_count_from_grid(
    grid_ids: Sequence[int] | Sequence[str],
) -> int:
    """
    Convenience helper: if you already have the list of unique IDs you expect models for,
    this returns the expected_count to pass into `detect_incomplete_model_files`.

    Parameters
    ----------
    grid_ids:
        Sequence of IDs.

    Returns
    -------
    int
    """
    return int(len(set(grid_ids)))


def select_doys_to_fix(report: IncompleteModelReport) -> List[int]:
    """Return the DOYs to retrain (alias for report.incomplete_days)."""
    return list(report.incomplete_days)
