"""
tdew_estimation.checks

Checks for the anomaly-model pipeline, focused on detecting *incomplete DOYs*
in anomaly coefficient outputs and deriving a list of DOYs to retrain.

Context
-------
In the original Colab-derived workflow, anomaly model coefficients are trained for
many spatial IDs and for each day-of-year (DOY). A run can partially fail for some
DOYs (e.g., due to transient worker errors, OOM, etc.). The practical check is:

Option A (as requested):
- For each DOY, count distinct IDs with coefficients
- If actual_count != expected_count -> DOY is incomplete and should be retrained

This module intentionally:
- is path-agnostic (no hard-coded paths)
- does not assume "one parquet file per DOY" (GPU model style). Instead, it assumes
  a *combined anomaly coefficients parquet* (or dataset) that contains columns:
    - ID
    - doy
  plus coefficient columns.

Typical usage
-------------
from pathlib import Path
from tdew_estimation.grid import expected_count_from_grid
from tdew_estimation.checks import detect_incomplete_anomaly_doys

coeffs_path = Path("/path/to/results/llr_coeffs_anomaly_final_direct.parquet")
grid_path = Path("/path/to/PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf")

expected = expected_count_from_grid(grid_path, id_column="ID")
report = detect_incomplete_anomaly_doys(coeffs_path, expected_count=expected)

print(report.incomplete_doys)
# -> [13, 54, 65, ...]

Then retrain ONLY those DOYs and patch results back into the coefficient dataset.

Design notes
------------
- This module reads only required columns by default to stay fast.
- Counting uses nunique(ID) per doy. If your coefficients can contain multiple
  rows per (ID, doy) (unlikely), the nunique still behaves correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import pandas as pd
from pandas import DataFrame, Series

PathLike = Union[str, Path]


@dataclass(frozen=True)
class IncompleteAnomalyDoyReport:
    """Summary report for DOY completeness in anomaly coefficients."""

    coeffs_path: Path
    expected_count: int
    doy_range: Tuple[int, int]
    id_col: str
    doy_col: str

    # DOYs where actual_count != expected_count
    incomplete_doys: List[int]

    # Per-DOY counts (only for DOYs present in the file unless include_missing=True)
    counts_by_doy: Dict[int, int]

    # DOYs completely missing from the coeffs dataset (only set when include_missing=True)
    missing_doys: List[int]

    @property
    def ok(self) -> bool:
        return len(self.incomplete_doys) == 0

    def to_fix_list(self) -> List[int]:
        """Copy-paste-friendly list of DOYs to retrain."""
        return list(self.incomplete_doys)


def _validate_doy_range(doy_range: Tuple[int, int]) -> None:
    start, end = doy_range
    if start < 1 or end > 366 or start > end:
        raise ValueError(f"Invalid doy_range={doy_range}. Expected within 1..366.")


def _iter_doys(doy_range: Tuple[int, int]) -> Iterable[int]:
    _validate_doy_range(doy_range)
    start, end = doy_range
    return range(start, end + 1)


def detect_incomplete_anomaly_doys(
    coeffs_path: PathLike,
    expected_count: int,
    *,
    id_col: str = "ID",
    doy_col: str = "doy",
    doy_range: Tuple[int, int] = (1, 366),
    include_missing: bool = True,
    strict_inequality: bool = True,
    verbose: bool = True,
    parquet_engine: Optional[str] = None,
) -> IncompleteAnomalyDoyReport:
    """
    Detect incomplete DOYs in an anomaly coefficient dataset.

    Parameters
    ----------
    coeffs_path:
        Path to the anomaly coefficient parquet (e.g., llr_coeffs_anomaly_final_direct.parquet).
    expected_count:
        Expected number of unique IDs per DOY (e.g., derived from the grid file).
    id_col:
        Column name for spatial/location ID. Default "ID".
    doy_col:
        Column name for day-of-year. Default "doy".
    doy_range:
        Inclusive range of DOYs to check. Default (1, 366).
    include_missing:
        If True, DOYs that do not appear at all in the coeffs file are treated as incomplete.
    strict_inequality:
        If True, mark DOY incomplete when actual_count != expected_count.
        If False, mark incomplete when actual_count < expected_count (less strict).
    verbose:
        Print a compact human-readable report.
    parquet_engine:
        Optional engine override for pandas.read_parquet.

    Returns
    -------
    IncompleteAnomalyDoyReport
    """
    _validate_doy_range(doy_range)
    if expected_count <= 0:
        raise ValueError("expected_count must be a positive integer.")

    p = Path(coeffs_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Coefficient file not found: {p}")

    # Read only needed columns for speed
    read_kwargs = {}
    if parquet_engine:
        read_kwargs["engine"] = parquet_engine

    df = pd.read_parquet(p, columns=[id_col, doy_col], **read_kwargs)

    if id_col not in df.columns or doy_col not in df.columns:
        raise KeyError(
            f"Expected columns {id_col!r} and {doy_col!r} in {p}, "
            f"found: {list(df.columns)}"
        )

    # Normalize doy values to ints where possible, drop missing doys
    df = df.dropna(subset=[doy_col, id_col]).copy()

    # Normalize DOY values robustly:
    # - coerce to numeric (invalid -> NaN)
    # - drop missing
    # - cast to Python int
    #
    # Avoid pandas-stubs issues by keeping the transformation explicitly as a Series:
    df = cast(DataFrame, df)
    doy_series = cast(Series, pd.to_numeric(cast(Series, df[doy_col]), errors="coerce"))
    doy_series = cast(Series, doy_series.dropna())
    # Some type checkers incorrectly infer scalars/NA unions here; runtime behavior is correct.
    df[doy_col] = doy_series.astype("int64")  # type: ignore[attr-defined]

    # Restrict to doy_range
    start, end = doy_range
    df = df[(df[doy_col] >= start) & (df[doy_col] <= end)]

    # Count unique IDs per DOY
    counts = (
        df.groupby(doy_col, dropna=False)[id_col]
        .nunique(dropna=True)
        .astype(int)  # type: ignore[attr-defined]
        .to_dict()
    )

    missing_doys: List[int] = []
    if include_missing:
        present = set(counts.keys())
        missing_doys = [d for d in _iter_doys(doy_range) if d not in present]

    incomplete: List[int] = []
    for doy in _iter_doys(doy_range):
        if doy in missing_doys:
            incomplete.append(doy)
            continue
        actual = counts.get(doy)
        if actual is None:
            # Should only happen if include_missing=False and doy absent
            continue
        if strict_inequality:
            if actual != expected_count:
                incomplete.append(doy)
        else:
            if actual < expected_count:
                incomplete.append(doy)

    report = IncompleteAnomalyDoyReport(
        coeffs_path=p,
        expected_count=expected_count,
        doy_range=doy_range,
        id_col=id_col,
        doy_col=doy_col,
        incomplete_doys=sorted(incomplete),
        counts_by_doy={int(k): int(v) for k, v in counts.items()},
        missing_doys=sorted(missing_doys),
    )

    if verbose:
        _print_incomplete_report(report, strict_inequality=strict_inequality)

    return report


def _print_incomplete_report(
    report: IncompleteAnomalyDoyReport, *, strict_inequality: bool
) -> None:
    print("\n--- Incomplete Anomaly DOY Check ---")
    print(f"Coefficients file: {report.coeffs_path}")
    print(f"DOY range:         {report.doy_range[0]}..{report.doy_range[1]}")
    print(f"Expected IDs/DOY:  {report.expected_count}")
    print(
        f"Check:             actual_count {'!=' if strict_inequality else '<'} expected_count"
    )

    if report.ok:
        print("✅ All checked DOYs appear complete.")
        return

    print(f"❌ Found {len(report.incomplete_doys)} incomplete DOY(s).")

    if report.missing_doys:
        print(f"- Missing DOYs (no rows at all): {len(report.missing_doys)}")
        print(f"  DOYs: {report.missing_doys}")

    # Show a small preview of wrong counts
    wrong = []
    for doy in report.incomplete_doys:
        if doy in report.counts_by_doy:
            wrong.append((doy, report.counts_by_doy[doy]))
    if wrong:
        preview = wrong[:25]
        preview_str = ", ".join([f"{d}:{c}" for d, c in preview])
        more = "" if len(wrong) <= 25 else f" ... (+{len(wrong) - 25} more)"
        print(f"- Incomplete counts (doy:actual): {preview_str}{more}")

    print("\n# Copy-paste friendly list for retraining")
    print(f"DOYS_TO_FIX = {report.incomplete_doys}")


def derive_doys_to_retrain(
    coeffs_path: PathLike,
    expected_count: int,
    *,
    id_col: str = "ID",
    doy_col: str = "doy",
    doy_range: Tuple[int, int] = (1, 366),
    include_missing: bool = True,
    strict_inequality: bool = True,
) -> List[int]:
    """
    Convenience wrapper returning only the list of DOYs to retrain.

    This is useful when you already log/print elsewhere and only need the list.
    """
    report = detect_incomplete_anomaly_doys(
        coeffs_path,
        expected_count,
        id_col=id_col,
        doy_col=doy_col,
        doy_range=doy_range,
        include_missing=include_missing,
        strict_inequality=strict_inequality,
        verbose=False,
    )
    return report.to_fix_list()
