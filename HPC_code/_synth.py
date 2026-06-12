#!/usr/bin/env python3
"""
HPC_code._synth

Tiny synthetic raw dataset generator for the D4 benchmark/scaling harness.

The point is to exercise the *real* preparation pipeline rather than hand-write
bucketed shards: we emit raw monthly TD/TMIN parquet files in the exact layout the
production readers expect
(``{base}/{var}/Outputs/{var}_daily_YYYY_MM.parquet`` with columns ``ID, FECHA,
Value``), then call the real ``tdew_estimation`` prep functions
(``calculate_and_save_climatology_chunked`` -> ``build_bucketed_training_dataset`` +
``shard_climatology_by_bucket`` [+ ``shard_future_tmin_by_bucket``]) so the bucketed
inputs are guaranteed schema-correct. This both produces benchmark inputs and smoke
-tests the prep path.

This module imports nothing GPU/RAPIDS and runs anywhere the base venv runs.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Allow `import tdew_estimation` regardless of cwd / install state (mirror entrypoint).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from tdew_estimation.bucketed_data import (  # noqa: E402
    build_bucketed_training_dataset,
    shard_climatology_by_bucket,
    shard_future_tmin_by_bucket,
)
from tdew_estimation.climatology import (  # noqa: E402
    calculate_and_save_climatology_chunked,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticResult:
    base: Path
    results: Path
    year_range: Tuple[int, int]
    num_buckets: int
    n_ids: int
    with_forecast: bool
    pred_years: Tuple[int, int] | None


def _write_monthly_var(
    *,
    var_dir: Path,
    variable: str,
    year: int,
    month: int,
    ids: np.ndarray,
    values_by_day: dict[pd.Timestamp, np.ndarray],
) -> None:
    """Write one ``{variable}_daily_YYYY_MM.parquet`` file (columns ID, FECHA, Value)."""
    var_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for day, vals in values_by_day.items():
        frames.append(pd.DataFrame({"ID": ids, "FECHA": day, "Value": vals}))
    df = pd.concat(frames, ignore_index=True)
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    out = var_dir / f"{variable}_daily_{year:04d}_{month:02d}.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)


def generate_synthetic_raw(
    base: Path,
    *,
    n_ids: int,
    years: Tuple[int, int],
    seed: int = 1234,
    td_var: str = "td",
    tmin_var: str = "tmin_v1",
    future_tmin_var: str | None = None,
    future_years: Tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Write synthetic raw monthly TD/TMIN parquet under ``base``.

    A deterministic seasonal signal (sinusoid in day-of-year) plus a per-ID offset
    plus small Gaussian noise. TD and TMIN share the exact ``(ID, FECHA)`` grid so the
    inner merge in :func:`build_bucketed_training_dataset` keeps every row. TMIN is
    generated a few degrees below TD.

    If ``future_tmin_var`` and ``future_years`` are given, additionally emits a future
    TMIN variable folder for the forecast phase.

    Returns the array of generated IDs.
    """
    base = Path(base)
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n_ids + 1, dtype=int)
    id_offset = rng.normal(0.0, 3.0, size=n_ids)  # per-ID climatological offset

    def daily_values(day: pd.Timestamp, *, base_level: float, amp: float) -> np.ndarray:
        doy = int(day.dayofyear)
        seasonal = amp * np.sin(2.0 * np.pi * doy / 365.0)
        noise = rng.normal(0.0, 1.0, size=n_ids)
        return base_level + seasonal + id_offset + noise

    start_year, end_year = years
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_start = pd.Timestamp(year=year, month=month, day=1)
            month_end = month_start + pd.offsets.MonthEnd(1)
            days = pd.date_range(month_start, month_end, freq="D")
            td_by_day = {d: daily_values(d, base_level=15.0, amp=8.0) for d in days}
            # TMIN tied to TD's shape but a few degrees lower (independent noise is fine).
            tmin_by_day = {d: daily_values(d, base_level=8.0, amp=6.0) for d in days}
            _write_monthly_var(
                var_dir=base / td_var / "Outputs",
                variable=td_var,
                year=year,
                month=month,
                ids=ids,
                values_by_day=td_by_day,
            )
            _write_monthly_var(
                var_dir=base / tmin_var / "Outputs",
                variable=tmin_var,
                year=year,
                month=month,
                ids=ids,
                values_by_day=tmin_by_day,
            )

    if future_tmin_var and future_years:
        fstart, fend = future_years
        for year in range(fstart, fend + 1):
            for month in range(1, 13):
                month_start = pd.Timestamp(year=year, month=month, day=1)
                month_end = month_start + pd.offsets.MonthEnd(1)
                days = pd.date_range(month_start, month_end, freq="D")
                tmin_by_day = {
                    d: daily_values(d, base_level=8.0, amp=6.0) for d in days
                }
                _write_monthly_var(
                    var_dir=base / future_tmin_var / "Outputs",
                    variable=future_tmin_var,
                    year=year,
                    month=month,
                    ids=ids,
                    values_by_day=tmin_by_day,
                )

    return ids


def build_synthetic_results(
    base: Path,
    results: Path,
    *,
    n_ids: int = 40,
    year_range: Tuple[int, int] = (2010, 2014),
    num_buckets: int = 8,
    seed: int = 1234,
    td_var: str = "td",
    tmin_var: str = "tmin_v1",
    with_forecast: bool = False,
    pred_years: Tuple[int, int] = (2015, 2015),
    future_tmin_var: str = "tmin",
    overwrite: bool = True,
) -> SyntheticResult:
    """
    Generate raw synthetic data under ``base`` and run the real prep pipeline to
    populate ``results`` with the same layout ``Local/run_pipeline.sh`` produces
    (``daily_climatology.parquet``, ``bucketed_training_data/``,
    ``climatology_by_bucket/`` [+ ``future_tmin_by_bucket/``]).
    """
    base = Path(base)
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)

    log.info("Generating synthetic raw data under %s", base)
    generate_synthetic_raw(
        base,
        n_ids=n_ids,
        years=year_range,
        seed=seed,
        td_var=td_var,
        tmin_var=tmin_var,
        future_tmin_var=future_tmin_var if with_forecast else None,
        future_years=pred_years if with_forecast else None,
    )

    clim_path = results / "daily_climatology.parquet"
    log.info("Computing climatology -> %s", clim_path)
    calculate_and_save_climatology_chunked(
        year_range,
        base,
        clim_path,
        td_var=td_var,
        tmin_var=tmin_var,
    )

    log.info("Building bucketed training dataset (num_buckets=%s)", num_buckets)
    build_bucketed_training_dataset(
        year_range=year_range,
        base_path=base,
        output_dir=results / "bucketed_training_data",
        td_var=td_var,
        tmin_var=tmin_var,
        num_buckets=num_buckets,
        overwrite=overwrite,
    )

    log.info("Sharding climatology by bucket")
    shard_climatology_by_bucket(
        climatology_path=clim_path,
        output_dir=results / "climatology_by_bucket",
        num_buckets=num_buckets,
        overwrite=overwrite,
    )

    if with_forecast:
        log.info("Sharding future TMIN by bucket for pred_years=%s", pred_years)
        shard_future_tmin_by_bucket(
            prediction_years=pred_years,
            base_path=base,
            output_dir=results / "future_tmin_by_bucket",
            future_tmin_var=future_tmin_var,
            num_buckets=num_buckets,
            overwrite=overwrite,
        )

    return SyntheticResult(
        base=base,
        results=results,
        year_range=year_range,
        num_buckets=num_buckets,
        n_ids=n_ids,
        with_forecast=with_forecast,
        pred_years=pred_years if with_forecast else None,
    )


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Generate a tiny synthetic TDEW dataset.")
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--n-ids", type=int, default=40)
    p.add_argument("--train-start", type=int, default=2010)
    p.add_argument("--train-end", type=int, default=2014)
    p.add_argument("--num-buckets", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--with-forecast", action="store_true")
    p.add_argument("--pred-start", type=int, default=2015)
    p.add_argument("--pred-end", type=int, default=2015)
    a = p.parse_args()
    res = build_synthetic_results(
        a.base,
        a.results,
        n_ids=a.n_ids,
        year_range=(a.train_start, a.train_end),
        num_buckets=a.num_buckets,
        seed=a.seed,
        with_forecast=a.with_forecast,
        pred_years=(a.pred_start, a.pred_end),
    )
    print(f"[synth] built {res.n_ids} ids x {res.year_range} -> {res.results}")
