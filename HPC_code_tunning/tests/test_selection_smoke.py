"""P3: backward-stepwise keeps a planted informative feature and drops a noise decoy.

Raw TD is generated as ``TD_anom ≈ 2 * TMIN_anom + small noise`` (strong contemporaneous
coupling), while daily noise is i.i.d. so lagged anomalies carry no predictive power. With
the candidate pool ``[const, TMIN_anom, TD_anom_lag30]`` selection should retain
``TMIN_anom`` almost everywhere and drop the ``TD_anom_lag30`` decoy in most doys.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_HPC = _ROOT / "HPC_code"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HPC))

cp = pytest.importorskip("cupy")


def _has_gpu() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="no CUDA device")

from HPC_code_tunning.feature_spec import AnomalyTrainingConfig, TuningConfig  # noqa: E402
from HPC_code_tunning.selection import run_selection  # noqa: E402
from tdew_estimation.bucketed_data import (  # noqa: E402
    build_bucketed_training_dataset,
    shard_climatology_by_bucket,
)
from tdew_estimation.bucket_layout import discover_bucket_ids  # noqa: E402
from tdew_estimation.climatology import calculate_and_save_climatology_chunked  # noqa: E402


def _write_var(var_dir: Path, var: str, year: int, month: int, ids, values_by_day) -> None:
    var_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.DataFrame({"ID": ids, "FECHA": d, "Value": v}) for d, v in values_by_day.items()]
    df = pd.concat(frames, ignore_index=True)
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df.to_parquet(var_dir / f"{var}_daily_{year:04d}_{month:02d}.parquet", index=False)


def _generate_coupled_raw(base: Path, ids, years, *, td_var="td", tmin_var="tmin_v1", seed=11):
    """TD strongly driven by contemporaneous TMIN; daily noise i.i.d. (no autocorrelation)."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    id_off = rng.normal(0, 3.0, size=n)
    for year in range(years[0], years[1] + 1):
        for month in range(1, 13):
            start = pd.Timestamp(year=year, month=month, day=1)
            days = pd.date_range(start, start + pd.offsets.MonthEnd(1), freq="D")
            td_by_day, tmin_by_day = {}, {}
            for d in days:
                doy = int(d.dayofyear)
                seas_t = 6.0 * np.sin(2 * np.pi * doy / 365.0)
                seas_d = 8.0 * np.sin(2 * np.pi * doy / 365.0)
                shared = rng.normal(0, 1.0, size=n)          # the informative signal
                tmin = 8.0 + seas_t + id_off + shared
                td = 15.0 + seas_d + id_off + 2.0 * shared + rng.normal(0, 0.25, size=n)
                tmin_by_day[d] = tmin
                td_by_day[d] = td
            _write_var(base / td_var / "Outputs", td_var, year, month, ids, td_by_day)
            _write_var(base / tmin_var / "Outputs", tmin_var, year, month, ids, tmin_by_day)


@pytest.fixture(scope="module")
def coupled_results(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    results = tmp_path_factory.mktemp("results")
    ids = np.arange(1, 41, dtype=int)
    years = (2010, 2016)
    _generate_coupled_raw(base, ids, years)
    clim_path = results / "daily_climatology.parquet"
    calculate_and_save_climatology_chunked(years, base, clim_path, td_var="td", tmin_var="tmin_v1")
    build_bucketed_training_dataset(
        year_range=years, base_path=base, output_dir=results / "bucketed_training_data",
        td_var="td", tmin_var="tmin_v1", num_buckets=4, overwrite=True,
    )
    shard_climatology_by_bucket(
        climatology_path=clim_path, output_dir=results / "climatology_by_bucket",
        num_buckets=4, overwrite=True,
    )
    return {"results": results, "ids": ids, "years": years}


def test_selection_keeps_signal_drops_decoy(coupled_results):
    results = coupled_results["results"]
    ids = coupled_results["ids"]
    prepared_root = results / "bucketed_training_data"
    clim_root = results / "climatology_by_bucket"

    assert discover_bucket_ids(prepared_root), "no training buckets were built"
    zone_table = pd.DataFrame({"ID": ids, "zone_id": 0, "zone_label": "Z0"})
    sample = {0: np.array(ids, dtype=int)}

    base_cfg = AnomalyTrainingConfig(
        base_path=results, td_var="td", tmin_var="tmin_v1",
        train_year_range=coupled_results["years"], kernel="Tricube", min_samples=15,
    )
    tuning = TuningConfig(
        base=base_cfg,
        candidate_pool=("const", "TMIN_anom", "TD_anom_lag30"),
        h_grid=(11,),
        tol=0.0,               # drop a feature whenever removing it does not hurt skill
        granularity="doy",
        per_zone_n=40,
        id_chunk=40,
        seed=0,
    )

    manifest = run_selection(zone_table, sample, prepared_root, clim_root, tuning)
    assert not manifest.empty, "selection produced no recipes"

    has_signal = manifest["feature_list"].str.contains("TMIN_anom")
    has_decoy = manifest["feature_list"].str.contains("TD_anom_lag30")

    # The planted signal is retained essentially everywhere.
    assert has_signal.mean() > 0.9, f"TMIN_anom retained in only {has_signal.mean():.0%} of doys"
    # The decoy is dropped in a clear majority of doys.
    assert has_decoy.mean() < 0.5, f"decoy retained in {has_decoy.mean():.0%} of doys"
    # Signal-bearing recipes score well (strong coupling -> high cosine skill).
    assert float(manifest.loc[has_signal, "skill"].median()) > 0.5
