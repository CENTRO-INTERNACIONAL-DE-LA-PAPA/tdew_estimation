"""P4/P5 integration: a manifest drives a single zoned training pass to tidy coeffs.

Builds a synthetic bucketed dataset, writes a hand recipe (zone-wide ``const,TMIN_anom``),
trains the whole grid in one pass, and checks the tidy output only carries the retained
features and that the coefficients equal a direct 2-feature GPU solve of the same systems.
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

import _synth  # noqa: E402
import gpu_train  # noqa: E402
from HPC_code_tunning.assemble_generic import assemble_grams_noyear, convolve_doy  # noqa: E402
from HPC_code_tunning.feature_spec import (  # noqa: E402
    AnomalyTrainingConfig,
    FeatureRegistry,
    TuningConfig,
    build_feature_frame,
)
from HPC_code_tunning.manifest import ZoneManifest  # noqa: E402
from HPC_code_tunning.selection import ALL_DOYS  # noqa: E402
from HPC_code_tunning.train_zoned import TIDY_COLUMNS, run_zoned_training  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402
from tdew_estimation.parquet_io import read_parquet_any  # noqa: E402

H = 11
POOL = ("const", "TMIN_anom", "TD_anom_lag1", "TD_anom_lag2", "TMIN_anom_lag1")
SELECTED = ("const", "TMIN_anom")


@pytest.fixture(scope="module")
def synth_results(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    results = tmp_path_factory.mktemp("results")
    _synth.build_synthetic_results(
        base, results, n_ids=20, year_range=(2010, 2014), num_buckets=4, seed=5
    )
    return results


def _tuning(results) -> TuningConfig:
    base = AnomalyTrainingConfig(
        base_path=results, td_var="td", tmin_var="tmin_v1",
        train_year_range=(2010, 2014), kernel="Tricube", min_samples=15, h=H,
    )
    return TuningConfig(base=base, candidate_pool=POOL, h_grid=(H,))


def test_zoned_training_tidy_output_and_values(synth_results, tmp_path):
    results = synth_results
    prepared_root = results / "bucketed_training_data"
    clim_root = results / "climatology_by_bucket"
    coeffs_root = tmp_path / "zoned_coeffs"

    # Every ID -> zone 0; a single zone-wide recipe keeping const + TMIN_anom.
    ids = []
    for bid in discover_bucket_ids(prepared_root):
        t = read_parquet_any(bucket_dir(prepared_root, bid))
        ids.extend(pd.unique(t["ID"]).tolist())
    ids = sorted(int(i) for i in ids)
    zone_table = pd.DataFrame({"ID": ids, "zone_id": 0, "zone_label": "Z0"})
    manifest = ZoneManifest(pd.DataFrame([{
        "zone_id": 0, "zone_label": "Z0", "doy": ALL_DOYS, "h": H,
        "feature_list": ",".join(SELECTED), "n_features": len(SELECTED), "skill": 0.9,
    }]))

    tuning = _tuning(results)
    summary = run_zoned_training(
        prepared_training_root=prepared_root,
        bucketed_climatology_root=clim_root,
        coeffs_output_root=coeffs_root,
        zone_table=zone_table, manifest=manifest, tuning=tuning, overwrite=True,
    )
    assert int(summary["rows"].sum()) > 0

    tidy = pd.concat(
        [pd.read_parquet(bucket_dir(coeffs_root, b) / "coeffs.parquet")
         for b in discover_bucket_ids(coeffs_root)],
        ignore_index=True,
    )
    assert list(tidy.columns) == TIDY_COLUMNS
    # Only the retained features appear; every (ID,doy) got exactly the 2 retained features.
    assert set(tidy["feature_name"].unique()) == set(SELECTED)
    per_cell = tidy.groupby(["ID", "doy"]).size()
    assert (per_cell == len(SELECTED)).all()
    assert (tidy["h"] == H).all()

    # Value check on one bucket. train_zoned assembles over the *full pool* (assemble-once)
    # and index-selects the retained sub-block, so the oracle must do the same: pool dropna,
    # then solve the [const, TMIN_anom] = columns [0, 1] sub-block.
    bid = discover_bucket_ids(prepared_root)[0]
    train = read_parquet_any(bucket_dir(prepared_root, bid))
    clim = pd.read_parquet(bucket_dir(clim_root, bid) / "climatology.parquet")
    reg_pool = FeatureRegistry(POOL)
    frame_pool = build_feature_frame(train, clim, reg_pool)
    Sxx, Sxy, Syy, cnt, id_values = assemble_grams_noyear(frame_pool, reg_pool)
    A, b, syyw, nbr = convolve_doy(Sxx, Sxy, Syy, cnt, h=H, kernel="Tricube", axis=1)
    n_id = int(id_values.shape[0])
    sub = cp.asarray([0, 1])
    A = A[:, :, sub][:, :, :, sub]
    b = b[:, :, sub]
    valid = (nbr >= 15).reshape(n_id * 366)
    idx = cp.where(valid)[0]
    beta, _ = gpu_train.solve_bucket_reference(
        A.reshape(n_id * 366, 2, 2)[idx], b.reshape(n_id * 366, 2)[idx],
        syyw.reshape(n_id * 366)[idx],
    )
    idx_h = cp.asnumpy(idx)
    direct = pd.DataFrame({
        "ID": id_values[(idx_h // 366).astype(int)].astype(int),
        "doy": (idx_h % 366).astype(int) + 1,
        "TMIN_anom": cp.asnumpy(beta)[:, 1],
    })
    got = tidy[(tidy["feature_name"] == "TMIN_anom") & (tidy["ID"].isin(direct["ID"]))]
    merged = got.merge(direct, on=["ID", "doy"])
    assert len(merged) > 0
    assert np.max(np.abs(merged["coeff"].to_numpy() - merged["TMIN_anom"].to_numpy())) < 1e-6
