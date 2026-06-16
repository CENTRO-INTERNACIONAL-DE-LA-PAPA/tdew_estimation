"""
Smoke + correctness test for the D5 dataset-comparison tool (compare_datasets.py).

Builds two small bucketed coefficient datasets (and two prediction datasets) with
*injected, known* differences, then checks that compare_datasets reports exactly those
differences: coverage counts, paired Δ statistics, and prediction RMSE/MAE/bias. Also
verifies the markdown report and PNG plots are written. Pure CPU/pandas — no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make HPC_code/ importable (its modules are scripts, not an installed package).
_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

import compare_datasets as cd  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir  # noqa: E402

COEFF_COLS = [
    "const_anom",
    "TMIN_anom_coeff",
    "TD_anom_lag1",
    "TD_anom_lag2",
    "TMIN_anom_lag1",
    "doy",
    "r_squared_anom",
    "ID",
]


def _write_bucketed_coeffs(root: Path, df: pd.DataFrame, *, num_buckets: int = 4) -> None:
    """Write a coeffs frame as a bucketed root: id_bucket=XXXX/coeffs.parquet."""
    for bid in range(num_buckets):
        part = df[df["ID"] % num_buckets == bid]
        if part.empty:
            continue
        d = bucket_dir(root, bid)
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / "coeffs.parquet", engine="pyarrow", index=False)


def _make_coeffs(ids, doys, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in ids:
        for d in doys:
            rows.append(
                {
                    "ID": int(i),
                    "doy": int(d),
                    "const_anom": rng.normal(),
                    "TMIN_anom_coeff": rng.normal(),
                    "TD_anom_lag1": rng.normal(),
                    "TD_anom_lag2": rng.normal(),
                    "TMIN_anom_lag1": rng.normal(),
                    "r_squared_anom": rng.uniform(0, 1),
                }
            )
    return pd.DataFrame(rows)[COEFF_COLS]


def test_coeff_comparison_reports_injected_difference(tmp_path):
    ids = list(range(12))
    doys = [1, 50, 100, 200, 300, 366]

    a = _make_coeffs(ids, doys, seed=1)
    # B = A shifted by a known constant on one coefficient, plus one extra-only key in each.
    b = a.copy()
    SHIFT = 0.25
    b["const_anom"] = b["const_anom"] + SHIFT

    # Drop one (ID,doy) from A and one *different* one from B -> a_only=1, b_only=1.
    a = a.drop(index=a.index[0]).reset_index(drop=True)  # removes (ID=0, doy=1) from A
    b = b.drop(index=b.index[-1]).reset_index(drop=True)  # removes (ID=11, doy=366) from B

    res = cd.compare_coeffs(a, b, label_a="v1", label_b="v2")

    cov = res["coverage"]
    assert cov["v1_total"] == len(a)
    assert cov["v2_total"] == len(b)
    assert cov["v1_only"] == 1  # (ID=11,doy=366): in A, dropped from B
    assert cov["v2_only"] == 1  # (ID=0,doy=1): in B, dropped from A
    assert cov["common"] == len(ids) * len(doys) - 2

    delta = res["delta"].set_index("coeff")
    # const_anom shifted by exactly SHIFT on every common fit.
    assert delta.loc["const_anom", "mean_delta"] == pytest.approx(SHIFT, abs=1e-9)
    assert delta.loc["const_anom", "std_delta"] == pytest.approx(0.0, abs=1e-9)
    assert delta.loc["const_anom", "max_abs_delta"] == pytest.approx(SHIFT, abs=1e-9)
    # untouched coefficients are identical between A and B.
    assert delta.loc["TMIN_anom_coeff", "max_abs_delta"] == pytest.approx(0.0, abs=1e-9)
    assert delta.loc["TD_anom_lag1", "pearson_r"] == pytest.approx(1.0, abs=1e-9)
    # identical column (untouched) -> cosine of a vector with itself is exactly 1.
    assert "cosine" in delta.columns
    assert delta.loc["TMIN_anom_coeff", "cosine"] == pytest.approx(1.0, abs=1e-9)


def test_prediction_comparison_metrics(tmp_path):
    ids = [1, 2, 3]
    dates = pd.date_range("2017-01-01", periods=120, freq="D")
    rng = np.random.default_rng(7)
    base = pd.DataFrame(
        {
            "ID": np.repeat(ids, len(dates)),
            "FECHA": np.tile(dates, len(ids)),
            "TD_predicted": rng.normal(10, 3, size=len(ids) * len(dates)),
        }
    )
    a = base.copy()
    b = base.copy()
    BIAS = 0.5
    b["TD_predicted"] = b["TD_predicted"] + BIAS  # constant bias -> RMSE=MAE=bias=BIAS

    res = cd.compare_predictions(a, b, label_a="v1", label_b="v2")
    ag = res["agreement"]
    assert ag["n"] == len(a)
    assert ag["bias"] == pytest.approx(BIAS, abs=1e-9)
    assert ag["mae"] == pytest.approx(BIAS, abs=1e-9)
    assert ag["rmse"] == pytest.approx(BIAS, abs=1e-9)
    assert ag["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    assert "cosine" in ag and ag["cosine"] == pytest.approx(1.0, abs=1e-3)
    # monthly table covers the months present (Jan–Apr for 120 days from Jan 1).
    assert set(res["monthly"]["month"]) <= set(range(1, 13))
    assert (res["monthly"]["mean_abs_delta"] - BIAS).abs().max() == pytest.approx(0.0, abs=1e-9)


def test_load_dataset_dir_and_file_and_outputs(tmp_path):
    """load_dataset handles a bucketed dir and a single file; full run writes artifacts."""
    ids = list(range(12))
    doys = [10, 20, 30]
    a = _make_coeffs(ids, doys, seed=3)
    b = _make_coeffs(ids, doys, seed=4)

    a_root = tmp_path / "coeffs_a"  # bucketed directory layout
    _write_bucketed_coeffs(a_root, a, num_buckets=4)
    b_file = tmp_path / "coeffs_b.parquet"  # single combined file
    b.to_parquet(b_file, engine="pyarrow", index=False)

    loaded_a = cd.load_dataset(a_root, label="v1")
    loaded_b = cd.load_dataset(b_file, label="v2")
    assert len(loaded_a) == len(a)
    assert len(loaded_b) == len(b)

    out_dir = tmp_path / "plots"
    md_out = tmp_path / "report.md"
    rc = cd.main(
        [
            "--coeffs-a",
            str(a_root),
            "--coeffs-b",
            str(b_file),
            "--label-a",
            "v1",
            "--label-b",
            "v2",
            "--out-dir",
            str(out_dir),
            "--md-out",
            str(md_out),
        ]
    )
    assert rc == 0
    assert md_out.exists() and md_out.stat().st_size > 0
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= len(cd.COEFF_METRICS)  # one Δ hist per coeff (+ r² scatter)
    assert (out_dir / "r2_scatter.png").exists()
