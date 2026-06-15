"""
Smoke + correctness test for the held-out accuracy evaluator (evaluate_accuracy.py).

Crafts observed TD and predictions with a *known* error, and checks the reported RMSE /
MAE / bias / Pearson r, the two-run "which is better" verdict, and that the markdown report
and PNG plots are written. Pure CPU/pandas — no GPU, no pipeline run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

import evaluate_accuracy as ea  # noqa: E402


def _obs(ids, dates, *, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ID": np.repeat(ids, len(dates)),
            "FECHA": np.tile(dates, len(ids)),
            "Value": rng.normal(10, 3, size=len(ids) * len(dates)),
        }
    )


def test_constant_bias_metrics(tmp_path):
    ids = [1, 2, 3]
    dates = pd.date_range("2017-01-01", periods=90, freq="D")
    obs = _obs(ids, dates)
    BIAS = 0.7
    pred = obs.rename(columns={"Value": "TD_predicted"}).copy()
    pred["TD_predicted"] = pred["TD_predicted"] + BIAS  # pred = obs + BIAS

    obs_loaded = obs.rename(columns={"Value": "td_obs"})
    obs_loaded["FECHA"] = pd.to_datetime(obs_loaded["FECHA"])
    res = ea.score(pred, obs_loaded, label="v1")
    o = res["overall"]
    assert o["n_scored"] == len(obs)
    assert o["bias"] == pytest.approx(BIAS, abs=1e-9)
    assert o["mae"] == pytest.approx(BIAS, abs=1e-9)   # constant positive error
    assert o["rmse"] == pytest.approx(BIAS, abs=1e-9)
    assert o["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    # monthly covers Jan–Mar (90 days from Jan 1); each month's bias == BIAS.
    assert (res["monthly"]["bias"] - BIAS).abs().max() == pytest.approx(0.0, abs=1e-9)


def test_two_runs_picks_lower_rmse(tmp_path):
    ids = [1, 2]
    dates = pd.date_range("2017-01-01", periods=60, freq="D")
    obs = _obs(ids, dates)
    obs_path = tmp_path / "obs.parquet"
    obs.to_parquet(obs_path, index=False)

    rng = np.random.default_rng(1)
    good = obs.rename(columns={"Value": "TD_predicted"}).copy()
    good["TD_predicted"] += rng.normal(0, 0.2, size=len(good))   # small error
    bad = obs.rename(columns={"Value": "TD_predicted"}).copy()
    bad["TD_predicted"] += rng.normal(0, 2.0, size=len(bad))     # large error
    pa = tmp_path / "pred_good.parquet"
    pb = tmp_path / "pred_bad.parquet"
    good.to_parquet(pa, index=False)
    bad.to_parquet(pb, index=False)

    out_dir = tmp_path / "plots"
    md_out = tmp_path / "acc.md"
    rc = ea.main(
        [
            "--pred-a", str(pa),
            "--pred-b", str(pb),
            "--obs", str(obs_path),
            "--label-a", "good",
            "--label-b", "bad",
            "--out-dir", str(out_dir),
            "--md-out", str(md_out),
        ]
    )
    assert rc == 0
    text = md_out.read_text()
    assert "Lower RMSE: `good`" in text
    # residual hist + scatter per run (4) + monthly bar (1)
    assert (out_dir / "residual_hist_good.png").exists()
    assert (out_dir / "pred_vs_obs_bad.png").exists()
    assert (out_dir / "monthly_rmse.png").exists()


def test_load_observed_renames_value(tmp_path):
    ids = [5, 6]
    dates = pd.date_range("2017-06-01", periods=10, freq="D")
    obs = _obs(ids, dates)
    p = tmp_path / "td.parquet"
    obs.to_parquet(p, index=False)
    loaded = ea.load_observed(p, value_col="Value")
    assert list(loaded.columns) == ["ID", "FECHA", "td_obs"]
    assert len(loaded) == len(obs)
