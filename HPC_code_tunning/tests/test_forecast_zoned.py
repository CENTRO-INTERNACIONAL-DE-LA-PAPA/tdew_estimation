"""P6/B1: the zoned autoregressive forecast is correct and self-consistent.

Three checks, all on tiny in-memory synthetic frames (no GPU, no disk, no real data):

1. **Parity** -- the vectorised bucket path (``forecast_bucket_zoned``) reproduces the
   transparent single-ID reference recursion (``generate_zoned_forecast_for_prepared_id``)
   to floating-point noise. This is the invariant that lets the fast path stand in for the
   oracle on the full grid.
2. **Independent recursion** -- a hand-written recursion (re-deriving the anomaly algebra
   from the training convention: ``<VAR>_anom_lag<k> = VAR(t-k) - VAR_clim(doy(t-k))``,
   with the *lag day's own* doy) matches the module day-for-day.
3. **Autoregression** -- perturbing a predicted day shifts the next day's ``TD_anom_lag1``
   input, proving predictions feed their own future lags (not observed TD).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from HPC_code_tunning.feature_spec import parse_feature  # noqa: E402
from HPC_code_tunning.forecast_zoned import (  # noqa: E402
    forecast_bucket_zoned,
    generate_zoned_forecast_for_prepared_id,
)

FEATURES = ("const", "TMIN_anom", "TD_anom_lag1", "TD_anom_lag2", "TMIN_anom_lag7")
YEAR = 2001


def _synth_frames(ids=(10, 20), seed=0):
    """Build coeffs/clim/history/future for a synthetic 1-year horizon (doy-varying clim)."""
    rng = np.random.default_rng(seed)
    doys = np.arange(1, 367)
    # Non-trivial, doy-varying climatology so lag-day clim differs from current-day clim.
    td_clim = 20.0 + 4.0 * np.sin(2 * np.pi * doys / 366.0)
    tmin_clim = 10.0 + 3.0 * np.cos(2 * np.pi * doys / 366.0)

    coeff_rows, clim_rows = [], []
    for idv in ids:
        # Distinct but stable per-(ID,doy) coefficients for every feature.
        base = {
            "const": 0.1, "TMIN_anom": 0.4, "TD_anom_lag1": 0.5,
            "TD_anom_lag2": -0.1, "TMIN_anom_lag7": 0.05,
        }
        for d in doys:
            jitter = 0.01 * np.sin(d / 30.0 + idv)
            for f in FEATURES:
                coeff_rows.append({
                    "ID": idv, "zone_id": 0, "doy": int(d),
                    "feature_name": f, "coeff": base[f] + jitter,
                    "r_squared_anom": 0.5, "h": 11,
                })
            clim_rows.append({
                "ID": idv, "doy": int(d),
                "TD_clim": float(td_clim[d - 1]), "TMIN_clim": float(tmin_clim[d - 1]),
            })
    coeffs = pd.DataFrame(coeff_rows)
    clim = pd.DataFrame(clim_rows)

    # History = all of year-1 (seeds any lag up to 365 days); future TMIN = all of `YEAR`.
    hist_dates = pd.date_range(f"{YEAR-1}-01-01", f"{YEAR-1}-12-31", freq="D")
    fut_dates = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
    hist_rows, fut_rows = [], []
    for idv in ids:
        hist_rows.append(pd.DataFrame({
            "ID": idv, "FECHA": hist_dates,
            "TD": 20.0 + rng.normal(0, 2, len(hist_dates)),
            "TMIN": 10.0 + rng.normal(0, 2, len(hist_dates)),
        }))
        fut_rows.append(pd.DataFrame({
            "ID": idv, "FECHA": fut_dates,
            "TMIN": 10.0 + rng.normal(0, 2, len(fut_dates)),
        }))
    history = pd.concat(hist_rows, ignore_index=True)
    future = pd.concat(fut_rows, ignore_index=True)
    return coeffs, clim, history, future


def test_vectorised_matches_reference():
    ids = (10, 20)
    coeffs, clim, history, future = _synth_frames(ids)
    vec = forecast_bucket_zoned(coeffs, clim, history, future, prediction_years=(YEAR, YEAR))

    refs = []
    for idv in ids:
        r = generate_zoned_forecast_for_prepared_id(
            idv,
            coeffs_id_df=coeffs[coeffs.ID == idv],
            clim_id_df=clim[clim.ID == idv],
            history_df=history[history.ID == idv],
            future_tmin_df=future[future.ID == idv],
            prediction_years=(YEAR, YEAR),
        )
        refs.append(r)
    ref = pd.concat(refs, ignore_index=True)

    m = vec.merge(ref, on=["ID", "FECHA"], suffixes=("_v", "_r"), how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), "row sets differ between paths"
    assert len(m) == len(ids) * 365
    assert np.max(np.abs(m.TD_predicted_v - m.TD_predicted_r)) < 1e-9


def test_matches_independent_recursion():
    coeffs, clim, history, future = _synth_frames((10,))
    idv = 10
    vec = forecast_bucket_zoned(coeffs, clim, history, future, prediction_years=(YEAR, YEAR))

    tdc = {int(r.doy): r.TD_clim for r in clim.itertuples()}
    tmc = {int(r.doy): r.TMIN_clim for r in clim.itertuples()}
    tmin_obs = {pd.Timestamp(f).normalize(): float(v)
                for f, v in zip(history.FECHA, history.TMIN)}
    tmin_obs.update({pd.Timestamp(f).normalize(): float(v)
                     for f, v in zip(future.FECHA, future.TMIN)})
    td = {pd.Timestamp(f).normalize(): float(v) for f, v in zip(history.FECHA, history.TD)}

    cd_by_doy = {int(d): dict(zip(g.feature_name, g.coeff))
                 for d, g in coeffs.groupby("doy")}

    manual = {}
    for day in pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D"):
        d = int(day.dayofyear)
        acc = 0.0
        for name, c in cd_by_doy[d].items():
            s = parse_feature(name)
            if s.kind == "const":
                x = 1.0
            elif s.kind == "anom":
                x = tmin_obs[day] - tmc[d]
            else:
                ld = day - pd.Timedelta(days=s.lag)
                x = (tmin_obs[ld] - tmc[int(ld.dayofyear)]) if s.var == "TMIN" \
                    else (td[ld] - tdc[int(ld.dayofyear)])
            acc += c * x
        pred = acc + tdc[d]
        td[day] = pred  # autoregressive feed, mirroring the module
        manual[day.normalize()] = pred

    got = {pd.Timestamp(f).normalize(): float(v)
           for f, v in zip(vec.FECHA, vec.TD_predicted)}
    diffs = [abs(got[k] - manual[k]) for k in manual]
    assert max(diffs) < 1e-9


def test_autoregressive_feed():
    """Day t+1's TD_anom_lag1 must come from day t's *prediction*, not observed TD."""
    coeffs, clim, history, future = _synth_frames((10,))
    vec = forecast_bucket_zoned(coeffs, clim, history, future, prediction_years=(YEAR, YEAR))
    v = vec.sort_values("FECHA").reset_index(drop=True)

    tdc = {int(r.doy): r.TD_clim for r in clim.itertuples()}
    tmc = {int(r.doy): r.TMIN_clim for r in clim.itertuples()}
    tmin_obs = {pd.Timestamp(f).normalize(): float(v_) for f, v_ in zip(future.FECHA, future.TMIN)}
    cd_by_doy = {int(d): dict(zip(g.feature_name, g.coeff)) for d, g in coeffs.groupby("doy")}

    # Reconstruct day 2 (2001-01-02) using day 1's *predicted* TD as the lag1 input.
    day1 = pd.Timestamp(f"{YEAR}-01-01")
    day2 = pd.Timestamp(f"{YEAR}-01-02")
    pred1 = float(v[v.FECHA == day1].TD_predicted.iloc[0])
    d2 = int(day2.dayofyear)
    hist = {pd.Timestamp(f).normalize(): float(t) for f, t in zip(history.FECHA, history.TD)}
    acc = 0.0
    for name, c in cd_by_doy[d2].items():
        s = parse_feature(name)
        if s.kind == "const":
            x = 1.0
        elif s.kind == "anom":
            x = tmin_obs[day2] - tmc[d2]
        elif s.var == "TMIN":
            ld = day2 - pd.Timedelta(days=s.lag)
            hm = {pd.Timestamp(f).normalize(): float(t) for f, t in zip(history.FECHA, history.TMIN)}
            hm.update(tmin_obs)
            x = hm[ld] - tmc[int(ld.dayofyear)]
        elif s.lag == 1:                 # lag1 -> day1, which is a PREDICTION
            x = pred1 - tdc[int(day1.dayofyear)]
        else:                            # lag2 -> 2000-12-31, observed history
            ld = day2 - pd.Timedelta(days=s.lag)
            x = hist[ld] - tdc[int(ld.dayofyear)]
        acc += c * x
    expected_day2 = acc + tdc[d2]
    assert abs(float(v[v.FECHA == day2].TD_predicted.iloc[0]) - expected_day2) < 1e-9
