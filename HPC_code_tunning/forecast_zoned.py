"""Autoregressive Td forecast for the per-``(zone x doy)`` tuned recipes (plan P6).

The production fixed-model forecast (``tdew_estimation.forecast``) hard-codes the
canonical 5-feature *wide* coefficient schema
``[const_anom, TMIN_anom_coeff, TD_anom_lag1, TD_anom_lag2, TMIN_anom_lag1]`` and a
2-day rolling window. The tuned pipeline instead produces **long/tidy** coefficients
``[ID, zone_id, doy, feature_name, coeff, r_squared_anom, h]`` (``train_zoned``) with a
**variable** feature set per grid point x doy and TD lags of up to 30 days. This module
generalises the recursion to consume those coefficients + climatology directly.

Semantics (must match ``feature_spec.build_feature_frame``, which produced the coeffs):

* ``const``                -> 1.
* ``<VAR>_anom``           -> ``VAR(t) - VAR_clim(doy(t))``   (only ``TMIN_anom`` is a
  feature; ``TD_anom`` is the regression target, never an input).
* ``<VAR>_anom_lag<k>``    -> ``VAR(t-k) - VAR_clim(doy(t-k))`` (the anomaly is formed at
  the *lag day's own* doy, then shifted -- exactly the training convention).

The forecast is **autoregressive** in TD: on day ``t`` the model predicts ``TD_anom(t)``
and that value feeds the ``TD_anom_lag<k>`` inputs of days ``t+1 .. t+k``. TMIN over the
horizon is the *observed* (known) exogenous v1.2 Tmin, so its anomalies/lags need no
recursion.

Two entry points, kept in sync by a parity test (``tests``/``__main__`` self-check):

* :func:`generate_zoned_forecast_for_prepared_id` -- a transparent single-ID reference
  recursion (the correctness oracle).
* :func:`forecast_bucket_zoned` -- the workhorse: vectorised across every ID in a bucket
  with a single Python loop over horizon days (the TD recursion is sequential in time but
  embarrassingly parallel across IDs), returning a tidy ``[ID, FECHA, TD_predicted]``
  frame. This is what the bucketed driver calls.

Design goals mirror the rest of the package: path-agnostic, no hidden globals, numpy-only
core (the sampled backtest is ~10^4 IDs; GPU vectorisation is deferred to the full-grid
fill, plan B3).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids
from tdew_estimation.parquet_io import as_path, read_parquet_any

from .feature_spec import parse_feature

PathLike = Union[str, Path]
logger = logging.getLogger(__name__)

DOY_AXIS = 366
OUTPUT_COLUMNS = ["FECHA", "ID", "TD_predicted"]


# ---------------------------------------------------------------------------------------
# Feature-name bookkeeping
# ---------------------------------------------------------------------------------------
def _feature_universe(feature_names: Sequence[str]) -> List[str]:
    """Ordered feature list for the dense coeff cube: ``const`` first, then sorted rest."""
    names = set(feature_names)
    ordered = ["const"] if "const" in names else []
    ordered.extend(sorted(n for n in names if n != "const"))
    return ordered


def _max_lag(feature_names: Sequence[str]) -> int:
    """Largest lag ``k`` appearing across a set of ``<VAR>_anom_lag<k>`` feature names."""
    return max((parse_feature(n).lag for n in feature_names), default=0)


# ---------------------------------------------------------------------------------------
# Single-ID reference recursion (the correctness oracle)
# ---------------------------------------------------------------------------------------
def generate_zoned_forecast_for_prepared_id(
    location_id: int,
    *,
    coeffs_id_df: pd.DataFrame,
    clim_id_df: pd.DataFrame,
    history_df: pd.DataFrame,
    future_tmin_df: pd.DataFrame,
    prediction_years: Tuple[int, int],
) -> Optional[pd.DataFrame]:
    """Recursive TD forecast for one ID from already-loaded per-ID frames (long coeffs).

    Parameters
    ----------
    coeffs_id_df:
        Long/tidy coeffs for this ID: ``[doy, feature_name, coeff]`` (extra cols ignored).
    clim_id_df:
        Climatology for this ID: ``[doy, TD_clim, TMIN_clim]``.
    history_df:
        Observed ``[FECHA, TD, TMIN]`` used to seed the lag window; needs at least
        ``max_lag`` days ending just before the horizon start.
    future_tmin_df:
        Exogenous ``[FECHA, TMIN]`` over the forecast horizon (observed v1.2 Tmin).
    prediction_years:
        ``(start_year, end_year)`` inclusive forecast horizon.

    Returns
    -------
    ``[FECHA, ID, TD_predicted]`` frame, an empty frame with those columns if no day was
    predictable, or ``None`` if required inputs are missing/malformed.
    """
    if coeffs_id_df is None or coeffs_id_df.empty or clim_id_df is None or clim_id_df.empty:
        return None
    if not {"doy", "feature_name", "coeff"}.issubset(coeffs_id_df.columns):
        return None
    if not {"doy", "TD_clim", "TMIN_clim"}.issubset(clim_id_df.columns):
        return None
    if history_df is None or future_tmin_df is None or future_tmin_df.empty:
        return None

    # doy -> {feature_name: coeff}
    coeff_by_doy: Dict[int, Dict[str, float]] = {}
    for d, g in coeffs_id_df.groupby("doy"):
        coeff_by_doy[int(d)] = dict(zip(g["feature_name"].astype(str), g["coeff"].astype(float)))
    if not coeff_by_doy:
        return None
    max_lag = _max_lag({f for m in coeff_by_doy.values() for f in m})

    # doy -> (TD_clim, TMIN_clim), dropping NaN climatology days.
    clim = clim_id_df.dropna(subset=["TD_clim", "TMIN_clim"])
    td_clim = {int(d): float(v) for d, v in zip(clim["doy"], clim["TD_clim"])}
    tmin_clim = {int(d): float(v) for d, v in zip(clim["doy"], clim["TMIN_clim"])}
    if not td_clim:
        return None

    # Per-day observed series (TMIN from history + future; TD from history only).
    tmin_obs: Dict[pd.Timestamp, float] = {}
    td: Dict[pd.Timestamp, float] = {}
    if not history_df.empty:
        h = history_df.copy()
        h["FECHA"] = pd.to_datetime(h["FECHA"])
        for f, tv, mv in zip(h["FECHA"], h["TD"], h["TMIN"]):
            f = pd.Timestamp(f).normalize()
            if pd.notna(mv):
                tmin_obs[f] = float(mv)
            if pd.notna(tv):
                td[f] = float(tv)
    ft = future_tmin_df.copy()
    ft["FECHA"] = pd.to_datetime(ft["FECHA"])
    for f, mv in zip(ft["FECHA"], ft["TMIN"]):
        if pd.notna(mv):
            tmin_obs[pd.Timestamp(f).normalize()] = float(mv)

    y0, y1 = prediction_years
    horizon = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")

    def _tmin_anom(day: pd.Timestamp) -> Optional[float]:
        m = tmin_obs.get(day)
        c = tmin_clim.get(int(day.dayofyear))
        if m is None or c is None:
            return None
        return m - c

    def _td_anom(day: pd.Timestamp) -> Optional[float]:
        v = td.get(day)
        c = td_clim.get(int(day.dayofyear))
        if v is None or c is None:
            return None
        return v - c

    out: List[Dict[str, object]] = []
    for day in horizon:
        d = int(day.dayofyear)
        if d not in coeff_by_doy or d not in td_clim:
            continue
        coeffs = coeff_by_doy[d]
        acc = 0.0
        ok = True
        for name, c in coeffs.items():
            if c == 0.0:
                continue  # dropped/zero feature contributes nothing (and skips NaN lookups)
            spec = parse_feature(name)
            if spec.kind == "const":
                x = 1.0
            elif spec.kind == "anom":  # only TMIN_anom is ever a feature
                x = _tmin_anom(day) if spec.var == "TMIN" else None
            else:  # lag
                lag_day = day - pd.Timedelta(days=spec.lag)
                x = _tmin_anom(lag_day) if spec.var == "TMIN" else _td_anom(lag_day)
            if x is None or not np.isfinite(x):
                ok = False
                break
            acc += c * float(x)
        if not ok:
            continue
        predicted_td = acc + td_clim[d]
        td[day] = predicted_td  # feed the autoregressive TD lags of later days
        out.append({"FECHA": day, "ID": int(location_id), "TD_predicted": float(predicted_td)})

    if not out:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(out)[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------------------
# Vectorised-across-IDs bucket forecaster (the workhorse)
# ---------------------------------------------------------------------------------------
def _pivot_to_matrix(
    long_df: pd.DataFrame,
    value_col: str,
    id_index: Dict[int, int],
    date_index: Dict[pd.Timestamp, int],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Scatter a long ``[ID, FECHA, <value_col>]`` frame into a dense ``[n_id, n_days]``."""
    mat = np.full(shape, np.nan, dtype=np.float64)
    if long_df is None or long_df.empty:
        return mat
    df = long_df[["ID", "FECHA", value_col]].copy()
    df["FECHA"] = pd.to_datetime(df["FECHA"]).dt.normalize()
    rows, cols, vals = [], [], []
    for idv, f, v in zip(df["ID"], df["FECHA"], df[value_col]):
        i = id_index.get(int(idv))
        j = date_index.get(pd.Timestamp(f))
        if i is not None and j is not None and pd.notna(v):
            rows.append(i)
            cols.append(j)
            vals.append(float(v))
    if rows:
        mat[np.asarray(rows), np.asarray(cols)] = np.asarray(vals)
    return mat


def forecast_bucket_zoned(
    coeffs_df: pd.DataFrame,
    clim_df: pd.DataFrame,
    history_df: pd.DataFrame,
    future_tmin_df: pd.DataFrame,
    *,
    prediction_years: Tuple[int, int],
) -> pd.DataFrame:
    """Autoregressive TD forecast for every ID in a bucket, vectorised across IDs.

    Reads the long coeffs, builds a dense ``[n_id, 366, F]`` coefficient cube and per-day
    TMIN/TD matrices, then loops **once over horizon days** applying the recursion to all
    IDs at once. Returns a tidy ``[FECHA, ID, TD_predicted]`` frame (empty if nothing was
    predictable).
    """
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if coeffs_df is None or coeffs_df.empty or clim_df is None or clim_df.empty:
        return empty
    if future_tmin_df is None or future_tmin_df.empty:
        return empty

    # IDs that actually have a recipe (land cells); order them deterministically.
    ids = np.sort(coeffs_df["ID"].astype(int).unique())
    n_id = int(ids.shape[0])
    if n_id == 0:
        return empty
    id_index = {int(v): i for i, v in enumerate(ids.tolist())}

    feat_list = _feature_universe(coeffs_df["feature_name"].astype(str).unique())
    feat_index = {n: j for j, n in enumerate(feat_list)}
    nf = len(feat_list)
    specs = [parse_feature(n) for n in feat_list]
    max_lag = _max_lag(feat_list)

    # Dense coeff cube C[n_id, 366, F] (0 where a feature is dropped for that ID x doy).
    C = np.zeros((n_id, DOY_AXIS, nf), dtype=np.float64)
    cf = coeffs_df[["ID", "doy", "feature_name", "coeff"]].copy()
    ci = cf["ID"].map(id_index).to_numpy()
    cd = cf["doy"].astype(int).to_numpy() - 1
    cj = cf["feature_name"].astype(str).map(feat_index).to_numpy()
    keep = (~pd.isna(ci)) & (~pd.isna(cj)) & (cd >= 0) & (cd < DOY_AXIS)
    C[ci[keep].astype(int), cd[keep].astype(int), cj[keep].astype(int)] = cf["coeff"].to_numpy()[keep]

    # Climatology arrays [n_id, 366] (NaN where absent).
    td_clim = np.full((n_id, DOY_AXIS), np.nan)
    tmin_clim = np.full((n_id, DOY_AXIS), np.nan)
    cl = clim_df[["ID", "doy", "TD_clim", "TMIN_clim"]].copy()
    li = cl["ID"].map(id_index)
    keep_cl = li.notna() & cl["doy"].between(1, DOY_AXIS)
    li_v = li[keep_cl].astype(int).to_numpy()
    ld_v = cl["doy"][keep_cl].astype(int).to_numpy() - 1
    td_clim[li_v, ld_v] = cl["TD_clim"][keep_cl].to_numpy()
    tmin_clim[li_v, ld_v] = cl["TMIN_clim"][keep_cl].to_numpy()

    # Daily timeline: max_lag seed days + the horizon.
    y0, y1 = prediction_years
    horizon = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    seed_start = horizon[0] - pd.Timedelta(days=max_lag)
    dates = pd.date_range(seed_start, horizon[-1], freq="D")
    n_days = len(dates)
    horizon_start = max_lag  # index of horizon[0] within `dates`
    date_index = {pd.Timestamp(d): j for j, d in enumerate(dates)}
    doy_of_day = np.asarray([int(d.dayofyear) for d in dates]) - 1  # 0-based

    # Per-day TMIN (history seed + future horizon) and TD (history seed only).
    tmin_src = pd.concat(
        [
            history_df[["ID", "FECHA", "TMIN"]] if history_df is not None and not history_df.empty
            else pd.DataFrame(columns=["ID", "FECHA", "TMIN"]),
            future_tmin_df[["ID", "FECHA", "TMIN"]],
        ],
        ignore_index=True,
    )
    TMIN = _pivot_to_matrix(tmin_src, "TMIN", id_index, date_index, (n_id, n_days))
    TD = _pivot_to_matrix(
        history_df[["ID", "FECHA", "TD"]] if history_df is not None and not history_df.empty
        else pd.DataFrame(columns=["ID", "FECHA", "TD"]),
        "TD", id_index, date_index, (n_id, n_days),
    )

    # Anomaly matrices; TD horizon columns are filled as the recursion advances.
    tmin_clim_day = tmin_clim[:, doy_of_day]   # [n_id, n_days]
    td_clim_day = td_clim[:, doy_of_day]
    TMIN_anom = TMIN - tmin_clim_day
    TD_anom = TD - td_clim_day

    for j in range(horizon_start, n_days):
        d = doy_of_day[j]                      # 0-based doy of this horizon day
        Cd = C[:, d, :]                        # [n_id, F] coeffs for this doy per ID
        X = np.zeros((n_id, nf), dtype=np.float64)
        for f, spec in enumerate(specs):
            if spec.kind == "const":
                X[:, f] = 1.0
            elif spec.kind == "anom":
                X[:, f] = TMIN_anom[:, j] if spec.var == "TMIN" else np.nan
            else:  # lag
                src = TMIN_anom if spec.var == "TMIN" else TD_anom
                X[:, f] = src[:, j - spec.lag]
        # Dropped features (coeff 0) contribute exactly 0 -- and must not let a NaN input
        # poison the sum. Keep NaN only where a *used* feature is genuinely missing.
        contrib = np.where(Cd == 0.0, 0.0, Cd * X)
        pred_anom = contrib.sum(axis=1)
        TD_anom[:, j] = pred_anom
        TD[:, j] = pred_anom + td_clim_day[:, j]

    # Emit horizon predictions; drop cells with no climatology / non-finite results.
    pred = TD[:, horizon_start:]               # [n_id, len(horizon)]
    finite = np.isfinite(pred)
    if not finite.any():
        return empty
    ii, jj = np.nonzero(finite)
    out = pd.DataFrame(
        {
            "FECHA": horizon.to_numpy()[jj],
            "ID": ids[ii].astype(int),
            "TD_predicted": pred[ii, jj].astype(float),
        }
    )
    return out.sort_values(["ID", "FECHA"]).reset_index(drop=True)[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------------------
# Bucketed disk driver
# ---------------------------------------------------------------------------------------
def _read_history_shard(prepared_root: Path, bucket_id: int, history_end_year: int) -> pd.DataFrame:
    """Read the prepared shard for ``history_end_year`` (its December seeds the lags)."""
    tdir = bucket_dir(prepared_root, bucket_id)
    df = read_parquet_any(tdir)
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    return df[df["FECHA"].dt.year == int(history_end_year)][["ID", "FECHA", "TD", "TMIN"]]


def forecast_bucket_zoned_from_disk(
    *,
    bucket_id: int,
    coeffs_root: PathLike,
    climatology_root: PathLike,
    prepared_training_root: PathLike,
    future_tmin_root: PathLike,
    predictions_output_root: PathLike,
    prediction_years: Tuple[int, int],
    history_end_year: int,
    overwrite: bool = False,
) -> dict:
    """Forecast one bucket end-to-end from disk and write ``pred.parquet``.

    Input shards (each read once):
      * coeffs:      ``coeffs_root/id_bucket=XXXX/coeffs.parquet`` (tidy/long)
      * climatology: ``climatology_root/id_bucket=XXXX/climatology.parquet``
      * history:     ``prepared_training_root/id_bucket=XXXX/`` filtered to history_end_year
      * future TMIN: ``future_tmin_root/id_bucket=XXXX/`` (horizon exogenous, col ``TMIN``)
    """
    coeffs_root = as_path(coeffs_root)
    climatology_root = as_path(climatology_root)
    prepared_root = as_path(prepared_training_root)
    future_tmin_root = as_path(future_tmin_root)
    pred_dir = bucket_dir(as_path(predictions_output_root), bucket_id)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "pred.parquet"
    if pred_file.exists() and not overwrite:
        return {"bucket_id": int(bucket_id), "status": "skipped", "rows": 0}

    coeffs_file = bucket_dir(coeffs_root, bucket_id) / "coeffs.parquet"
    clim_file = bucket_dir(climatology_root, bucket_id) / "climatology.parquet"
    if not coeffs_file.exists() or not clim_file.exists():
        return {"bucket_id": int(bucket_id), "status": "missing_inputs", "rows": 0}

    coeffs_df = pd.read_parquet(coeffs_file)
    clim_df = pd.read_parquet(clim_file)
    history_df = _read_history_shard(prepared_root, bucket_id, history_end_year)
    future_tmin_df = read_parquet_any(bucket_dir(future_tmin_root, bucket_id))
    future_tmin_df["FECHA"] = pd.to_datetime(future_tmin_df["FECHA"])

    pred = forecast_bucket_zoned(
        coeffs_df, clim_df, history_df, future_tmin_df, prediction_years=prediction_years
    )
    if not pred.empty:
        pred.to_parquet(pred_file, engine="pyarrow", index=False)
        return {"bucket_id": int(bucket_id), "status": "ok", "rows": int(len(pred))}
    if overwrite and pred_file.exists():
        pred_file.unlink()
    return {"bucket_id": int(bucket_id), "status": "empty", "rows": 0}


def run_bucketed_zoned_forecast(
    *,
    coeffs_root: PathLike,
    climatology_root: PathLike,
    prepared_training_root: PathLike,
    future_tmin_root: PathLike,
    predictions_output_root: PathLike,
    prediction_years: Tuple[int, int],
    history_end_year: int,
    bucket_ids: Optional[Sequence[int]] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Forecast a set of buckets sequentially in-process; return a per-bucket summary.

    Sequential is intentional for the sampled backtest (few buckets, cheap per bucket).
    A Dask/CUDA fan-out can be layered on later for the full-grid fill (plan B3), mirroring
    ``forecast.run_bucketed_forecast_dask``.
    """
    coeffs_root_p = as_path(coeffs_root)
    buckets = (
        discover_bucket_ids(coeffs_root_p) if bucket_ids is None
        else sorted({int(b) for b in bucket_ids})
    )
    if not buckets:
        raise ValueError(f"No buckets found under {coeffs_root_p}")
    summaries: List[dict] = []
    for b in buckets:
        try:
            summaries.append(
                forecast_bucket_zoned_from_disk(
                    bucket_id=int(b),
                    coeffs_root=coeffs_root_p,
                    climatology_root=climatology_root,
                    prepared_training_root=prepared_training_root,
                    future_tmin_root=future_tmin_root,
                    predictions_output_root=predictions_output_root,
                    prediction_years=prediction_years,
                    history_end_year=history_end_year,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad bucket must not abort the run
            logger.warning("forecast bucket %s failed: %s: %s", b, type(exc).__name__, exc)
            summaries.append({"bucket_id": int(b), "status": "error", "rows": 0})
    return pd.DataFrame(summaries).sort_values("bucket_id").reset_index(drop=True)
