"""
Smoke + correctness test for the Phase-0 prep driver (prep_inputs.py).

Builds a tiny synthetic base (a few IDs, two years, monthly td + tmin_v11 parquet), runs the
prep both sequentially (--n-workers 1) and in parallel (--n-workers 2), and checks:

* all four artifacts are produced (daily_climatology.parquet, bucketed_training_data/,
  climatology_by_bucket/, future_tmin_by_bucket/);
* the per-year bucket shards exist (train_2001 + train_2002);
* the parallel build is byte-for-content identical to the sequential build (the whole point
  of the per-year parallelism — same files, only faster).

No GPU, no SLURM, no real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

import prep_inputs as pi  # noqa: E402

IDS = list(range(8))
YEARS = (2001, 2002)


def _write_base(base: Path) -> None:
    rng = np.random.default_rng(0)
    for var, lo, hi in (("td", 0.0, 15.0), ("tmin_v11", 2.0, 18.0)):
        out = base / var / "Outputs"
        out.mkdir(parents=True, exist_ok=True)
        for y in range(YEARS[0], YEARS[1] + 1):
            for m in range(1, 13):
                dates = pd.date_range(f"{y}-{m:02d}-01", periods=28, freq="D")
                df = pd.DataFrame(
                    {
                        "ID": np.repeat(IDS, len(dates)),
                        "FECHA": np.tile(dates, len(IDS)),
                        "Value": rng.uniform(lo, hi, size=len(IDS) * len(dates)).astype("float32"),
                    }
                )
                df.to_parquet(out / f"{var}_daily_{y}_{m:02d}.parquet", index=False)


def _run(base: Path, results: Path, n_workers: int) -> None:
    pi.prep(
        base=base,
        results=results,
        td_var="td",
        tmin_var="tmin_v11",
        train_year_range=YEARS,
        num_buckets=2,
        pred_year_range=(2002, 2002),
        overwrite=False,
        n_workers=n_workers,
    )


def _concat_bucket_train(results: Path) -> pd.DataFrame:
    root = results / "bucketed_training_data"
    parts = sorted(root.rglob("train_*.parquet"))
    frames = [pd.read_parquet(p).assign(_src=p.relative_to(root).as_posix()) for p in parts]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["_src", "ID", "FECHA"]).reset_index(drop=True)


def test_prep_parallel_matches_sequential(tmp_path):
    base = tmp_path / "base"
    _write_base(base)

    seq = tmp_path / "res_seq"
    par = tmp_path / "res_par"
    _run(base, seq, n_workers=1)
    _run(base, par, n_workers=2)

    # All four artifacts present (parallel run).
    assert (par / "daily_climatology.parquet").exists()
    assert (par / "bucketed_training_data").is_dir()
    assert (par / "climatology_by_bucket").is_dir()
    assert (par / "future_tmin_by_bucket").is_dir()

    # Per-year shards exist for both years, in 2 buckets.
    shards = {p.relative_to(par / "bucketed_training_data").as_posix()
              for p in (par / "bucketed_training_data").rglob("train_*.parquet")}
    for b in ("id_bucket=0000", "id_bucket=0001"):
        assert f"{b}/train_2001.parquet" in shards
        assert f"{b}/train_2002.parquet" in shards

    # Parallel == sequential, content-wise.
    seq_df = _concat_bucket_train(seq)
    par_df = _concat_bucket_train(par)
    pd.testing.assert_frame_equal(seq_df, par_df)
