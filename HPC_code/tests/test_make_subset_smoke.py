"""
Smoke test for the ID-subset builder (make_subset.py).

Builds a tiny synthetic "full" base (more IDs than we keep, two years, two vars), subsets to the
first k IDs over a year range, and checks: the kept IDs are exactly 0..k-1, the schema/filenames
are preserved (drop-in --base), and the year filter is applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPC))
sys.path.insert(0, str(_HPC.parent))

import make_subset as ms  # noqa: E402


def _write_full(base: Path, n_ids: int, years) -> None:
    rng = np.random.default_rng(0)
    for var in ("td", "tmin_v12"):
        out = base / var / "Outputs"
        out.mkdir(parents=True, exist_ok=True)
        for y in years:
            for m in range(1, 13):
                dates = pd.date_range(f"{y}-{m:02d}-01", periods=10, freq="D")
                df = pd.DataFrame(
                    {
                        "ID": np.repeat(range(n_ids), len(dates)),
                        "FECHA": np.tile(dates, n_ids),
                        "Value": rng.uniform(0, 20, size=n_ids * len(dates)).astype("float32"),
                        "source_file": f"{var}_daily_{y}_{m:02d}.nc",
                    }
                )
                df.to_parquet(out / f"{var}_daily_{y}_{m:02d}.parquet", index=False)


def test_subset_ids_years_and_schema(tmp_path):
    full = tmp_path / "full"
    _write_full(full, n_ids=50, years=(2001, 2002, 2003))

    out = tmp_path / "sub"
    written = ms.make_subset(
        base=full, out=out, n_ids=10, variables=["td", "tmin_v12"], year_range=(2001, 2002)
    )

    # year filter: only 2001+2002 (2 years × 12 months) copied per var.
    files = sorted((out / "td" / "Outputs").glob("td_daily_*.parquet"))
    assert len(files) == 24
    assert not (out / "td" / "Outputs" / "td_daily_2003_01.parquet").exists()

    df = pd.read_parquet(files[0])
    assert sorted(df["ID"].unique()) == list(range(10))   # IDs 0..9 only
    assert list(df.columns) == ["ID", "FECHA", "Value"]   # default columns (drop-in --base)
    # 10 IDs × 10 days per monthly file.
    assert written["td"] == 10 * 10 * 24
    assert written["tmin_v12"] == 10 * 10 * 24
