"""LIRF long-delay hedge recalibration -> v22 (reports/lirf_hedge_preregistration.md, PROGRESS.md §55).

For LIRF departures with 1 h < T - SOBT <= 6 h, cells = delay band {1-3 h, 3-6 h}
x NM-unmatched (AOBT_3_flt null): pred' = pred + n/(n + 50) * mean(taxi - pred),
then clip to [0, 140,000]. The shifts are fitted on the v21 holdout prediction
(cross-fit, as in tests/lirf_hedge_test.py) over Jan+Jul together, and applied
to the uploaded v21 submission.

Run:  .venv/Scripts/python.exe src/post/lirf_hedge.py
  -> data/submissions/smart-jigsaw_v22.parquet
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import lirf_hedge_test as T  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

SUB = ROOT / "data" / "submissions"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
V21, V22 = SUB / "smart-jigsaw_v21.parquet", SUB / "smart-jigsaw_v22.parquet"
EDGES = T.BANDS["primary"]


def main() -> None:
    ho = A.load()
    ho = ho.with_columns(pred=pl.Series(A.v21_pred(ho)), band=T.band_expr(EDGES))
    shifts = T.fit(ho).sort("band", "nmu")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        print("shifts (fit on Jan+Jul):\n", shifts)

    v21 = pl.read_parquet(V21)
    rk = (pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col("MVT_ID_mvt").cast(v21["MVT_ID_mvt"].dtype), "ADEP_mvt",
                  offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                  nmu=pl.col("AOBT_3_flt").is_null()))
    d = (v21.join(rk, on="MVT_ID_mvt", how="left").with_columns(band=T.band_expr(EDGES))
         .join(shifts.select("band", "nmu", "shift"), on=["band", "nmu"], how="left"))
    assert d.height == v21.height and d["ADEP_mvt"].null_count() == 0
    new = (pl.col("TAXITIME_SEC_mvt") + pl.col("shift").fill_null(0)).clip(0, T.LIRF_CEIL).round().cast(pl.Int32)
    d = d.with_columns(new=new)
    ch = d.filter(pl.col("shift").is_not_null())
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        print(f"changed rows: {ch.height:,} of {d.height:,}")
        print(ch.group_by("band", "nmu").agg(pl.len().alias("n"), (pl.col("new") - pl.col("TAXITIME_SEC_mvt"))
              .mean().alias("mean_shift")).sort("band", "nmu"))

    tpl = pl.read_parquet(TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(d.select("MVT_ID_mvt", pl.col("new").alias("TAXITIME_SEC_mvt")),
                                        on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height, "row count differs from template"
    assert out["TAXITIME_SEC_mvt"].null_count() == 0, "missing predictions"
    assert out["MVT_ID_mvt"].n_unique() == out.height, "duplicate ids"
    assert (out["TAXITIME_SEC_mvt"] >= 0).all()
    out = out.with_columns(pl.col("TAXITIME_SEC_mvt").cast(v21["TAXITIME_SEC_mvt"].dtype))
    out.write_parquet(V22)
    diff = (out["TAXITIME_SEC_mvt"].cast(pl.Float64) - tpl.select("MVT_ID_mvt").join(v21, on="MVT_ID_mvt")
            ["TAXITIME_SEC_mvt"].cast(pl.Float64))
    print(f"wrote {V22.name}: rows changed {(diff != 0).sum():,}; RMS diff vs v21 {np.sqrt((diff ** 2).mean()):.2f} s")


if __name__ == "__main__":
    main()
