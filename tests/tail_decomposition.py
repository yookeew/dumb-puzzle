"""Untrimmed error decomposition of v27's method on the Jan/Jul 2025 holdout (cross-fit).

Every row, no label trim: where does the squared error sit (label band, airport,
NM-unmatched, echo), and what does the pipeline predict for labels > 5 h?
No fitting beyond reproducing v27's cross-fit; no decision rides on it.

Run:  .venv/Scripts/python.exe tests/tail_decomposition.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_12m_test as T  # noqa: E402
import adsb_combiner_fullyear_test as TF  # noqa: E402
import adsb_combiner_test as C  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

CACHE = ROOT / "cache" / "tail_decomposition_v27.parquet"


def v27_holdout() -> pl.DataFrame:
    if CACHE.exists():
        return pl.read_parquet(CACHE)
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    T.MONTHS = TF.FULL
    pred, _ = T.combiner_12m(d4, T.oof_frame(), v23, use_holdout=True)
    raw = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
                                                 "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "AIRCRAFT_OPERATOR_flt"])
                     for p in sorted((ROOT / "data" / "raw").glob("training_2025-0[17]-*.parquet"))])
    raw = raw.select(pl.col("MVT_ID_mvt").cast(pl.Int64), op="AIRCRAFT_OPERATOR_flt",
                     off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                     um=pl.col("AOBT_3_flt").is_null(),
                     echo=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() < 30))
    out = d4.select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "ym", "taxi", "day").with_columns(
        pred=pl.Series(pred)).join(raw, on="MVT_ID_mvt", how="left")
    out.write_parquet(CACHE)
    return out


def share(df: pl.DataFrame, by: str | list[str], tot: float) -> pl.DataFrame:
    return df.group_by(by).agg(n=pl.len(), sse_pct=(pl.col("se").sum() / tot * 100),
                               rmse=pl.col("se").mean().sqrt(), mean_label=pl.col("taxi").mean(),
                               mean_pred=pl.col("pred").mean()).sort("sse_pct", descending=True)


def main() -> None:
    d = v27_holdout().with_columns(se=(pl.col("pred") - pl.col("taxi")) ** 2)
    pl.Config.set_tbl_rows(60); pl.Config.set_float_precision(1); pl.Config.set_tbl_width_chars(220)
    tot = d["se"].sum()
    for ym in (A.JAN, A.JUL, None):
        s = d if ym is None else d.filter(pl.col("ym") == ym)
        print(f"{ym or 'pooled'}: n={s.height:,} full RMSE {np.sqrt(s['se'].mean()):.2f}, "
              f"trimmed (<=5h) {np.sqrt(s.filter(pl.col('taxi') <= C.TRIM)['se'].mean()):.2f}")
    band = (pl.when(pl.col("taxi") <= 1800).then(pl.lit("a <=30m")).when(pl.col("taxi") <= 3600).then(pl.lit("b 30-60m"))
            .when(pl.col("taxi") <= 7200).then(pl.lit("c 1-2h")).when(pl.col("taxi") <= C.TRIM).then(pl.lit("d 2-5h"))
            .otherwise(pl.lit("e >5h")))
    d = d.with_columns(band=band)
    print("\n== SSE share by LABEL band"); print(share(d, "band", tot).sort("band"))
    print("\n== SSE share by airport x (label > 5h)"); print(share(d, ["ADEP_mvt", pl.col("taxi") > C.TRIM], tot).head(20))
    print("\n== SSE share by NM-unmatched x echo"); print(share(d, ["um", "echo"], tot))
    srt = np.sort(d["se"].to_numpy())[::-1]
    for k in (1, 5, 10, 31, 100, 1000):
        print(f"top {k:>5} rows: {srt[:k].sum() / tot * 100:5.1f}% of SSE; RMSE without them "
              f"{np.sqrt(srt[k:].sum() / d.height):.2f}")
    tail = d.filter(pl.col("taxi") > C.TRIM).sort("se", descending=True)
    print(f"\n== the {tail.height} rows with label > 5 h")
    print(tail.select("ym", "ADEP_mvt", "um", "echo", "off", "taxi", "pred",
                      se_pct=pl.col("se") / tot * 100))
    # what the pipeline predicts for rows that LOOK like tail candidates (no label used for selection)
    cand = d.filter(pl.col("off") >= 5 * 3600)
    print(f"\n== rows with offset >= 5 h (selected without labels): {cand.height}, "
          f"{cand['se'].sum() / tot * 100:.1f}% of SSE")
    print(cand.group_by("um").agg(n=pl.len(), label_gt5h=(pl.col("taxi") > C.TRIM).mean(), echo=pl.col("echo").mean(),
                                  sse_pct=pl.col("se").sum() / tot * 100, mean_label=pl.col("taxi").mean(),
                                  mean_pred=pl.col("pred").mean()))


if __name__ == "__main__":
    main()
