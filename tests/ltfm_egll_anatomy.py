"""LTFM and EGLL error anatomy (PROGRESS.md §45 item 2, result §53).

Same recipe as reports/lfpg_investigation.md: where the squared error sits,
what kinds of labels carry it, and whether any recording pattern shows up.
No training. The prediction is the v21 pipeline cross-fit on the Jan+Jul 2025
holdout: NNLS(lgb, catcorr), then the quality-modulated ADS-B blend, then the
partial-track estimate, each fitted on the other month.

Run:  .venv/Scripts/python.exe tests/ltfm_egll_anatomy.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from post import adsb_blend, adsb_partial  # noqa: E402

EVAL = ROOT / "cache" / "eval"
DET = ROOT / "cache" / "adsb_pushback"
RAW = ROOT / "data" / "raw"
JAN, JUL = "2025-01", "2025-07"
AIRPORTS = ["LTFM", "EGLL"]


def load() -> pl.DataFrame:
    ev = {e: pl.read_parquet(EVAL / f"{e}_mixed_holdout_ev.parquet") for e in ("lgb", "catcorr")}
    df = ev["lgb"].select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", pl.col("ym").cast(pl.Utf8),
                          pl.col("taxi").cast(pl.Float64), pl.col("pred").alias("lgb")).join(
        ev["catcorr"].select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias("catcorr")),
        on="MVT_ID_mvt", how="inner")
    raw = (pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
           .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).collect())
    det = pl.concat([pl.read_parquet(p) for p in sorted(DET.glob("day=2025-0[17]*.parquet"))])
    df = (df.join(raw.drop("ADEP_mvt"), on="MVT_ID_mvt", how="left")
          .join(det.drop("airport"), on="MVT_ID_mvt", how="left"))
    return df.with_columns(
        mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
        day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
        offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
        echo=pl.col("BLOCK_TIME_UTC_mvt") == pl.col("SCHED_TIME_UTC_mvt"),
        nmu=pl.col("AOBT_3_flt").is_null(),
        aobt3_taxi=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
    ).with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts"))


def v21_pred(df: pl.DataFrame) -> np.ndarray:
    eng = ["lgb", "catcorr"]
    out = np.empty(df.height)
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        f = df.filter(pl.col("ym") == fm)
        w, _ = nnls(f.select(eng).to_numpy(), f["taxi"].to_numpy())
        f = f.with_columns(pred=pl.Series(f.select(eng).to_numpy() @ w))
        p = adsb_blend.fit(f, per_airport=True)
        q = adsb_blend.fit_quality(f, p)
        f = adsb_partial.with_inputs(f.with_columns(base=adsb_blend.apply(f, p, q)))
        pp = adsb_partial.fit(f)
        d = df.with_columns(pred=pl.Series(df.select(eng).to_numpy() @ w))
        d = adsb_partial.with_inputs(d.with_columns(base=adsb_blend.apply(d, p, q)))
        b = adsb_partial.apply(d, pp)
        m = (df["ym"] == am).to_numpy()
        out[m] = b[m]
    return out


def rmse(e: pl.Expr) -> pl.Expr:
    return e.pow(2).mean().sqrt()


def by(df: pl.DataFrame, key, total_sse: float, n: int = 12) -> pl.DataFrame:
    """SSE share, RMSE, bias and the share removable by a per-group shift (n * bias^2)."""
    return (df.group_by(key).agg(pl.len().alias("n"), (pl.col("r").pow(2).sum() / total_sse).alias("sse_share"),
                                 rmse(pl.col("r")).alias("rmse"), pl.col("r").mean().alias("bias"),
                                 pl.col("taxi").median().alias("taxi_p50"))
            .with_columns(bias_share=pl.col("n") * pl.col("bias").pow(2) / total_sse)
            .sort("sse_share", descending=True).head(n))


def anatomy(df: pl.DataFrame, ap: str, all_sse: float) -> None:
    a = df.filter(pl.col("ADEP_mvt") == ap)
    tot = float((a["r"] ** 2).sum())
    print(f"\n{'=' * 78}\n{ap}: n={a.height:,}  RMSE={np.sqrt(tot / a.height):.1f}  "
          f"share of all-airport SSE={tot / all_sse:.3f}  taxi sd={a['taxi'].std():.0f}")
    for ym in (JAN, JUL):
        s = a.filter(pl.col("ym") == ym)
        print(f"  {ym}: n={s.height:,}  RMSE={np.sqrt((s['r'] ** 2).mean()):.1f}  bias={s['r'].mean():+.1f}")

    se = np.sort((a["r"] ** 2).to_numpy())[::-1]
    print("  top-k SSE share / RMSE without them:",
          {k: (round(se[:k].sum() / tot, 3), round(float(np.sqrt(se[k:].mean())), 1)) for k in (1, 2, 10, 100, 1000)})
    print(f"  rows with |r| > 1800 s: {(a['r'].abs() > 1800).sum()} carry "
          f"{float((a.filter(pl.col('r').abs() > 1800)['r'] ** 2).sum()) / tot:.3f} of SSE")

    a = a.with_columns(tbin=pl.col("taxi").cut([300, 600, 900, 1200, 1800, 3600],
                                                labels=["<5m", "5-10", "10-15", "15-20", "20-30", "30-60", "60m+"]),
                       has_adsb=pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False))
    for key in ("tbin", "echo", "nmu", "has_adsb", "RUNWAY_mvt", "hour", "AIRCRAFT_OPERATOR_flt",
                "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "day"):
        print(f"\n  by {key}:")
        print(by(a, key, tot, 24 if key == "hour" else 10))
    a = a.with_columns(stand_grp=pl.col("STAND_mvt").str.slice(0, 1))
    print("\n  by stand prefix:")
    print(by(a, "stand_grp", tot))

    print("\n  top 15 rows by squared error:")
    print(a.sort(pl.col("r").abs(), descending=True).head(15).select(
        "MVT_ID_mvt", "day", "hour", "taxi", pl.col("pred").round(0), pl.col("offset").cast(pl.Int64),
        "echo", "nmu", pl.col("aobt3_taxi").cast(pl.Int64), "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
        "STAND_mvt", "RUNWAY_mvt", "adsb_tier"))

    # label sanity: AOBT_3-based taxi vs label, and offset vs label
    m = a.filter(~pl.col("nmu"))
    d = (m["taxi"] - m["aobt3_taxi"]).to_numpy()
    print(f"\n  label - (T - AOBT_3) on NM-matched rows: p5/p50/p95 = "
          f"{np.percentile(d, [5, 50, 95]).round(0)}; |diff| > 600 s: {(np.abs(d) > 600).mean():.3f}")
    print(f"  label == offset (echo-like) rows: {(a['taxi'] == a['offset']).sum()}; "
          f"label > offset + 60: {(a['taxi'] > a['offset'] + 60).sum()}; taxi <= 60: {(a['taxi'] <= 60).sum()}")


def main() -> None:
    df = load()
    df = df.with_columns(pred=pl.Series(v21_pred(df))).with_columns(r=pl.col("taxi") - pl.col("pred"))
    all_sse = float((df["r"] ** 2).sum())
    print(f"holdout rows {df.height:,}; v21 cross-fit RMSE {np.sqrt(all_sse / df.height):.2f}")
    trim = df.filter(pl.col("taxi") <= 18000)
    print(f"trimmed (labels <= 5 h, {df.height - trim.height} rows out) RMSE "
          f"{np.sqrt((trim['r'] ** 2).mean()):.2f}")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=30, float_precision=3, tbl_width_chars=200,
                   tbl_cols=20):
        print(by(df, "ADEP_mvt", all_sse))
        print("\ntrimmed SSE share by airport:")
        print(by(trim, "ADEP_mvt", float((trim["r"] ** 2).sum())))
        for ap in AIRPORTS:
            anatomy(df, ap, all_sse)


if __name__ == "__main__":
    main()
