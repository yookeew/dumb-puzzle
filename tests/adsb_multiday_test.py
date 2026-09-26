"""Score the stand-appearance ADS-B detector across all collected days.

Reads cache/adsb_pushback/day=*.parquet (src/link/adsb_pushback.py) and, for
2025 days, joins truth (BLOCK_TIME_UTC_mvt) and AOBT_3_flt. Reports per day x
airport: recovery (appear+dwell / DEP), ADS-B pushback error, AOBT_3 error on
the identical rows. Where the day lies in the Jan+Jul 2025 holdout, also the
production model's taxi error on those rows (cached holdout predictions from
tests/adsb_vs_model_test.py). 2026 days: recovery only (no truth).

Run:  .venv/Scripts/python.exe src/link/adsb_pushback.py   (first)
      .venv/Scripts/python.exe tests/adsb_multiday_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
EV = ROOT / "cache" / "eval" / "prod_mixed_holdout_ev.parquet"
FMT = dict(tbl_rows=100, tbl_cols=30, tbl_width_chars=300, tbl_formatting="ASCII_MARKDOWN",
           float_precision=1)


def rmse(e: pl.Expr) -> pl.Expr:
    return e.pow(2).mean().sqrt()


def main() -> None:
    # day per movement (also for unrecovered rows) comes from the file partition
    parts = []
    for p in sorted((ROOT / "cache" / "adsb_pushback").glob("day=*.parquet")):
        parts.append(pl.read_parquet(p).with_columns(day=pl.lit(p.stem.split("=")[1])))
    det = pl.concat(parts)
    good = pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False)

    print("Recovery (appear+dwell as % of DEP movements), by day x airport")
    rec = det.group_by("day", "airport").agg(
        (good.mean() * 100).alias("rec%")).pivot(on="airport", index="day", values="rec%").sort("day")
    with pl.Config(**FMT):
        print(rec.select("day", *sorted(c for c in rec.columns if c != "day")))
        print("\nTier mix, all days:")
        print(det.filter(pl.col("adsb_tier").is_not_null()).group_by("day").agg(
            *[(pl.col("adsb_tier") == t).sum().alias(t) for t in ("appear", "dwell", "pass")]).sort("day"))

    truth = (pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
             .select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                     block_ts=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                     aobt3_ts=pl.col("AOBT_3_flt").dt.epoch("ms") / 1000.0)
             .collect())
    t = det.filter(good, pl.col("day") < "2026").join(truth, on="MVT_ID_mvt", how="inner")
    ev = pl.read_parquet(EV).select(pl.col("MVT_ID_mvt").cast(pl.Int64), "pred", "taxi")
    t = t.join(ev, on="MVT_ID_mvt", how="left").with_columns(
        e_adsb=pl.col("block_ts") - pl.col("adsb_pushback_ts"),   # taxi error sign
        e_aobt3=pl.col("block_ts") - pl.col("aobt3_ts"),
        e_model=pl.col("pred") - pl.col("taxi"))
    t = t.with_columns(e_mean=(pl.col("e_adsb") + pl.col("e_model")) / 2)

    aggs = [pl.len().alias("n"),
            (-pl.col("e_adsb")).median().alias("adsb_med_s"),
            ((pl.col("e_adsb").abs() <= 60).mean() * 100).alias("adsb<=60%"),
            ((pl.col("e_adsb").abs() <= 300).mean() * 100).alias("adsb<=300%"),
            rmse(pl.col("e_adsb")).alias("adsb_rmse"),
            ((pl.col("e_aobt3").abs() <= 60).mean() * 100).alias("aobt3<=60%"),
            rmse(pl.col("e_aobt3")).alias("aobt3_rmse"),
            pl.col("e_model").is_not_null().sum().alias("n_model"),
            rmse(pl.col("e_model")).alias("model_rmse"),
            rmse(pl.col("e_mean")).alias("mean_rmse")]
    with pl.Config(**FMT):
        print("\nAccuracy on appear+dwell rows, 2025 days, by day x airport "
              "(model columns only where the day is in the Jan+Jul holdout)")
        print(t.group_by("day", "airport").agg(aggs).sort("day", "airport"))
        print("\nBy airport, all 2025 days pooled")
        print(t.group_by("airport").agg(aggs).sort("airport"))
        print("\nBy tier, all 2025 days pooled")
        print(t.group_by("adsb_tier").agg(aggs))
        print("\nAll pooled")
        print(t.select(aggs))
    t.write_parquet(ROOT / "logs" / "adsb_multiday_scored.parquet")


if __name__ == "__main__":
    main()
