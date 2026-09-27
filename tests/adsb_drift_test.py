"""Label-free drift check: does ADS-B pushback timing transfer from 2025 to 2026?

The §39 blend's per-airport lag (adsb_taxi - taxi) and tier weights are fit
on the 2025 holdout and applied to 2026 ranking rows, but the receiver network
changed between the years. Truth (BLOCK_TIME) is blanked in 2026, so compare
against a reference that exists in both years and is independent of ADS-B:
AOBT_3_flt (NM's flown-trajectory off-block).

Per airport x tier x month (Jan/Jul 2025 vs Jan/Jul 2026):
  - recovery share (appear+dwell / DEP) and tier mix
  - median and IQR of (adsb_pushback - AOBT_3_flt), in seconds
A shift of the median between years at an airport means its 2025 lag won't
transfer as-is. Also reports, for 2025 only, how well the AOBT_3-relative
median tracks the truth-relative lag, which justifies using it as a proxy.

Run:  .venv/Scripts/python.exe tests/adsb_drift_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "cache" / "adsb_pushback"
MONTHS = ["2025-01", "2025-07", "2026-01", "2026-07"]
FMT = dict(tbl_rows=200, tbl_cols=20, tbl_width_chars=250, tbl_formatting="ASCII_MARKDOWN",
           float_precision=0)


def main() -> None:
    det = pl.concat([pl.read_parquet(p).with_columns(day=pl.lit(p.stem.split("=")[1]))
                     for p in sorted(DET.glob("day=*.parquet"))])
    det = det.with_columns(ym=pl.col("day").str.slice(0, 7)).filter(pl.col("ym").is_in(MONTHS))
    cols = ["MVT_ID_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"]
    src = pl.concat([
        pl.scan_parquet(str(ROOT / "data" / "raw" / "training_*.parquet")),
        pl.scan_parquet(ROOT / "data" / "ranking" / "ranking.parquet"),
    ], how="diagonal_relaxed").filter(pl.col("PHASE_mvt") == "DEP").select(cols).collect()
    src = src.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64),
                           aobt3=pl.col("AOBT_3_flt").dt.epoch("ms") / 1000.0,
                           block=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms") / 1000.0)
    d = det.join(src.select("MVT_ID_mvt", "aobt3", "block"), on="MVT_ID_mvt", how="left")
    good = pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False)

    rec = d.group_by("airport", "ym").agg((good.mean() * 100).alias("rec%"))
    with pl.Config(**FMT):
        print("Recovery % (appear+dwell / DEP)")
        print(rec.pivot(on="ym", index="airport", values="rec%").sort("airport")
              .select("airport", *MONTHS))

    e = d.filter(good, pl.col("aobt3").is_not_null()).with_columns(
        vs_aobt3=pl.col("adsb_pushback_ts") - pl.col("aobt3"),
        vs_block=pl.col("adsb_pushback_ts") - pl.col("block"))
    g = e.group_by("airport", "adsb_tier", "ym").agg(
        pl.len().alias("n"),
        pl.col("vs_aobt3").median().alias("med"),
        (pl.col("vs_aobt3").quantile(0.75) - pl.col("vs_aobt3").quantile(0.25)).alias("iqr"),
        pl.col("vs_block").median().alias("med_vs_block"))
    g = g.filter(pl.col("n") >= 50)
    with pl.Config(**FMT):
        print("\nmedian(adsb_pushback - AOBT_3_flt), seconds; cells with n >= 50")
        print(g.pivot(on="ym", index=["airport", "adsb_tier"], values="med")
              .sort("airport", "adsb_tier").select("airport", "adsb_tier",
                                                   *[m for m in MONTHS if m in g["ym"].unique()]))
        print("\nIQR of the same, seconds")
        print(g.pivot(on="ym", index=["airport", "adsb_tier"], values="iqr")
              .sort("airport", "adsb_tier").select("airport", "adsb_tier",
                                                   *[m for m in MONTHS if m in g["ym"].unique()]))
        print("\n2025 only: median vs AOBT_3 alongside median vs true block (the proxy check)")
        print(g.filter(pl.col("ym").str.starts_with("2025"))
              .select("airport", "adsb_tier", "ym", "n", "med", "med_vs_block")
              .sort("airport", "adsb_tier", "ym"))
        # year-over-year shift, same calendar month
        yoy = []
        for m25, m26 in (("2025-01", "2026-01"), ("2025-07", "2026-07")):
            a = g.filter(pl.col("ym") == m25).select("airport", "adsb_tier", pl.col("med").alias("m25"),
                                                     pl.col("n").alias("n25"))
            b = g.filter(pl.col("ym") == m26).select("airport", "adsb_tier", pl.col("med").alias("m26"),
                                                     pl.col("n").alias("n26"))
            yoy.append(a.join(b, on=["airport", "adsb_tier"], how="inner")
                       .with_columns(pair=pl.lit(f"{m25[5:]}: 25->26"),
                                     shift=pl.col("m26") - pl.col("m25")))
        print("\nYear-over-year shift in median(adsb - AOBT_3), same month")
        print(pl.concat(yoy).sort("pair", "airport", "adsb_tier"))
        tm = d.filter(good).group_by("airport", "ym").agg(
            ((pl.col("adsb_tier") == "appear").mean() * 100).alias("appear%"))
        print("\nTier mix: appear share of recovered rows, %")
        print(tm.pivot(on="ym", index="airport", values="appear%").sort("airport")
              .select("airport", *[m for m in MONTHS if m in tm["ym"].unique()]))


if __name__ == "__main__":
    main()
