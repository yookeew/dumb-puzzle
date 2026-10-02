"""Scoping for the v3 ADS-B detector (inferred stands + fallback match), PROGRESS.md §56.

Non-holdout days only (2025-09-15, 2025-11-15). Compares the production
detector cache with cache/adsb_pushback_v3/: how many rows gain a pushback,
and how accurate the new pushbacks are against the true taxi label, next to
the existing appear/dwell tiers. Base = mean of the month-wise OOF LightGBM
and CatBoost predictions (honest on these days).

Run:  .venv/Scripts/python.exe tests/adsb_v3_scope.py
"""
import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DAYS = ("2025-09-15", "2025-11-15")


def load() -> pl.DataFrame:
    v1 = pl.concat([pl.read_parquet(ROOT / f"cache/adsb_pushback/day={d}.parquet") for d in DAYS])
    v3 = pl.concat([pl.read_parquet(ROOT / f"cache/adsb_pushback_v3/day={d}.parquet") for d in DAYS])
    oof = None
    for e in ("lgb", "cat"):
        s = pl.concat([pl.read_parquet(ROOT / f"cache/oof/{e}_mixed/fold={m}.parquet") for m in ("2025-09", "2025-11")])
        s = s.select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias(e))
        oof = s if oof is None else oof.join(s, on="MVT_ID_mvt")
    mvt = (pl.scan_parquet(str(ROOT / "data/raw/training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                   T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0).collect())
    df = (v3.select("MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts", "adsb_fallback", "adsb_first_own_m")
          .join(v1.select("MVT_ID_mvt", tier1="adsb_tier", own1="adsb_first_own_m"), on="MVT_ID_mvt")
          .join(mvt, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt"))
    return df.with_columns(base=(pl.col("lgb") + pl.col("cat")) / 2,
                           adsb_taxi=pl.col("T") - pl.col("adsb_pushback_ts"))


def main() -> None:
    df = load()
    obs1 = pl.col("tier1").is_in(["appear", "dwell"]).fill_null(False)
    obs3 = pl.col("adsb_tier").is_in(["appear", "dwell", "fb_dwell", "fb_appear"]).fill_null(False)
    df = df.with_columns(src=pl.when(obs1).then(pl.col("tier1"))
                         .when(obs3 & pl.col("adsb_fallback")).then(pl.col("adsb_tier"))
                         .when(obs3).then(pl.lit("new_via_inferred_stand")).otherwise(None))
    df = df.filter(pl.col("ADEP_mvt") != "LIRF")
    lag = (df.filter(pl.col("src").is_in(["appear", "dwell"])).group_by("ADEP_mvt")
           .agg(lag=(pl.col("adsb_taxi") - pl.col("taxi")).median()))
    df = df.join(lag, on="ADEP_mvt", how="left").with_columns(err=pl.col("adsb_taxi") - pl.col("lag").fill_null(0) - pl.col("taxi"))
    print(f"rows (non-LIRF) {df.height:,}; observed v1 {df.select(obs1.sum()).item():,}; "
          f"observed v3 {df.select(obs3.sum()).item():,}")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=40, float_precision=1):
        print("\naccuracy by source (err = ADS-B taxi - airport lag - label):")
        print(df.filter(pl.col("src").is_not_null()).group_by("src").agg(
            pl.len(), pl.col("err").median().alias("err_p50"), (pl.col("err").abs() <= 60).mean().alias("within60"),
            (pl.col("err").abs() <= 300).mean().alias("within300"), pl.col("err").pow(2).mean().sqrt().alias("rmse_adsb"),
            (pl.col("base") - pl.col("taxi")).pow(2).mean().sqrt().alias("rmse_base")).sort("src"))
        print("\nnew rows by airport:")
        print(df.filter(pl.col("src").is_in(["fb_dwell", "fb_appear", "new_via_inferred_stand"]))
              .group_by("ADEP_mvt", "src").agg(pl.len(), pl.col("err").median().alias("err_p50"),
              pl.col("err").pow(2).mean().sqrt().alias("rmse_adsb"),
              (pl.col("base") - pl.col("taxi")).pow(2).mean().sqrt().alias("rmse_base")).sort("ADEP_mvt", "src"))
    for s in ("fb_dwell", "fb_appear", "new_via_inferred_stand"):
        g = df.filter(pl.col("src") == s)
        if g.height < 20:
            continue
        r = (g["taxi"] - g["base"]).to_numpy(); dl = (g["adsb_taxi"] - g["lag"].fill_null(0) - g["base"]).to_numpy()
        w = float(np.clip((r * dl).sum() / (dl * dl).sum(), 0, 1))
        print(f"{s:<24} n={g.height:>4}  in-sample LS weight {w:.2f}: base {np.sqrt(np.mean(r ** 2)):.1f} -> blend "
              f"{np.sqrt(np.mean((r - w * dl) ** 2)):.1f}")


if __name__ == "__main__":
    main()
