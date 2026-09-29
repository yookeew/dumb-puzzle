"""§51 Part B3b: B3 restricted to stands MISSING from Gateway (no "disagreeing" overrides).

B3 (tests/step51_partB_stands_and_parked.py) overrode Gateway where it was missing
OR > 50 m from the empirical position. Moved rows at stands where Gateway exists
and "disagrees" (EDDM/EGLL/EHAM/LEBL) came out biased early (median -170 to -360 s),
while rows at stands missing from Gateway (LSZH I01-I05/F70/F71/501..., EDDF V151/V153,
LEMD 259/263/T20) were centred near 0. This sizes the missing-only variant:
label-free tier changes on every day, accuracy on 2025-09-15 / 2025-11-15 only.

Input: data/derived/stands_empirical.csv (written by the B script).

Run:  .venv/Scripts/python.exe tests/step51_partB3b_missing_only.py > logs/step51_partB3b_missing_only.log
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import step51_partB_stands_and_parked as B  # noqa: E402
from src.ingest.normalise_adsb import available_days  # noqa: E402
from src.ingest.stands import load_stands, stand_key, stand_positions  # noqa: E402
from src.link.adsb_pushback import departures, detect_day  # noqa: E402


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(1)
    src = ROOT / "data" / "derived" / "stands_empirical.csv"
    print(f"input {src.relative_to(ROOT)} mtime {dt.datetime.fromtimestamp(src.stat().st_mtime):%Y-%m-%d %H:%M}")
    emp = pl.read_csv(src)
    ov = emp.filter(pl.col("keep") & ~pl.col("gateway_resolved"))
    print(f"override stands (kept, missing in Gateway): {ov.height}")
    print(ov.select("airport", "STAND_mvt", "n", "spread_m", "nearest_gateway", "nearest_gateway_m"))
    pos = stand_positions(load_stands())
    ovk = ov.select("airport", key=stand_key(pl.col("STAND_mvt")), lat="lat", lon="lon")
    pos_new = pl.concat([pos.join(ovk.select("airport", "key"), on=["airport", "key"], how="anti"), ovk])

    changes, acc = [], []
    lab = pl.read_parquet(B.LAB, columns=["MVT_ID_mvt", "taxi"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    oof = pl.concat([pl.read_parquet(B.OOF / f"fold={d[:7]}.parquet") for d in B.TRAIN_DAYS]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias("base"))
    for day in available_days():
        old = B.baseline(day, pos)
        new = detect_day(day, pos_new)
        j = (old.select("MVT_ID_mvt", "airport", old_tier="adsb_tier", old_pb="adsb_pushback_ts")
             .join(new.select("MVT_ID_mvt", new_tier="adsb_tier", new_pb="adsb_pushback_ts"), on="MVT_ID_mvt")
             .with_columns(day=pl.lit(day.isoformat()),
                           old_ad=pl.col("old_tier").is_in(["appear", "dwell"]).fill_null(False),
                           new_ad=pl.col("new_tier").is_in(["appear", "dwell"]).fill_null(False)))
        changes.append(j.filter(pl.col("old_tier").fill_null("-") != pl.col("new_tier").fill_null("-")))
        if day.isoformat() in B.TRAIN_DAYS:
            mv = departures(day).select("MVT_ID_mvt", "mvt_ts")
            acc.append(j.join(mv, on="MVT_ID_mvt").join(lab, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt"))
    ch = pl.concat(changes).with_columns(
        grp=pl.when(pl.col("day").is_in(list(B.TRAIN_DAYS))).then(pl.lit("2025 Sep/Nov"))
        .otherwise(pl.col("day").str.slice(0, 4) + pl.lit(" Jan/Jul")))
    print("rows moving INTO / OUT OF appear/dwell vs the unchanged detector:")
    print(ch.group_by("airport", "grp").agg((pl.col("new_ad") & ~pl.col("old_ad")).sum().alias("into"),
                                            (pl.col("old_ad") & ~pl.col("new_ad")).sum().alias("out"))
          .pivot(on="grp", index="airport", values=["into", "out"]).sort("airport"))
    a = pl.concat(acc).with_columns(
        lag=pl.col("airport").replace_strict(B.SHIPPED_LAGS, default=-39.5, return_dtype=pl.Float64))
    m = a.filter(pl.col("new_ad") & ~pl.col("old_ad")).with_columns(
        e=pl.col("mvt_ts") - pl.col("new_pb") - pl.col("lag") - pl.col("taxi"),
        eb=pl.col("base") - pl.col("taxi"))
    print("training days, moved rows: ADS-B error after the shipped lag vs the OOF base on the same rows")
    print(m.group_by("airport", "day").agg(
        pl.len().alias("n"), pl.col("e").median().alias("median"), pl.col("e").pow(2).mean().sqrt().alias("rmse"),
        (100 * (pl.col("e").abs() <= 60).mean()).alias("pm60%"), pl.col("eb").pow(2).mean().sqrt().alias("base_rmse"))
          .sort("airport", "day"))
    for d in B.TRAIN_DAYS:
        g = m.filter(pl.col("day") == d)
        e, eb = g["e"].to_numpy(), g["eb"].to_numpy()
        print(f"  {d}: n={len(e)}  ADS-B rmse {np.sqrt(np.mean(e ** 2)):.1f}  base rmse {np.sqrt(np.mean(eb ** 2)):.1f}  "
              f"median {np.median(e):+.1f}")
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
