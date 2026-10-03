"""Infer coordinates for stands missing from the Gateway table, from ADS-B (PROGRESS.md §56).

Split-blind: uses only the detector cache (cache/adsb_pushback/, built from
takeoff time and STAND_mvt), the raw adsb.lol points and STAND_mvt. No label
and no off-block time is read.

A matched departure run whose first sample is stationary (gs <= 1 kt) starts
at the aircraft's parking position: the transponder was on before pushback.
For each (airport, stand key) the inferred position is the coordinate-wise
median of those first samples over all ADS-B days, kept when there are
>= MIN_N of them and their median distance to that position is <= MAX_SPREAD_M.

The same estimator is run on stands that do have Gateway coordinates, and the
distance to the Gateway position is reported: that is the validation.

Output data/external/stands_inferred.csv: airport, key, lat, lon, n, spread_m
(derived from adsb.lol, ODbL 1.0; see DATA_SOURCES.md).

Run:  .venv/Scripts/python.exe src/ingest/stand_infer.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.normalise_adsb import available_days, load_day  # noqa: E402
from src.ingest.stands import attach_stand_coords, load_stands, stand_key, stand_positions  # noqa: E402
from src.link.adsb_pushback import OUT as DET, departures  # noqa: E402

OUT = ROOT / "data" / "external" / "stands_inferred.csv"
STATIONARY_GS = 1.0
MIN_N = 3
MAX_SPREAD_M = 30.0
EXCLUDED = ("LFPG",)   # validation fails there: 1 of 7 Gateway stands within 100 m


def first_positions() -> pl.DataFrame:
    """One row per matched departure whose run starts stationary: airport, key, lat, lon, has_gateway."""
    pos = stand_positions(load_stands())
    rows = []
    for day in available_days():
        p = DET / f"day={day.isoformat()}.parquet"
        if not p.exists():
            continue
        det = pl.read_parquet(p).filter(pl.col("adsb_matched") & (pl.col("adsb_first_gs") <= STATIONARY_GS))
        if det.height == 0:
            continue
        mv = attach_stand_coords(departures(day), pos).select(
            "MVT_ID_mvt", "STAND_mvt", has_gateway=pl.col("stand_lat").is_not_null())
        adsb = load_day(day).select("airport", "ts", "gs", "lat", "lon")
        d = (det.select("MVT_ID_mvt", "airport", ts="adsb_first_ts", gs="adsb_first_gs")
             .join(mv, on="MVT_ID_mvt", how="inner")
             .join(adsb, on=["airport", "ts", "gs"], how="inner"))
        # a (ts, gs) pair shared by several aircraft is ambiguous: drop it
        d = d.filter(pl.len().over("MVT_ID_mvt") == 1)
        rows.append(d.select("airport", "STAND_mvt", "lat", "lon", "has_gateway"))
    return pl.concat(rows).with_columns(key=stand_key(pl.col("STAND_mvt")))


def infer(fp: pl.DataFrame) -> pl.DataFrame:
    g = fp.group_by("airport", "key").agg(
        pl.len().alias("n"), pl.col("lat").median().alias("lat"), pl.col("lon").median().alias("lon"),
        pl.col("has_gateway").first(), pl.col("lat").alias("_lats"), pl.col("lon").alias("_lons"))
    spread = []
    for r in g.iter_rows(named=True):
        dy = (np.array(r["_lats"]) - r["lat"]) * 110540.0
        dx = (np.array(r["_lons"]) - r["lon"]) * 111320.0 * np.cos(np.radians(r["lat"]))
        spread.append(float(np.median(np.hypot(dx, dy))))
    return g.drop("_lats", "_lons").with_columns(spread_m=pl.Series(spread))


def validate(inf: pl.DataFrame) -> None:
    gw = stand_positions(load_stands()).rename({"lat": "gw_lat", "lon": "gw_lon"})
    v = (inf.filter(pl.col("has_gateway") & (pl.col("n") >= MIN_N) & (pl.col("spread_m") <= MAX_SPREAD_M))
         .join(gw, on=["airport", "key"], how="inner"))
    if v.height == 0:
        print("validation: no stands with Gateway coords matched by exact key")
        return
    dy = (v["lat"] - v["gw_lat"]).to_numpy() * 110540.0
    dx = (v["lon"] - v["gw_lon"]).to_numpy() * 111320.0 * np.cos(np.radians(v["lat"].to_numpy()))
    d = np.hypot(dx, dy)
    print(f"validation on {v.height} Gateway stands (exact key): distance inferred-Gateway "
          f"p50 {np.median(d):.0f} m, p90 {np.percentile(d, 90):.0f} m, share <= 50 m {np.mean(d <= 50):.2f}, "
          f"<= 100 m {np.mean(d <= 100):.2f}")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12):
        print(v.with_columns(dist_m=pl.Series(d)).group_by("airport").agg(
            pl.len(), pl.col("dist_m").median().alias("p50_m"), (pl.col("dist_m") <= 100).mean().alias("le100"))
            .sort("airport"))


def main() -> None:
    fp = first_positions()
    print(f"stationary first samples: {fp.height:,} ({fp['has_gateway'].mean():.2f} at Gateway stands)")
    inf = infer(fp)
    validate(inf)
    keep = inf.filter(~pl.col("has_gateway") & (pl.col("n") >= MIN_N) & (pl.col("spread_m") <= MAX_SPREAD_M)
                     & ~pl.col("airport").is_in(EXCLUDED))
    cand = inf.filter(~pl.col("has_gateway"))
    print(f"stands without Gateway coords seen: {cand.height}; kept (n >= {MIN_N}, spread <= {MAX_SPREAD_M:.0f} m): "
          f"{keep.height}")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12):
        print(keep.group_by("airport").agg(pl.len(), pl.col("n").sum().alias("obs")).sort("airport"))
    keep.select("airport", "key", "lat", "lon", "n", pl.col("spread_m").round(1)).sort("airport", "key").write_csv(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


def load_inferred() -> pl.DataFrame:
    """(airport, key, lat, lon) for inferred stands, in stand_positions() form."""
    return pl.read_csv(OUT).select("airport", pl.col("key").cast(pl.Utf8), "lat", "lon")


if __name__ == "__main__":
    main()
