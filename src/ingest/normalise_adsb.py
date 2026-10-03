"""Normalise raw adsb.lol day extracts into one partitioned parquet store.

Input:  external-data/adsb-fetch/adsb_YYYYMMDD.parquet (src/ingest/fetch_adsb.py)
        and external-data/adsb/adsb_YYYYMMDD.parquet (earlier Colab extracts);
        for a day present in both, adsb-fetch wins.
Output: data/external/adsb/day=YYYY-MM-DD/part-0.parquet, schema

        hex str | reg str | ts f64 (epoch s) | lat f64 | lon f64 | gs f64 (kt)
        alt_baro f64 (ft, null when on ground) | is_ground bool | airport str
        track f64 (deg) | src str (readsb position source: adsb_icao, mlat, ...)

`track` and `src` exist only in extracts written by src/ingest/fetch_adsb.py;
they are null for the older Colab extracts.

Two raw schemas exist:
  - old (2025-01-15): `alt` int, -1 = surface flag, box +-0.06 lat / +-0.09 lon
  - new (later days): `alt_raw` + `is_ground`, box +-0.10 / +-0.15
Downstream code uses gs < 40 as the surface test, not is_ground/alt: the
altitude field also carries genuine negative barometric values.

The raw files other than 2025-01-15 and 2026-01-15 also contain points from
*other* collected days (the Colab work directory was not cleared between
days, so leftover trace files were re-read). Each day's own files overwrite
same-name leftovers, so keeping only points whose UTC date equals the
nominal day recovers that day intact. Rows are also de-duplicated on
(airport, hex, ts).

Run:  .venv/Scripts/python.exe src/ingest/normalise_adsb.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW_DIRS = [ROOT / "external-data" / "adsb", ROOT / "external-data" / "adsb-fetch",
            ROOT / "external-data" / "adsb-restofyear"]  # later wins; restofyear = other 2025 months (§62)
OUT = ROOT / "data" / "external" / "adsb"
COLS = ["hex", "reg", "ts", "lat", "lon", "gs", "alt_baro", "is_ground", "airport",
        "track", "src"]


def normalise(path: Path) -> tuple[dt.date, pl.DataFrame, int]:
    day = dt.datetime.strptime(path.stem.split("_")[1], "%Y%m%d").date()
    df = pl.read_parquet(path)
    if "alt" in df.columns:  # old schema
        df = df.with_columns(is_ground=pl.col("alt") == -1,
                             alt_baro=pl.when(pl.col("alt") == -1).then(None)
                             .otherwise(pl.col("alt")).cast(pl.Float64))
    else:
        df = df.with_columns(alt_baro=pl.when(pl.col("is_ground")).then(None)
                             .otherwise(pl.col("alt_raw")).cast(pl.Float64))
    for c, typ in (("track", pl.Float64), ("src", pl.Utf8)):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=typ).alias(c))
    n_raw = df.height
    df = (df.filter(pl.from_epoch(pl.col("ts"), time_unit="s").dt.date() == day)
          .select(COLS)
          .unique(["airport", "hex", "ts"], keep="first", maintain_order=True)
          .sort("airport", "hex", "ts"))
    return day, df, n_raw


def main() -> None:
    latest = {p.name: p for d in RAW_DIRS for p in sorted(d.glob("adsb_*.parquet"))}
    for path in (latest[k] for k in sorted(latest)):
        day, df, n_raw = normalise(path)
        if n_raw == 0:   # e.g. 2025-10-04: release held a single empty trace
            print(f"{day} [{path.parent.name}]: empty extract, skipped")
            continue
        dest = OUT / f"day={day.isoformat()}"
        dest.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dest / "part-0.parquet")
        print(f"{day} [{path.parent.name}]: {n_raw:>9,} raw -> {df.height:>9,} kept "
              f"({(1 - df.height / n_raw) * 100:4.1f}% off-day/duplicate dropped)")


def load_day(day: dt.date) -> pl.DataFrame:
    return pl.read_parquet(OUT / f"day={day.isoformat()}" / "part-0.parquet")


def available_days() -> list[dt.date]:
    return sorted(dt.date.fromisoformat(p.name.split("=")[1]) for p in OUT.glob("day=*"))


if __name__ == "__main__":
    main()
