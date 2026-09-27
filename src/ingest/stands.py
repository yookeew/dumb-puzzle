"""Load Gateway stand coordinates and key them to STAND_mvt naming.

data/external/stands.csv (src/ingest/fetch_stands.py) names stands the way
the scenery author did ("Gate A04", "Stand W 22", "202-(C)", "F6"); the
movements use the airport's own IDs ("A04", "202", "F06"). stand_key()
normalises both sides to the same form; the join then tries the exact key
and falls back to the key with a trailing L/R/A/B/C variant letter dropped
(Gateway splits MARS stands into 115A/115B, movements log 115).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STANDS = ROOT / "data" / "external" / "stands.csv"


def stand_key(col: pl.Expr) -> pl.Expr:
    """Normalise a stand-name column: uppercase, drop words/suffixes/spaces, unpad digits."""
    return (
        col.cast(pl.Utf8).str.to_uppercase()
        .str.replace(r"-\([A-Z]\)$", "")                       # LTFM "202-(C)"
        .str.replace(r"^(GATE|STAND|RAMP|PARKING|APRON|POS)\s*", "")
        .str.replace_all(r"[\s_\-]", "")
        .str.replace_all(r"(^|[A-Z])0+(\d)", "$1$2")            # "F06" -> "F6", "008" -> "8"
    )


def variant_base(col: pl.Expr) -> pl.Expr:
    """Key with a trailing variant letter removed ("115B" -> "115", "G10L" -> "G10")."""
    return col.str.replace(r"(\d)[LRABC]$", "$1")


def load_stands() -> pl.DataFrame:
    """One row per Gateway stand, with `key` and `key_base` for joining to STAND_mvt."""
    s = pl.read_csv(STANDS, schema_overrides={"airlines": pl.Utf8, "width_code": pl.Utf8})
    return s.with_columns(key=stand_key(pl.col("stand_name"))).with_columns(
        key_base=variant_base(pl.col("key")))


def stand_positions(stands: pl.DataFrame) -> pl.DataFrame:
    """(airport, key) -> mean lat/lon, over both exact keys and variant bases.

    Exact keys win over variant bases when both exist for the same string.
    """
    exact = stands.group_by("airport", "key").agg(
        pl.col("lat").mean(), pl.col("lon").mean(), prio=pl.lit(0))
    base = stands.group_by("airport", key="key_base").agg(
        pl.col("lat").mean(), pl.col("lon").mean(), prio=pl.lit(1))
    return (pl.concat([exact, base]).sort("prio")
            .unique(["airport", "key"], keep="first").drop("prio"))


def attach_stand_coords(mvt: pl.DataFrame, positions: pl.DataFrame,
                        airport_col: str = "ADEP_mvt") -> pl.DataFrame:
    """Left-join stand lat/lon onto movements by normalised STAND_mvt (exact, then variant base)."""
    m = mvt.with_columns(_k=stand_key(pl.col("STAND_mvt"))).with_columns(
        _kb=variant_base(pl.col("_k")))
    pos = positions.rename({"airport": airport_col})
    m = m.join(pos.rename({"key": "_k", "lat": "stand_lat", "lon": "stand_lon"}),
               on=[airport_col, "_k"], how="left")
    m = m.join(pos.rename({"key": "_kb", "lat": "_lat2", "lon": "_lon2"}),
               on=[airport_col, "_kb"], how="left")
    return m.with_columns(
        stand_lat=pl.coalesce("stand_lat", "_lat2"),
        stand_lon=pl.coalesce("stand_lon", "_lon2"),
    ).drop("_k", "_kb", "_lat2", "_lon2")
