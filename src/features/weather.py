"""Family 7 -- METAR weather/de-icing context (Iowa Environmental Mesonet
archive of NOAA/NWS hourly surface observations). See DATA_SOURCES.md.

Hourly, not daily -- this is the genuinely-new-information class flagged by
PROGRESS.md S20/S21 (EUROCONTROL ATFM was capped at ~2.07s because it's
published per (airport, day); the oracle bound at (airport, day, hour)
granularity is ~10.82s, reconfirmed on the current production model by
tests/metar_ceiling.py).

Joined on (ADEP_mvt, hour bucket of MVT_TIME_UTC_mvt=T) -- takeoff time,
never blanked, same anchor family 2's congestion features use. Every
departure at an airport in the same UTC hour sees the same weather row
(the raw METAR observation itself is hourly), unlike ATFM's per-day
resolution.

Feature priority, per the pre-registered mechanism (low-vis procedures,
de-icing, present weather, then runway config via wind) mirrors CLAUDE.md's
challenge write-up and the contestants' own reported finding (PROGRESS.md
S5): temperature/dewpoint spread beats a binary snow flag by a distance,
so it's emitted as a continuous column, not collapsed into a flag.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "external-data" / "metar"

STATIONS = [
    "EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
    "LEMD", "LFPG", "LIRF", "LTFM", "LSZH",
]

# Present-weather groups (METAR intensity+descriptor+phenomenon codes) that
# match the pre-registered mechanisms. Substring match against `wxcodes`,
# which IEM emits as space-separated raw groups (e.g. "-SN BR", "+TSRA").
_SNOW_TOKENS = ("SN", "SG", "PL", "GS", "GR")  # snow/ice-pellet/hail family
_FREEZING_TOKENS = ("FZ",)                       # freezing rain/drizzle/fog
_TSTORM_TOKENS = ("TS",)
_LOWVIS_TOKENS = ("FG", "BR", "HZ", "FU", "VA", "DU", "SA")  # obscuration


def _has_any(col: str, tokens: tuple[str, ...]) -> pl.Expr:
    e = pl.lit(False)
    for t in tokens:
        e = e | pl.col(col).str.contains(t, literal=True)
    return e.fill_null(False)


# Explicit dtypes: an all-missing ("M") column in any single monthly file
# (e.g. a station with zero third-layer sky-cover reports that month) makes
# polars infer String for that file, which then poisons the vertical concat
# across files/stations with a float/string dtype clash.
_FLOAT_COLS = [
    "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby",
    "skyl1", "skyl2", "skyl3", "p01i",
]
_SCHEMA_OVERRIDES = {c: pl.Float64 for c in _FLOAT_COLS} | {
    "station": pl.String, "valid": pl.String,
    "skyc1": pl.String, "skyc2": pl.String, "skyc3": pl.String,
    "wxcodes": pl.String,
}


def _read_station(station: str, ext: Path) -> pl.DataFrame:
    frames = [
        pl.read_csv(p, null_values=["M"], schema_overrides=_SCHEMA_OVERRIDES)
        for p in sorted(ext.glob(f"{station}_*.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"no METAR CSVs for {station} under {ext}")
    return pl.concat(frames, how="vertical_relaxed").unique(subset=["station", "valid"])


@lru_cache(maxsize=1)
def load_metar_hourly(external_dir: str = str(EXTERNAL_DIR)) -> pl.DataFrame:
    """One row per (ICAO station, UTC hour). Raw METAR is already ~hourly
    (report_type=3 = routine obs); floor `valid` to the hour and keep the
    last observation in that hour (there's occasionally more than one raw
    row per clock hour, e.g. a corrected METAR)."""
    ext = Path(external_dir)
    raw = pl.concat([_read_station(s, ext) for s in STATIONS], how="vertical_relaxed")

    # explicit UTC tz -- fetched with tz=Etc/UTC (see fetch_metar_data.py), and
    # `T`/MVT_TIME_UTC_mvt on the movements side is tz-aware UTC too. Leaving
    # this naive risks the exact tz-aware-vs-naive silent-null join bug another
    # team reported losing 19 submissions to (PROGRESS.md S5).
    raw = raw.with_columns(
        valid=pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC"),
    ).with_columns(
        hour=pl.col("valid").dt.truncate("1h"),
    )

    # Ceiling: lowest BKN/OVC/VV (obscured-sky vertical visibility) layer
    # among the 3 reported. FEW/SCT/CLR/NSC don't constitute a ceiling.
    def _layer_ceiling(cov: str, hgt: str) -> pl.Expr:
        return pl.when(pl.col(cov).is_in(["BKN", "OVC", "VV"])).then(pl.col(hgt))

    ceiling_ft = pl.min_horizontal(
        _layer_ceiling("skyc1", "skyl1"),
        _layer_ceiling("skyc2", "skyl2"),
        _layer_ceiling("skyc3", "skyl3"),
    )

    out = raw.with_columns(
        vis_mi=pl.col("vsby"),
        ceiling_ft=ceiling_ft,
        temp_c=(pl.col("tmpf") - 32) / 1.8,
        dewpoint_c=(pl.col("dwpf") - 32) / 1.8,
        wind_dir=pl.col("drct"),
        wind_kt=pl.col("sknt"),
        gust_kt=pl.col("gust"),
        precip_1h_in=pl.col("p01i"),
        wx_snow=_has_any("wxcodes", _SNOW_TOKENS),
        wx_freezing=_has_any("wxcodes", _FREEZING_TOKENS),
        wx_tstorm=_has_any("wxcodes", _TSTORM_TOKENS),
        wx_obscuration=_has_any("wxcodes", _LOWVIS_TOKENS),
    ).with_columns(
        temp_dewpoint_spread_c=pl.col("temp_c") - pl.col("dewpoint_c"),
        below_freezing=(pl.col("temp_c") < 0.0),
    ).with_columns(
        deicing_risk=(
            pl.col("below_freezing")
            & ((pl.col("precip_1h_in").fill_null(0.0) > 0.0)
               | pl.col("wx_snow") | pl.col("wx_freezing"))
        ),
        # standard US flight-category thresholds (vis in sm, ceiling in ft)
        flight_category=pl.when(
            (pl.col("vis_mi") < 1) | (pl.col("ceiling_ft").fill_null(99999) < 500)
        ).then(pl.lit("LIFR")).when(
            (pl.col("vis_mi") < 3) | (pl.col("ceiling_ft").fill_null(99999) < 1000)
        ).then(pl.lit("IFR")).when(
            (pl.col("vis_mi") < 5) | (pl.col("ceiling_ft").fill_null(99999) < 3000)
        ).then(pl.lit("MVFR")).otherwise(pl.lit("VFR")),
    )

    keep = [
        "station", "hour", "vis_mi", "ceiling_ft", "temp_c", "dewpoint_c",
        "temp_dewpoint_spread_c", "below_freezing", "deicing_risk",
        "wx_snow", "wx_freezing", "wx_tstorm", "wx_obscuration",
        "flight_category", "wind_dir", "wind_kt", "gust_kt", "precip_1h_in",
    ]
    return (
        out.select(keep)
        .sort(["station", "hour"])
        .group_by(["station", "hour"], maintain_order=True)
        .last()
    )
