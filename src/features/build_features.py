"""Stage 2 — feature builder (Option A: internal features only, no OSM).

Single split-blind entry point. `build_features(df, priors)` takes ONE dataframe
in ranking schema (arrivals + departures together, DEP off-block/taxi already
nulled) and returns one row per departure of features keyed on MVT_ID_mvt. It
never reads a departure off-block time.

Families implemented here:
  1. physical baseline  — fitted unimpeded-taxi percentile per stand / runway
                          (via `priors`, fit on the training split only)
  2. congestion         — takeoff-anchored rolling counts, inter-departure gaps,
                          saturation runs
  3. runway config      — active departure/arrival runway set per 5-min bin,
                          time since it last changed, mode share
  4. rotation & schedule— Stage 1 stand link: inbound delay, ground time,
                          L_sec (soft), schedule/EOBT/IOBT/LOBT deltas
  + calendar / categorical passthrough
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.encode import (  # noqa: E402,F401
    CAT_COLS,
    apply_priors,
    feature_matrix,
    fit_priors,
)
from link.stand_link import build_stand_links  # noqa: E402

# back-compat alias — earlier code imported the private name
_apply_priors = apply_priors


def _runway_time(df: pl.DataFrame) -> pl.DataFrame:
    """Best time the aircraft is on the runway: takeoff for DEP, landing for ARR."""
    return df.with_columns(
        airport=pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt")),
        rt=pl.col("MVT_TIME_UTC_mvt"),
        stand_group=pl.col("STAND_mvt").str.extract(r"^([A-Za-z]+)").fill_null("_"),
    )


# --------------------------------------------------------------------------- 2
def _roll_count(
    src: pl.DataFrame, by: list[str], period: str, offset: str, name: str
) -> pl.DataFrame:
    """One row per src row (keyed MVT_ID_mvt) with a rolling window count.

    Window is [rt + offset, rt + offset + period]. `rolling` preserves input row
    order within each group, so we attach by position on the same sorted frame.
    """
    s = src.sort(by + ["rt"])
    r = s.rolling(index_column="rt", period=period, offset=offset, group_by=by).agg(
        pl.len().alias(name)
    )
    return s.with_columns(r[name]).select("MVT_ID_mvt", name)


def _congestion(frame: pl.DataFrame) -> pl.DataFrame:
    """Takeoff-anchored rolling counts. Returns per-DEP columns keyed MVT_ID_mvt."""
    f = _runway_time(frame).select(
        "MVT_ID_mvt", "PHASE_mvt", "airport", "RUNWAY_mvt", "rt"
    )
    dep = f.filter(pl.col("PHASE_mvt") == "DEP")
    arr = f.filter(pl.col("PHASE_mvt") == "ARR")

    out = dep.select("MVT_ID_mvt")
    for by, period, offset, nm in [
        (["airport"], "30m", "-30m", "n_dep_30m_prev"),
        (["airport"], "60m", "-60m", "n_dep_60m_prev"),
        (["airport"], "15m", "0s", "n_dep_15m_next"),
        (["airport", "RUNWAY_mvt"], "30m", "-30m", "n_deprwy_30m_prev"),
        (["airport", "RUNWAY_mvt"], "15m", "0s", "n_deprwy_15m_next"),
    ]:
        out = out.join(_roll_count(dep, by, period, offset, nm), on="MVT_ID_mvt", how="left")

    # arrivals landing within +/-30 min of this takeoff
    both = pl.concat([
        dep.select("MVT_ID_mvt", "airport", "rt", is_dep=pl.lit(True)),
        arr.select("MVT_ID_mvt", "airport", "rt", is_dep=pl.lit(False)),
    ]).sort(["airport", "rt"])
    ar = both.rolling(
        index_column="rt", period="60m", offset="-30m", group_by="airport"
    ).agg((~pl.col("is_dep")).sum().alias("n_arr_60m_around"))
    both = both.with_columns(ar["n_arr_60m_around"])
    out = out.join(
        both.filter(pl.col("is_dep")).select("MVT_ID_mvt", "n_arr_60m_around"),
        on="MVT_ID_mvt", how="left",
    )

    # inter-departure gap on the same runway + saturation run length
    gap = (
        dep.sort(["airport", "RUNWAY_mvt", "rt"])
        .with_columns(
            prev_gap=(pl.col("rt") - pl.col("rt").shift(1))
            .dt.total_seconds()
            .over(["airport", "RUNWAY_mvt"])
        )
        .with_columns(
            _brk=(pl.col("prev_gap").fill_null(9999) >= 120)
            .cum_sum()
            .over(["airport", "RUNWAY_mvt"])
        )
        .with_columns(
            sat_run=pl.int_range(pl.len()).over(["airport", "RUNWAY_mvt", "_brk"])
        )
        .select("MVT_ID_mvt", "prev_gap", "sat_run")
    )
    out = out.join(gap, on="MVT_ID_mvt", how="left")

    return out.with_columns(dep_pressure=pl.col("n_dep_30m_prev") / 30.0)


# --------------------------------------------------------------------------- 3
def _runway_config(frame: pl.DataFrame) -> pl.DataFrame:
    f = _runway_time(frame).select(
        "MVT_ID_mvt", "PHASE_mvt", "airport", "RUNWAY_mvt", "rt"
    ).with_columns(bin=pl.col("rt").dt.truncate("5m"))

    # active runway sets per 5-min bin
    def config(phase: str, cfg: str, ncol: str) -> pl.DataFrame:
        return (
            f.filter(pl.col("PHASE_mvt") == phase)
            .group_by("airport", "bin")
            .agg(
                pl.col("RUNWAY_mvt").drop_nulls().unique().sort().str.join("+").alias(cfg),
                pl.col("RUNWAY_mvt").n_unique().alias(ncol),
            )
        )

    depc = config("DEP", "dep_rwy_config", "n_active_dep_rwy")
    arrc = config("ARR", "arr_rwy_config", "n_active_arr_rwy")

    bins = f.select("airport", "bin").unique().sort(["airport", "bin"])
    bins = bins.join(depc, on=["airport", "bin"], how="left").join(
        arrc, on=["airport", "bin"], how="left"
    )
    bins = bins.with_columns(
        pl.col("dep_rwy_config").fill_null("?"),
        pl.col("arr_rwy_config").fill_null("?"),
        pl.col("n_active_dep_rwy").fill_null(0),
        pl.col("n_active_arr_rwy").fill_null(0),
    ).with_columns(
        _chg=(
            pl.col("dep_rwy_config") != pl.col("dep_rwy_config").shift(1)
        ).over("airport"),
    ).with_columns(
        _grp=pl.col("_chg").fill_null(True).cum_sum().over("airport")
    ).with_columns(
        mins_since_cfg_change=(
            (pl.col("bin") - pl.col("bin").first().over(["airport", "_grp"]))
            .dt.total_seconds() / 60.0
        )
    )

    dep = f.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "airport", "bin", "RUNWAY_mvt"
    )
    dep = dep.join(
        bins.select("airport", "bin", "dep_rwy_config", "arr_rwy_config",
                    "n_active_dep_rwy", "n_active_arr_rwy", "mins_since_cfg_change"),
        on=["airport", "bin"], how="left",
    )
    return dep.select(
        "MVT_ID_mvt", "dep_rwy_config", "arr_rwy_config",
        "n_active_dep_rwy", "n_active_arr_rwy", "mins_since_cfg_change",
    )


# --------------------------------------------------------------------------- 4
def _rotation_schedule(frame: pl.DataFrame) -> pl.DataFrame:
    links = build_stand_links(frame)

    arr = frame.filter(pl.col("PHASE_mvt") == "ARR").select(
        inbound_mvt_id="MVT_ID_mvt",
        inbound_sched="SCHED_TIME_UTC_mvt",
        inbound_inblock="BLOCK_TIME_UTC_mvt",
        inbound_actype="AIRCRAFT_TYPE_mvt",
        inbound_wk="WK_TBL_CAT_flt",
    )
    dep = frame.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
        "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "AOBT_3_flt",
        "WK_TBL_CAT_flt",
    )

    d = dep.join(links, on="MVT_ID_mvt", how="left").join(
        arr, on="inbound_mvt_id", how="left"
    )

    def secs(a, b):
        return (pl.col(a) - pl.col(b)).dt.total_seconds()

    return d.with_columns(
        # the offset we add back: taxi_hat = (T - SOBT) - d_hat
        sched_to_takeoff=secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        eobt_delay=secs("EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
        iobt_delay=secs("IOBT_flt", "SCHED_TIME_UTC_mvt"),
        lobt_delay=secs("LOBT_flt", "SCHED_TIME_UTC_mvt"),
        inbound_arr_delay=secs("inbound_inblock", "inbound_sched"),
        sched_ground=secs("SCHED_TIME_UTC_mvt", "inbound_sched"),
        actual_ground=pl.col("ground_time_sec"),
        # toggleable AOBT_3 block
        aobt3_taxi=secs("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        aobt3_vs_eobt=secs("AOBT_3_flt", "EOBT_1_flt"),
        wake_match=(pl.col("WK_TBL_CAT_flt") == pl.col("inbound_wk")),
    ).select(
        "MVT_ID_mvt", "inbound_mvt_id", "link_confidence", "type_match",
        "has_next", "bound_binding", "L_sec", "U_sec",
        "sched_to_takeoff", "eobt_delay", "iobt_delay", "lobt_delay",
        "inbound_arr_delay", "sched_ground", "actual_ground",
        "aobt3_taxi", "aobt3_vs_eobt", "wake_match", "inbound_actype",
    )


# ------------------------------------------------------------------ entrypoint
def build_features(
    df: pl.DataFrame, priors: dict[str, pl.DataFrame] | None = None
) -> pl.DataFrame:
    """One row per departure. `priors` from fit_priors() on the training split."""
    base = _runway_time(df).filter(pl.col("PHASE_mvt") == "DEP")

    feats = base.select(
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "stand_group",
        "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
        "FLIGHT_RULE_mvt", "AIRCRAFT_OPERATOR_flt",
        T="MVT_TIME_UTC_mvt", SOBT="SCHED_TIME_UTC_mvt",
    ).with_columns(
        hour=pl.col("T").dt.hour(),
        dow=pl.col("T").dt.weekday(),
        month=pl.col("T").dt.month(),
        doy=pl.col("T").dt.ordinal_day(),
        is_weekend=(pl.col("T").dt.weekday() >= 6).cast(pl.Int8),
        minute_of_day=pl.col("T").dt.hour() * 60 + pl.col("T").dt.minute(),
        sched_takeoff_offset=(pl.col("T") - pl.col("SOBT")).dt.total_seconds(),
    )

    feats = feats.join(_congestion(df), on="MVT_ID_mvt", how="left")
    feats = feats.join(_runway_config(df), on="MVT_ID_mvt", how="left")
    feats = feats.join(_rotation_schedule(df), on="MVT_ID_mvt", how="left")

    if priors is not None:
        feats = apply_priors(feats, priors)

    return feats
