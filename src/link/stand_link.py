"""Stage 1 — stand linking + interval bounds.

For every departure, pair it with the inbound aircraft that last occupied its
stand, using only split-blind information (arrival in-block times, departure
takeoff times). Emit the hard interval bounds on taxi-out implied by stand
occupancy (CLAUDE.md "core insight" #3):

    off-block in [A_own, A_next]
      A_own  = in-block of the arrival that brought this aircraft
      A_next = in-block of the next arrival to occupy the stand
    => L = max(0, T - A_next) <= taxi <= T - A_own = U     (T = takeoff)

This module NEVER reads a departure off-block time. It works on one dataframe in
ranking schema (arrivals + departures together) and is safe to run on train and
ranking alike.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

# Departure off-block is unknown; order departures among arrivals by an estimated
# off-block = takeoff minus a nominal taxi. Only needs to be roughly right — stand
# inter-arrival gaps are typically >1 h.
NOMINAL_TAXI_SEC = 720

# Plausible turnaround window between inbound in-block and outbound off-block.
GROUND_MIN_SEC = 10 * 60
GROUND_MAX_SEC = 16 * 3600


def _prep(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        airport=pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt")),
        actype=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
    )


def build_stand_links(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per departure with linking + bounds columns.

    Output columns (keyed on MVT_ID_mvt):
      inbound_mvt_id, A_own, A_next, L_sec, U_sec, interval_sec,
      ground_time_sec, type_match, has_next, bound_binding, link_confidence
    """
    d = _prep(df)

    arr = (
        d.filter(
            (pl.col("PHASE_mvt") == "ARR")
            & pl.col("STAND_mvt").is_not_null()
            & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
        )
        .select(
            "airport",
            "STAND_mvt",
            "actype",
            a_mvt_id="MVT_ID_mvt",
            a_inblock="BLOCK_TIME_UTC_mvt",
        )
        .sort("a_inblock")
    )

    dep = (
        d.filter(
            (pl.col("PHASE_mvt") == "DEP")
            & pl.col("STAND_mvt").is_not_null()
            & pl.col("MVT_TIME_UTC_mvt").is_not_null()
        )
        .select(
            "MVT_ID_mvt",
            "airport",
            "STAND_mvt",
            "actype",
            T="MVT_TIME_UTC_mvt",
        )
        .with_columns(
            est_offblock=pl.col("T") - pl.duration(seconds=NOMINAL_TAXI_SEC)
        )
        .sort("est_offblock")
    )

    # --- A_own, same aircraft type: last matching arrival before est_offblock ---
    own_type = (
        arr.select(
            "airport",
            "STAND_mvt",
            "actype",
            own_type_mvt_id="a_mvt_id",
            A_own_type="a_inblock",
        )
        .sort("A_own_type")
    )
    dep = dep.join_asof(
        own_type,
        left_on="est_offblock",
        right_on="A_own_type",
        by=["airport", "STAND_mvt", "actype"],
        strategy="backward",
    )

    # --- A_own, any type: fallback when type is missing / doesn't match ---
    own_any = arr.select(
        "airport", "STAND_mvt", own_any_mvt_id="a_mvt_id", A_own_any="a_inblock"
    ).sort("A_own_any")
    dep = dep.sort("est_offblock").join_asof(
        own_any,
        left_on="est_offblock",
        right_on="A_own_any",
        by=["airport", "STAND_mvt"],
        strategy="backward",
    )

    # --- A_next, any type: first arrival to occupy the stand after est_offblock ---
    nxt_any = arr.select(
        "airport", "STAND_mvt", A_next="a_inblock"
    ).sort("A_next")
    dep = dep.sort("est_offblock").join_asof(
        nxt_any,
        left_on="est_offblock",
        right_on="A_next",
        by=["airport", "STAND_mvt"],
        strategy="forward",
    )

    out = dep.with_columns(
        type_match=pl.col("A_own_type").is_not_null(),
        A_own=pl.coalesce("A_own_type", "A_own_any"),
        inbound_mvt_id=pl.coalesce("own_type_mvt_id", "own_any_mvt_id"),
    ).with_columns(
        U_sec=(pl.col("T") - pl.col("A_own")).dt.total_seconds(),
        raw_L_sec=(pl.col("T") - pl.col("A_next")).dt.total_seconds(),
        ground_time_sec=(pl.col("est_offblock") - pl.col("A_own")).dt.total_seconds(),
        has_next=pl.col("A_next").is_not_null(),
    ).with_columns(
        L_sec=pl.max_horizontal(pl.lit(0), pl.col("raw_L_sec")).fill_null(0),
    ).with_columns(
        interval_sec=pl.col("U_sec") - pl.col("L_sec"),
        bound_binding=pl.col("raw_L_sec") > 0,
        plausible_ground=pl.col("ground_time_sec").is_between(
            GROUND_MIN_SEC, GROUND_MAX_SEC
        ),
    ).with_columns(
        link_confidence=pl.when(pl.col("A_own").is_null())
        .then(pl.lit("none"))
        .when(
            pl.col("type_match")
            & pl.col("plausible_ground")
            & pl.col("has_next")
        )
        .then(pl.lit("high"))
        .when(pl.col("type_match") | pl.col("plausible_ground"))
        .then(pl.lit("med"))
        .otherwise(pl.lit("low"))
    )

    return out.select(
        "MVT_ID_mvt",
        "airport",
        "STAND_mvt",
        "inbound_mvt_id",
        "A_own",
        "A_next",
        "L_sec",
        "U_sec",
        "interval_sec",
        "ground_time_sec",
        "type_match",
        "has_next",
        "bound_binding",
        "plausible_ground",
        "link_confidence",
    )
