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
  5. ATFM context       — EUROCONTROL slot-adherence/delay-cause/traffic data
                          (features/atfm.py), joined on both ADEP_mvt and
                          ADES_mvt per day; never touches a blanked column.
                          OFF BY DEFAULT (`atfm=True` to enable) — tested and
                          not adopted, see PROGRESS.md S20.
  6. realised queue     — pushback-anchored queue ahead, from AOBT_3_flt.
                          OFF BY DEFAULT (`queue=True` to enable) — tested and
                          not adopted, see PROGRESS.md S23.
  7. weather / de-icing — METAR (features/weather.py), joined on
                          (ADEP_mvt, hour bucket of takeoff time T); never
                          touches a blanked column. OFF BY DEFAULT
                          (`weather=True` to enable) pending the pre-registered
                          test — see tests/metar_weather_test.py.
  + calendar / categorical passthrough
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.atfm import load_atfm_daily, panel_icaos  # noqa: E402
from features.weather import load_metar_hourly  # noqa: E402
from features.encode import (  # noqa: E402,F401
    CAT_COLS,
    apply_priors,
    feature_matrix,
    fit_priors,
)
from link.stand_link import build_stand_links  # noqa: E402

# back-compat alias — earlier code imported the private name
_apply_priors = apply_priors

# Any bin-to-bin gap wider than this forces a runway-config "change" even if the
# config string is unchanged — guards mins_since_cfg_change against bridging a
# real data gap (e.g. ranking.parquet's Jan->Jul hole). Legitimate overnight
# lulls at these airports are well under a day; only a missing-month gap crosses it.
CFG_GAP_RESET_MIN = 24 * 60


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
        # `frame` isn't guaranteed continuous (ranking.parquet bundles Jan+Jul with
        # a 5-month hole between them) — without the gap check, a same-string
        # config either side of the hole reads as "no change" and
        # mins_since_cfg_change silently measures back across months of missing
        # data instead of resetting.
        _gap_min=(
            (pl.col("bin") - pl.col("bin").shift(1)).dt.total_seconds() / 60.0
        ).over("airport"),
    ).with_columns(
        _chg=(
            (pl.col("dep_rwy_config") != pl.col("dep_rwy_config").shift(1)).over("airport")
            | (pl.col("_gap_min") > CFG_GAP_RESET_MIN)
        ),
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


# --------------------------------------------------------------------------- 5
def _atfm_context(frame: pl.DataFrame) -> pl.DataFrame:
    """EUROCONTROL ATFM/traffic context, joined twice per departure: once on
    the departure airport (ADEP_mvt, shifts the whole day) and once on the
    destination (ADES_mvt, varies flight to flight -- this is the one that
    matters most, since a regulation protecting the destination is what
    actually holds a flight at its origin stand). See features/atfm.py.

    `ades_in_panel` flags whether ADES_mvt is one of the ~377 ICAOs
    airport_traffic covers at all -- a destination outside the panel gets no
    ATFM context (nulls, not a silently-imputed zero), and this flag is what
    lets the model tell "genuinely quiet destination" apart from "no data".
    """
    atfm = load_atfm_daily()
    panel = panel_icaos()
    ctx_cols = [c for c in atfm.columns if c not in ("APT_ICAO", "date")]

    dep = frame.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt",
        date=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    )
    dep = dep.join(
        atfm.rename({c: f"dep_atfm_{c}" for c in ctx_cols}),
        left_on=["ADEP_mvt", "date"], right_on=["APT_ICAO", "date"], how="left",
    ).join(
        atfm.rename({c: f"des_atfm_{c}" for c in ctx_cols}),
        left_on=["ADES_mvt", "date"], right_on=["APT_ICAO", "date"], how="left",
    ).with_columns(
        ades_in_panel=pl.col("ADES_mvt").is_in(panel).cast(pl.Int8),
    )
    return dep.select(
        "MVT_ID_mvt", "ades_in_panel",
        *[f"dep_atfm_{c}" for c in ctx_cols],
        *[f"des_atfm_{c}" for c in ctx_cols],
    )


# --------------------------------------------------------------------------- 6
def _realised_queue(frame: pl.DataFrame) -> pl.DataFrame:
    """Departure queue AHEAD of this flight at the moment it pushed back.

    Family 2's congestion counts are TAKEOFF-anchored: they count movements
    around T. That is a different quantity from "who was already on the
    ground ahead of me when I started taxiing", which is the actual physical
    driver of taxi-out delay.

    CAUSALITY (the thing that is easy to get subtly wrong):
      * anchored on this flight's own pushback `a_i = AOBT_3_flt`, NOT on T.
        Anchoring on T would, for a long-taxi flight, place the window
        entirely AFTER its pushback -- counting aircraft behind it and
        calling them a queue ahead of it. That correlates with taxi
        in-sample and need not transfer.
      * only aircraft with `AOBT_3_j < a_i` are counted, strictly. Never
        after. Self is excluded automatically by the strict inequality.
      * neighbours' pushback uses AOBT_3_flt, never BLOCK_TIME_UTC_mvt --
        the latter is nulled on every DEP row by `blind()` before any
        feature code runs (CLAUDE.md's hard invariant), so it is structurally
        unavailable here rather than merely avoided by convention.

    AOBT_3_flt is populated for ~98.5% of ranking DEP rows, so this is
    deployable as-is; rows without it get nulls.

    Columns (all per departure, keyed MVT_ID_mvt):
      q_ahead          aircraft pushed back but not yet airborne at a_i
      q_ahead_rwy      same, restricted to this flight's departure runway
      q_push_15m       pushbacks in the 15 min before a_i
      q_ahead_mean_wait  mean seconds those ahead had already been taxiing
    """
    import numpy as np

    dep = (
        _runway_time(frame)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("MVT_ID_mvt", "airport", "RUNWAY_mvt",
                a=pl.col("AOBT_3_flt").dt.epoch("s"),
                t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"))
    )

    def _counts(g: pl.DataFrame, anchor: np.ndarray) -> tuple:
        """(active, mean_wait, pushed_15m) for each anchor time, using only
        rows of `g` whose pushback strictly precedes that anchor."""
        a = g["a"].to_numpy().astype("float64")
        t = g["t"].to_numpy().astype("float64")
        # Only valid taxi intervals may be COUNTED in anyone's queue. Rows
        # with takeoff at or before pushback (known data corruption, see
        # reports/lfpg_investigation.md) are not intervals at all, and
        # leaving them in makes `started - ended` go negative, which the
        # clamp below then hides -- and hides asymmetrically between the
        # full and per-runway populations, producing q_ahead_rwy > q_ahead.
        ok = ~np.isnan(a) & ~np.isnan(t) & (t > a)
        a, t = a[ok], t[ok]
        if a.size == 0:
            z = np.zeros(anchor.shape)
            return z, np.full(anchor.shape, np.nan), z
        starts = np.sort(a)
        cum_start = np.concatenate([[0.0], np.cumsum(starts)])
        order = np.argsort(t)
        ends, a_by_end = t[order], a[order]
        cum_end = np.concatenate([[0.0], np.cumsum(a_by_end)])
        # strict: started BEFORE the anchor; finished at or before the anchor
        ns = np.searchsorted(starts, anchor, side="left")
        ne = np.searchsorted(ends, anchor, side="right")
        active = np.maximum(ns - ne, 0)
        sum_a = cum_start[ns] - cum_end[ne]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_wait = np.where(active > 0, anchor - sum_a / np.maximum(active, 1),
                                 np.nan)
        n15 = ns - np.searchsorted(starts, anchor - 900.0, side="left")
        return active.astype("float64"), mean_wait, np.maximum(n15, 0).astype("float64")

    out = []
    for (ap,), g in dep.group_by(["airport"], maintain_order=True):
        anchor = g["a"].to_numpy().astype("float64")
        safe = np.where(np.isnan(anchor), -np.inf, anchor)
        act, wait, n15 = _counts(g, safe)
        rwy_act = np.full(anchor.shape, np.nan)
        idx = np.arange(g.height)
        for (_rw,), gr in g.group_by(["RUNWAY_mvt"], maintain_order=True):
            sel = idx[(g["RUNWAY_mvt"] == _rw).to_numpy()]
            ra, _, _ = _counts(gr, safe[sel])
            rwy_act[sel] = ra
        nan = np.isnan(anchor)
        act[nan] = np.nan
        n15[nan] = np.nan
        rwy_act[nan] = np.nan
        out.append(g.select("MVT_ID_mvt").with_columns(
            q_ahead=pl.Series(act),
            q_ahead_rwy=pl.Series(rwy_act),
            q_push_15m=pl.Series(n15),
            q_ahead_mean_wait=pl.Series(wait),
        ))
    # emit nulls, not NaN: polars treats them differently (NaN propagates
    # through mean/std, so NaN silently poisons any downstream summary)
    return pl.concat(out, how="vertical_relaxed").with_columns(
        pl.col("q_ahead", "q_ahead_rwy", "q_push_15m", "q_ahead_mean_wait")
        .fill_nan(None)
    ).with_columns(
        # A single aircraft holding for many hours stays "active" and drags
        # this mean into a region training barely samples: >3600s is 0.001%
        # of 2025-03 rows but 0.142% of 2026 ranking rows (p99.9 1460s vs
        # 5707s). Trees are order-invariant so the tail itself is harmless,
        # but 140x more mass where the model has no training support is not.
        # Clipping maps it onto a well-sampled boundary instead.
        pl.col("q_ahead_mean_wait").clip(0, 3600)
    )


# --------------------------------------------------------------------------- 7
def _weather_context(frame: pl.DataFrame) -> pl.DataFrame:
    """METAR weather/de-icing context, joined on (ADEP_mvt, hour bucket of T)
    -- takeoff time, never blanked, same anchor family 2 uses. See
    features/weather.py for the mechanism and column definitions.

    `ceiling_ft` is null whenever there's no BKN/OVC/VV layer reported (a
    real "unlimited ceiling" condition, not missing data) -- `has_ceiling`
    makes that explicit for the model instead of leaving a silent null that
    looks the same as a genuinely missing METAR hour.
    """
    wx = load_metar_hourly()
    wx_cols = [c for c in wx.columns if c not in ("station", "hour")]

    dep = frame.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "ADEP_mvt",
        hour=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h"),
    )
    dep = dep.join(
        wx, left_on=["ADEP_mvt", "hour"], right_on=["station", "hour"], how="left",
    ).with_columns(
        has_ceiling=pl.col("ceiling_ft").is_not_null().cast(pl.Int8),
        ceiling_ft=pl.col("ceiling_ft").fill_null(99999.0),
    )
    return dep.select("MVT_ID_mvt", "has_ceiling", *wx_cols)


# ------------------------------------------------------------------ entrypoint
def build_features(
    df: pl.DataFrame, priors: dict[str, pl.DataFrame] | None = None,
    atfm: bool = False, queue: bool = False, weather: bool = False,
) -> pl.DataFrame:
    """One row per departure. `priors` from fit_priors() on the training split.

    `queue` enables family 6 (realised pushback-anchored queue). Off by
    default: tested and NOT adopted (PROGRESS.md S23 -- +1.81s, 95% CI
    [-3.61, +8.64], July regressed, and no gain gradient with queue depth,
    which was the pre-registered mechanism test). Implementation is verified
    causal (tests/queue_causality_check.py) and kept switchable because the
    code is correct; it is the hypothesis that failed, not the build.

    `atfm` enables family 5 (EUROCONTROL ATFM context). Off by default:
    tested and NOT adopted (PROGRESS.md S20 -- paired delta -0.38s, 95% CI
    [-7.98, +7.98], July regressed). It is kept switchable rather than
    deleted because the ingest path is correct and the bound that killed it
    is about granularity, not correctness: the data is published per
    (airport, day), and 98.6% of this model's residual variance lives WITHIN
    an airport-day, capping any such feature at ~2s. Requires the CSVs under
    external-data/ (src/ingest/fetch_atfm_data.py).

    `weather` enables family 7 (METAR de-icing/low-vis context). Off by
    default pending the pre-registered test (tests/metar_weather_test.py,
    see reports/eval/). Unlike ATFM, weather is published hourly, not
    daily -- the (airport, day, hour) oracle ceiling is ~10.82s vs ATFM's
    ~2.07s daily cap (tests/metar_ceiling.py, PROGRESS.md S20/S24). Requires
    the CSVs under external-data/metar/ (src/ingest/fetch_metar_data.py).
    """
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
    if queue:
        feats = feats.join(_realised_queue(df), on="MVT_ID_mvt", how="left")
    if atfm:
        feats = feats.join(_atfm_context(df), on="MVT_ID_mvt", how="left")
    if weather:
        feats = feats.join(_weather_context(df), on="MVT_ID_mvt", how="left")

    if priors is not None:
        feats = apply_priors(feats, priors)

    return feats
