"""Stand-appearance ADS-B pushback detector -> per-departure parquet.

Split-blind: reads only MVT_TIME_UTC_mvt (takeoff) and STAND_mvt from the
movements, both unblanked for ranking DEP rows. Never touches BLOCK_TIME.

Per (airport, hex) day-track from data/external/adsb/day=*/ (gs < 40 kt =
surface, independent of the altitude encoding):
  1. Split into surface runs at gaps > 20 min; a run followed within 120 s by
     gs >= 40 is a candidate departure, its last surface sample the takeoff.
  2. Match candidates 1:1 to DEP movements of the same airport and UTC day,
     takeoff within +-180 s of MVT_TIME, greedy, preferring runs that come
     within 100 m of the movement's own stand, then smallest time gap.
  3. Pushback from points within 100 m of own stand:
       dwell  : a stationary (gs <= 1) at-stand sample -> first sample after
                the last one
       appear : run's first sample is at the stand -> that sample
                (transponder switched on at pushback)
       pass   : reaches the stand only after starting elsewhere -> first
                at-stand sample (low quality; validated as useless)
     No at-stand sample -> matched but censored (tier null).

Output cache/adsb_pushback/day=YYYY-MM-DD.parquet, one row per DEP movement
of that day at every challenge airport:
  MVT_ID_mvt, airport, adsb_day_coverage (airport had >= 20 aircraft with
  surface points that day), adsb_matched, adsb_tier, adsb_pushback_ts (epoch s),
  adsb_takeoff_gap_s (ADS-B takeoff - MVT_TIME), adsb_n_pts (samples in run),
  and for every matched run (used by the partial-track estimate):
  adsb_first_ts (first surface sample of the run), adsb_first_gs (its speed, kt),
  adsb_first_own_m (its distance to the own stand), adsb_min_own_m (closest
  approach to the own stand over the run); the two distances are null when
  the stand has no coordinates.
  Pushback quality, for rows with a tier: adsb_pb_gap_s (for dwell, the
  sampling gap between the last stationary at-stand sample and the next
  sample; for appear/pass, the gap between the pushback sample and the next
  sample), adsb_pb_dist_m (pushback sample's distance to the own stand),
  adsb_pb_gs (its ground speed, kt).

Run:  .venv/Scripts/python.exe src/link/adsb_pushback.py [--v3] [YYYY-MM-DD ...]

--v3 adds the inferred stands (src/ingest/stand_infer.py) and the fallback
match for unmatched departures (tiers fb_dwell / fb_appear, flag
adsb_fallback), writing to cache/adsb_pushback_v3/ (PROGRESS.md §56). The
default output is unchanged. --v4 (§57) also lets matched rows without an
appear/dwell tier take a fallback event that precedes their run, writing to
cache/adsb_pushback_v4/. --v5 (§60) adds, before the fallback, a second match pass for
departures still unmatched: surface runs ending on a runway near takeoff without a
received climb-out (flag adsb_rwy_end), writing to cache/adsb_pushback_v5/.
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.normalise_adsb import available_days, load_day  # noqa: E402
from src.ingest.stands import attach_stand_coords, load_stands, stand_positions  # noqa: E402

RAW = ROOT / "data" / "raw"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
OUT = ROOT / "cache" / "adsb_pushback"
OUT_V3 = ROOT / "cache" / "adsb_pushback_v3"   # --v3: inferred stands + fallback (§56)
OUT_V4 = ROOT / "cache" / "adsb_pushback_v4"   # --v4: v3 + fallback for matched, untiered rows (§57)
OUT_V5 = ROOT / "cache" / "adsb_pushback_v5"   # --v5: v4 + runway-ending runs as matches (§60)
RUNWAYS = ROOT / "data" / "external" / "runways.csv"

SURFACE_GS = 40.0
STATIONARY_GS = 1.0
GAP_S = 1200.0
TAKEOFF_LOOKAHEAD_S = 120.0
MATCH_TOL_S = 180.0
R_STAND = 100.0
MIN_COVERAGE_AIRCRAFT = 20

SCHEMA = {"MVT_ID_mvt": pl.Int64, "airport": pl.Utf8, "adsb_day_coverage": pl.Boolean,
          "adsb_matched": pl.Boolean, "adsb_tier": pl.Utf8, "adsb_pushback_ts": pl.Float64,
          "adsb_takeoff_gap_s": pl.Float64, "adsb_n_pts": pl.Int64,
          "adsb_first_ts": pl.Float64, "adsb_first_gs": pl.Float64,
          "adsb_first_own_m": pl.Float64, "adsb_min_own_m": pl.Float64,
          "adsb_pb_gap_s": pl.Float64, "adsb_pb_dist_m": pl.Float64, "adsb_pb_gs": pl.Float64}


def to_xy(lat, lon, lat0):
    return np.c_[np.asarray(lon) * 111320.0 * np.cos(np.radians(lat0)),
                 np.asarray(lat) * 110540.0]


def surface_runs(ts, gs):
    """(i, j) index ranges of surface runs ending in a takeoff."""
    n = len(ts)
    surf = gs < SURFACE_GS
    i = 0
    while i < n:
        if not surf[i]:
            i += 1
            continue
        j = i + 1
        while j < n and surf[j] and ts[j] - ts[j - 1] <= GAP_S:
            j += 1
        if j < n and not surf[j] and ts[j] - ts[j - 1] <= TAKEOFF_LOOKAHEAD_S:
            yield i, j
        i = j


def pushback(ts, gs, d_own):
    """(pushback_ts, tier, k) for one surface run; tier None = censored.

    k indexes the sample the quality fields describe: the last stationary
    at-stand sample for dwell, the pushback sample itself otherwise."""
    at = np.flatnonzero(d_own < R_STAND)
    if len(at) == 0:
        return np.nan, None, None
    stat = at[gs[at] <= STATIONARY_GS]
    if len(stat):
        k = stat[-1]
        return (ts[k + 1] if k + 1 < len(ts) else ts[k]), "dwell", int(k)
    if at[0] == 0:
        return ts[0], "appear", 0
    return ts[at[0]], "pass", int(at[0])


def departures(day: dt.date) -> pl.DataFrame:
    """DEP movements whose takeoff falls on `day`: id, airport, stand, takeoff only."""
    src = RANKING if day.year >= 2026 else next(RAW.glob(f"training_{day:%Y-%m}-01_*.parquet"))
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt")
            .filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == day)
            .collect()
            .with_columns(MVT_ID_mvt=pl.col("MVT_ID_mvt").cast(pl.Int64),
                          mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0))


def detect_day(day: dt.date, pos: pl.DataFrame, fallback: bool = False,
               fb_matched: bool = False, rwy: pl.DataFrame | None = None) -> pl.DataFrame:
    adsb = load_day(day)
    mv = attach_stand_coords(departures(day), pos)
    cov = dict(adsb.filter(pl.col("gs") < SURFACE_GS).group_by("airport")
               .agg(pl.col("hex").n_unique()).iter_rows())
    out = []
    for (ap,), m_ap in mv.group_by("ADEP_mvt"):
        recs = m_ap.to_dicts()
        base = {"airport": ap, "adsb_day_coverage": cov.get(ap, 0) >= MIN_COVERAGE_AIRCRAFT}
        res = {r["MVT_ID_mvt"]: dict(MVT_ID_mvt=r["MVT_ID_mvt"], **base, adsb_matched=False,
                                     adsb_tier=None, adsb_pushback_ts=None,
                                     adsb_takeoff_gap_s=None, adsb_n_pts=None,
                                     adsb_first_ts=None, adsb_first_gs=None,
                                     adsb_first_own_m=None, adsb_min_own_m=None,
                                     adsb_pb_gap_s=None, adsb_pb_dist_m=None,
                                     adsb_pb_gs=None) for r in recs}
        sub = adsb.filter(pl.col("airport") == ap)
        ap_pos = pos.filter(pl.col("airport") == ap)
        if sub.height and ap_pos.height:
            lat0 = ap_pos["lat"].mean()
            runs = []
            for _, g in sub.group_by("hex", maintain_order=True):
                ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
                gs = np.where(np.isnan(gs), SURFACE_GS, gs)  # unknown speed: not surface
                xy = to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0)
                for i, j in surface_runs(ts, gs):
                    runs.append((ts[i:j], gs[i:j], xy[i:j], ts[j - 1]))
            if runs:
                run_to = np.array([r[3] for r in runs])
                order = np.argsort(run_to)
                run_sorted = run_to[order]
                pairs = []
                for r in recs:
                    lo, hi = np.searchsorted(run_sorted, [r["mvt_ts"] - MATCH_TOL_S,
                                                          r["mvt_ts"] + MATCH_TOL_S])
                    sxy = (None if r["stand_lat"] is None
                           else to_xy([r["stand_lat"]], [r["stand_lon"]], lat0)[0])
                    for k in range(lo, hi):
                        ri = int(order[k])
                        d_own = None if sxy is None else np.hypot(*(runs[ri][2] - sxy).T)
                        visits = d_own is not None and bool((d_own < R_STAND).any())
                        pairs.append((not visits, abs(run_to[ri] - r["mvt_ts"]),
                                      r["MVT_ID_mvt"], ri, d_own, r["mvt_ts"]))
                pairs.sort(key=lambda p: (p[0], p[1]))
                used_m, used_r = set(), set()
                for _, _, mid, ri, d_own, mts in pairs:
                    if mid in used_m or ri in used_r:
                        continue
                    used_m.add(mid); used_r.add(ri)
                    ts, gs, _, to = runs[ri]
                    row = res[mid]
                    row.update(adsb_matched=True, adsb_takeoff_gap_s=float(to - mts),
                               adsb_n_pts=len(ts), adsb_first_ts=float(ts[0]),
                               adsb_first_gs=float(gs[0]))
                    if d_own is not None:
                        row.update(adsb_first_own_m=float(d_own[0]),
                                   adsb_min_own_m=float(d_own.min()))
                        pb, tier, k = pushback(ts, gs, d_own)
                        if tier is not None:
                            row.update(adsb_tier=tier, adsb_pushback_ts=float(pb),
                                       adsb_pb_gap_s=float(ts[k + 1] - ts[k]) if k + 1 < len(ts) else None,
                                       adsb_pb_dist_m=float(d_own[k]), adsb_pb_gs=float(gs[k]))
        if rwy is not None:
            for row in res.values():
                row["adsb_rwy_end"] = False
            if sub.height and ap_pos.height:
                _rwy_end_pass(recs, res, sub, lat0, rwy.filter(pl.col("airport") == ap))
        if fallback:
            for row in res.values():
                row["adsb_fallback"] = False
            if sub.height and ap_pos.height:
                _fallback(recs, res, sub, lat0, fb_matched)
        out.extend(res.values())
    schema = SCHEMA_V5 if rwy is not None else SCHEMA_V3 if fallback else SCHEMA
    return pl.DataFrame(out, schema=schema)


# ---- v5 runway-ending runs (PROGRESS.md §60). The takeoff match needs gs >= 40 within 120 s
# of the last surface sample; many tracks lose the aircraft at the runway before the climb-out
# is received. A departure still unmatched after that pass may take a surface run (split at
# gaps > GAP_S, not itself a takeoff run) whose last sample is within RWY_END_M of a runway
# centreline segment and in [T - RWY_END_BEFORE_S, T + RWY_END_AFTER_S]. Preference: runs that
# visit the own stand, then |last - (T - 30 s)|; 1:1 greedy. The matched row then gets the
# normal pushback tiering and first-sighting fields, flagged adsb_rwy_end.
RWY_END_M = 200.0
RWY_END_BEFORE_S = 300.0
RWY_END_AFTER_S = 30.0
SCHEMA_V5 = {**SCHEMA, "adsb_fallback": pl.Boolean, "adsb_rwy_end": pl.Boolean}


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
    return float(np.hypot(*(p - (a + t * ab))))


def _rwy_end_pass(recs: list[dict], res: dict, sub: pl.DataFrame, lat0: float, rwy: pl.DataFrame) -> None:
    open_recs = [r for r in recs if not res[r["MVT_ID_mvt"]]["adsb_matched"]]
    if not open_recs or rwy.height == 0:
        return
    by_name = {r["runway"]: r for r in rwy.iter_rows(named=True)}
    segs = []
    for r in by_name.values():
        o = by_name.get(r["opposite"])
        if o is not None:
            segs.append((to_xy([r["lat"]], [r["lon"]], lat0)[0], to_xy([o["lat"]], [o["lon"]], lat0)[0]))
    runs = []
    for _, g in sub.group_by("hex", maintain_order=True):
        ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
        gs = np.where(np.isnan(gs), SURFACE_GS, gs)
        xy = to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0)
        n, i = len(ts), 0
        while i < n:
            if gs[i] >= SURFACE_GS:
                i += 1
                continue
            j = i + 1
            while j < n and gs[j] < SURFACE_GS and ts[j] - ts[j - 1] <= GAP_S:
                j += 1
            takeoff = j < n and gs[j] >= SURFACE_GS and ts[j] - ts[j - 1] <= TAKEOFF_LOOKAHEAD_S
            if not takeoff and min(_seg_dist(xy[j - 1], a, b) for a, b in segs) <= RWY_END_M:
                runs.append((ts[i:j], gs[i:j], xy[i:j], ts[j - 1]))
            i = j
    if not runs:
        return
    last = np.array([r[3] for r in runs])
    pairs = []
    for r in open_recs:
        t = r["mvt_ts"]
        sxy = None if r["stand_lat"] is None else to_xy([r["stand_lat"]], [r["stand_lon"]], lat0)[0]
        for ri in np.flatnonzero((last >= t - RWY_END_BEFORE_S) & (last <= t + RWY_END_AFTER_S)):
            d_own = None if sxy is None else np.hypot(*(runs[ri][2] - sxy).T)
            visits = d_own is not None and bool((d_own < R_STAND).any())
            pairs.append((not visits, abs(last[ri] - (t - 30.0)), r["MVT_ID_mvt"], int(ri), d_own, t))
    pairs.sort(key=lambda p: (p[0], p[1]))
    used_m, used_r = set(), set()
    for _, _, mid, ri, d_own, mts in pairs:
        if mid in used_m or ri in used_r:
            continue
        used_m.add(mid); used_r.add(ri)
        ts, gs, _, to = runs[ri]
        row = res[mid]
        row.update(adsb_matched=True, adsb_rwy_end=True, adsb_takeoff_gap_s=float(to - mts),
                   adsb_n_pts=len(ts), adsb_first_ts=float(ts[0]), adsb_first_gs=float(gs[0]))
        if d_own is not None:
            row.update(adsb_first_own_m=float(d_own[0]), adsb_min_own_m=float(d_own.min()))
            pb, tier, k = pushback(ts, gs, d_own)
            if tier is not None:
                row.update(adsb_tier=tier, adsb_pushback_ts=float(pb),
                           adsb_pb_gap_s=float(ts[k + 1] - ts[k]) if k + 1 < len(ts) else None,
                           adsb_pb_dist_m=float(d_own[k]), adsb_pb_gs=float(gs[k]))


# ---- v3 fallback (PROGRESS.md §56): unmatched departures whose aircraft is seen at the own
# stand. The takeoff-time match above needs the takeoff roll in the track; many tracks lose
# the aircraft before it. Here a departure with no matched run takes the most recent
# at-stand event in [T - FB_WINDOW_S, T - FB_MIN_TAXI_S] from any aircraft:
#   fb_dwell : last stationary (gs <= 1) at-stand sample k, pushback = next sample (as dwell)
#   fb_appear: an aircraft's first sample after a > GAP_S silence is at the stand with
#              gs <= FB_APPEAR_GS (transponder on at pushback, as appear)
# The aircraft must then be seen away from the stand or moving (gs >= FB_MOVE_GS) by
# T + 60 s, and not stationary at the stand again before T. Events within FB_EVENT_TOL_S
# of a pushback already detected at the same stand are skipped, and pushback must follow
# the previous departure from the same stand (takeoff - FB_PREV_SLACK_S). Departures are
# served in takeoff order; an event is used once.
FB_WINDOW_S = 5400.0
FB_MIN_TAXI_S = 120.0
FB_APPEAR_GS = 5.0
FB_MOVE_GS = 3.0
FB_EVENT_TOL_S = 120.0
FB_PREV_SLACK_S = 300.0
SCHEMA_V3 = {**SCHEMA, "adsb_fallback": pl.Boolean}


def _events(sub: pl.DataFrame, sxy: np.ndarray, lat0: float, t: float) -> list[tuple]:
    """Candidate (pushback_ts, tier, hex, pb_dist_m, pb_gap_s, pb_gs) at one stand before takeoff t."""
    w = sub.filter((pl.col("ts") >= t - FB_WINDOW_S - GAP_S) & (pl.col("ts") <= t + 60))
    if w.height == 0:
        return []
    xy = to_xy(w["lat"].to_numpy(), w["lon"].to_numpy(), lat0)
    d_all = np.hypot(*(xy - sxy).T)
    near = d_all < R_STAND
    if not near.any():
        return []
    evs = []
    hexes, ts_all = w["hex"].to_numpy(), w["ts"].to_numpy()
    gs_all = np.nan_to_num(w["gs"].to_numpy(), nan=SURFACE_GS)
    for h in np.unique(hexes[near]):
        m = hexes == h
        ts, gs, d = ts_all[m], gs_all[m], d_all[m]
        o = np.argsort(ts)
        ts, gs, d = ts[o], gs[o], d[o]
        at = d < R_STAND
        win = (ts >= t - FB_WINDOW_S) & (ts <= t - FB_MIN_TAXI_S)
        stat = np.flatnonzero(at & (gs <= STATIONARY_GS) & win)
        cand = None
        if len(stat):
            k = stat[-1]
            if k + 1 < len(ts) and ts[k + 1] <= t - FB_MIN_TAXI_S:
                cand = (ts[k + 1], "fb_dwell", k, float(ts[k + 1] - ts[k]))
        else:
            starts = np.flatnonzero(np.r_[True, np.diff(ts) > GAP_S])
            ok = [i for i in starts if at[i] and win[i] and gs[i] <= FB_APPEAR_GS]
            if ok:
                k = ok[-1]
                cand = (ts[k], "fb_appear", k, float(ts[k + 1] - ts[k]) if k + 1 < len(ts) else np.nan)
        if cand is None:
            continue
        pb, tier, k, gap = cand
        after = ts > pb
        if not ((after & ((d >= R_STAND) | (gs >= FB_MOVE_GS))).any()):
            continue                                   # never seen leaving
        if (after & at & (gs <= STATIONARY_GS) & (ts <= t)).any() and tier == "fb_appear":
            continue                                   # parked again: not a departure
        evs.append((float(pb), tier, str(h), float(d[k]), gap, float(gs[k])))
    return evs


def _fallback(recs: list[dict], res: dict, sub: pl.DataFrame, lat0: float, fb_matched: bool = False) -> None:
    """fb_matched (v4): also serve matched rows without an appear/dwell tier, taking only events
    before the matched run's first sample (the stand visit sits in an earlier run of the track)."""
    by_stand: dict = {}
    for r in recs:
        if r["stand_lat"] is not None:
            by_stand.setdefault((r["stand_lat"], r["stand_lon"]), []).append(r)
    for (slat, slon), rs in by_stand.items():
        rs = sorted(rs, key=lambda r: r["mvt_ts"])
        sxy = to_xy([slat], [slon], lat0)[0]
        known = [res[r["MVT_ID_mvt"]]["adsb_pushback_ts"] for r in rs
                 if res[r["MVT_ID_mvt"]]["adsb_pushback_ts"] is not None]
        used: set = set()
        prev_t = -np.inf
        for r in rs:
            row = res[r["MVT_ID_mvt"]]
            t = r["mvt_ts"]
            open_ = not row["adsb_matched"] or (fb_matched and row["adsb_tier"] not in ("appear", "dwell"))
            if open_:
                first = row["adsb_first_ts"] if row["adsb_matched"] else np.inf
                evs = [e for e in _events(sub, sxy, lat0, t)
                       if (e[2], e[0]) not in used and e[0] < first
                       and e[0] > prev_t - FB_PREV_SLACK_S
                       and all(abs(e[0] - k) > FB_EVENT_TOL_S for k in known)]
                if evs:
                    pb, tier, h, dist, gap, gs = max(evs, key=lambda e: e[0])
                    used.add((h, pb))
                    row.update(adsb_tier=tier, adsb_pushback_ts=pb, adsb_fallback=True,
                               adsb_pb_dist_m=dist, adsb_pb_gap_s=None if np.isnan(gap) else gap,
                               adsb_pb_gs=gs)
            prev_t = t


def main(days: list[dt.date] | None = None, v3: bool = False, v4: bool = False,
         v5: bool = False) -> None:
    v4 = v4 or v5
    v3 = v3 or v4
    rwy = pl.read_csv(RUNWAYS) if v5 else None
    pos = stand_positions(load_stands())
    out_dir = OUT
    if v3:
        from src.ingest.stand_infer import load_inferred
        inf = load_inferred().join(pos.select("airport", "key"), on=["airport", "key"], how="anti")
        pos = pl.concat([pos, inf])
        out_dir = OUT_V5 if v5 else OUT_V4 if v4 else OUT_V3
    out_dir.mkdir(parents=True, exist_ok=True)
    for day in days or available_days():
        df = detect_day(day, pos, fallback=v3, fb_matched=v4, rwy=rwy)
        df.write_parquet(out_dir / f"day={day.isoformat()}.parquet")
        n_tier = df.filter(pl.col("adsb_tier").is_in(["appear", "dwell"])).height
        print(f"{day}: {df.height:,} DEP, matched {df['adsb_matched'].sum():,}, "
              f"appear+dwell {n_tier:,} ({n_tier / df.height * 100:.1f}%)")


def load_all() -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("day=*.parquet"))])


if __name__ == "__main__":
    days = [dt.date.fromisoformat(a) for a in sys.argv[1:] if not a.startswith("--")]
    main(days or None, v3="--v3" in sys.argv, v4="--v4" in sys.argv, v5="--v5" in sys.argv)
