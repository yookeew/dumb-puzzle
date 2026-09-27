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

Run:  .venv/Scripts/python.exe src/link/adsb_pushback.py
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
          "adsb_first_own_m": pl.Float64, "adsb_min_own_m": pl.Float64}


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
    """(pushback_ts, tier) for one surface run; tier None = censored."""
    at = np.flatnonzero(d_own < R_STAND)
    if len(at) == 0:
        return np.nan, None
    stat = at[gs[at] <= STATIONARY_GS]
    if len(stat):
        k = stat[-1]
        return (ts[k + 1] if k + 1 < len(ts) else ts[k]), "dwell"
    if at[0] == 0:
        return ts[0], "appear"
    return ts[at[0]], "pass"


def departures(day: dt.date) -> pl.DataFrame:
    """DEP movements whose takeoff falls on `day`: id, airport, stand, takeoff only."""
    src = RANKING if day.year >= 2026 else next(RAW.glob(f"training_{day:%Y-%m}-01_*.parquet"))
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt")
            .filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == day)
            .collect()
            .with_columns(MVT_ID_mvt=pl.col("MVT_ID_mvt").cast(pl.Int64),
                          mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0))


def detect_day(day: dt.date, pos: pl.DataFrame) -> pl.DataFrame:
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
                                     adsb_first_own_m=None, adsb_min_own_m=None) for r in recs}
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
                        pb, tier = pushback(ts, gs, d_own)
                        if tier is not None:
                            row.update(adsb_tier=tier, adsb_pushback_ts=float(pb))
        out.extend(res.values())
    return pl.DataFrame(out, schema=SCHEMA)


def main(days: list[dt.date] | None = None) -> None:
    pos = stand_positions(load_stands())
    OUT.mkdir(parents=True, exist_ok=True)
    for day in days or available_days():
        df = detect_day(day, pos)
        df.write_parquet(OUT / f"day={day.isoformat()}.parquet")
        n_tier = df.filter(pl.col("adsb_tier").is_in(["appear", "dwell"])).height
        print(f"{day}: {df.height:,} DEP, matched {df['adsb_matched'].sum():,}, "
              f"appear+dwell {n_tier:,} ({n_tier / df.height * 100:.1f}%)")


def load_all() -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("day=*.parquet"))])


if __name__ == "__main__":
    main()
