"""ADS-B pushback-recovery: flight-level recovery rate & accuracy check.

Follow-up to the ceiling check (tests/adsb_ceiling_test.py, PROGRESS.md S29):
that computed a best-case RMSE ceiling assuming an unspecified 50%/25%
coverage guess. This measures the actual number, on the one day of ADS-B
already on disk (external-data/adsb/adsb_20250115.parquet, 2025-01-15, the 6
airports with any ground coverage: EHAM, EGLL, EDDM, EDDF, LSZH, LEBL) --
costs nothing further, no download.

Method (no callsign/tail-to-flight key exists in either dataset, so matching
is by time, the same anchor the rest of this project uses -- MVT_TIME_UTC_mvt
is never blanked):

1. For each (airport, hex) aircraft, walk its sorted day-track. `alt == -1`
   is a categorical ADS-B surface-position flag (not a computed threshold),
   so ground/airborne segmentation is exact, not airport-elevation-dependent.
   Each maximal ground run that is immediately followed by an airborne run
   is a candidate departure; its last ground sample is the ADS-B takeoff
   proxy (`adsb_takeoff_ts`).
2. Within that ground run, find the LAST sub-run of >=3 min with
   ground speed <= 2 kt (a genuine gate/stand dwell, not a taxi-queue stop).
   The first sample after it ends is the pushback estimate
   (`adsb_pushback_ts`). If no such dwell exists in the run (track starts
   already moving -- e.g. picked up mid-taxi because gate-side coverage is
   poor), flag `censored=True`: we have a takeoff but no usable pushback
   estimate for that aircraft.
3. Match candidate ADS-B takeoffs to true DEP movements 1:1 by nearest
   `MVT_TIME_UTC_mvt`, same airport, within a 180s tolerance (greedy,
   closest pairs first).
4. Recovery rate = matched-and-uncensored / total true DEP movements that
   day. Accuracy = adsb_pushback_ts - true BLOCK_TIME_UTC_mvt, on the
   matched-and-uncensored set, reported the same way as the AOBT_3_flt check
   in reports/step0_audit.md SS4 so the two are directly comparable.

Run:  .venv/Scripts/python.exe tests/adsb_recovery_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ADSB = ROOT / "external-data" / "adsb" / "adsb_20250115.parquet"
TRAIN = ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet"

COVERED = ["EHAM", "EGLL", "EDDM", "EDDF", "LSZH", "LEBL"]
DAY = dt.date(2025, 1, 15)

STATIONARY_GS = 2.0       # kt -- below this counts as "not moving"
MIN_DWELL_S = 180.0       # a genuine gate dwell, not a taxi-queue stop
MATCH_TOL_S = 180.0       # takeoff-time match tolerance, seconds
RULE = "=" * 78


def _last_dwell_end(seg_ts: np.ndarray, seg_gs: np.ndarray) -> tuple[float, bool]:
    """Return (pushback_ts, censored) for one ground run ending in takeoff."""
    n = len(seg_ts)
    stat = seg_gs <= STATIONARY_GS
    best_end = None
    i = 0
    while i < n:
        if stat[i]:
            j = i
            while j < n and stat[j]:
                j += 1
            if seg_ts[j - 1] - seg_ts[i] >= MIN_DWELL_S:
                best_end = j - 1
            i = j
        else:
            i += 1
    if best_end is None:
        return seg_ts[0], True
    if best_end + 1 < n:
        return seg_ts[best_end + 1], False
    return seg_ts[best_end], True


def _detect_events(ts: np.ndarray, alt: np.ndarray, gs: np.ndarray) -> list[dict]:
    """Walk one hex's sorted day-track; return one dict per ground run that
    ends in a takeoff (adsb_takeoff_ts, adsb_pushback_ts, censored)."""
    ground = alt == -1
    n = len(ts)
    out = []
    i = 0
    while i < n:
        if not ground[i]:
            i += 1
            continue
        j = i
        while j < n and ground[j]:
            j += 1
        if j < n:  # ground run ended because the next sample is airborne
            pb_ts, censored = _last_dwell_end(ts[i:j], gs[i:j])
            out.append(dict(adsb_takeoff_ts=ts[j - 1], adsb_pushback_ts=pb_ts,
                             censored=censored))
        i = j
        while i < n and not ground[i]:
            i += 1
    return out


def _match_airport(mv_ts: np.ndarray, ev_ts: np.ndarray) -> dict[int, int]:
    """Greedy nearest-time 1:1 match within MATCH_TOL_S. Returns
    {mv_index: ev_index}."""
    pairs = []
    for mi, mt in enumerate(mv_ts):
        lo = np.searchsorted(ev_ts, mt - MATCH_TOL_S)
        hi = np.searchsorted(ev_ts, mt + MATCH_TOL_S)
        for ei in range(lo, hi):
            pairs.append((abs(ev_ts[ei] - mt), mi, ei))
    pairs.sort(key=lambda p: p[0])
    used_mv, used_ev, out = set(), set(), {}
    for _, mi, ei in pairs:
        if mi in used_mv or ei in used_ev:
            continue
        used_mv.add(mi)
        used_ev.add(ei)
        out[mi] = ei
    return out


def main() -> None:
    print(f"loading ADS-B probe: {ADSB.name}")
    adsb = pl.read_parquet(ADSB)

    print("loading true DEP movements for 2025-01-15 at covered airports")
    mv = (
        pl.scan_parquet(TRAIN)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt")
        .collect()
        .filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == DAY)
        .filter(pl.col("ADEP_mvt").is_in(COVERED))
        .with_columns(
            mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
            block_ts=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
        )
    )

    rows = []
    per_airport_summary = []
    for airport in COVERED:
        sub = adsb.filter(pl.col("airport") == airport).sort(["hex", "ts"])
        events = []
        for hex_id, g in sub.group_by("hex", maintain_order=True):
            ts = g["ts"].to_numpy()
            alt = g["alt"].to_numpy()
            gs = g["gs"].to_numpy()
            for e in _detect_events(ts, alt, gs):
                e["hex"] = hex_id[0]
                events.append(e)
        ev_df = pl.DataFrame(events) if events else pl.DataFrame(
            schema={"adsb_takeoff_ts": pl.Float64, "adsb_pushback_ts": pl.Float64,
                    "censored": pl.Boolean, "hex": pl.String})
        ev_df = ev_df.sort("adsb_takeoff_ts")
        ev_ts = ev_df["adsb_takeoff_ts"].to_numpy()

        mv_ap = mv.filter(pl.col("ADEP_mvt") == airport).sort("mvt_ts")
        mv_ts = mv_ap["mvt_ts"].to_numpy()
        block_ts = mv_ap["block_ts"].to_numpy()

        match = _match_airport(mv_ts, ev_ts)
        n_total = len(mv_ts)
        n_matched = len(match)
        n_censored_matched = sum(1 for ei in match.values() if ev_df["censored"][int(ei)])
        n_usable = n_matched - n_censored_matched

        for mi, ei in match.items():
            if ev_df["censored"][int(ei)]:
                continue
            diff = float(ev_df["adsb_pushback_ts"][int(ei)]) - float(block_ts[mi])
            rows.append(dict(airport=airport, diff_sec=diff))

        per_airport_summary.append(dict(
            airport=airport, n_total=n_total, n_candidates=len(ev_ts),
            n_matched=n_matched, n_censored_matched=n_censored_matched,
            n_usable=n_usable,
        ))

    print(f"\n{RULE}\nPer-airport recovery (2025-01-15 true DEP movements)\n{RULE}")
    print(f"{'airport':<8}{'n_dep':>8}{'n_adsb_cand':>13}{'n_matched':>11}"
          f"{'censored':>10}{'n_usable':>10}{'recovery%':>11}")
    tot_total = tot_usable = 0
    for s in per_airport_summary:
        rec = s["n_usable"] / s["n_total"] * 100
        tot_total += s["n_total"]
        tot_usable += s["n_usable"]
        print(f"{s['airport']:<8}{s['n_total']:>8}{s['n_candidates']:>13}"
              f"{s['n_matched']:>11}{s['n_censored_matched']:>10}"
              f"{s['n_usable']:>10}{rec:>10.1f}%")
    print(f"\nTOTAL: n_dep={tot_total}  n_usable={tot_usable}  "
          f"recovery={tot_usable / tot_total * 100:.1f}%")

    acc = pl.DataFrame(rows)
    if acc.height == 0:
        print("\nNo usable matched flights -- cannot compute accuracy.")
        return

    d = acc["diff_sec"].to_numpy()
    print(f"\n{RULE}\nAccuracy: adsb_pushback_ts - true BLOCK_TIME_UTC_mvt "
          f"(n={len(d)})\n{RULE}")
    print(f"  median = {np.median(d) / 60:.2f} min")
    print(f"  p10    = {np.percentile(d, 10) / 60:.2f} min")
    print(f"  p90    = {np.percentile(d, 90) / 60:.2f} min")
    print(f"  |d|<=60s   : {(np.abs(d) <= 60).mean() * 100:.1f}%")
    print(f"  |d|<=300s  : {(np.abs(d) <= 300).mean() * 100:.1f}%")
    print(f"  rmse   = {float(np.sqrt(np.mean(d ** 2))):.1f} s")
    print("\n  per airport:")
    for airport in COVERED:
        s = acc.filter(pl.col("airport") == airport)["diff_sec"].to_numpy()
        if len(s) == 0:
            print(f"    {airport:<6} n=0")
            continue
        print(f"    {airport:<6} n={len(s):>4}  median={np.median(s) / 60:6.2f}min  "
              f"|d|<=60s={( np.abs(s) <= 60).mean() * 100:5.1f}%  "
              f"|d|<=300s={(np.abs(s) <= 300).mean() * 100:5.1f}%  "
              f"rmse={float(np.sqrt(np.mean(s ** 2))):7.1f}s")

    print("\nFor comparison, reports/step0_audit.md SS4 (AOBT_3_flt vs true "
          "BLOCK_TIME_UTC_mvt, full year, all 10 airports): median=-0.8min, "
          "|d|<=60s=21.0%, |d|<=300s=74.6%.")


if __name__ == "__main__":
    main()
