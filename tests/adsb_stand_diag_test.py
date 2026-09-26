"""Diagnose why ADS-B surface points rarely sit near Gateway stands.

Before building the stand-aware pushback detector, rule out the four ways a
weak result could arise: (1) stand coordinates wrong (swap/sign/offset),
(2) proximity tolerance too tight (MLAT noise), (3) tracks start after
pushback (decimated / no gate reception), (4) genuine coverage.

Per departure on 2025-01-15 at the covered airports:
  - segment each hex's day-track into surface runs (gs < 40, split at gaps
    > 20 min); a run followed by gs >= 40 within 120 s is a candidate takeoff,
    its last surface sample the takeoff proxy;
  - match candidates 1:1 to true DEP by MVT_TIME_UTC_mvt (+-180 s, greedy);
  - for matched runs with own-stand coords (STAND_mvt -> stands.csv):
    first-point time vs true BLOCK_TIME (does the track exist before
    pushback?), distance of the first point and the closest point to own
    stand, nearest-any-stand distance, point count in the first 5 min.
Also writes one plot per airport of a matched departure's surface track
over the stand layout to logs/adsb_stand_diag_<ICAO>.png.

Run:  .venv/Scripts/python.exe tests/adsb_stand_diag_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingest.stands import attach_stand_coords, load_stands, stand_positions  # noqa: E402

ADSB = ROOT / "external-data" / "adsb" / "adsb_20250115.parquet"
TRAIN = ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet"
LOGS = ROOT / "logs"
COVERED = ["EHAM", "EGLL", "EDDM", "EDDF", "LSZH", "LEBL"]
DAY = dt.date(2025, 1, 15)
SURFACE_GS = 40.0
GAP_S = 1200.0
TAKEOFF_LOOKAHEAD_S = 120.0
MATCH_TOL_S = 180.0
RADII = [30, 60, 100, 150, 300]


def to_xy(lat, lon, lat0):
    return np.c_[np.asarray(lon) * 111320.0 * np.cos(np.radians(lat0)),
                 np.asarray(lat) * 110540.0]


def surface_runs(ts, gs):
    """Yield (i, j) index ranges of surface runs ending in a takeoff."""
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


def greedy_match(mv_ts, ev_ts):
    pairs = []
    order = np.argsort(ev_ts)
    ev_sorted = ev_ts[order]
    for mi, mt in enumerate(mv_ts):
        lo, hi = np.searchsorted(ev_sorted, [mt - MATCH_TOL_S, mt + MATCH_TOL_S])
        pairs += [(abs(ev_sorted[k] - mt), mi, int(order[k])) for k in range(lo, hi)]
    pairs.sort()
    um, ue, out = set(), set(), {}
    for _, mi, ei in pairs:
        if mi not in um and ei not in ue:
            um.add(mi); ue.add(ei); out[mi] = ei
    return out


def main() -> None:
    stands = load_stands()
    pos = stand_positions(stands)
    adsb = pl.read_parquet(ADSB).filter(pl.col("airport").is_in(COVERED))

    # Stationary-point proximity curve (tolerance question).
    print("Share of stationary (gs<=1) surface points within R of any stand")
    print(f"{'airport':<8}{'n_stat':>8}" + "".join(f"{'<' + str(r) + 'm':>9}" for r in RADII))
    for ap in COVERED:
        st = stands.filter(pl.col("airport") == ap)
        lat0 = st["lat"].mean()
        tree = cKDTree(to_xy(st["lat"], st["lon"], lat0))
        g = adsb.filter((pl.col("airport") == ap) & (pl.col("gs") <= 1))
        d, _ = tree.query(to_xy(g["lat"], g["lon"], lat0))
        print(f"{ap:<8}{len(d):>8}" + "".join(f"{(d < r).mean() * 100:>8.1f}%" for r in RADII))

    mv = (pl.scan_parquet(TRAIN).filter(pl.col("PHASE_mvt") == "DEP")
          .select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt")
          .collect()
          .filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == DAY, pl.col("ADEP_mvt").is_in(COVERED))
          .with_columns(mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                        block_ts=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms") / 1000.0))
    mv = attach_stand_coords(mv, pos)

    rows = []
    for ap in COVERED:
        st = stands.filter(pl.col("airport") == ap)
        lat0 = st["lat"].mean()
        tree = cKDTree(to_xy(st["lat"], st["lon"], lat0))
        tracks, events = {}, []
        for (hx,), g in adsb.filter(pl.col("airport") == ap).sort("hex", "ts").group_by("hex", maintain_order=True):
            ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
            tracks[hx] = (ts, gs, g["lat"].to_numpy(), g["lon"].to_numpy())
            for i, j in surface_runs(ts, gs):
                events.append((hx, i, j, ts[j - 1]))
        m_ap = mv.filter(pl.col("ADEP_mvt") == ap)
        match = greedy_match(m_ap["mvt_ts"].to_numpy(), np.array([e[3] for e in events]))
        for mi, ei in match.items():
            r = m_ap.row(mi, named=True)
            hx, i, j, _ = events[ei]
            ts, gs, la, lo = (a[i:j] for a in tracks[hx])
            xy = to_xy(la, lo, lat0)
            d_any, _ = tree.query(xy)
            row = dict(airport=ap, mvt_id=r["MVT_ID_mvt"], hex=hx, stand=r["STAND_mvt"],
                       n_pts=len(ts), first_minus_block=ts[0] - r["block_ts"],
                       pts_first5min=int((ts < ts[0] + 300).sum()),
                       first_any_stand_m=float(d_any[0]), min_any_stand_m=float(d_any.min()),
                       has_own=r["stand_lat"] is not None)
            if r["stand_lat"] is not None:
                d_own = np.hypot(*(xy - to_xy([r["stand_lat"]], [r["stand_lon"]], lat0)).T)
                row.update(first_own_m=float(d_own[0]), min_own_m=float(d_own.min()))
            rows.append(row)
    res = pl.DataFrame(rows)
    res.write_parquet(LOGS / "adsb_stand_diag.parquet")

    n_dep = mv.group_by("ADEP_mvt").len().rename({"ADEP_mvt": "airport", "len": "n_dep"})
    summ = (res.group_by("airport").agg(
        pl.len().alias("n_matched"),
        ((pl.col("first_minus_block") <= 0).mean() * 100).alias("start<=block%"),
        (pl.col("first_minus_block").median() / 60).alias("med_start-block_min"),
        pl.col("pts_first5min").median().alias("med_pts_1st5min"),
        pl.col("first_own_m").median().alias("med_first_own_m"),
        pl.col("min_own_m").median().alias("med_min_own_m"),
        *[((pl.col("min_own_m") < r).mean() * 100).alias(f"own<{r}m%") for r in RADII],
        ((pl.col("first_any_stand_m") < 60).mean() * 100).alias("1st_any<60m%"),
    )).join(n_dep, on="airport").sort("airport")
    with pl.Config(tbl_rows=20, tbl_cols=30, tbl_width_chars=300, tbl_formatting="ASCII_MARKDOWN",
                   float_precision=1):
        print("\nPer matched departure (surface run ending in takeoff, matched on MVT_TIME +-180s)")
        print(summ)
        print("\nstart<=block%: track's first surface sample is at/before true off-block.")
        print("own<Rm%: closest surface sample within R of the departure's own STAND_mvt position.")
        # Conditional: of tracks that start before block, how close to own stand is the first point?
        pre = res.filter(pl.col("first_minus_block") <= 0)
        print("\nTracks starting at/before off-block: first point distance to own stand")
        print(pre.group_by("airport").agg(
            pl.len().alias("n"),
            pl.col("first_own_m").quantile(0.25).alias("p25_m"),
            pl.col("first_own_m").median().alias("p50_m"),
            pl.col("first_own_m").quantile(0.75).alias("p75_m"),
            ((pl.col("first_own_m") < 100).mean() * 100).alias("<100m%"),
        ).sort("airport"))

    # One plot per airport: a matched departure whose track starts before block.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ap in ["EDDM", "EDDF", "EHAM", "EGLL", "LSZH", "LEBL"]:
        cand = res.filter((pl.col("airport") == ap) & pl.col("has_own")).sort(
            pl.col("first_minus_block"))
        if cand.height == 0:
            continue
        r = cand.row(cand.height // 4, named=True)  # an early-starting but not extreme track
        st = stands.filter(pl.col("airport") == ap)
        g = adsb.filter((pl.col("airport") == ap) & (pl.col("hex") == r["hex"])).sort("ts")
        mrow = mv.filter(pl.col("MVT_ID_mvt") == r["mvt_id"]).row(0, named=True)
        g = g.filter(pl.col("ts").is_between(mrow["block_ts"] - 3600, mrow["mvt_ts"] + 60),
                     pl.col("gs") < SURFACE_GS)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(st["lon"], st["lat"], s=6, c="0.6", label="Gateway stands")
        sc = ax.scatter(g["lon"], g["lat"], c=(g["ts"] - mrow["block_ts"]) / 60, cmap="viridis",
                        s=10, label="surface track")
        ax.scatter([mrow["stand_lon"]], [mrow["stand_lat"]], marker="*", s=250, c="red",
                   label=f"own stand {r['stand']}")
        plt.colorbar(sc, ax=ax, label="minutes relative to true off-block")
        ax.set_aspect(1 / np.cos(np.radians(st["lat"].mean())))
        ax.set_title(f"{ap} {r['mvt_id']} hex {r['hex']}: first pt {r['first_minus_block'] / 60:+.1f} min "
                     f"vs block, min dist own stand {r['min_own_m']:.0f} m")
        ax.legend(loc="best")
        fig.savefig(LOGS / f"adsb_stand_diag_{ap}.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
    print("\nplots -> logs/adsb_stand_diag_<ICAO>.png")


if __name__ == "__main__":
    main()
