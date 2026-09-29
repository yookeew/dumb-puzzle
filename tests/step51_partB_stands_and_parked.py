"""§51 Part B: section D, empirical stand positions, detector simulation, "parked anywhere" tier.

Needs the normalised ADS-B points (data/external/adsb/day=*/) and the detector
(src/link/adsb_pushback.py). Production code is not changed; the detector is
only called with a different `pos` table.

PASS 1 (label-free, all 126 days). For every departure matched to a surface run
(same matching as detect_day, asserted to reproduce its tiers), record the run's
stationary segments: maximal stretches of gs <= 1 kt lasting >= 180 s, with
their median position, start/end, and the first sample after the segment. The
"parked" segment of a run is the LAST such segment that ends before first
motion (first sample with gs >= 5 kt).

B1 (= §51 section D). Screen-1 population (matched, no appear/dwell, not
    partial-eligible): parked segment present? nearest Gateway stand
    (name, type, distance); does its name differ systematically from STAND_mvt?
B2. Empirical stand positions: per (airport, STAND_mvt), median parked position
    over matched departures (all days, any tier); keep n >= 10 and spread
    (median distance to the median point) <= 40 m. Agreement with Gateway where
    both exist; offset vectors for LSZH B/D/G/C/T and EDDF E/K. Written to
    data/derived/stands_empirical.csv.
B3. Detector simulation: Gateway positions replaced by empirical ones where
    Gateway is missing or > 50 m away. Label-free tier changes on every day;
    accuracy (labels) on 2025-09-15 / 2025-11-15 only.
B4. "Parked anywhere" tier on 2025-09-15 / 2025-11-15, rows still without
    appear/dwell after B3: pushback = first sample after the last >= 3 min
    stationary segment (before takeoff) within 60 m of any Gateway gate/tie-down
    stand (no misc/de-icing/hangar/helo-only), > 300 m from any runway.
    Pass bar: RMSE after lag removal <= ~250 s AND beats the OOF base on those
    rows, on both days.

Labels: training-month labels (Sep/Nov 2025) only, for B3/B4 accuracy. Jan/Jul
2025 labels are never read. Base = OOF CatBoost (mixed) fold predictions.

Run:  .venv/Scripts/python.exe tests/step51_partB_stands_and_parked.py > logs/step51_partB_stands_and_parked.log
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingest.normalise_adsb import available_days, load_day  # noqa: E402
from src.ingest.stands import attach_stand_coords, load_stands, stand_key, stand_positions  # noqa: E402
from src.link.adsb_pushback import (  # noqa: E402
    MATCH_TOL_S, R_STAND, SURFACE_GS, departures, detect_day, pushback, surface_runs, to_xy)

DET = ROOT / "cache" / "adsb_pushback"
SCRATCH = ROOT / "cache" / "step51"
DERIVED = ROOT / "data" / "derived"
RUNWAYS = ROOT / "data" / "external" / "runways.csv"
LAB = ROOT / "cache" / "features" / "labels2025.parquet"
OOF = ROOT / "cache" / "oof" / "cat_mixed"
TRAIN_DAYS = ("2025-09-15", "2025-11-15")
STAT_GS, STAT_MIN_S, MOTION_GS = 1.0, 180.0, 5.0
START_TOL_S = 30.0
EMP_MIN_N, EMP_MAX_SPREAD = 10, 40.0
DISAGREE_M = 50.0
NEAR_STAND_M, RWY_CLEAR_M = 60.0, 300.0
MONSTER_S = 5 * 3600
SHIPPED_LAGS = {"EDDF": -31.5, "EDDM": -48.7, "EGLL": -169.4, "EHAM": -22.2, "LEBL": -36.4,
                "LEMD": -135.8, "LFPG": -39.5, "LSZH": -27.1}   # v21 fit (logs/stack_submit_v21.log)
PARTIAL_D_MAX = 1000.0


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# --------------------------------------------------------------------------- pass 1
def baseline(day: dt.date, pos: pl.DataFrame) -> pl.DataFrame:
    """Unchanged-detector output for `day` on the local points. Identical to the cache
    for the 124 Jan/Jul days (tests/step51_partB0_detector_repro.py); the two re-fetched
    training days are recomputed instead of read from the cache."""
    if day.isoformat() in TRAIN_DAYS:
        return detect_day(day, pos)
    return pl.read_parquet(DET / f"day={day.isoformat()}.parquet")


def segments(ts, gs):
    """(i0, i1) inclusive index ranges of stationary stretches lasting >= STAT_MIN_S."""
    out, n, i = [], len(ts), 0
    while i < n:
        if gs[i] > STAT_GS:
            i += 1
            continue
        j = i
        while j + 1 < n and gs[j + 1] <= STAT_GS:
            j += 1
        if ts[j] - ts[i] >= STAT_MIN_S:
            out.append((i, j))
        i = j + 1
    return out


def matched_runs(day: dt.date, pos: pl.DataFrame):
    """Yield (MVT row dict, tier, ts, gs, lat, lon) for every matched departure.
    Mirrors detect_day's candidate runs and greedy 1:1 matching exactly."""
    adsb = load_day(day)
    mv = attach_stand_coords(departures(day), pos)
    for (ap,), m_ap in mv.group_by("ADEP_mvt"):
        recs = m_ap.to_dicts()
        sub = adsb.filter(pl.col("airport") == ap)
        ap_pos = pos.filter(pl.col("airport") == ap)
        if not (sub.height and ap_pos.height):
            continue
        lat0 = ap_pos["lat"].mean()
        runs = []
        for _, g in sub.group_by("hex", maintain_order=True):
            ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
            gs = np.where(np.isnan(gs), SURFACE_GS, gs)
            la, lo = g["lat"].to_numpy(), g["lon"].to_numpy()
            xy = to_xy(la, lo, lat0)
            for i, j in surface_runs(ts, gs):
                runs.append((ts[i:j], gs[i:j], xy[i:j], ts[j - 1], la[i:j], lo[i:j]))
        if not runs:
            continue
        run_to = np.array([r[3] for r in runs])
        order = np.argsort(run_to)
        run_sorted = run_to[order]
        pairs = []
        for r in recs:
            lo_, hi_ = np.searchsorted(run_sorted, [r["mvt_ts"] - MATCH_TOL_S, r["mvt_ts"] + MATCH_TOL_S])
            sxy = None if r["stand_lat"] is None else to_xy([r["stand_lat"]], [r["stand_lon"]], lat0)[0]
            for k in range(lo_, hi_):
                ri = int(order[k])
                d_own = None if sxy is None else np.hypot(*(runs[ri][2] - sxy).T)
                visits = d_own is not None and bool((d_own < R_STAND).any())
                pairs.append((not visits, abs(run_to[ri] - r["mvt_ts"]), r["MVT_ID_mvt"], ri, d_own, r))
        pairs.sort(key=lambda p: (p[0], p[1]))
        used_m, used_r = set(), set()
        for _, _, mid, ri, d_own, r in pairs:
            if mid in used_m or ri in used_r:
                continue
            used_m.add(mid)
            used_r.add(ri)
            ts, gs, _, _, la, lo = runs[ri]
            tier = None
            if d_own is not None:
                _, tier, _ = pushback(ts, gs, d_own)
            first_own = None if d_own is None else float(d_own[0])
            yield r, ap, tier, first_own, ts, gs, la, lo


def pass1(pos: pl.DataFrame) -> pl.DataFrame:
    """One row per stationary segment of every matched run (plus a row with seg=-1
    for runs without any), all days. Cached in cache/step51/segments.parquet."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out_p = SCRATCH / "segments.parquet"
    days = available_days()
    newest = max((ROOT / "data" / "external" / "adsb").glob("day=*/part-0.parquet"),
                 key=lambda p: p.stat().st_mtime).stat().st_mtime
    if out_p.exists() and out_p.stat().st_mtime > newest:
        print(f"pass 1: reusing {out_p.relative_to(ROOT)} (newer than every ADS-B day file)")
        return pl.read_parquet(out_p)
    recs, n_check = [], 0
    for day in days:
        ref = {r["MVT_ID_mvt"]: r["adsb_tier"] for r in
               baseline(day, pos).select("MVT_ID_mvt", "adsb_tier").to_dicts()}
        for r, ap, tier, first_own, ts, gs, la, lo in matched_runs(day, pos):
            if ref.get(r["MVT_ID_mvt"], "MISSING") != tier:
                raise SystemExit(f"{day} {r['MVT_ID_mvt']}: tier {tier} vs detector {ref.get(r['MVT_ID_mvt'])}")
            n_check += 1
            mot = np.flatnonzero(gs >= MOTION_GS)
            first_motion = ts[mot[0]] if len(mot) else ts[-1]
            base = dict(day=day.isoformat(), MVT_ID_mvt=r["MVT_ID_mvt"], ADEP_mvt=ap, STAND_mvt=r["STAND_mvt"],
                        mvt_ts=r["mvt_ts"], tier=tier, first_own_m=first_own, run_t0=float(ts[0]),
                        run_n=len(ts), first_motion_ts=float(first_motion),
                        stand_lat=r["stand_lat"], stand_lon=r["stand_lon"])
            segs = segments(ts, gs)
            if not segs:
                recs.append(dict(base, seg=-1))
            for k, (i0, i1) in enumerate(segs):
                recs.append(dict(base, seg=k, seg_t0=float(ts[i0]), seg_t1=float(ts[i1]),
                                 seg_next=float(ts[i1 + 1]) if i1 + 1 < len(ts) else float(ts[i1]),
                                 seg_lat=float(np.median(la[i0:i1 + 1])), seg_lon=float(np.median(lo[i0:i1 + 1])),
                                 seg_n=i1 - i0 + 1, before_motion=bool(ts[i1] < first_motion)))
        print(f"  pass 1 {day}: cumulative matched runs {n_check:,}", flush=True)
    df = pl.DataFrame(recs, infer_schema_length=None)
    df.write_parquet(out_p)
    print(f"pass 1: {n_check:,} matched runs, tiers identical to the detector cache; wrote {out_p.relative_to(ROOT)}")
    return df


# --------------------------------------------------------------------------- geometry
def nearest(points_lat, points_lon, ref_lat, ref_lon, lat0):
    """(index, distance m) of the nearest ref point for each point."""
    p = to_xy(points_lat, points_lon, lat0)
    r = to_xy(ref_lat, ref_lon, lat0)
    d = np.hypot(p[:, None, 0] - r[None, :, 0], p[:, None, 1] - r[None, :, 1])
    k = d.argmin(axis=1)
    return k, d[np.arange(len(k)), k]


def runway_clearance(lat, lon, rw: pl.DataFrame, lat0) -> np.ndarray:
    """Distance (m) from each point to the nearest runway centreline segment."""
    p = to_xy(lat, lon, lat0)
    best = np.full(len(p), np.inf)
    seen = set()
    by = {r["runway"]: r for r in rw.to_dicts()}
    for r in rw.to_dicts():
        o = by.get(r["opposite"])
        if o is None or (r["opposite"], r["runway"]) in seen:
            continue
        seen.add((r["runway"], r["opposite"]))
        a = to_xy([r["lat"]], [r["lon"]], lat0)[0]
        b = to_xy([o["lat"]], [o["lon"]], lat0)[0]
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.hypot(*(p - (a + t[:, None] * ab)).T))
    return best


def eligible_stands(stands: pl.DataFrame) -> pl.DataFrame:
    bad = r"(?i)de-?ic|hangar|hgr|heli"
    return stands.filter(pl.col("stand_type").is_in(["gate", "tie_down"])
                         & ~pl.col("stand_name").str.contains(bad)
                         & (pl.col("aircraft_classes").fill_null("") != "helos"))


def name_relation(own: str | None, near: str | None) -> str:
    if own is None or near is None:
        return "n/a"
    if own == near:
        return "same key"
    strip = lambda s: __import__("re").sub(r"(\d)[LRABC]$", r"\1", s)  # noqa: E731
    if strip(own) == strip(near):
        return "same after variant letter"
    d_own, d_near = "".join(c for c in own if c.isdigit()), "".join(c for c in near if c.isdigit())
    a_own, a_near = "".join(c for c in own if c.isalpha()), "".join(c for c in near if c.isalpha())
    if d_own and d_own == d_near:
        return "same digits, other letters"
    if a_own == a_near:
        return "same letters, other digits"
    return "different"


# --------------------------------------------------------------------------- main
def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(1)
    pl.Config.set_fmt_str_lengths(40)
    stands = load_stands()
    pos = stand_positions(stands)
    elig = eligible_stands(stands)
    rw = pl.read_csv(RUNWAYS)
    lat0s = dict(pos.group_by("airport").agg(pl.col("lat").mean()).iter_rows())

    seg = pass1(pos)
    runs = seg.unique("MVT_ID_mvt", keep="first").select(
        "day", "MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "mvt_ts", "tier", "first_own_m", "run_t0",
        "first_motion_ts", "stand_lat", "stand_lon")
    # "parked" = the >= 3 min stationary segment the run STARTS in (first sample within
    # START_TOL_S of the run's first sample). The spec's "last stationary segment before
    # first motion" was measured to pick the post-pushback engine-start hold instead
    # (median 125-170 m from known-good Gateway stands at EDDM/EGLL/EHAM/LEBL, vs 4-12 m
    # for run-start segments); see §51 Part B.
    parked = (seg.filter((pl.col("seg") == 0) & ((pl.col("seg_t0") - pl.col("run_t0")) <= START_TOL_S))
              .group_by("MVT_ID_mvt").first()
              .select("MVT_ID_mvt", "seg_lat", "seg_lon", "seg_t0", "seg_t1", "seg_next"))
    runs = runs.join(parked, on="MVT_ID_mvt", how="left").with_columns(
        year=pl.col("day").str.slice(0, 4), own_key=stand_key(pl.col("STAND_mvt")),
        ad=pl.col("tier").is_in(["appear", "dwell"]).fill_null(False))

    # nearest Gateway stand to each parked position (all stands, any type)
    near_rows = []
    for (ap,), g in runs.filter(pl.col("seg_lat").is_not_null()).group_by("ADEP_mvt"):
        st = stands.filter(pl.col("airport") == ap)
        if st.height == 0:
            continue
        k, d = nearest(g["seg_lat"].to_numpy(), g["seg_lon"].to_numpy(),
                       st["lat"].to_numpy(), st["lon"].to_numpy(), lat0s[ap])
        near_rows.append(g.select("MVT_ID_mvt").with_columns(
            near_m=pl.Series(d), near_name=st["stand_name"].gather(k), near_type=st["stand_type"].gather(k),
            near_key=st["key"].gather(k),
            rwy_m=pl.Series(runway_clearance(g["seg_lat"].to_numpy(), g["seg_lon"].to_numpy(),
                                             rw.filter(pl.col("airport") == ap), lat0s[ap]))))
    runs = runs.join(pl.concat(near_rows), on="MVT_ID_mvt", how="left")

    # ------------------------------------------------------------------ B1
    section("B1 (= §51 section D): Screen-1 population, parked position vs nearest Gateway stand")
    s1 = runs.filter(~pl.col("ad") & ~((pl.col("first_own_m") < PARTIAL_D_MAX).fill_null(False)
                                        & (pl.col("ADEP_mvt") != "LIRF")))
    print(f"Screen-1 rows (matched, no appear/dwell, not partial-eligible), all days: {s1.height:,}")
    print(s1.group_by("ADEP_mvt", "year").agg(
        pl.len().alias("n"), (100 * pl.col("seg_lat").is_not_null().mean()).alias("parked_seg%"),
        pl.col("near_m").median().alias("med_near_m"),
        (100 * (pl.col("near_m") < 60).mean()).alias("near<60m%_of_parked"),
        (100 * (pl.col("near_key") == pl.col("own_key")).mean()).alias("near==own%"),
        (100 * pl.col("stand_lat").is_null().mean()).alias("own_unresolved%"))
        .sort("ADEP_mvt", "year"))
    rel = s1.filter(pl.col("near_m") < 60).with_columns(
        rel=pl.struct("own_key", "near_key").map_elements(lambda s: name_relation(s["own_key"], s["near_key"]),
                                                          return_dtype=pl.Utf8))
    print("name relation, STAND_mvt vs nearest Gateway stand (parked within 60 m), % of rows:")
    print(rel.group_by("ADEP_mvt", "rel").len().with_columns(
        pct=100 * pl.col("len") / pl.col("len").sum().over("ADEP_mvt"))
        .pivot(on="rel", index="ADEP_mvt", values="pct").sort("ADEP_mvt"))
    print("top (STAND_mvt -> nearest Gateway stand) pairs, parked within 60 m, names differ:")
    print(rel.filter(pl.col("near_key") != pl.col("own_key")).group_by(
        "ADEP_mvt", "STAND_mvt", "near_name", "near_type").len().sort("len", descending=True).head(40))

    # ------------------------------------------------------------------ B2
    section("B2: EMPIRICAL STAND POSITIONS (median parked position per (airport, STAND_mvt), all days)")
    emp_rows = []
    for (ap, stn), g in runs.filter(pl.col("seg_lat").is_not_null() & pl.col("STAND_mvt").is_not_null()).group_by(
            "ADEP_mvt", "STAND_mvt"):
        if g.height < EMP_MIN_N or ap not in lat0s:
            continue
        mla, mlo = float(g["seg_lat"].median()), float(g["seg_lon"].median())
        xy = to_xy(g["seg_lat"].to_numpy(), g["seg_lon"].to_numpy(), lat0s[ap])
        c = to_xy([mla], [mlo], lat0s[ap])[0]
        spread = float(np.median(np.hypot(*(xy - c).T)))
        emp_rows.append(dict(airport=ap, STAND_mvt=stn, n=g.height, lat=mla, lon=mlo, spread_m=spread))
    emp = pl.DataFrame(emp_rows)
    emp = emp.with_columns(keep=(pl.col("spread_m") <= EMP_MAX_SPREAD))
    gw = attach_stand_coords(emp.select(pl.col("airport").alias("ADEP_mvt"), "STAND_mvt"), pos)
    emp = emp.join(gw.rename({"ADEP_mvt": "airport"}), on=["airport", "STAND_mvt"], how="left")
    off, nn_name, nn_m = [], [], []
    for r in emp.to_dicts():
        l0 = lat0s[r["airport"]]
        if r["stand_lat"] is not None:
            a = to_xy([r["lat"]], [r["lon"]], l0)[0] - to_xy([r["stand_lat"]], [r["stand_lon"]], l0)[0]
            off.append((float(a[0]), float(a[1]), float(np.hypot(*a))))
        else:
            off.append((None, None, None))
        st = stands.filter(pl.col("airport") == r["airport"])
        k, d = nearest([r["lat"]], [r["lon"]], st["lat"].to_numpy(), st["lon"].to_numpy(), l0)
        nn_name.append(st["stand_name"][int(k[0])])
        nn_m.append(float(d[0]))
    emp = emp.with_columns(off_e_m=pl.Series([o[0] for o in off], dtype=pl.Float64),
                           off_n_m=pl.Series([o[1] for o in off], dtype=pl.Float64),
                           gateway_offset_m=pl.Series([o[2] for o in off], dtype=pl.Float64),
                           nearest_gateway=pl.Series(nn_name), nearest_gateway_m=pl.Series(nn_m),
                           gateway_resolved=pl.col("stand_lat").is_not_null())
    allst = runs.filter(pl.col("STAND_mvt").is_not_null()).group_by("ADEP_mvt").agg(
        pl.col("STAND_mvt").n_unique().alias("stands_seen"))
    print(emp.group_by("airport").agg(
        pl.len().alias("with_n>=10"), pl.col("keep").sum().alias("kept_spread<=40"),
        (pl.col("keep") & ~pl.col("gateway_resolved")).sum().alias("kept_not_in_gateway"),
        pl.col("gateway_offset_m").filter(pl.col("keep")).median().alias("med_offset_m"),
        (100 * (pl.col("gateway_offset_m") <= 30).filter(pl.col("keep") & pl.col("gateway_resolved")).mean())
        .alias("within30m%"),
        (100 * (pl.col("gateway_offset_m") > DISAGREE_M).filter(pl.col("keep") & pl.col("gateway_resolved")).mean())
        .alias("offset>50m%"))
        .join(allst.rename({"ADEP_mvt": "airport"}), on="airport").sort("airport"))
    print("kept stands NOT in Gateway, per airport (up to 25):")
    print(emp.filter(pl.col("keep") & ~pl.col("gateway_resolved")).sort("n", descending=True)
          .group_by("airport", maintain_order=True).head(25)
          .select("airport", "STAND_mvt", "n", "spread_m", "nearest_gateway", "nearest_gateway_m"))
    for ap, fams in (("LSZH", "BDGCT"), ("EDDF", "EK")):
        print(f"\n{ap} families {fams}: empirical vs Gateway offset (east, north, total m) and nearest Gateway stand")
        print(emp.filter((pl.col("airport") == ap)
                         & stand_key(pl.col("STAND_mvt")).str.contains(f"^[{fams}]\\d"))
              .sort("STAND_mvt").select("STAND_mvt", "n", "spread_m", "keep", "off_e_m", "off_n_m",
                                        "gateway_offset_m", "nearest_gateway", "nearest_gateway_m"))
    DERIVED.mkdir(parents=True, exist_ok=True)
    emp.select("airport", "STAND_mvt", "n", "spread_m", "keep", "lat", "lon", "gateway_resolved",
               "gateway_offset_m", "off_e_m", "off_n_m", "nearest_gateway", "nearest_gateway_m").sort(
        "airport", "STAND_mvt").write_csv(DERIVED / "stands_empirical.csv")
    print(f"wrote {(DERIVED / 'stands_empirical.csv').relative_to(ROOT)} ({emp.height} stands, "
          f"{emp['keep'].sum()} kept)")

    # ------------------------------------------------------------------ B3
    section("B3: DETECTOR WITH EMPIRICAL POSITIONS (Gateway missing or > 50 m away)")
    ov = emp.filter(pl.col("keep") & (~pl.col("gateway_resolved") | (pl.col("gateway_offset_m") > DISAGREE_M)))
    print(f"override stands: {ov.height} ({(~ov['gateway_resolved']).sum()} missing in Gateway, "
          f"{ov['gateway_resolved'].sum()} disagreeing)")
    print(ov.group_by("airport").agg(pl.len().alias("stands"), (~pl.col("gateway_resolved")).sum().alias("missing"),
                                     pl.col("gateway_resolved").sum().alias("disagree")).sort("airport"))
    ovk = ov.select("airport", key=stand_key(pl.col("STAND_mvt")), lat="lat", lon="lon")
    pos_new = pl.concat([pos.join(ovk.select("airport", "key"), on=["airport", "key"], how="anti"),
                         ovk]).unique(["airport", "key"], keep="last")
    changes = []
    new_by_day = {}
    for day in available_days():
        new = detect_day(day, pos_new)
        old = baseline(day, pos)
        j = old.select("MVT_ID_mvt", "airport", old_tier="adsb_tier").join(
            new.select("MVT_ID_mvt", new_tier="adsb_tier"), on="MVT_ID_mvt")
        j = j.with_columns(day=pl.lit(day.isoformat()),
                           old_ad=pl.col("old_tier").is_in(["appear", "dwell"]).fill_null(False),
                           new_ad=pl.col("new_tier").is_in(["appear", "dwell"]).fill_null(False))
        changes.append(j.filter(pl.col("old_tier").fill_null("-") != pl.col("new_tier").fill_null("-")))
        if day.isoformat() in TRAIN_DAYS:
            new_by_day[day.isoformat()] = new
    ch = pl.concat(changes).with_columns(
        grp=pl.when(pl.col("day").is_in(list(TRAIN_DAYS))).then(pl.lit("2025 Sep/Nov"))
        .otherwise(pl.col("day").str.slice(0, 4) + pl.lit(" Jan/Jul")))
    print("rows changing tier (vs the unchanged detector on the same points), per airport:")
    print(ch.group_by("airport", "grp").agg(
        pl.len().alias("changed"), (pl.col("new_ad") & ~pl.col("old_ad")).sum().alias("into_AD"),
        (pl.col("old_ad") & ~pl.col("new_ad")).sum().alias("out_of_AD"),
        (pl.col("old_ad") & pl.col("new_ad")).sum().alias("AD_tier_swap"))
        .pivot(on="grp", index="airport", values=["into_AD", "out_of_AD"]).sort("airport"))

    # accuracy on training days
    lab = pl.read_parquet(LAB, columns=["MVT_ID_mvt", "taxi"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    oof = pl.concat([pl.read_parquet(OOF / f"fold={d[:7]}.parquet") for d in TRAIN_DAYS]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias("base"))
    mts = runs.select("MVT_ID_mvt", "mvt_ts")
    acc = []
    for d, new in new_by_day.items():
        base_det = baseline(dt.date.fromisoformat(d), pos)   # same points, Gateway positions
        f = (base_det.select("MVT_ID_mvt", "airport", old_tier="adsb_tier", old_pb="adsb_pushback_ts")
             .join(new.select("MVT_ID_mvt", new_tier="adsb_tier", new_pb="adsb_pushback_ts"), on="MVT_ID_mvt")
             .join(mts, on="MVT_ID_mvt", how="left").join(lab, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt")
             .with_columns(day=pl.lit(d),
                           old_ad=pl.col("old_tier").is_in(["appear", "dwell"]).fill_null(False),
                           new_ad=pl.col("new_tier").is_in(["appear", "dwell"]).fill_null(False),
                           lag=pl.col("airport").replace_strict(SHIPPED_LAGS, default=-39.5, return_dtype=pl.Float64)))
        f = f.with_columns(mvt_ts=pl.coalesce("mvt_ts", pl.lit(None)))
        acc.append(f)
    acc = pl.concat(acc)
    mv = pl.concat([departures(dt.date.fromisoformat(d)).select("MVT_ID_mvt", "mvt_ts") for d in TRAIN_DAYS])
    acc = acc.drop("mvt_ts").join(mv, on="MVT_ID_mvt", how="left").with_columns(
        err_old=pl.col("mvt_ts") - pl.col("old_pb") - pl.col("lag") - pl.col("taxi"),
        err_new=pl.col("mvt_ts") - pl.col("new_pb") - pl.col("lag") - pl.col("taxi"),
        err_base=pl.col("base") - pl.col("taxi"))

    def stats(df: pl.DataFrame, col: str) -> dict:
        e = df[col].drop_nulls().to_numpy()
        return dict(n=len(e), median=float(np.median(e)) if len(e) else np.nan,
                    rmse=float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
                    within60=float(100 * np.mean(np.abs(e) <= 60)) if len(e) else np.nan,
                    within300=float(100 * np.mean(np.abs(e) <= 300)) if len(e) else np.nan)

    rows = []
    for (ap, d), g in acc.group_by("airport", "day"):
        moved = g.filter(pl.col("new_ad") & ~pl.col("old_ad"))
        existing = g.filter(pl.col("old_ad") & pl.col("new_ad"))
        rows.append(dict(airport=ap, day=d, **{f"moved_{k}": v for k, v in stats(moved, "err_new").items()},
                         moved_base_rmse=stats(moved, "err_base")["rmse"],
                         **{f"exist_{k}": v for k, v in stats(existing, "err_old").items() if k in ("n", "rmse")}))
    print("\ntraining days: rows moved INTO appear/dwell -- ADS-B error after the shipped per-airport lag "
          "(s), vs existing appear/dwell rows and vs the OOF base on the same rows:")
    print(pl.DataFrame(rows).filter(pl.col("moved_n") > 0).sort("airport", "day"))
    tm = acc.filter(pl.col("new_ad") & ~pl.col("old_ad"))
    print(f"pooled moved rows: {stats(tm, 'err_new')} | OOF base on them: rmse {stats(tm, 'err_base')['rmse']:.1f}")

    # ------------------------------------------------------------------ B4
    section("B4: 'PARKED ANYWHERE' TIER on training days (rows without appear/dwell after B3)")
    still = acc.filter(~pl.col("new_ad")).select("MVT_ID_mvt", "day", "airport", "mvt_ts", "taxi", "base")
    sg = (seg.filter(pl.col("day").is_in(list(TRAIN_DAYS)) & (pl.col("seg") >= 0)
                     & (pl.col("seg_t1") < pl.col("mvt_ts")))
          .join(still.select("MVT_ID_mvt"), on="MVT_ID_mvt"))
    # tag each segment with the nearest eligible stand and runway clearance
    tagged = []
    for (ap,), g in sg.group_by("ADEP_mvt"):
        st = elig.filter(pl.col("airport") == ap)
        if st.height == 0 or ap not in lat0s:
            continue
        k, d = nearest(g["seg_lat"].to_numpy(), g["seg_lon"].to_numpy(), st["lat"].to_numpy(),
                       st["lon"].to_numpy(), lat0s[ap])
        tagged.append(g.with_columns(
            st_m=pl.Series(d), st_name=st["stand_name"].gather(k),
            rwy_m=pl.Series(runway_clearance(g["seg_lat"].to_numpy(), g["seg_lon"].to_numpy(),
                                             rw.filter(pl.col("airport") == ap), lat0s[ap]))))
    tg = pl.concat(tagged).with_columns(ok=(pl.col("st_m") <= NEAR_STAND_M) & (pl.col("rwy_m") > RWY_CLEAR_M))
    okseg = tg.filter("ok").sort("seg")
    pick = okseg.group_by("MVT_ID_mvt").agg(
        pl.col("seg_next").last().alias("pb_ts"), pl.col("st_name").last().alias("stand_used"),
        pl.col("st_name").n_unique().alias("n_stands_visited"), pl.len().alias("n_parked_segs"))
    b4 = still.join(pick, on="MVT_ID_mvt", how="left").with_columns(
        covered=pl.col("pb_ts").is_not_null(), tow=(pl.col("n_stands_visited") > 1).fill_null(False),
        raw_err=pl.col("mvt_ts") - pl.col("pb_ts") - pl.col("taxi"))
    print("coverage: share of still-uncovered matched rows that get a parked-anywhere pushback")
    print(b4.group_by("airport", "day").agg(pl.len().alias("uncovered"), pl.col("covered").sum().alias("covered"),
                                            (100 * pl.col("covered").mean()).alias("cov%")).sort("airport", "day"))
    c = b4.filter("covered")
    # lag: in-sample per (airport, day) median, and cross-day (fit on the other day)
    med = c.group_by("airport", "day").agg(pl.col("raw_err").median().alias("lag_in"))
    cross = med.join(med.rename({"day": "oday", "lag_in": "lag_x"}), on="airport").filter(pl.col("day") != pl.col("oday"))
    c = c.join(med, on=["airport", "day"]).join(cross.select("airport", "day", "lag_x"), on=["airport", "day"], how="left")
    c = c.with_columns(e_in=pl.col("raw_err") - pl.col("lag_in"),
                       e_x=pl.col("raw_err") - pl.col("lag_x").fill_null(pl.col("lag_in")),
                       e_base=pl.col("base") - pl.col("taxi"))
    res = c.group_by("airport", "day").agg(
        pl.len().alias("n"), pl.col("raw_err").median().alias("med_raw"),
        pl.col("e_in").pow(2).mean().sqrt().alias("rmse_lag_in"),
        pl.col("e_x").pow(2).mean().sqrt().alias("rmse_lag_crossday"),
        (100 * (pl.col("e_x").abs() <= 60).mean()).alias("pm60%"),
        (100 * (pl.col("e_x").abs() <= 300).mean()).alias("pm300%"),
        (100 * pl.col("tow").mean()).alias("tow%"),
        pl.col("e_base").pow(2).mean().sqrt().alias("base_rmse")).sort("airport", "day")
    print(res)
    pooled = c.group_by("day").agg(pl.len().alias("n"), pl.col("e_x").pow(2).mean().sqrt().alias("rmse_lag_crossday"),
                                   pl.col("e_in").pow(2).mean().sqrt().alias("rmse_lag_in"),
                                   pl.col("e_base").pow(2).mean().sqrt().alias("base_rmse"),
                                   (100 * pl.col("tow").mean()).alias("tow%")).sort("day")
    print("pooled per day:")
    print(pooled)
    print("tow vs single-stand segments (pooled days):")
    print(c.group_by("tow").agg(pl.len(), pl.col("e_x").pow(2).mean().sqrt().alias("rmse"),
                                pl.col("e_base").pow(2).mean().sqrt().alias("base_rmse")))
    ok = all((r["rmse_lag_crossday"] <= 250) and (r["rmse_lag_crossday"] < r["base_rmse"]) for r in pooled.to_dicts())
    print(f"\nPASS BAR (per day: RMSE after cross-day lag <= 250 s AND below the OOF base on those rows): "
          f"{'PASS' if ok else 'FAIL'}")
    print("reference: appear 119 s / dwell 225 s RMSE on 2025-01-15 (§37); moved-row numbers in B3 above.")
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
