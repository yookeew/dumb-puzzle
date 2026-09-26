"""Stand-aware ADS-B pushback detector: recovery and accuracy vs AOBT_3_flt.

Replaces the §29 "last >=3-min dwell before takeoff" rule, which could not
tell a gate from a de-icing pad. tests/adsb_stand_diag_test.py showed the
stand coordinates are right and that transponders typically switch on at
pushback -- a departure's track often *starts* at its stand -- so the signal
is the aircraft's presence at its own stand, not the end of a long dwell.

Inputs used are all available at ranking time for DEP rows: MVT_TIME_UTC_mvt
(takeoff) and STAND_mvt. BLOCK_TIME_UTC_mvt is used only for scoring.

Per (airport, hex) day-track on 2025-01-15:
  1. Surface = gs < 40 kt (alt == -1 agreement reported for reference).
     Split into surface runs at gaps > 20 min; a run immediately followed
     (<=120 s) by gs >= 40 is a candidate departure; its last surface sample
     is the takeoff proxy.
  2. Match candidates 1:1 to true DEP movements, same airport, takeoff within
     +-180 s of MVT_TIME. Greedy, preferring runs that come within R_STAND of
     the movement's own stand, then smallest takeoff-time gap.
  3. Pushback within the matched run, using points within R_STAND of own
     stand:
       dwell  : a stationary (gs <= 1) at-stand sample exists -> pushback =
                first sample after the last stationary at-stand sample
       appear : run's first sample is at the stand, none stationary ->
                pushback = first sample (transponder on at pushback)
       pass   : track reaches the stand only after starting elsewhere ->
                pushback = first at-stand sample (low confidence)
     No at-stand sample -> censored.
  4. Score diff = adsb_pushback - true BLOCK_TIME, and AOBT_3_flt - true
     BLOCK_TIME on the same rows (taxi error equals diff exactly, since
     MVT_TIME is exact).

Run:  .venv/Scripts/python.exe tests/adsb_stand_detector_test.py
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
from src.ingest.stands import attach_stand_coords, load_stands, stand_positions  # noqa: E402

ADSB = ROOT / "external-data" / "adsb" / "adsb_20250115.parquet"
TRAIN = ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet"
OUT = ROOT / "logs" / "adsb_stand_detector.parquet"
COVERED = ["EHAM", "EGLL", "EDDM", "EDDF", "LSZH", "LEBL"]
DAY = dt.date(2025, 1, 15)
SURFACE_GS = 40.0
STATIONARY_GS = 1.0
GAP_S = 1200.0
TAKEOFF_LOOKAHEAD_S = 120.0
MATCH_TOL_S = 180.0
R_STAND = 100.0
RULE = "=" * 78


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


def summarise(d: np.ndarray) -> dict:
    return {"n": len(d),
            "median_s": float(np.median(d)) if len(d) else np.nan,
            "p10_s": float(np.percentile(d, 10)) if len(d) else np.nan,
            "p90_s": float(np.percentile(d, 90)) if len(d) else np.nan,
            "le60%": float((np.abs(d) <= 60).mean() * 100) if len(d) else np.nan,
            "le300%": float((np.abs(d) <= 300).mean() * 100) if len(d) else np.nan,
            "rmse_s": float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan}


def main() -> None:
    pos = stand_positions(load_stands())
    adsb = pl.read_parquet(ADSB).filter(pl.col("airport").is_in(COVERED))

    agree = adsb.select(
        both=((pl.col("gs") < SURFACE_GS) & (pl.col("alt") == -1)).sum(),
        gs_only=((pl.col("gs") < SURFACE_GS) & (pl.col("alt") != -1)).sum(),
        alt_only=((pl.col("gs") >= SURFACE_GS) & (pl.col("alt") == -1)).sum())
    print(f"surface flag agreement (6 airports): {agree.row(0, named=True)}")

    mv = (pl.scan_parquet(TRAIN).filter(pl.col("PHASE_mvt") == "DEP")
          .select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
                  "BLOCK_TIME_UTC_mvt", "AOBT_3_flt")
          .collect()
          .filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == DAY, pl.col("ADEP_mvt").is_in(COVERED))
          .with_columns(*[(pl.col(c).dt.epoch("ms") / 1000.0).alias(a) for c, a in
                          [("MVT_TIME_UTC_mvt", "mvt_ts"), ("BLOCK_TIME_UTC_mvt", "block_ts"),
                           ("AOBT_3_flt", "aobt3_ts")]]))
    mv = attach_stand_coords(mv, pos)

    rows, per_ap = [], []
    for ap in COVERED:
        lat0 = pos.filter(pl.col("airport") == ap)["lat"].mean()
        runs = []  # (hex, ts, gs, xy, takeoff_ts)
        for (hx,), g in adsb.filter(pl.col("airport") == ap).sort("hex", "ts").group_by("hex", maintain_order=True):
            ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
            xy = to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0)
            for i, j in surface_runs(ts, gs):
                runs.append((hx, ts[i:j], gs[i:j], xy[i:j], ts[j - 1]))
        run_to = np.array([r[4] for r in runs])
        order = np.argsort(run_to)
        run_sorted = run_to[order]

        m_ap = mv.filter(pl.col("ADEP_mvt") == ap)
        recs = m_ap.to_dicts()
        pairs = []
        for mi, r in enumerate(recs):
            lo, hi = np.searchsorted(run_sorted, [r["mvt_ts"] - MATCH_TOL_S, r["mvt_ts"] + MATCH_TOL_S])
            sxy = None if r["stand_lat"] is None else to_xy([r["stand_lat"]], [r["stand_lon"]], lat0)[0]
            for k in range(lo, hi):
                ri = int(order[k])
                d_own = None if sxy is None else np.hypot(*(runs[ri][3] - sxy).T)
                visits = d_own is not None and bool((d_own < R_STAND).any())
                pairs.append((not visits, abs(run_to[ri] - r["mvt_ts"]), mi, ri, d_own))
        pairs.sort(key=lambda p: (p[0], p[1]))
        used_m, used_r = set(), set()
        n_matched = 0
        for _, _, mi, ri, d_own in pairs:
            if mi in used_m or ri in used_r:
                continue
            used_m.add(mi); used_r.add(ri); n_matched += 1
            r = recs[mi]
            if d_own is None:
                continue
            hx, ts, gs, _, _ = runs[ri]
            pb, tier = pushback(ts, gs, d_own)
            if tier is None:
                continue
            rows.append(dict(airport=ap, mvt_id=r["MVT_ID_mvt"], hex=hx, tier=tier,
                             adsb_pb=pb, block=r["block_ts"], aobt3=r["aobt3_ts"],
                             diff_adsb=pb - r["block_ts"],
                             diff_aobt3=(r["aobt3_ts"] - r["block_ts"]) if r["aobt3_ts"] is not None else None))
        per_ap.append(dict(airport=ap, n_dep=len(recs),
                           has_stand=sum(r["stand_lat"] is not None for r in recs),
                           n_matched=n_matched))

    res = pl.DataFrame(rows)
    res.write_parquet(OUT)
    rec = res.group_by("airport").agg(pl.len().alias("usable"),
                                      *[(pl.col("tier") == t).sum().alias(t) for t in ("dwell", "appear", "pass")])
    tab = pl.DataFrame(per_ap).join(rec, on="airport", how="left").with_columns(
        (pl.col("usable") / pl.col("n_dep") * 100).round(1).alias("recovery%"))
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, tbl_formatting="ASCII_MARKDOWN",
                   float_precision=1):
        print(f"\n{RULE}\nRecovery, 2025-01-15 true DEP movements\n{RULE}")
        print(tab)
        tot = tab.select(pl.col("n_dep").sum(), pl.col("usable").sum()).row(0)
        print(f"TOTAL usable {tot[1]}/{tot[0]} = {tot[1] / tot[0] * 100:.1f}%  (§29 naive: 670/2629 = 25.5%)")

        def block(df, label):
            both = df.filter(pl.col("diff_aobt3").is_not_null())
            out = []
            for name, col in (("ADS-B", "diff_adsb"), ("AOBT_3_flt", "diff_aobt3")):
                out.append({"set": label, "source": name, **summarise(both[col].to_numpy())})
            return out

        print(f"\n{RULE}\nAccuracy vs true BLOCK_TIME, same rows (rows lacking AOBT_3_flt dropped)\n{RULE}")
        acc = []
        acc += block(res, "ALL")
        for t in ("dwell", "appear", "pass"):
            acc += block(res.filter(pl.col("tier") == t), t)
        for ap in COVERED:
            acc += block(res.filter(pl.col("airport") == ap), ap)
        print(pl.DataFrame(acc))

        # Does ADS-B beat AOBT_3 row by row, and would a simple rule combine them?
        b = res.filter(pl.col("diff_aobt3").is_not_null(), pl.col("tier") != "pass")
        closer = (b["diff_adsb"].abs() < b["diff_aobt3"].abs()).mean() * 100
        avg = ((b["diff_adsb"] + b["diff_aobt3"]) / 2).to_numpy()
        print(f"\ndwell+appear rows: ADS-B closer than AOBT_3 on {closer:.1f}% of rows; "
              f"RMSE ADS-B {summarise(b['diff_adsb'].to_numpy())['rmse_s']:.1f}s, "
              f"AOBT_3 {summarise(b['diff_aobt3'].to_numpy())['rmse_s']:.1f}s, "
              f"simple mean of both {summarise(avg)['rmse_s']:.1f}s (n={b.height})")


if __name__ == "__main__":
    main()
