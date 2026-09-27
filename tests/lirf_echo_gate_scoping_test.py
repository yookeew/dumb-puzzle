"""Scoping checks before building the has_aobt3=False echo-blend fix
(reports/lirf_investigation.md SS8). Per review feedback: don't tune on 10
rows. Three free/cheap checks first:

1. Rows-4-9-only oracle (not all 10) -- the real number to decide against,
   since rows 1-3 (genuine ~24h anomalies) aren't in scope for this fix.
2. Cross-airport holdout population count for the candidate gate condition
   -- is this LIRF-specific or general? Uses the already-cached
   `cache/lirf_ceilfix_mixed_ev.parquet`, no retrain.
3. Ranking-set population count for what's actually DETECTABLE at
   prediction time (has_aobt3=False + near-total _flt nullity + large
   `sched_takeoff_offset` -- offset is never blanked, unlike `d`/taxi, so
   this is a real, checkable-today count, not a guess). echo_prob itself
   needs a classifier scored on ranking -- deferred to a follow-up script
   if this count says the lane is worth pursuing.

Run:  .venv/Scripts/python.exe tests/lirf_echo_gate_scoping_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EV_PATH = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
RULE = "=" * 78

# _flt fields checked for "near-total nullity" in the top-10 read.
FLT_NULL_COLS = ["CALLSIGN_flt", "AIRCRAFT_TYPE_flt", "MARKET_SEGMENT_flt",
                 "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt"]


def part1_rows49_oracle() -> None:
    print(f"{RULE}\n1. Rows-4-9-ONLY oracle (not all 10)\n{RULE}")
    ev = pl.read_parquet(EV_PATH)
    total_se = float((ev["pred"] - ev["taxi"]).pow(2).sum())
    n = ev.height
    base_rmse = (total_se / n) ** 0.5

    lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF").with_columns(
        se=(pl.col("pred") - pl.col("taxi")).pow(2)).sort("se", descending=True)
    top10 = lirf.head(10)
    rows49 = top10[3:9]   # positions 4-9 (0-indexed 3:9), the 6 "normal taxi, big delay" rows
    rows13_10 = pl.concat([top10[0:3], top10[9:10]])

    for label, subset in [("rows 4-9 only (6 rows)", rows49),
                           ("rows 1-3 + row 10 (4 rows, for comparison)", rows13_10),
                           ("all 10 (sanity check vs SS8's -14.53s)", top10)]:
        se_before = float(subset["se"].sum())
        se_after = 0.0  # oracle: predict truth exactly on this subset
        new_total_se = total_se - se_before + se_after
        new_rmse = (new_total_se / n) ** 0.5
        print(f"  {label}: subset SSE={se_before:,.0f} "
              f"({se_before / total_se * 100:.2f}% of everything)  "
              f"oracle new RMSE={new_rmse:.2f}s  "
              f"recovered={base_rmse - new_rmse:+.2f}s "
              f"({(base_rmse - new_rmse) / base_rmse * 100:+.2f}%)")


def part2_holdout_population() -> None:
    print(f"\n{RULE}\n2. Cross-airport holdout population for the candidate gate\n{RULE}")
    ev = pl.read_parquet(EV_PATH)
    print("  (using has_aobt3=False & AIRCRAFT_OPERATOR_flt null as the proxy "
        "for 'near-total _flt nullity' -- cached ev doesn't carry every _flt "
        "column, this one is a reasonable stand-in, checked against the raw "
        "top-10 read where operator was null on all 10)")

    null_lane = ev.filter(pl.col("has_aobt3") == False).filter(  # noqa: E712
        pl.col("AIRCRAFT_OPERATOR_flt").is_null())
    print(f"\n  has_aobt3=False & operator-null: n={null_lane.height:,} "
          f"({null_lane.height / ev.height * 100:.2f}% of holdout)")

    # is_echo=False but echo_prob elevated anyway -- the exact rows-4-9 signature
    mis_scored = null_lane.filter(
        (pl.col("is_echo") == False) & (pl.col("echo_prob") > 0.5))  # noqa: E712
    print(f"  ...of those, is_echo=False but echo_prob>0.5 "
          f"(the rows-4-9 signature): n={mis_scored.height:,}")

    print("\n  by airport (has_aobt3=False & operator-null & is_echo=False & "
        "echo_prob>0.5):")
    g = (mis_scored.group_by("ADEP_mvt").agg(pl.len().alias("n"))
         .sort("n", descending=True))
    print(g)

    if mis_scored.height:
        se = (mis_scored["pred"] - mis_scored["taxi"]).pow(2).sum()
        total_se = float((ev["pred"] - ev["taxi"]).pow(2).sum())
        print(f"\n  this population's total SSE share: {se / total_se * 100:.2f}% "
              f"of everything (n={mis_scored.height:,}, vs the 6 LIRF rows read by hand)")
        d = mis_scored["d"].to_numpy()
        taxi = mis_scored["taxi"].to_numpy()
        pred = mis_scored["pred"].to_numpy()
        print(f"  d stats: median={np.median(d):.0f}  p10={np.percentile(d,10):.0f}  "
              f"p90={np.percentile(d,90):.0f}")
        print(f"  true taxi median={np.median(taxi):.0f}  pred median={np.median(pred):.0f}  "
              f"(if pred >> taxi broadly, matches the rows-4-9 over-prediction pattern)")


def part3_ranking_detectable_population() -> None:
    print(f"\n{RULE}\n3. Ranking-set population -- what's DETECTABLE at prediction "
          f"time (no echo_prob yet, no d/taxi -- both blanked for DEP)\n{RULE}")
    rk = pl.scan_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "AOBT_3_flt", *FLT_NULL_COLS
    ).collect()
    rk = rk.with_columns(
        has_aobt3=pl.col("AOBT_3_flt").is_not_null(),
        offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
        n_flt_null=pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in FLT_NULL_COLS]),
    )
    print(f"  ranking DEP rows: n={rk.height:,}")
    print(f"  has_aobt3=False: n={(~rk['has_aobt3']).sum():,} "
          f"({(~rk['has_aobt3']).sum() / rk.height * 100:.2f}%)")

    near_null = rk.filter(~pl.col("has_aobt3")).filter(
        pl.col("n_flt_null") >= len(FLT_NULL_COLS) - 1)  # all-or-all-but-one null
    print(f"  has_aobt3=False & near-total _flt nullity "
          f"(>= {len(FLT_NULL_COLS)-1}/{len(FLT_NULL_COLS)} null): "
          f"n={near_null.height:,} ({near_null.height / rk.height * 100:.3f}%)")

    print("\n  by airport:")
    g = (near_null.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True))
    print(g)

    print("\n  offset distribution within this lane (the rows-4-9-detectable "
          "signal -- large offset without needing d/block time):")
    for thr in (3600, 7200, 10800, 20000):
        n_big = near_null.filter(pl.col("offset") > thr).height
        print(f"    offset > {thr:>6}s: n={n_big:>4}  "
              f"({n_big / max(near_null.height,1) * 100:.1f}% of the null lane, "
              f"{n_big / rk.height * 100:.3f}% of all ranking DEP)")

    extreme = near_null.filter(pl.col("offset") > 30000)
    print(f"\n  offset > 30,000s (rows-1-3-scale, for the SEPARATE flag, not "
          f"bundled with this fix): n={extreme.height:,}")
    if extreme.height:
        print(extreme.select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                             "SCHED_TIME_UTC_mvt", "offset").sort("offset", descending=True))


def main() -> None:
    part1_rows49_oracle()
    part2_holdout_population()
    part3_ranking_detectable_population()


if __name__ == "__main__":
    main()
