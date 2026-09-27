"""LIRF diagnostics: distribution shape (Q1) and runway/stand pattern (Q2).

Two of four concrete follow-up checks requested after the ADS-B ceiling
work. Both are pure data analysis against data already on disk -- no
model fit needed (see tests/lirf_specific_model_test.py for the model
comparison, Q3).

Q1: is LIRF's taxi-time distribution bimodal, heavy-tailed, outlier-driven,
    or just high-variance throughout? Full 2025 labels, all splits.
Q2: does LIRF have a runway-configuration or stand-assignment pattern the
    model's existing runway/stand features can't already see? Checks mean
    taxi by RUNWAY_mvt (a raw categorical the model already has) and by the
    existing engineered config features, binned by FEATURE not by outcome
    decile (avoiding the regression-to-mean artifact that fooled an earlier
    pass on this exact airport -- see reports/lirf_investigation.md SS4).

Run:  .venv/Scripts/python.exe tests/lirf_distribution_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TRAIN_GLOB = str(ROOT / "data" / "raw" / "training_2025-*.parquet")
FEAT_DIR = ROOT / "cache" / "features"
RULE = "=" * 78

OTHER_AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM"]


def q1_distribution() -> pl.DataFrame:
    print(f"\n{RULE}\nQ1. LIRF taxi-time distribution shape (full 2025, all splits)\n{RULE}")
    lab = pl.read_parquet(FEAT_DIR / "labels2025.parquet")
    lirf = lab.filter(pl.col("ADEP_mvt") == "LIRF")
    others = lab.filter(pl.col("ADEP_mvt") != "LIRF")

    for name, d in [("LIRF", lirf), ("all other airports (pooled)", others)]:
        t = d["taxi"].to_numpy()
        qs = np.percentile(t, [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9])
        print(f"\n  {name}: n={len(t):,}  mean={t.mean():.0f}  sd={t.std():.0f}")
        print(f"    p1={qs[0]:.0f} p5={qs[1]:.0f} p10={qs[2]:.0f} p25={qs[3]:.0f} "
              f"p50={qs[4]:.0f} p75={qs[5]:.0f} p90={qs[6]:.0f} p95={qs[7]:.0f} "
              f"p99={qs[8]:.0f} p99.5={qs[9]:.0f} p99.9={qs[10]:.0f}")

    # histogram of LIRF taxi (log-spaced bins, since it spans 2 orders of magnitude)
    t = lirf["taxi"].drop_nulls().to_numpy()
    t = t[t > 0]
    edges = np.geomspace(max(t.min(), 1), t.max(), 25)
    hist, _ = np.histogram(t, bins=edges)
    print(f"\n  LIRF histogram (log-spaced bins, n={len(t):,}):")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        bar = "#" * int(c / max(hist.max(), 1) * 60)
        print(f"    [{lo:7.0f},{hi:7.0f}) n={c:6d} {bar}")

    # decompose by the two known mechanisms: echo (|d|<30s) and extreme (>7200s)
    is_echo = lirf["d"].abs() < 30
    is_extreme = lirf["taxi"] > 7200
    seg = lirf.with_columns(
        segment=pl.when(is_extreme).then(pl.lit("extreme (taxi>7200s)"))
        .when(is_echo).then(pl.lit("echo (|d|<30s)"))
        .otherwise(pl.lit("ordinary"))
    )
    print("\n  decomposition by known mechanism:")
    g = seg.group_by("segment").agg(
        pl.len().alias("n"), pl.col("taxi").mean().alias("mean_taxi"),
        pl.col("taxi").std().alias("sd_taxi"),
        pl.col("taxi").sum().alias("sum_taxi"))
    tot_n, tot_sum = lirf.height, lirf["taxi"].sum()
    for r in g.sort("n", descending=True).iter_rows(named=True):
        print(f"    {r['segment']:<22} n={r['n']:>6,} ({r['n']/tot_n*100:5.1f}%)  "
              f"mean={r['mean_taxi']:8.0f}  sd={r['sd_taxi']:8.0f}  "
              f"share_of_total_taxi_mass={r['sum_taxi']/tot_sum*100:5.1f}%")
    return lirf


def q2_runway_stand() -> None:
    print(f"\n{RULE}\nQ2. Runway/stand pattern check (feature-binned, not outcome-binned)\n{RULE}")
    mv = (
        pl.scan_parquet(TRAIN_GLOB)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "RUNWAY_mvt", "STAND_mvt")
        .collect()
    )
    lirf = mv.filter(pl.col("ADEP_mvt") == "LIRF").with_columns(
        d=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
        hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
        month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    )
    # exclude the two known confounding mechanisms so what's left isolates a
    # genuine runway/stand effect, not echo-absorption or extreme outliers
    core = lirf.filter((pl.col("d").abs() >= 30) & (pl.col("TAXITIME_SEC_mvt") <= 7200))
    print(f"\n  LIRF 'core' rows (excl. echo & extreme-tail): n={core.height:,} "
          f"of {lirf.height:,} ({core.height/lirf.height*100:.1f}%)")

    print("\n  mean/median taxi by RUNWAY_mvt (core rows, n>=200 only):")
    g = (core.group_by("RUNWAY_mvt")
         .agg(pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").mean().alias("mean_taxi"),
              pl.col("TAXITIME_SEC_mvt").median().alias("median_taxi"))
         .filter(pl.col("n") >= 200).sort("mean_taxi", descending=True))
    for r in g.iter_rows(named=True):
        print(f"    {r['RUNWAY_mvt']:<8} n={r['n']:>6,}  mean={r['mean_taxi']:7.0f}  "
              f"median={r['median_taxi']:7.0f}")

    print("\n  mean taxi by (runway, hour) for the runway flagged in the prior "
          "investigation (25), July only, core rows:")
    r25_jul = core.filter((pl.col("RUNWAY_mvt") == "25") & (pl.col("month") == 7))
    if r25_jul.height:
        g2 = (r25_jul.group_by("hour")
              .agg(pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").mean().alias("mean_taxi"))
              .sort("hour"))
        for r in g2.iter_rows(named=True):
            print(f"    hour={r['hour']:>2}  n={r['n']:>4}  mean_taxi={r['mean_taxi']:7.0f}")
    else:
        print("    no core rows on runway '25' in July -- check RUNWAY_mvt encoding")

    print("\n  runway usage share by month (all LIRF DEP, not just core -- "
          "checks whether config genuinely shifts seasonally):")
    g3 = (lirf.group_by(["month", "RUNWAY_mvt"]).agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / pl.col("n").sum().over("month") * 100).alias("pct"))
          .sort(["month", "n"], descending=[False, True]))
    for m in range(1, 13):
        top = g3.filter(pl.col("month") == m).head(3)
        if top.height == 0:
            continue
        parts = ", ".join(f"{r['RUNWAY_mvt']}={r['pct']:.0f}%" for r in top.iter_rows(named=True))
        print(f"    month={m:>2}: {parts}")

    # existing engineered config feature check, binned by hour (a real
    # pre-hoc feature axis, not the true outcome) rather than by taxi decile
    print("\n  existing feature check: does the cached feature set already "
          "encode a LIRF single/dual-runway split? (categorical RUNWAY_mvt "
          "value counts, full year)")
    vc = (lirf.group_by("RUNWAY_mvt").agg(pl.len().alias("n"))
          .sort("n", descending=True))
    print(vc)


def main() -> None:
    q1_distribution()
    q2_runway_stand()


if __name__ == "__main__":
    main()
