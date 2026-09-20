"""Can the echo population be identified from ARRIVAL-side evidence?

Why this matters. S15/S16 measured an oracle-routing ceiling of ~51.7s
(337.6 -> 285.8) for perfectly choosing between the echo reconstruction
(taxi = sched_takeoff_offset) and the model prediction, and closed the lever
because post-hoc blending on `echo_prob` couldn't capture it -- explicitly
"with the features currently available to the classifier". The classifier's
recall is 9.6%.

The untried idea: an "echo" is a DATA-FEED artifact -- the block-time feed
copied the schedule instead of reporting a measurement. If that feed is
degraded for an airport on a given day, it should show up on ARRIVALS too
(ARR rows have BLOCK_TIME == SCHED_TIME by the same mechanism). And arrival
block times are NOT blanked -- they are fully visible in ranking.parquet.

So a live, per-(airport, day) arrival-side echo rate is computable directly
on the 2026 ranking set, unlike the current label-derived group encodings
which can only carry 2025 behaviour forward.

This script checks, for free:
  A. does the arrival-side echo phenomenon exist at all?
  B. does arrival echo rate predict DEPARTURE echo rate at (airport, day)?
  C. is the signal present in the 2026 ranking data?
  D. what is perfect echo routing actually worth on our holdout?

Run:  .venv/Scripts/python.exe tests/echo_arrival_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
TRAIN_GLOB = str(ROOT / "data" / "raw" / "training_2025-*.parquet")
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"

ECHO_ABS = 30  # |BLOCK - SCHED| below this = the feed echoed the schedule
RULE = "=" * 78

COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_OPERATOR_flt"]


def _prep(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        airport=pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")),
        day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        echo=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
              .dt.total_seconds().abs() < ECHO_ABS),
    )


def part_a(tr: pl.DataFrame) -> None:
    print(f"\n{RULE}\nA. Does the ARRIVAL-side echo phenomenon exist?\n{RULE}")
    for phase in ("DEP", "ARR"):
        d = tr.filter((pl.col("PHASE_mvt") == phase)
                      & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
        print(f"\n  {phase}: echo rate = {d['echo'].mean() * 100:5.2f}%   n={d.height:,}")
        g = d.group_by("airport").agg(pl.col("echo").mean().alias("r"),
                                      pl.len().alias("n")).sort("r", descending=True)
        print("    by airport: " + "  ".join(
            f"{r['airport']}={r['r'] * 100:.1f}%" for r in g.iter_rows(named=True)))


def part_b(tr: pl.DataFrame) -> None:
    print(f"\n{RULE}\nB. Does ARRIVAL echo rate predict DEPARTURE echo rate?\n{RULE}")
    print("  per (airport, day): arrivals are visible in ranking.parquet,")
    print("  departures are what we need to predict.")
    d = tr.filter(pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    arr = d.filter(pl.col("PHASE_mvt") == "ARR").group_by("airport", "day").agg(
        pl.col("echo").mean().alias("arr_echo"), pl.len().alias("n_arr"))
    dep = d.filter(pl.col("PHASE_mvt") == "DEP").group_by("airport", "day").agg(
        pl.col("echo").mean().alias("dep_echo"), pl.len().alias("n_dep"))
    j = arr.join(dep, on=["airport", "day"], how="inner").filter(
        (pl.col("n_arr") >= 20) & (pl.col("n_dep") >= 20))
    r = j.select(pl.corr("arr_echo", "dep_echo")).item()
    print(f"\n  overall correlation: r = {r:.3f}   ({j.height:,} airport-days)")
    print("\n  per airport:")
    for row in (j.group_by("airport").agg(
            pl.corr("arr_echo", "dep_echo").alias("r"),
            pl.col("dep_echo").mean().alias("dep_rate"),
            pl.col("arr_echo").mean().alias("arr_rate"),
            pl.len().alias("days")).sort("airport").iter_rows(named=True)):
        rv = row["r"]
        print(f"    {row['airport']:<6} r={rv if rv is not None else float('nan'):+.3f}  "
              f"dep_echo={row['dep_rate'] * 100:5.1f}%  "
              f"arr_echo={row['arr_rate'] * 100:5.1f}%  days={row['days']}")

    print("\n  departure echo rate by ARRIVAL-echo decile (the usable form):")
    jj = j.with_columns(
        b=((pl.col("arr_echo").rank("ordinal") - 1) * 10 // pl.len()))
    for row in (jj.group_by("b").agg(
            pl.col("arr_echo").min().alias("lo"), pl.col("arr_echo").max().alias("hi"),
            pl.col("dep_echo").mean().alias("dep"), pl.len().alias("n")
    ).sort("b").iter_rows(named=True)):
        print(f"    arr_echo d{row['b']} [{row['lo']:.3f},{row['hi']:.3f}] "
              f"-> dep_echo={row['dep']* 100:5.1f}%  ({row['n']} airport-days)")


def part_c(rk: pl.DataFrame) -> None:
    print(f"\n{RULE}\nC. Is the signal present in the 2026 RANKING set?\n{RULE}")
    arr = rk.filter((pl.col("PHASE_mvt") == "ARR")
                    & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    dep = rk.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"\n  ranking ARR rows with block time: {arr.height:,} "
          f"({arr.height / rk.filter(pl.col('PHASE_mvt') == 'ARR').height * 100:.1f}% of arrivals)")
    print(f"  ranking ARR echo rate: {arr['echo'].mean() * 100:.2f}%")
    print(f"  ranking DEP rows (block time blanked, as expected): {dep.height:,}, "
          f"block non-null = {dep['BLOCK_TIME_UTC_mvt'].is_not_null().sum()}")
    g = arr.group_by("airport", "day").agg(
        pl.col("echo").mean().alias("arr_echo"), pl.len().alias("n")).filter(
        pl.col("n") >= 20)
    print(f"\n  usable (airport, day) arrival-echo cells in ranking: {g.height:,}")
    print(f"  arr_echo spread: min={g['arr_echo'].min():.3f} "
          f"p50={g['arr_echo'].median():.3f} max={g['arr_echo'].max():.3f}")
    print("  (non-degenerate spread => the feature varies and is usable live)")


def part_d() -> None:
    print(f"\n{RULE}\nD. What is PERFECT echo routing worth on our holdout?\n{RULE}")
    ev = pl.read_parquet(PROD_EV)
    base = ev.select((pl.col("pred") - pl.col("taxi")).pow(2).mean()).item() ** 0.5
    orc = ev.with_columns(
        p=pl.when(pl.col("is_echo"))
        .then(pl.col("sched_takeoff_offset")).otherwise(pl.col("pred"))
    ).select((pl.col("p") - pl.col("taxi")).pow(2).mean()).item() ** 0.5
    # best-of-two oracle (routing, not just echo substitution)
    both = ev.with_columns(
        e=(pl.col("sched_takeoff_offset") - pl.col("taxi")).abs(),
        m=(pl.col("pred") - pl.col("taxi")).abs(),
    ).with_columns(
        p=pl.when(pl.col("e") < pl.col("m"))
        .then(pl.col("sched_takeoff_offset")).otherwise(pl.col("pred"))
    ).select((pl.col("p") - pl.col("taxi")).pow(2).mean()).item() ** 0.5
    print(f"\n  production                              {base:7.1f}s")
    print(f"  + perfect is_echo -> predict offset      {orc:7.1f}s   "
          f"(gain {base - orc:.1f}s)")
    print(f"  + perfect best-of-two routing            {both:7.1f}s   "
          f"(gain {base - both:.1f}s)")
    board = 302.0 / 337.6
    print(f"\n  board calibration: our 337.6 local -> 302 actual (x{board:.3f})")
    print(f"  implied board score at perfect echo routing: {orc * board:.0f}s")
    print(f"  implied board score at best-of-two routing:  {both * board:.0f}s")
    print("  leaderboard front is ~245s.")


def main() -> None:
    tr = _prep(pl.read_parquet(TRAIN_GLOB, columns=COLS))
    part_a(tr)
    part_b(tr)
    part_c(_prep(pl.read_parquet(RANKING, columns=COLS)))
    part_d()


if __name__ == "__main__":
    main()
