"""Do the Stage 1 stand-occupancy bounds discriminate where it actually matters?

CLAUDE.md dropped Stage 4 (the [L, U] constraint layer) because median
U - L ~= 104 min ~= 14x the taxi sd -- too wide to constrain. But that is a
WHOLE-POPULATION median, dominated by ordinary flights whose ground time is
long and whose taxi is easy anyway.

tests/extreme_rows.py shows the error is not there. For offset >= 7200s
(9,101 rows, 23.8% of total MSE) the population is a near-binary mixture:
~98% pushed back very late and taxied normally (~800-1,300s), ~1.3% sat at
the stand (median taxi 9,931s). Getting one of the latter wrong costs
~8,800s. The existing echo classifier separates the arms (echo_prob 0.559
vs 0.005) but not confidently, and under squared loss an uncertain
probability is already optimally blended -- so the ONLY way to win is more
information, not recalibration (which S16 established the hard way).

Stand occupancy is physically exactly that information: if another aircraft
arrived at your stand, you must have pushed back before it. L_sec =
max(0, T - A_next) is already computed and already fed to the regressor as a
soft feature -- but nobody has asked whether it discriminates the two arms
on the population that carries the error.

Run:  .venv/Scripts/python.exe tests/bounds_on_extremes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"
RULE = "=" * 78

BIG_OFFSET = 7200


def _load() -> pl.DataFrame:
    f = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", "L_sec", "U_sec", "bound_binding", "has_next",
        "link_confidence", "actual_ground", "inbound_arr_delay")
    ev = pl.read_parquet(PROD_EV).join(f, on="MVT_ID_mvt", how="left")
    return ev.with_columns(
        se=(pl.col("pred") - pl.col("taxi")).pow(2),
        offset=pl.col("sched_takeoff_offset"),
        ratio=pl.col("taxi") / pl.col("sched_takeoff_offset"),
    )


def coverage(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n1. BOUND COVERAGE AND WIDTH, BY OFFSET BAND\n{RULE}")
    print("\n  L_sec is a LOWER bound on taxi implied by the next arrival at")
    print("  the same stand. What matters is not its width in absolute terms")
    print("  but how much it constrains RELATIVE to the offset in play.")
    print(f"\n  {'offset band':<20}{'n':>8}{'L>0':>8}{'med L':>10}"
          f"{'med L/offset':>14}{'% MSE':>8}")
    tot = ev["se"].sum()
    for lo, hi in [(0, 1800), (1800, 3600), (3600, 7200), (7200, 14400),
                   (14400, 10**9)]:
        b = ev.filter((pl.col("offset") >= lo) & (pl.col("offset") < hi))
        if b.height < 50:
            continue
        pos = b.filter(pl.col("L_sec") > 0)
        lab = f"[{lo:,}, {hi:,})" if hi < 10**9 else f">= {lo:,}"
        medl = pos["L_sec"].median() if pos.height else 0
        medr = (pos.select((pl.col("L_sec") / pl.col("offset")).median()).item()
                if pos.height else 0)
        print(f"  {lab:<20}{b.height:>8,}{pos.height / b.height * 100:>7.1f}%"
              f"{medl:>10,.0f}{medr:>13.2f}{b['se'].sum() / tot * 100:>8.1f}%")


def discrimination(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n2. DOES L_sec SEPARATE THE TWO ARMS? (offset >= 7200s)\n{RULE}")
    big = ev.filter(pl.col("offset") >= BIG_OFFSET)
    late = big.filter(pl.col("ratio") < 0.5)      # pushed back late, normal taxi
    sat = big.filter(pl.col("ratio") >= 0.9)      # sat at the stand, huge taxi
    print(f"\n  pushed-late arm : n={late.height:,}")
    print(f"  sat-at-stand arm: n={sat.height:,}")
    print(f"\n  {'metric':<28}{'pushed-late':>14}{'sat-at-stand':>15}")
    for col, lab in [("L_sec", "median L_sec"),
                     ("echo_prob", "mean echo_prob"),
                     ("actual_ground", "median actual_ground"),
                     ("inbound_arr_delay", "median inbound_arr_delay")]:
        a = late[col].median() if lab.startswith("median") else late[col].mean()
        b = sat[col].median() if lab.startswith("median") else sat[col].mean()
        a = a if a is not None else float("nan")
        b = b if b is not None else float("nan")
        print(f"  {lab:<28}{a:>14,.3f}{b:>15,.3f}")
    for lab, expr in [("has_next", pl.col("has_next")),
                      ("L_sec > 0", pl.col("L_sec") > 0),
                      ("L_sec > 0.5*offset", pl.col("L_sec") > 0.5 * pl.col("offset")),
                      ("L_sec > 0.8*offset", pl.col("L_sec") > 0.8 * pl.col("offset"))]:
        a = late.select(expr.mean()).item()
        b = sat.select(expr.mean()).item()
        a = a if a is not None else float("nan")
        b = b if b is not None else float("nan")
        print(f"  {lab:<28}{a * 100:>13.1f}%{b * 100:>14.1f}%")
    print("\n  A large gap on the L_sec rows = stand occupancy is telling us")
    print("  which arm the flight is in, and the classifier isn't using it.")


def oracle_value(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n3. WHAT WOULD USING THE BOUND BE WORTH?\n{RULE}")
    base = ev.select(pl.col("se").mean()).item() ** 0.5
    # A lower bound on taxi is a HARD physical constraint: clip predictions up
    # to L_sec wherever the link is trustworthy.
    for conf in (["high"], ["high", "med"]):
        c = ev.with_columns(
            p=pl.when(pl.col("link_confidence").is_in(conf)
                      & (pl.col("L_sec") > 0)
                      & (pl.col("L_sec") <= pl.col("offset")))
            .then(pl.max_horizontal("pred", "L_sec"))
            .otherwise(pl.col("pred"))
        )
        n_moved = c.filter(pl.col("p") != pl.col("pred")).height
        new = c.select((pl.col("p") - pl.col("taxi")).pow(2).mean()).item() ** 0.5
        print(f"\n  clip pred up to L_sec  (link_confidence in {conf}):")
        print(f"    rows moved: {n_moved:,}   RMSE {base:.1f} -> {new:.1f}  "
              f"({new - base:+.1f}s)")
    # how often is the bound actually violated by the truth (sanity on L_sec)
    v = ev.filter((pl.col("L_sec") > 0) & (pl.col("link_confidence") == "high"))
    if v.height:
        viol = v.filter(pl.col("taxi") < pl.col("L_sec")).height
        print(f"\n  sanity: true taxi < L_sec on {viol:,}/{v.height:,} "
              f"high-confidence rows ({viol / v.height * 100:.2f}%)")
        print("  (a low violation rate means L_sec is a trustworthy hard floor)")


def main() -> None:
    ev = _load()
    coverage(ev)
    discrimination(ev)
    oracle_value(ev)


if __name__ == "__main__":
    main()
