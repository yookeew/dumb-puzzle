"""Dissect the concentrated error population: recoverable or irreducible?

Context. Error is extraordinarily concentrated (tests/error_floor.py): the
top 1,000 holdout rows (0.29%) carry 58.5% of MSE, and solving them would
take local 337.6 -> 217.4. But perfect ECHO routing only reaches 296.7
(tests/echo_arrival_signal.py part D), so echo handling is not the mechanism
that gets there. This asks what those rows actually are.

Identity used throughout: taxi = offset - d, where offset =
sched_takeoff_offset = T - SOBT is ALWAYS observable (never blanked) and d =
AOBT - SOBT is the unknown the flip target models. So for a given offset,
all taxi error is d error.

The decisive question is whether d | offset is:
  (a) predictable structure the model is missing        -> a real lever, or
  (b) a bimodal/heavy-tailed mixture that is genuinely
      indistinguishable from observables                -> the conditional
      mean is already optimal under RMSE and the error is irreducible.

Run:  .venv/Scripts/python.exe tests/extreme_rows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"
RANKING_FEAT = FEAT_DIR / "ranking.parquet"
RULE = "=" * 78


def _load() -> pl.DataFrame:
    ev = pl.read_parquet(PROD_EV)
    return ev.with_columns(
        resid=pl.col("taxi") - pl.col("pred"),
        se=(pl.col("taxi") - pl.col("pred")).pow(2),
        offset=pl.col("sched_takeoff_offset"),
        # what the model implicitly believes d is, given it predicted `pred`
        d_hat=pl.col("sched_takeoff_offset") - pl.col("pred"),
    )


def direction(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n1. DIRECTION AND SHAPE OF THE WORST ROWS\n{RULE}")
    s = ev.sort("se", descending=True)
    for k in (100, 1000):
        t = s.head(k)
        under = (t["resid"] > 0).mean()
        print(f"\n  top {k}: model UNDER-predicts {under * 100:.0f}% of them")
        print(f"    median true taxi {t['taxi'].median():>9,.0f}s   "
              f"median pred {t['pred'].median():>9,.0f}s")
        print(f"    median offset    {t['offset'].median():>9,.0f}s   "
              f"median true d {t['d'].median():>9,.0f}s")
        print(f"    median d_hat     {t['d_hat'].median():>9,.0f}s")


def d_given_offset(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n2. IS d PREDICTABLE GIVEN offset?\n{RULE}")
    print("\n  taxi = offset - d, and offset is always observable. So for each")
    print("  offset band: how spread out is d, and how much of that spread is")
    print("  the model already capturing?")
    bands = [(0, 1800), (1800, 3600), (3600, 7200), (7200, 14400),
             (14400, 10**9)]
    print(f"\n  {'offset band':<20}{'n':>8}{'sd(d)':>10}{'sd(resid)':>11}"
          f"{'sd(taxi)':>10}{'% MSE':>8}")
    tot = ev["se"].sum()
    for lo, hi in bands:
        b = ev.filter((pl.col("offset") >= lo) & (pl.col("offset") < hi)
                      & pl.col("d").is_not_null())
        if b.height < 50:
            continue
        lab = f"[{lo:,}, {hi:,})" if hi < 10**9 else f">= {lo:,}"
        print(f"  {lab:<20}{b.height:>8,}{b['d'].std():>10,.0f}"
              f"{b['resid'].std():>11,.0f}{b['taxi'].std():>10,.0f}"
              f"{b['se'].sum() / tot * 100:>7.1f}%")
    print("\n  sd(resid) << sd(d) means the model IS explaining d in that band.")
    print("  sd(resid) ~= sd(d) means it is predicting ~a constant and the")
    print("  spread is unexplained.")


def bimodality(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n3. IS THE LARGE-OFFSET POPULATION BIMODAL?\n{RULE}")
    print("\n  For large offsets a flight either pushed back near schedule and")
    print("  then sat (d ~ 0, taxi ~ offset) or pushed back very late (d ~")
    print("  offset, taxi ~ normal). If those are indistinguishable from")
    print("  observables, the conditional mean is already RMSE-optimal.")
    big = ev.filter((pl.col("offset") >= 7200) & pl.col("d").is_not_null())
    print(f"\n  offset >= 7200s: n={big.height:,} "
          f"({big['se'].sum() / ev['se'].sum() * 100:.1f}% of total MSE)")
    print(f"\n  {'taxi/offset ratio':<22}{'n':>8}{'% of band':>11}"
          f"{'median taxi':>13}{'median d':>11}")
    r = big.with_columns(ratio=pl.col("taxi") / pl.col("offset"))
    cuts = [(0.0, 0.1, "~0  (pushed late)"), (0.1, 0.5, "0.1-0.5"),
            (0.5, 0.9, "0.5-0.9"), (0.9, 1.01, "~1  (sat at stand)")]
    for lo, hi, lab in cuts:
        c = r.filter((pl.col("ratio") >= lo) & (pl.col("ratio") < hi))
        if c.height == 0:
            continue
        print(f"  {lab:<22}{c.height:>8,}{c.height / r.height * 100:>10.1f}%"
              f"{c['taxi'].median():>13,.0f}{c['d'].median():>11,.0f}")
    print("\n  A U-shape (mass at both ends) = mixture. The question is whether")
    print("  anything observable separates the two arms.")

    # can the existing echo classifier separate them?
    lo_arm = r.filter(pl.col("ratio") < 0.5)
    hi_arm = r.filter(pl.col("ratio") >= 0.9)
    if lo_arm.height and hi_arm.height:
        print(f"\n  existing echo_prob on each arm:")
        print(f"    pushed-late arm  (ratio<0.5):  mean echo_prob="
              f"{lo_arm['echo_prob'].mean():.3f}  n={lo_arm.height:,}")
        print(f"    sat-at-stand arm (ratio>=0.9): mean echo_prob="
              f"{hi_arm['echo_prob'].mean():.3f}  n={hi_arm.height:,}")
        print("    (well separated => the classifier already distinguishes them;")
        print("     similar => this is the unexploited signal)")


def ranking_exposure() -> None:
    print(f"\n{RULE}\n4. HOW EXPOSED IS THE ACTUAL RANKING SET?\n{RULE}")
    r = pl.read_parquet(RANKING_FEAT).select("MVT_ID_mvt", "sched_takeoff_offset")
    ho = pl.read_parquet(PROD_EV).select(off="sched_takeoff_offset")
    print(f"\n  {'offset band':<20}{'ranking n':>12}{'ranking %':>12}"
          f"{'holdout %':>12}")
    for lo, hi in [(0, 1800), (1800, 3600), (3600, 7200), (7200, 14400),
                   (14400, 10**9)]:
        rn = r.filter((pl.col("sched_takeoff_offset") >= lo)
                      & (pl.col("sched_takeoff_offset") < hi)).height
        hn = ho.filter((pl.col("off") >= lo) & (pl.col("off") < hi)).height
        lab = f"[{lo:,}, {hi:,})" if hi < 10**9 else f">= {lo:,}"
        print(f"  {lab:<20}{rn:>12,}{rn / r.height * 100:>11.2f}%"
              f"{hn / ho.height * 100:>11.2f}%")
    print("\n  If ranking exposure ~= holdout exposure, holdout gains transfer.")


def main() -> None:
    ev = _load()
    direction(ev)
    d_given_offset(ev)
    bimodality(ev)
    ranking_exposure()


if __name__ == "__main__":
    main()
