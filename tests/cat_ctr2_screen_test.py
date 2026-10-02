"""Pre-registered stage-1 screen: CatBoost with categorical pairs (max_ctr_complexity=2).

reports/cat_ctr2_preregistration.md. Uncorrected models: cross-fit NNLS(lgb, cat_ctr2)
vs NNLS(lgb, cat), where cat is the rerun CatBoost v21 is built on. Decides on
trimmed RMSE (holdout labels > 5 h excluded) with the full-RMSE guard.

Needs cache/eval/{lgb_mixed,cat_mixed_rerun,cat_ctr2_mixed}_holdout_ev.parquet.

Run:  .venv/Scripts/python.exe tests/cat_ctr2_screen_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402
from stack_r20k_test import EVAL, JAN, JUL, MONSTER_S, ROOT, cross, ev  # noqa: E402

N_MONSTER_EXPECTED = 31


def main() -> None:
    base = pl.read_parquet(EVAL / "lgb_mixed_holdout_ev.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", pl.col("ym").cast(pl.Utf8), "taxi", pl.col("pred").alias("lgb"))
    day = pl.read_parquet(ROOT / "cache" / "features" / "holdout_gap2025.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), day=pl.col("T").dt.date())
    df = (base.join(day, on="MVT_ID_mvt", how="left")
          .join(ev("cat_mixed_rerun", "cat"), on="MVT_ID_mvt")
          .join(ev("cat_ctr2_mixed", "ctr2"), on="MVT_ID_mvt"))
    if df.height != base.height:
        raise SystemExit(f"holdout join lost rows: {base.height} -> {df.height}")
    t = df["taxi"].to_numpy()
    keep = t <= MONSTER_S
    if int((~keep).sum()) != N_MONSTER_EXPECTED:
        raise SystemExit(f"expected {N_MONSTER_EXPECTED} monster rows, got {int((~keep).sum())}")
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()
    alln = np.ones(df.height, bool)

    old, wo = cross(df, ["lgb", "cat"])
    new, wn = cross(df, ["lgb", "ctr2"])
    print(f"rows {df.height:,}; monster labels excluded from trimmed: {(~keep).sum()}")
    print(f"stack weights  control {wo}\n               ctr2    {wn}")

    def score(a: np.ndarray, b: np.ndarray, mask: np.ndarray, label: str):
        d = df.with_columns(se_b=pl.Series((a - t) ** 2), se_t=pl.Series((b - t) ** 2)).filter(pl.Series(mask))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), 3000, 0)
        print(f"  {label:<26} base={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
              f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    print("\nSCREEN: NNLS(lgb, cat_ctr2) vs NNLS(lgb, cat) (cross-fit)")
    print(" TRIMMED (decides):")
    d_jul, _ = score(old, new, jul & keep, "Jul trimmed (fit Jan)")
    d_jan, _ = score(old, new, jan & keep, "Jan trimmed (fit Jul)")
    d_pool, p_pool = score(old, new, keep, "POOLED trimmed")
    print(" FULL (guard):")
    score(old, new, jul, "Jul full")
    score(old, new, jan, "Jan full")
    _, p_full = score(old, new, alln, "POOLED full")
    ok = d_pool < 0 and p_pool < 0.05 and d_jan < 0 and d_jul < 0 and p_full < 0.9
    print(f"\nrule: pooled trimmed delta<0 & P<0.05 [{d_pool:+.2f}, {p_pool:.3f}], both months trimmed<0 "
          f"[Jan {d_jan:+.2f}, Jul {d_jul:+.2f}], full guard P<0.9 [{p_full:.3f}]"
          f"\n  -> {'PASS (stage 2, if the timing rule allows)' if ok else 'FAIL (stop)'}")

    print("\nDIAGNOSTICS (not decisive)")
    c, c2 = df["cat"].to_numpy(), df["ctr2"].to_numpy()
    score(c, c2, keep, "single: ctr2 vs cat, trim")
    score(c, c2, alln, "single: ctr2 vs cat, full")
    d2 = df.with_columns(old=pl.Series(old), new=pl.Series(new))
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=2, tbl_rows=20):
        agg = lambda c: ((pl.col(c) - pl.col("taxi")) ** 2).mean().sqrt()  # noqa: E731
        for lbl, frame in (("trimmed", d2.filter(pl.Series(keep))), ("full", d2)):
            print(f"stack per airport, {lbl}:")
            print(frame.group_by("ADEP_mvt").agg(agg("old").alias("control"), agg("new").alias("ctr2"),
                                                 agg("cat").alias("cat_single"), agg("ctr2").alias("ctr2_single"),
                                                 pl.len())
                  .with_columns(delta=pl.col("ctr2") - pl.col("control")).sort("ADEP_mvt"))
        print("stack per decile of true taxi (all rows):")
        print(d2.with_columns(dec=pl.col("taxi").qcut(10, labels=[str(i) for i in range(1, 11)]))
              .group_by("dec").agg(pl.col("taxi").max().alias("taxi_max"), agg("old").alias("control"),
                                   agg("new").alias("ctr2"), pl.len())
              .with_columns(delta=pl.col("ctr2") - pl.col("control")).sort("taxi_max"))


if __name__ == "__main__":
    main()
