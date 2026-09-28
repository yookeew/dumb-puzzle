"""Pre-registered follow-up: does CatBoost with a 20k-round cap improve the stack?

reports/stack_preregistration.md ("Follow-up: CatBoost with a 20,000-round
cap"). Compares the cross-fit NNLS stacks lgb+cat_r20k vs lgb+cat on the
Jan+Jul 2025 holdout. Also reports trimmed RMSE (holdout labels > 5 h
excluded, a fixed set), per PROGRESS.md §46.

Run:  .venv/Scripts/python.exe tests/stack_r20k_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402

EVAL = ROOT / "cache" / "eval"
JAN, JUL = "2025-01", "2025-07"
MONSTER_S = 5 * 3600


def ev(name: str, col: str) -> pl.DataFrame:
    return pl.read_parquet(EVAL / f"{name}_holdout_ev.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias(col))


def cross(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    out, ws = np.empty(df.height), {}
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        f = df.filter(pl.col("ym") == fm)
        w, _ = nnls(f.select(cols).to_numpy(), f["taxi"].to_numpy())
        ws[f"fit {fm}"] = dict(zip(cols, np.round(w, 3)))
        m = (df["ym"] == am).to_numpy()
        out[m] = df.filter(pl.col("ym") == am).select(cols).to_numpy() @ w
    return out, ws


def main() -> None:
    base = pl.read_parquet(EVAL / "lgb_mixed_holdout_ev.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", pl.col("ym").cast(pl.Utf8), "taxi",
        pl.col("pred").alias("lgb"), day=pl.lit(None))
    lab = pl.read_parquet(ROOT / "cache" / "features" / "holdout_gap2025.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), day=pl.col("T").dt.date())
    df = (base.drop("day").join(lab, on="MVT_ID_mvt", how="left")
          .join(ev("cat_mixed", "cat"), on="MVT_ID_mvt").join(ev("cat_mixed_r20k", "cat20"), on="MVT_ID_mvt"))
    t = df["taxi"].to_numpy()
    rm = lambda x, m=None: float(np.sqrt(np.mean((x - t) ** 2 if m is None else (x[m] - t[m]) ** 2)))  # noqa: E731
    keep = t <= MONSTER_S
    print(f"rows {df.height:,}; monster labels (> 5 h) excluded from trimmed RMSE: {(~keep).sum()}")
    print("single models, full / trimmed:")
    for c in ("lgb", "cat", "cat20"):
        x = df[c].to_numpy()
        print(f"  {c:<6} {rm(x):7.2f} / {rm(x, keep):7.2f}")

    old, wo = cross(df, ["lgb", "cat"])
    new, wn = cross(df, ["lgb", "cat20"])
    print(f"\nstack weights  old {wo}\n               new {wn}")

    def score(mask: np.ndarray, label: str) -> tuple[float, float]:
        d = df.with_columns(se_b=pl.Series((old - t) ** 2), se_t=pl.Series((new - t) ** 2)).filter(pl.Series(mask))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), 3000, 0)
        print(f"  {label:<26} old={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
              f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()
    print("\nstack lgb+cat20 vs lgb+cat (cross-fit), FULL:")
    a, _ = score(jul, "A  Jul (fit Jan)")
    b, _ = score(jan, "B  Jan (fit Jul)")
    _, pw = score(np.ones(df.height, bool), "POOLED")
    print("TRIMMED (labels <= 5 h):")
    score(jul & keep, "A  Jul trimmed")
    score(jan & keep, "B  Jan trimmed")
    score(keep, "POOLED trimmed")
    print(f"\nrule (both months < 0 and pooled P(worse) < 0.05, full RMSE): "
          f"{'ADOPT cat_r20k' if a < 0 and b < 0 and pw < 0.05 else 'KEEP current stack'}")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=2, tbl_rows=12):
        print(df.with_columns(old=pl.Series(old), new=pl.Series(new)).group_by("ADEP_mvt").agg(
            ((pl.col("old") - pl.col("taxi")).pow(2).mean().sqrt()).alias("old"),
            ((pl.col("new") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new"))
            .with_columns(delta=pl.col("new") - pl.col("old")).sort("ADEP_mvt"))


if __name__ == "__main__":
    main()
