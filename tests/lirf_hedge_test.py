"""LIRF long-delay hedge recalibration, per reports/lirf_hedge_preregistration.md.

Base: v21 cross-fit on the Jan+Jul 2025 holdout (tests/ltfm_egll_anatomy.py::v21_pred).
Per-cell shrunk mean-residual shift for LIRF departures with 1 h < T - SOBT <= 6 h,
cells = delay band x NM-unmatched, fit on one month and scored on the other.
Decision on full RMSE; trimmed RMSE is the guard.

Run:  .venv/Scripts/python.exe tests/lirf_hedge_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

JAN, JUL = A.JAN, A.JUL
SHRINK = 50
LIRF_CEIL = 140000
TRIM = 18000
N_RES, SEED = 3000, 0
BANDS = {"primary": [3600, 10800, 21600], "variant_6h_plus": [3600, 10800, 21600, float("inf")]}


def band_expr(edges: list[float]) -> pl.Expr:
    e = pl.lit(None, dtype=pl.Utf8)
    for lo, hi in reversed(list(zip(edges[:-1], edges[1:]))):
        e = pl.when((pl.col("offset") > lo) & (pl.col("offset") <= hi)).then(pl.lit(f"{lo / 3600:g}-{hi / 3600:g}h")).otherwise(e)
    return pl.when(pl.col("ADEP_mvt") == "LIRF").then(e).otherwise(None)


def fit(f: pl.DataFrame) -> pl.DataFrame:
    return (f.filter(pl.col("band").is_not_null()).group_by("band", "nmu")
            .agg(pl.len().alias("n_fit"), (pl.col("taxi") - pl.col("pred")).mean().alias("mean_r"))
            .with_columns(shift=pl.col("n_fit") / (pl.col("n_fit") + SHRINK) * pl.col("mean_r")))


def cross(df: pl.DataFrame) -> tuple[np.ndarray, dict]:
    out, shifts = df["pred"].to_numpy().copy(), {}
    d = df.with_row_index("i")
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = fit(df.filter(pl.col("ym") == fm))
        shifts[f"fit {fm}"] = s.sort("band", "nmu")
        t = d.filter((pl.col("ym") == am) & pl.col("band").is_not_null()).join(s, on=["band", "nmu"], how="left")
        idx = t["i"].to_numpy()
        out[idx] = np.clip(out[idx] + t["shift"].fill_null(0).to_numpy(), 0, LIRF_CEIL)
    return out, shifts


def score(df: pl.DataFrame, new: np.ndarray, label: str) -> tuple[float, float]:
    d = df.with_columns(se_b=(pl.col("pred") - pl.col("taxi")).pow(2),
                        se_t=pl.Series((new - df["taxi"].to_numpy()) ** 2))
    cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
    pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), N_RES, SEED)
    print(f"  {label:<30} base={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
          f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
    return pt, pw


def evaluate(df: pl.DataFrame, name: str) -> bool:
    print(f"\n=== {name} ===")
    out, shifts = cross(df)
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        for k, s in shifts.items():
            print(f"  {k}:\n{s}")
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()
    trim = (df["taxi"] <= TRIM).to_numpy()
    print(" FULL RMSE (decides):")
    ptA, _ = score(df.filter(pl.Series(jul)), out[jul], "A  Jul (fit Jan)")
    ptB, _ = score(df.filter(pl.Series(jan)), out[jan], "B  Jan (fit Jul)")
    _, pwP = score(df, out, "POOLED")
    print(" TRIMMED RMSE (guard):")
    _, pwT = score(df.filter(pl.Series(trim)), out[trim], "POOLED trimmed")
    ok = ptA < 0 and ptB < 0 and pwP < 0.05 and pwT < 0.9
    t = df.with_columns(new=pl.Series(out)).filter(pl.col("band").is_not_null())
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        print(" per cell (population rows):")
        print(t.group_by("band", "nmu").agg(pl.len().alias("n"),
              ((pl.col("pred") - pl.col("taxi")).pow(2).mean().sqrt()).alias("base"),
              ((pl.col("new") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new")).sort("band", "nmu"))
    print(f"  rule: {'ADOPT' if ok else 'REJECT'}")
    return ok


def main() -> None:
    df = A.load()
    df = df.with_columns(pred=pl.Series(A.v21_pred(df)))
    print(f"holdout rows {df.height:,}; v21 cross-fit full RMSE "
          f"{np.sqrt(((df['pred'] - df['taxi']) ** 2).mean()):.2f}")
    for name, edges in BANDS.items():
        d = df.with_columns(band=band_expr(edges))
        print(f"\n{name}: population rows {d['band'].is_not_null().sum():,}")
        ok = evaluate(d, name)
        if name == "primary":
            print(f"\nPRIMARY DECISION: {'ADOPT -> build v22, board A/B' if ok else 'REJECT -> stop'}")


if __name__ == "__main__":
    main()
