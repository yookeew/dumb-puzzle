"""Cross-fit NNLS stack of LightGBM / XGBoost / CatBoost holdout predictions.

Implements reports/stack_preregistration.md. Inputs: holdout evs written by
the Colab cell (cache/eval/<engine>_mixed_holdout_ev.parquet; lgb falls back
to prod_mixed_holdout_ev.parquet) and cache/adsb_blend_ev.parquet's
ADS-B columns for the "ADS-B on top" check.

Run:  .venv/Scripts/python.exe tests/stack_test.py
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
from adsb_blend_test import apply as adsb_apply, fit as adsb_fit, load as adsb_load  # noqa: E402

EVAL = ROOT / "cache" / "eval"
ENGINES = ["lgb", "xgb", "cat"]
JAN, JUL = "2025-01", "2025-07"
N_RES, SEED = 3000, 0


def load_preds() -> pl.DataFrame:
    df = None
    for e in ENGINES:
        p = EVAL / f"{e}_mixed_holdout_ev.parquet"
        if e == "lgb" and not p.exists():
            p = EVAL / "prod_mixed_holdout_ev.parquet"
        if not p.exists():
            print(f"  {e}: missing ({p.name}), skipped")
            continue
        ev = pl.read_parquet(p).select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt",
                                       pl.col("ym").cast(pl.Utf8), "taxi",
                                       pl.col("pred").alias(e))
        df = ev if df is None else df.join(ev.select("MVT_ID_mvt", e), on="MVT_ID_mvt", how="inner")
        print(f"  {e}: {p.name}, n={ev.height:,}")
    return df


def score(df: pl.DataFrame, base: np.ndarray, new: np.ndarray, label: str) -> tuple[float, float]:
    d = df.with_columns(se_b=pl.Series((base - df["taxi"].to_numpy()) ** 2),
                        se_t=pl.Series((new - df["taxi"].to_numpy()) ** 2))
    cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(),
                                           pl.len().alias("n"))
    pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(),
                                       cl["n"].to_numpy(), N_RES, SEED)
    print(f"  {label:<30} base={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}"
          f"  delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
    return pt, pw


def cross_fit_stack(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    out = np.empty(df.height)
    ws = {}
    for fit_m, app_m in ((JAN, JUL), (JUL, JAN)):
        f = df.filter(pl.col("ym") == fit_m)
        w, _ = nnls(f.select(cols).to_numpy(), f["taxi"].to_numpy())
        ws[f"fit {fit_m}"] = dict(zip(cols, np.round(w, 3)))
        m = (df["ym"] == app_m).to_numpy()
        out[m] = df.filter(pl.col("ym") == app_m).select(cols).to_numpy() @ w
    return out, ws


def main() -> None:
    print("loading holdout predictions")
    df = load_preds()
    days = adsb_load().select("MVT_ID_mvt", "day", "adsb_tier", "adsb_taxi", "eligible")
    df = df.join(days, on="MVT_ID_mvt", how="left")
    have = [e for e in ENGINES if e in df.columns]
    base = df["lgb"].to_numpy()
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()

    print("\nsingle models vs lgb (diagnostic):")
    for e in have[1:]:
        score(df, base, df[e].to_numpy(), e)
    print("\n2-model subsets (diagnostic, cross-fit):")
    for e in have[1:]:
        s, ws = cross_fit_stack(df, ["lgb", e])
        print(f"  weights {ws}")
        score(df, base, s, f"lgb+{e}")

    print(f"\nPRIMARY stack {'+'.join(have)} (cross-fit NNLS, no intercept):")
    stack, ws = cross_fit_stack(df, have)
    print(f"  weights {ws}")
    ptA, _ = score(df.filter(pl.col("ym") == JUL), base[jul], stack[jul], "A  Jul (fit Jan)")
    ptB, _ = score(df.filter(pl.col("ym") == JAN), base[jan], stack[jan], "B  Jan (fit Jul)")
    ptP, pwP = score(df, base, stack, "POOLED")
    adopt = ptA < 0 and ptB < 0 and pwP < 0.05
    print(f"  rule: {'ADOPT' if adopt else 'REJECT'}")
    print("\n  per airport:")
    with pl.Config(tbl_rows=20, tbl_formatting="ASCII_MARKDOWN", float_precision=2):
        print(df.with_columns(stack=pl.Series(stack)).group_by("ADEP_mvt").agg(
            ((pl.col("lgb") - pl.col("taxi")).pow(2).mean().sqrt()).alias("lgb"),
            ((pl.col("stack") - pl.col("taxi")).pow(2).mean().sqrt()).alias("stack"),
        ).with_columns(delta=pl.col("stack") - pl.col("lgb")).sort("ADEP_mvt"))

    if not adopt:
        return
    print("\nADS-B blend on top of the stack (refit, cross-fit, LIRF excluded):")
    d = df.with_columns(pred=pl.Series(stack))
    mask = (pl.col("ADEP_mvt") != "LIRF") & pl.col("eligible").fill_null(False)
    out = np.empty(d.height)
    for fit_m, app_m in ((JAN, JUL), (JUL, JAN)):
        p = adsb_fit(d.filter(pl.col("ym") == fit_m).filter(mask), per_airport=True)
        b = adsb_apply(d, p, mask).to_numpy()
        m = (d["ym"] == app_m).to_numpy()
        out[m] = b[m]
    a1, _ = score(d.filter(pl.col("ym") == JUL), stack[jul], out[jul], "stack+ADS-B vs stack, Jul")
    a2, _ = score(d.filter(pl.col("ym") == JAN), stack[jan], out[jan], "stack+ADS-B vs stack, Jan")
    score(d, base, out, "stack+ADS-B vs lgb, POOLED")
    print(f"  keep ADS-B on top: {a1 < 0 and a2 < 0}")


if __name__ == "__main__":
    main()
