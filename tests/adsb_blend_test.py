"""ADS-B pushback blend on the Jan+Jul 2025 holdout, per the pre-registration.

Implements reports/adsb_blend_preregistration.md (committed d3ed616 before
any result) exactly:

  0. Data check: detector outputs old box vs new box on the 4 overlap days.
  1. Primary blend: per-airport lag median + one weight per tier (appear,
     dwell), LIRF excluded; month cross-fit (A: Jan->Jul, B: Jul->Jan);
     paired RMSE over all holdout rows, (airport, day) cluster bootstrap.
  2. Secondary: per-airport-tier weights (cells >= 200 fit rows).
  3. Winter-only terms W1 (within-Jan odd/even day split) and W2 (drop the 3
     best Jan days) -- always computed, only *decisive* if the summer case
     arises.
  4. LIRF: H_echo diagnosis, then variant L (echo_prob < 0.5 gate) if it holds.
  5. EDDM with its 3 worst days (largest model error) shown separately.

Inputs: cache/eval/prod_mixed_holdout_ev.parquet (tests/adsb_vs_model_test.py),
cache/adsb_pushback/ (src/link/adsb_pushback.py), data/raw/ for MVT_TIME.

Run:  .venv/Scripts/python.exe tests/adsb_blend_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402

EV = ROOT / "cache" / "eval" / "prod_mixed_holdout_ev.parquet"
DET = ROOT / "cache" / "adsb_pushback"
OLD = ROOT / "cache" / "adsb_pushback_oldbox"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "cache" / "adsb_blend_ev.parquet"
JAN, JUL = "2025-01", "2025-07"
TIERS = ("appear", "dwell")
MIN_LAG_ROWS = 100
MIN_CELL_ROWS = 200
ECHO_GATE = 0.5
N_RES, SEED = 3000, 0
RULE = "=" * 78


# ------------------------------------------------------------------ data
def load() -> pl.DataFrame:
    ev = pl.read_parquet(EV).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64),
                                          pl.col("ym").cast(pl.Utf8))
    mvt = (pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                   mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                   day=pl.col("MVT_TIME_UTC_mvt").dt.date())
           .collect())
    det = pl.concat([pl.read_parquet(p) for p in sorted(DET.glob("day=*.parquet"))]).select(
        "MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts")
    df = ev.join(mvt, on="MVT_ID_mvt", how="left").join(det, on="MVT_ID_mvt", how="left")
    assert df.height == ev.height
    df = df.with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts"),
                         eligible=pl.col("adsb_tier").is_in(TIERS).fill_null(False))
    return df


# ------------------------------------------------------------------ blend
def fit(f: pl.DataFrame, per_airport: bool = False) -> dict:
    """Lag medians per airport and tier weights (optionally per airport-tier) on eligible rows."""
    f = f.filter("eligible")
    pooled = f.select((pl.col("adsb_taxi") - pl.col("taxi")).median()).item()
    lags = {a: (g.select((pl.col("adsb_taxi") - pl.col("taxi")).median()).item()
                if g.height >= MIN_LAG_ROWS else pooled)
            for (a,), g in f.group_by("ADEP_mvt")}
    f = _correct(f, {"lags": lags, "pooled": pooled})

    def w(g: pl.DataFrame) -> float:
        dlt = (g["adsb_c"] - g["pred"]).to_numpy()
        r = (g["taxi"] - g["pred"]).to_numpy()
        return float(np.clip((r * dlt).sum() / max((dlt * dlt).sum(), 1e-9), 0, 1))

    weights = {t: w(f.filter(pl.col("adsb_tier") == t)) for t in TIERS
               if f.filter(pl.col("adsb_tier") == t).height}
    cells = {}
    if per_airport:
        for (a, t), g in f.group_by("ADEP_mvt", "adsb_tier"):
            if g.height >= MIN_CELL_ROWS:
                cells[(a, t)] = w(g)
    return {"lags": lags, "pooled": pooled, "w": weights, "cells": cells}


def _correct(df: pl.DataFrame, p: dict) -> pl.DataFrame:
    lag = pl.col("ADEP_mvt").replace_strict(p["lags"], default=p["pooled"], return_dtype=pl.Float64)
    return df.with_columns(adsb_c=pl.col("adsb_taxi") - lag)


def apply(df: pl.DataFrame, p: dict, mask: pl.Expr) -> pl.Series:
    """Blended prediction for every row of df; rows outside `mask` keep pred."""
    d = _correct(df, p)
    wt = pl.col("adsb_tier").replace_strict(p["w"], default=0.0, return_dtype=pl.Float64)
    if p["cells"]:
        key = pl.concat_str("ADEP_mvt", pl.lit("|"), "adsb_tier")
        cell = {f"{a}|{t}": v for (a, t), v in p["cells"].items()}
        wt = key.replace_strict(cell, default=None, return_dtype=pl.Float64).fill_null(wt)
    out = d.select(pl.when(mask).then(pl.col("pred") + wt * (pl.col("adsb_c") - pl.col("pred")))
                   .otherwise(pl.col("pred")).alias("blend"))
    return out["blend"]


# ------------------------------------------------------------------ scoring
def score(df: pl.DataFrame, blend: pl.Series, label: str) -> tuple[float, float]:
    d = df.with_columns(blend=blend).with_columns(
        se_b=(pl.col("pred") - pl.col("taxi")).pow(2), se_t=(pl.col("blend") - pl.col("taxi")).pow(2))
    cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(),
                                           pl.len().alias("n"))
    pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(),
                                       cl["n"].to_numpy(), N_RES, SEED)
    rb, rt = d["se_b"].mean() ** 0.5, d["se_t"].mean() ** 0.5
    print(f"  {label:<34} model={rb:7.2f}  blend={rt:7.2f}  delta={pt:+6.2f}  "
          f"CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}  n={d.height:,}")
    return pt, pw


def breakdown(df: pl.DataFrame, blend: pl.Series, by: str) -> None:
    d = df.with_columns(blend=blend)
    t = d.group_by(by).agg(
        pl.len().alias("n"), pl.col("eligible").sum().alias("n_elig"),
        ((pl.col("pred") - pl.col("taxi")).pow(2).mean().sqrt()).alias("model"),
        ((pl.col("blend") - pl.col("taxi")).pow(2).mean().sqrt()).alias("blend"),
    ).with_columns(delta=pl.col("blend") - pl.col("model")).sort(by)
    with pl.Config(tbl_rows=40, tbl_formatting="ASCII_MARKDOWN", float_precision=2):
        print(t)


def cross_fit(df: pl.DataFrame, mask: pl.Expr, per_airport: bool = False
              ) -> tuple[pl.Series, dict, dict]:
    jan, jul = df["ym"] == JAN, df["ym"] == JUL
    pA = fit(df.filter(pl.col("ym") == JAN).filter(mask), per_airport)   # A: fit Jan -> apply Jul
    pB = fit(df.filter(pl.col("ym") == JUL).filter(mask), per_airport)   # B: fit Jul -> apply Jan
    bA, bB = apply(df, pA, mask), apply(df, pB, mask)
    blend = pl.Series("blend", np.where(jul.to_numpy(), bA.to_numpy(), bB.to_numpy()))
    return blend, pA, pB


def show_params(tag: str, p: dict) -> None:
    lags = ", ".join(f"{a}:{v:+.0f}" for a, v in sorted(p["lags"].items()))
    print(f"  {tag}: w={ {k: round(v, 3) for k, v in p['w'].items()} }  "
          f"lag(adsb-taxi, s)=[{lags}]  pooled={p['pooled']:+.0f}"
          + (f"  cells={ {f'{a}/{t}': round(v, 2) for (a, t), v in sorted(p['cells'].items())} }"
             if p["cells"] else ""))


# ------------------------------------------------------------------ main
def box_check() -> None:
    print(f"{RULE}\n0. Data check: old box vs new box detector outputs (4 overlap days)\n{RULE}")
    for p in sorted(OLD.glob("day=*.parquet")):
        new_p = DET / p.name
        if not new_p.exists():
            continue
        o = pl.read_parquet(p).filter(pl.col("adsb_tier").is_in(TIERS))
        n = pl.read_parquet(new_p).filter(pl.col("adsb_tier").is_in(TIERS))
        j = o.join(n, on="MVT_ID_mvt", how="inner", suffix="_new")
        dpb = (j["adsb_pushback_ts_new"] - j["adsb_pushback_ts"]).abs()
        print(f"  {p.stem[4:]}: old {o.height:,}  new {n.height:,}  both {j.height:,}  "
              f"old-only {o.height - j.height:,}  new-only {n.height - j.height:,}  "
              f"|dpushback|=0 on {(dpb == 0).mean() * 100:.1f}%  >60s on {(dpb > 60).sum()} rows  "
              f"tier changed {(j['adsb_tier'] != j['adsb_tier_new']).sum()}")


def main() -> None:
    box_check()
    df = load()
    non_lirf = (pl.col("ADEP_mvt") != "LIRF") & pl.col("eligible")
    print(f"\nholdout rows {df.height:,}; eligible (appear+dwell) {df['eligible'].sum():,}; "
          f"eligible excl LIRF {df.select(non_lirf.sum()).item():,}")
    with pl.Config(tbl_rows=40, tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        print(df.group_by("ym", "ADEP_mvt").agg((pl.col("eligible").mean() * 100).alias("elig%"))
              .pivot(on="ADEP_mvt", index="ym", values="elig%").sort("ym"))

    print(f"\n{RULE}\n1. PRIMARY: lag per airport + tier weights, LIRF excluded, month cross-fit\n{RULE}")
    blend, pA, pB = cross_fit(df, non_lirf)
    show_params("A (fit Jan)", pA)
    show_params("B (fit Jul)", pB)
    jan, jul = pl.col("ym") == JAN, pl.col("ym") == JUL
    sel = lambda e: df.with_row_index("_i").filter(e)["_i"].to_numpy()  # noqa: E731
    b = blend.to_numpy()
    ptA, pwA = score(df.filter(jul), pl.Series(b[sel(jul)]), "A  Jul 2025 (fit Jan)")
    ptB, pwB = score(df.filter(jan), pl.Series(b[sel(jan)]), "B  Jan 2025 (fit Jul)")
    ptP, pwP = score(df, blend, "POOLED")
    adopt = ptA < 0 and ptB < 0 and pwP < 0.05
    print(f"  standing rule (both months < 0 and pooled P(worse) < 0.05): "
          f"{'ADOPT' if adopt else 'REJECT'}")
    print("\n  per airport:")
    breakdown(df, blend, "ADEP_mvt")
    print("  per tier (eligible rows only):")
    e = df.with_columns(blend=blend).filter(non_lirf)
    breakdown(e.drop("blend"), e["blend"], "adsb_tier")
    df.with_columns(blend=blend).select("MVT_ID_mvt", "ADEP_mvt", "ym", "day", "adsb_tier",
                                        "eligible", "taxi", "pred", "adsb_taxi", "blend",
                                        "is_echo", "echo_prob").write_parquet(OUT)

    print(f"\n{RULE}\n2. SECONDARY: per-airport-tier weights (cells >= {MIN_CELL_ROWS})\n{RULE}")
    blend2, qA, qB = cross_fit(df, non_lirf, per_airport=True)
    show_params("A (fit Jan)", qA)
    show_params("B (fit Jul)", qB)
    b2 = blend2.to_numpy()
    rm = lambda x, e_: float(np.sqrt(np.mean((x[sel(e_)] - df.filter(e_)["taxi"].to_numpy()) ** 2)))  # noqa: E731
    beatA, beatB = rm(b2, jul) < rm(b, jul), rm(b2, jan) < rm(b, jan)
    print(f"  vs primary: Jul {rm(b, jul):.2f} -> {rm(b2, jul):.2f}  Jan {rm(b, jan):.2f} -> "
          f"{rm(b2, jan):.2f}  => {'ADOPT over primary' if beatA and beatB else 'not adopted'}")

    print(f"\n{RULE}\n3. Winter-only terms (decisive only if Jan gains and Jul shows nothing)\n{RULE}")
    summer_case = (ptB < 0 and pwB < 0.05) and (ptA >= -0.5 or pwA >= 0.10)
    print(f"  summer case triggered: {summer_case}  (B delta {ptB:+.2f} P(worse) {pwB:.3f}; "
          f"A delta {ptA:+.2f} P(worse) {pwA:.3f})")
    dj = df.filter(jan)
    odd = pl.col("day").dt.day() % 2 == 1
    p_odd = fit(dj.filter(odd & non_lirf))
    p_even = fit(dj.filter(~odd & non_lirf))
    bo, be = apply(dj, p_even, non_lirf).to_numpy(), apply(dj, p_odd, non_lirf).to_numpy()
    w1 = pl.Series(np.where(dj.select(odd).to_series().to_numpy(), bo, be))
    _, pw1 = score(dj, w1, "W1 Jan odd/even day split")
    per_day = dj.with_columns(blend=pl.Series(b[sel(jan)])).group_by("day").agg(
        ((pl.col("blend") - pl.col("taxi")).pow(2) - (pl.col("pred") - pl.col("taxi")).pow(2))
        .sum().alias("dse")).sort("dse")
    top3 = per_day.head(3)["day"].to_list()
    dj3 = dj.with_columns(blend=pl.Series(b[sel(jan)])).filter(~pl.col("day").is_in(top3))
    pt2, _ = score(dj3.drop("blend"), dj3["blend"], f"W2 Jan minus best 3 days {[str(d) for d in top3]}")
    print(f"  W1 {'PASS' if pw1 < 0.05 else 'FAIL'}   W2 {'PASS' if pt2 < 0 else 'FAIL'}"
          f"   W3 (Jan-2026-only application) is a deployment rule, not a test")

    print(f"\n{RULE}\n4. LIRF diagnosis (H_echo) and variant L\n{RULE}")
    li = df.filter(pl.col("ADEP_mvt") == "LIRF", pl.col("eligible"))
    nl = df.filter(non_lirf)
    rm_nl = float(np.sqrt(((nl["adsb_taxi"] - nl["taxi"]) ** 2).mean()))
    for flag, g in li.group_by("is_echo"):
        err = (g["adsb_taxi"] - g["taxi"]).to_numpy()
        print(f"  LIRF is_echo={flag[0]}: n={g.height:,}  ADS-B RMSE={np.sqrt(np.mean(err ** 2)):.0f}s  "
              f"median(adsb-taxi)={np.median(err):+.0f}s  share of LIRF ADS-B SSE="
              f"{(err ** 2).sum() / ((li['adsb_taxi'] - li['taxi']) ** 2).sum() * 100:.0f}%  "
              f"model RMSE={np.sqrt(((g['pred'] - g['taxi']) ** 2).mean()):.0f}s")
    ne = li.filter(~pl.col("is_echo").cast(pl.Boolean))
    rm_ne = float(np.sqrt(((ne["adsb_taxi"] - ne["taxi"]) ** 2).mean())) if ne.height else float("inf")
    echo_share = (((li.filter(pl.col("is_echo").cast(pl.Boolean))["adsb_taxi"]
                    - li.filter(pl.col("is_echo").cast(pl.Boolean))["taxi"]) ** 2).sum()
                  / max(((li["adsb_taxi"] - li["taxi"]) ** 2).sum(), 1e-9))
    h_echo = rm_ne <= 1.5 * rm_nl and echo_share > 0.5
    print(f"  pooled non-LIRF ADS-B RMSE {rm_nl:.0f}s; non-echo LIRF {rm_ne:.0f}s "
          f"(<= {1.5 * rm_nl:.0f}? {rm_ne <= 1.5 * rm_nl}); echo share {echo_share * 100:.0f}% (> 50%? "
          f"{echo_share > 0.5})  => H_echo {'HOLDS' if h_echo else 'FAILS'}")
    if h_echo:
        gate = non_lirf | ((pl.col("ADEP_mvt") == "LIRF") & pl.col("eligible")
                           & (pl.col("echo_prob") < ECHO_GATE))
        blendL, lA, lB = cross_fit(df, gate)
        show_params("L-A", lA)
        show_params("L-B", lB)
        bl = blendL.to_numpy()
        lirf = pl.col("ADEP_mvt") == "LIRF"
        for tag, e_ in (("Jul", jul & lirf), ("Jan", jan & lirf)):
            print(f"  LIRF {tag}: model {rm(df['pred'].to_numpy(), e_):.1f}  primary(excl) "
                  f"{rm(b, e_):.1f}  variant L {rm(bl, e_):.1f}")
        score(df, blendL, "variant L POOLED")

    print(f"\n{RULE}\n5. EDDM, with its 3 worst model-error days split out\n{RULE}")
    ed = df.with_columns(blend=blend).filter(pl.col("ADEP_mvt") == "EDDM")
    worst = (ed.group_by("day").agg((pl.col("pred") - pl.col("taxi")).pow(2).mean().sqrt().alias("r"))
             .sort("r", descending=True).head(3)["day"].to_list())
    for tag, g in (("worst-3 days " + ",".join(str(d) for d in worst), ed.filter(pl.col("day").is_in(worst))),
                   ("other days", ed.filter(~pl.col("day").is_in(worst)))):
        print(f"  EDDM {tag}: n={g.height:,} elig={g['eligible'].sum():,}  "
              f"model={np.sqrt(((g['pred'] - g['taxi']) ** 2).mean()):.1f}  "
              f"blend={np.sqrt(((g['blend'] - g['taxi']) ** 2).mean()):.1f}")


if __name__ == "__main__":
    main()
