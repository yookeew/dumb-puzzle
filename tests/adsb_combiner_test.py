"""Learned ADS-B combiner gate, per reports/adsb_combiner_preregistration.md.

Base: v23's pipeline cross-fit on the Jan+Jul 2025 holdout (tests/adsb_v3_test.py).
Treatment: s = NNLS(lgb, catcorr) from the fit month, plus a LightGBM on
(taxi - s) from the stack prediction and every ADS-B detector field (v4 cache),
fit on one month and scored on the other. LIRF keeps the base value.

Run:  .venv/Scripts/python.exe tests/adsb_combiner_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ltfm_egll_anatomy as A  # noqa: E402
import adsb_v3_test as V  # noqa: E402

JAN, JUL = A.JAN, A.JUL
V4 = ROOT / "cache" / "adsb_pushback_v4"
TRIM = 18000
TIERS = ["none", "appear", "dwell", "pass", "fb_dwell", "fb_appear"]
PARAMS = dict(objective="regression", num_leaves=31, learning_rate=0.03, min_data_in_leaf=500,
              lambda_l2=10.0, feature_fraction=0.9, max_bin=127, seed=42, deterministic=True,
              force_row_wise=True, verbose=-1)
ES_PATIENCE, MAX_ROUNDS = 200, 5000
FEATS = ["airport", "hour", "s", "tier", "matched", "fallback", "coverage", "a_rel", "adsb_taxi",
         "adsb_pb_dist_m", "adsb_pb_gs", "adsb_pb_gap_s", "L_rel", "L", "adsb_first_own_m",
         "adsb_first_gs", "adsb_min_own_m", "adsb_n_pts", "adsb_takeoff_gap_s"]
CATS = ["airport", "tier"]


def stack(df: pl.DataFrame, fit_month: str) -> np.ndarray:
    eng = ["lgb", "catcorr"]
    f = df.filter(pl.col("ym") == fit_month)
    w, _ = nnls(f.select(eng).to_numpy(), f["taxi"].to_numpy())
    return df.select(eng).to_numpy() @ w


def features(df: pl.DataFrame, s: np.ndarray, airports: list[str]) -> pl.DataFrame:
    d = df.with_columns(s=pl.Series(s))
    return d.select(
        airport=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(airports)}, default=-1,
                                                  return_dtype=pl.Int32),
        hour=pl.col("hour").cast(pl.Float64), s=pl.col("s"),
        tier=pl.col("adsb_tier").fill_null("none").replace_strict({t: i for i, t in enumerate(TIERS)},
                                                                   default=0, return_dtype=pl.Int32),
        matched=pl.col("adsb_matched").cast(pl.Float64), fallback=pl.col("adsb_fallback").cast(pl.Float64),
        coverage=pl.col("adsb_day_coverage").cast(pl.Float64),
        a_rel=pl.col("adsb_taxi") - pl.col("s"), adsb_taxi=pl.col("adsb_taxi"),
        adsb_pb_dist_m="adsb_pb_dist_m", adsb_pb_gs="adsb_pb_gs", adsb_pb_gap_s="adsb_pb_gap_s",
        L_rel=pl.col("mvt_ts") - pl.col("adsb_first_ts") - pl.col("s"),
        L=pl.col("mvt_ts") - pl.col("adsb_first_ts"),
        adsb_first_own_m="adsb_first_own_m", adsb_first_gs="adsb_first_gs", adsb_min_own_m="adsb_min_own_m",
        adsb_n_pts=pl.col("adsb_n_pts").cast(pl.Float64), adsb_takeoff_gap_s="adsb_takeoff_gap_s",
    ).select(FEATS)


def fit_combiner(X: np.ndarray, y: np.ndarray, inner: np.ndarray) -> tuple[lgb.Booster, int]:
    cat_idx = [FEATS.index(c) for c in CATS]
    tr = lgb.Dataset(X[~inner], y[~inner], categorical_feature=cat_idx, free_raw_data=False)
    va = lgb.Dataset(X[inner], y[inner], categorical_feature=cat_idx, reference=tr, free_raw_data=False)
    m = lgb.train(PARAMS, tr, MAX_ROUNDS, valid_sets=[va],
                  callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    best = max(m.best_iteration, 1)
    full = lgb.train(PARAMS, lgb.Dataset(X, y, categorical_feature=cat_idx), best)
    return full, best


def combiner(df: pl.DataFrame, base: np.ndarray) -> tuple[np.ndarray, dict]:
    airports = sorted(df["ADEP_mvt"].unique().to_list())
    out, info = base.copy(), {}
    non_lirf = (df["ADEP_mvt"] != "LIRF").to_numpy()
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = stack(df, fm)
        X = features(df, s, airports).to_numpy().astype(np.float64)
        fit_rows = ((df["ym"] == fm) & (df["taxi"] <= TRIM)).to_numpy() & non_lirf
        y = df["taxi"].to_numpy() - s
        inner = (df["day"].dt.day() % 5 == 0).to_numpy()[fit_rows]
        m, best = fit_combiner(X[fit_rows], y[fit_rows], inner)
        app = (df["ym"] == am).to_numpy() & non_lirf
        out[app] = s[app] + m.predict(X[app])
        imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda t: -t[1])
        info[f"fit {fm}"] = {"best_iter": best, "top_gain": [f"{k}:{v / sum(x[1] for x in imp):.2f}"
                                                              for k, v in imp[:8]]}
    return out, info


def main() -> None:
    df = A.load()
    v3, v4 = V.read_det(V.V3), V.read_det(V4)
    d3, d4 = V.with_det(df, v3), V.with_det(df, v4)
    base, _ = V.pipeline(d3, fb_stage=True, gate=True)
    tr = (df["taxi"] <= TRIM).to_numpy()
    print(f"holdout rows {df.height:,}; v23 cross-fit trimmed RMSE "
          f"{np.sqrt(np.mean((base - df['taxi'].to_numpy())[tr] ** 2)):.2f}")

    new, info = combiner(d4, base)
    print("combiner (v4):", info)
    V.evaluate(d4, base, new, "PRIMARY: learned combiner on v4 detections", decisive=True)
    grp = (pl.when(pl.col("adsb_tier").is_in(["appear", "dwell"])).then(pl.lit("appear/dwell"))
           .when(pl.col("adsb_tier").is_in(["fb_dwell", "fb_appear"]) & pl.col("adsb_matched")).then(pl.lit("fb (matched, v4)"))
           .when(pl.col("adsb_tier").is_in(["fb_dwell", "fb_appear"])).then(pl.lit("fb (unmatched)"))
           .when(pl.col("adsb_matched")).then(pl.lit("matched, no pushback")).otherwise(pl.lit("no ADS-B")))
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12, float_precision=2):
        t = d4.with_columns(b=pl.Series(base), n=pl.Series(new), g=grp).filter(pl.col("taxi") <= TRIM)
        print(t.group_by("g").agg(pl.len(), ((pl.col("b") - pl.col("taxi")).pow(2).mean().sqrt()).alias("base"),
                                  ((pl.col("n") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new")).sort("g"))

    new3, info3 = combiner(d3, base)
    print("\ncombiner (v3):", info3)
    V.evaluate(d3, base, new3, "reported: same combiner on v3 detections")


if __name__ == "__main__":
    main()
