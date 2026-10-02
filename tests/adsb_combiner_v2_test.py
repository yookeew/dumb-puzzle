"""ADS-B combiner v2 gate, per reports/adsb_combiner_v2_preregistration.md.

Base: the v24 combiner (tests/adsb_combiner_test.py) restricted to rows with ADS-B
information, v23 pipeline value elsewhere. Treatment: v2 = fit and apply only on
those rows, with stronger regularisation. Cross-fit Jan<->Jul, trimmed RMSE decides.

Run:  .venv/Scripts/python.exe tests/adsb_combiner_v2_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_test as C  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

JAN, JUL = A.JAN, A.JUL
PARAMS_V2 = {**C.PARAMS, "num_leaves": 15, "min_data_in_leaf": 2000, "lambda_l2": 50.0,
             "feature_fraction": 0.8}


def has_adsb() -> pl.Expr:
    return (pl.col("adsb_matched").fill_null(False)
            | pl.col("adsb_tier").is_in(["fb_dwell", "fb_appear"]).fill_null(False))


def fit_v2(X: np.ndarray, y: np.ndarray, inner: np.ndarray) -> tuple[lgb.Booster, int]:
    cat_idx = [C.FEATS.index(c) for c in C.CATS]
    tr = lgb.Dataset(X[~inner], y[~inner], categorical_feature=cat_idx, free_raw_data=False)
    va = lgb.Dataset(X[inner], y[inner], categorical_feature=cat_idx, reference=tr, free_raw_data=False)
    m = lgb.train(PARAMS_V2, tr, C.MAX_ROUNDS, valid_sets=[va],
                  callbacks=[lgb.early_stopping(C.ES_PATIENCE, verbose=False)])
    best = max(m.best_iteration, 1)
    return lgb.train(PARAMS_V2, lgb.Dataset(X, y, categorical_feature=cat_idx), best), best


def combiner_v2(df: pl.DataFrame, base: np.ndarray) -> tuple[np.ndarray, dict]:
    airports = sorted(df["ADEP_mvt"].unique().to_list())
    out, info = base.copy(), {}
    pop = (df["ADEP_mvt"] != "LIRF").to_numpy() & df.select(has_adsb()).to_series().to_numpy()
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = C.stack(df, fm)
        X = C.features(df, s, airports).to_numpy().astype(np.float64)
        fit_rows = ((df["ym"] == fm) & (df["taxi"] <= C.TRIM)).to_numpy() & pop
        y = df["taxi"].to_numpy() - s
        inner = (df["day"].dt.day() % 5 == 0).to_numpy()[fit_rows]
        m, best = fit_v2(X[fit_rows], y[fit_rows], inner)
        app = (df["ym"] == am).to_numpy() & pop
        out[app] = s[app] + m.predict(X[app])
        imp = sorted(zip(C.FEATS, m.feature_importance("gain")), key=lambda t: -t[1])
        info[f"fit {fm}"] = {"best_iter": best, "n_fit": int(fit_rows.sum()),
                             "top_gain": [f"{k}:{v / sum(x[1] for x in imp):.2f}" for k, v in imp[:8]]}
    return out, info


def main() -> None:
    df = A.load()
    d3, d4 = V.with_det(df, V.read_det(V.V3)), V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    v24, info24 = C.combiner(d4, v23)
    pop = d4.select(has_adsb()).to_series().to_numpy()
    base = np.where(pop, v24, v23)          # v24 combiner restricted to ADS-B rows
    print("v24 combiner:", info24)
    print(f"rows with ADS-B information: {pop.sum():,} of {len(pop):,}")

    new, info = combiner_v2(d4, v23)
    print("v2 combiner:", info)
    V.evaluate(d4, base, new, "PRIMARY: combiner v2 vs v24-restricted", decisive=True)
    V.evaluate(d4, v24, new, "reported: v2 vs unrestricted v24")


if __name__ == "__main__":
    main()
