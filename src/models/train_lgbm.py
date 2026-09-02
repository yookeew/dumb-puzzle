"""Stage 3a — global LightGBM on d = AOBT - SOBT.

taxi = (T - SOBT) - d   =>   predict d, recover taxi, clip [FLOOR, CEIL].

  holdout : fit on the 10 non-holdout months, score Jan+Jul 2025
  submit  : refit on all 2025, predict ranking DEP -> data/submissions/lgbm_v3

Run:  .venv/Scripts/python.exe src/models/train_lgbm.py

History
  lgbm_v1  holdout 301 s  — single global model, label window [60, 5400]
  lgbm_v2  holdout 313 s (global) / 332 s (+per-airport residual heads)
           REGRESSION — raising the label ceiling to 14400 to learn LIRF's tail
           hurt the 9 well-behaved airports (L2 chases the tail); the per-airport
           residual heads overfit the 2-fold oof residuals and did not transfer
           to the held-out season. Both abandoned. See reports/lirf_investigation.md
           and reports/stage3a_resid.md.
  This file: back to the single global model. LIRF tail / echo handling and any
  robust-loss or quantile work happens on Colab (local CPU fit is ~4 min, too
  slow to iterate — see notebooks/).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.holdout import HOLDOUT_MONTHS, TRAIN_GLOB, report, rmse  # noqa: E402
from features.build_features import build_features  # noqa: E402
from features.encode import apply_priors, feature_matrix, fit_priors  # noqa: E402
from post.submit import write_submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
CACHE = ROOT / "cache" / "features"

FLOOR, CEIL = 0, 10800          # prediction clip: no positive floor
LABEL_LO, LABEL_HI = 30, 7200   # drop poison + the most extreme tail from training
ROUNDS = 1500

LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.03, num_leaves=255,
    min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, max_bin=127, deterministic=True, force_row_wise=True,
    seed=42, num_threads=0, verbose=-1,
)

FRAME_COLS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_TYPE_flt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_mvt",
    "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "AOBT_3_flt",
]


def blind(df: pl.DataFrame) -> pl.DataFrame:
    dep = pl.col("PHASE_mvt") == "DEP"
    return df.with_columns(
        pl.when(dep).then(None).otherwise(pl.col("BLOCK_TIME_UTC_mvt")).alias("BLOCK_TIME_UTC_mvt"),
        pl.when(dep).then(None).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"),
    )


def base_features(frame: pl.DataFrame, tag: str) -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{tag}.parquet"
    if fp.exists():
        return pl.read_parquet(fp)
    feats = build_features(blind(frame), priors=None)
    feats.write_parquet(fp)
    return feats


def labels(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
        taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Int64),
        d=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
        ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    )


def matrix(feats, categories=None):
    X, names, cats, categories = feature_matrix(feats, categories)
    return X.with_columns([pl.col(c).fill_null(-1) for c in cats]).to_pandas(), names, cats, categories


def fit(feats_l, lab, categories):
    """feats_l already priors-applied and row-aligned to lab."""
    X, names, cats, _ = matrix(feats_l, categories)
    keep = lab["taxi"].is_between(LABEL_LO, LABEL_HI).to_numpy()
    ds = lgb.Dataset(X[keep], label=lab["d"].to_numpy()[keep],
                     categorical_feature=cats, free_raw_data=False)
    return lgb.train(LGB_PARAMS, ds, num_boost_round=ROUNDS), names


def predict_taxi(model, names, feats_l, offset, categories):
    X, _, _, _ = matrix(feats_l, categories)
    d_hat = model.predict(X[names])
    return np.clip(offset - d_hat, FLOOR, CEIL)


def main() -> None:
    t0 = time.time()
    frame25 = pl.read_parquet(TRAIN_GLOB, columns=FRAME_COLS)
    feats25 = base_features(frame25, "train2025")
    lab25 = labels(frame25)
    off25 = feats25.select("MVT_ID_mvt", "sched_takeoff_offset")
    print(f"loaded {feats25.height:,} DEP rows ({time.time() - t0:.0f}s)")

    is_ho = pl.col("ym").is_in(HOLDOUT_MONTHS)
    tr_lab, ho_lab = lab25.filter(~is_ho), lab25.filter(is_ho)
    priors = fit_priors(tr_lab)

    def prep(lab_):
        f = apply_priors(feats25.join(lab_.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
        return f, f.select("MVT_ID_mvt").join(lab_, on="MVT_ID_mvt")

    f_tr, tr_lab = prep(tr_lab)
    f_ho, ho_lab = prep(ho_lab)
    _, _, _, categories = matrix(f_tr)

    model, names = fit(f_tr, tr_lab, categories)
    print(f"fit done ({time.time() - t0:.0f}s)")

    ho_off = ho_lab.join(off25, on="MVT_ID_mvt")["sched_takeoff_offset"].to_numpy()
    ev = ho_lab.join(off25, on="MVT_ID_mvt").with_columns(
        pred=pl.Series(predict_taxi(model, names, f_ho, ho_off, categories))
    ).filter(pl.col("taxi").is_between(0, 4 * 3600))
    report(ev)

    ref = tr_lab.filter(pl.col("taxi").is_between(60, 5400)).group_by("ADEP_mvt").agg(
        pl.col("taxi").mean().alias("tm"))
    r = ev.join(ref, on="ADEP_mvt")
    print(f"\nreference RMSE (per-airport mean taxi): {rmse(r['tm'], r['taxi']):.1f} s")

    imp = sorted(zip(names, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top 15:", ", ".join(n for n, _ in imp[:15]))

    print(f"\n=== refit on all 2025, predict ranking ({time.time() - t0:.0f}s) ===")
    priors_all = fit_priors(lab25)
    f_all = apply_priors(feats25, priors_all)
    lab_all = f_all.select("MVT_ID_mvt").join(lab25, on="MVT_ID_mvt")
    _, _, _, cats_all = matrix(f_all)
    model_a, names_a = fit(f_all, lab_all, cats_all)

    frame_r = pl.read_parquet(RANKING, columns=FRAME_COLS)
    f_r = apply_priors(base_features(frame_r, "ranking"), priors_all)
    r_off = f_r["sched_takeoff_offset"].to_numpy()
    preds = f_r.select("MVT_ID_mvt").with_columns(
        TAXITIME_SEC_mvt=pl.Series(predict_taxi(model_a, names_a, f_r, r_off, cats_all))
    )
    write_submission(preds, "lgbm_v3")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
