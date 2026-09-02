"""Portable model stage — runs from the cache/features/ parquets alone.

No dependency on raw data or the Polars family builders: only `features.encode`
(categorical encoding + taxi priors). Same code path locally and on Colab.

    from models.fit import run
    run(engine="lgb")          # local CPU
    run(engine="xgb")          # Colab, uses device="cuda" if a GPU is present

Inputs (see features/export_model_inputs.py):
    <feat_dir>/train2025.parquet   base features, all 2025 departures
    <feat_dir>/ranking.parquet     base features, ranking departures
    <feat_dir>/labels2025.parquet  MVT_ID_mvt, ADEP/RUNWAY/STAND, taxi, d, ym
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.encode import apply_priors, feature_matrix, fit_priors  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "cache" / "features"
SUB_DIR = ROOT / "data" / "submissions"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"

HOLDOUT_MONTHS = ("2025-01", "2025-07")
FLOOR, CEIL = 0, 10800
LABEL_LO, LABEL_HI = 30, 7200
ROUNDS = 1500


# --------------------------------------------------------------------- engines
def _fit_lgb(X, y, cats, seed=42):
    import lightgbm as lgb

    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03, num_leaves=255,
        min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, max_bin=127, deterministic=True, force_row_wise=True,
        seed=seed, num_threads=0, verbose=-1,
    )
    ds = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
    m = lgb.train(params, ds, num_boost_round=ROUNDS)
    return m, (lambda M, A: M.predict(A))


def _fit_xgb(X, y, cats, seed=42):
    import xgboost as xgb

    try:
        gpu = "cuda" if xgb.build_info().get("USE_CUDA") else "cpu"
    except Exception:
        gpu = "cpu"
    dtrain = xgb.QuantileDMatrix(X, label=y, enable_categorical=True)
    params = dict(
        objective="reg:squarederror", eval_metric="rmse", eta=0.03, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, max_bin=127, device=gpu,
        tree_method="hist", seed=seed,
    )
    m = xgb.train(params, dtrain, num_boost_round=ROUNDS)
    return m, (lambda M, A: M.predict(xgb.DMatrix(A, enable_categorical=True)))


ENGINES = {"lgb": _fit_lgb, "xgb": _fit_xgb}


# ------------------------------------------------------------------ utilities
def _matrix(feats, categories=None):
    X, names, cats, categories = feature_matrix(feats, categories)
    return X.with_columns([pl.col(c).fill_null(-1) for c in cats]).to_pandas(), names, cats, categories


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _report(df: pl.DataFrame, pred="pred", true="taxi"):
    e = (pl.col(pred).cast(float) - pl.col(true).cast(float))
    print(f"OVERALL RMSE {df.select(e.pow(2).mean()).item() ** 0.5:.1f} s  (n={df.height:,})")
    for key in ("ADEP_mvt", "ym"):
        g = df.group_by(key).agg(e.pow(2).mean().sqrt().alias("r"), pl.len().alias("n")).sort(key)
        print("  " + "  ".join(f"{r[key]}:{r['r']:.0f}" for r in g.iter_rows(named=True)))


def _write_submission(mvt_ids, taxi, name):
    template = pl.read_parquet(TEMPLATE)
    preds = pl.DataFrame({"MVT_ID_mvt": mvt_ids,
                          "TAXITIME_SEC_mvt": np.round(taxi).astype("int32")})
    merged = template.select("MVT_ID_mvt").join(preds, on="MVT_ID_mvt", how="left")
    assert merged["TAXITIME_SEC_mvt"].null_count() == 0, "missing predictions"
    assert merged.height == template.height
    assert (merged["TAXITIME_SEC_mvt"] >= 0).all()
    SUB_DIR.mkdir(exist_ok=True)
    merged.write_parquet(SUB_DIR / f"{name}.parquet")
    print(f"wrote {name}.parquet  n={merged.height:,}  "
          f"median={merged['TAXITIME_SEC_mvt'].median()}")


# ------------------------------------------------------------------------ run
def run(engine: str = "lgb", feat_dir: Path = FEAT_DIR, name: str | None = None):
    t0 = time.time()
    fitter = ENGINES[engine]
    name = name or f"{engine}_colab"

    feats = pl.read_parquet(feat_dir / "train2025.parquet")
    lab = pl.read_parquet(feat_dir / "labels2025.parquet")
    off = feats.select("MVT_ID_mvt", "sched_takeoff_offset")

    is_ho = pl.col("ym").is_in(HOLDOUT_MONTHS)
    tr_lab, ho_lab = lab.filter(~is_ho), lab.filter(is_ho)

    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    f_ho = apply_priors(feats.join(ho_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    ho_lab = f_ho.select("MVT_ID_mvt").join(ho_lab, on="MVT_ID_mvt")

    Xtr, names, cats, categories = _matrix(f_tr)
    keep = tr_lab["taxi"].is_between(LABEL_LO, LABEL_HI).to_numpy()
    model, pred_fn = fitter(Xtr[keep], tr_lab["d"].to_numpy()[keep], cats)
    print(f"holdout fit {time.time() - t0:.0f}s")

    Xho, _, _, _ = _matrix(f_ho, categories)
    ho_off = ho_lab.join(off, on="MVT_ID_mvt")["sched_takeoff_offset"].to_numpy()
    taxi_hat = np.clip(ho_off - pred_fn(model, Xho[names]), FLOOR, CEIL)
    ev = ho_lab.join(off, on="MVT_ID_mvt").with_columns(pred=pl.Series(taxi_hat)).filter(
        pl.col("taxi").is_between(0, 4 * 3600))
    _report(ev)

    # refit on all 2025 + ranking submission
    priors_a = fit_priors(lab)
    f_all = apply_priors(feats, priors_a)
    lab_a = f_all.select("MVT_ID_mvt").join(lab, on="MVT_ID_mvt")
    Xall, names_a, cats_a, cats_map = _matrix(f_all)
    keep = lab_a["taxi"].is_between(LABEL_LO, LABEL_HI).to_numpy()
    model_a, pred_a = fitter(Xall[keep], lab_a["d"].to_numpy()[keep], cats_a)

    f_r = apply_priors(pl.read_parquet(feat_dir / "ranking.parquet"), priors_a)
    Xr, _, _, _ = _matrix(f_r, cats_map)
    r_off = f_r["sched_takeoff_offset"].to_numpy()
    taxi_r = np.clip(r_off - pred_a(model_a, Xr[names_a]), FLOOR, CEIL)
    _write_submission(f_r["MVT_ID_mvt"].to_list(), taxi_r, name)
    print(f"total {time.time() - t0:.0f}s")
    return model, ev


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "lgb")
