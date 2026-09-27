"""Portable model stage — runs from the cache/features/ parquets alone.

No dependency on raw data or the Polars family builders: only `features.encode`
(categorical encoding + taxi priors). Same code path locally and on Colab.

    from models.fit import run
    run(engine="lgb")                    # local CPU, huber loss alpha=800 (default)
    run(engine="xgb")                    # Colab, uses device="cuda" if a GPU is present
    run(engine="lgb", loss="l2")         # plain L2/MSE, the old default
    run(engine="lgb", target="direct")   # fit taxi directly, skip the d-flip

Inputs (see features/export_model_inputs.py):
    <feat_dir>/train2025.parquet       base features, all 2025 departures
    <feat_dir>/holdout_gap2025.parquet base features, isolated Jan+Jul 2025 --
                                        mirrors ranking.parquet's Jan->Jul gap
                                        instead of a continuous 12-month frame
    <feat_dir>/ranking.parquet         base features, ranking departures
    <feat_dir>/labels2025.parquet      MVT_ID_mvt, ADEP/RUNWAY/STAND, taxi, d, ym
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
from features.encode import (  # noqa: E402
    _ECHO_ABS_D,
    add_group_encodings_oof,
    apply_group_encodings,
    apply_priors,
    feature_matrix,
    fit_group_encodings,
    fit_priors,
)

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "cache" / "features"
SUB_DIR = ROOT / "data" / "submissions"
REPORT_DIR = ROOT / "reports" / "eval"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"

HOLDOUT_MONTHS = ("2025-01", "2025-07")
# CEIL=10800 (3h). TRIED raising this to 140000 on the theory that echo rows
# (BLOCK_TIME ~= SCHED_TIME) have a correct, large true-taxi reconstruction
# via sched_takeoff_offset that the ceiling was clipping away -- see
# PROGRESS.md. REVERTED: it was a net regression (303.1s -> 560.7s holdout).
# LightGBM's cumulative tree output can drift far from typical values for a
# handful of unusual feature combinations, and 10800 was quietly acting as a
# safety backstop against that for far more rows than just the genuine
# echoes -- only 1 row actually hit the new 140000 ceiling, yet several
# majority-lane airports (EDDF 290->1036s) got wrecked. A fix for the echo
# lane needs to be targeted (flag likely-echo rows specifically), not a
# blanket ceiling raise on every prediction.
FLOOR, CEIL = 0, 10800
# Sanity bound for the echo-hypothesis reconstruction only (see
# _reconstruct_taxi) -- just above the max observed real label (131167s).
# Never applied to the ordinary model reconstruction, only to the
# echo_rate-weighted blend component, which is itself gated to the
# has_aobt3=False lane -- this is the "targeted, not blanket" version of the
# CEIL raise above that failed.
ECHO_WIDE_CEIL = 140000
# LIRF-only relaxation of the base-model clip ceiling (see PROGRESS.md /
# reports/lirf_investigation.md SS4-5). The blanket CEIL=140000 raise tried
# in SS6 was a regression because raw taxi_model_raw > 10800 is tree-
# extrapolation noise for every other airport (checked directly: raw RMSE
# 8-84x worse than the clipped blend on EDDF/EDDM/EGLL/EHAM/LEBL/LFPG's
# rare over-CEIL rows). LIRF is the opposite: its 22 holdout rows (0.08% of
# LIRF, 54.5% of LIRF's total squared error under the old CEIL=10800) have
# raw RMSE 16190 vs blend RMSE 21975 -- the raw prediction is *better*,
# because LIRF is the only airport where genuine 10-24h+ ground delays
# occur and a confident large raw prediction usually reflects that, not
# extrapolation instability. Gated strictly on ADEP_mvt=="LIRF" so every
# other airport's reconstruction is byte-identical to before.
LIRF_RAW_CEIL = ECHO_WIDE_CEIL
LABEL_LO, LABEL_HI = 30, 7200

# Training budget. ROUNDS is a ceiling — early stopping on an inner-validation
# month (VALID_MONTH, carved out of the training split) picks the real count,
# then the all-data refit runs ~1.1x that many rounds. Lower ETA + higher
# ROUNDS trades compute for a little accuracy; crank via run(rounds=, eta=).
# Huber (the default loss, see LOSS_OBJECTIVE below) converges slower than L2 --
# it still hadn't plateaued at 8000 rounds in testing (best_iter=7998/8000) --
# so this ceiling is a local-CPU compromise, not a confirmed plateau. Push it
# higher on Colab/GPU if you want to find where it actually stops improving.
ROUNDS = 8000
ETA = 0.02
EARLY_STOP = 150
VALID_MONTH = "2025-06"

# Echo classifier: P(BLOCK_TIME ~= SCHED_TIME), trained on the same feature
# matrix as the d-regressor with label is_echo = |d| < _ECHO_ABS_D (see
# features.encode).
# v1 (eta=0.05, 100-round patience) early-stopped at iter 174 -- too fast to
# discriminate a 5.5%-positive-rate class well: even its own confidently-
# flagged rows had 1987s RMSE (a large share of LIRF's remaining error, see
# PROGRESS.md). Slower eta + more patience, matching the main model's
# cadence, gives it more rounds to actually separate the classes -- a real
# but small win (397.3s -> 392.9s holdout overall, LIRF 883 -> 857s).
# NOTE: also tried `is_unbalance=True` alongside this -- REGRESSION, do not
# re-add (see PROGRESS.md): it reweights the training gradient toward the
# rare class, but early stopping still watches plain binary_logloss on the
# natural class balance, so validation loss looked like it was getting
# worse from round 1 and stopped at iter 7 with zero predicted echoes.
CLF_ROUNDS = 5000
CLF_ETA = 0.02
CLF_EARLY_STOP = 150
CLF_OBJECTIVE = {"lgb": "binary", "xgb": "binary:logistic", "cat": "Logloss"}
CLF_METRIC = {"lgb": "binary_logloss", "xgb": "logloss", "cat": "Logloss"}


# --------------------------------------------------------------------- engines
def _fit_lgb(X, y, cats, *, eta=ETA, rounds=ROUNDS, es=EARLY_STOP, valid=None, seed=42,
             objective="regression", alpha=None, metric="rmse"):
    import lightgbm as lgb

    # metric stays rmse regardless of objective -- early stopping and reporting
    # always select on the competition metric, only the training gradient changes.
    # (the echo classifier below passes metric="binary_logloss" instead.)
    params = dict(
        objective=objective, metric=metric, learning_rate=eta, num_leaves=255,
        min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, max_bin=127, deterministic=True, force_row_wise=True,
        seed=seed, num_threads=0, verbose=-1,
    )
    if alpha is not None:
        params["alpha"] = alpha
    ds = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
    valid_sets, cbs = [], []
    if valid is not None and es:
        vds = lgb.Dataset(valid[0], label=valid[1], reference=ds,
                          categorical_feature=cats, free_raw_data=False)
        valid_sets, cbs = [vds], [lgb.early_stopping(es, verbose=False)]
    m = lgb.train(params, ds, num_boost_round=rounds, valid_sets=valid_sets, callbacks=cbs)
    best = m.best_iteration or rounds

    def predict(M, A):
        bi = getattr(M, "best_iteration", 0)
        return M.predict(A, num_iteration=bi if bi and bi > 0 else None)

    return m, predict, best


def _fit_xgb(X, y, cats, *, eta=ETA, rounds=ROUNDS, es=EARLY_STOP, valid=None, seed=42,
             objective="reg:squarederror", alpha=None, metric="rmse"):
    import xgboost as xgb

    try:
        gpu = "cuda" if xgb.build_info().get("USE_CUDA") else "cpu"
    except Exception:
        gpu = "cpu"
    dtrain = xgb.QuantileDMatrix(X, label=y, enable_categorical=True, max_bin=127)
    params = dict(
        objective=objective, eval_metric=metric, eta=eta, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, max_bin=127, device=gpu,
        tree_method="hist", seed=seed,
    )
    if alpha is not None and objective == "reg:pseudohubererror":
        params["huber_slope"] = alpha
    evals, es_arg = [], None
    if valid is not None and es:
        dvalid = xgb.QuantileDMatrix(valid[0], label=valid[1], ref=dtrain,
                                     enable_categorical=True, max_bin=127)
        evals, es_arg = [(dvalid, "valid")], es
    m = xgb.train(params, dtrain, num_boost_round=rounds, evals=evals,
                  early_stopping_rounds=es_arg, verbose_eval=False)
    best = getattr(m, "best_iteration", rounds - 1) + 1

    def predict(M, A):
        d = xgb.DMatrix(A, enable_categorical=True)
        bi = getattr(M, "best_iteration", None)
        return M.predict(d, iteration_range=(0, bi + 1)) if bi is not None else M.predict(d)

    return m, predict, best


def _fit_cat(X, y, cats, *, eta=ETA, rounds=ROUNDS, es=EARLY_STOP, valid=None, seed=42,
             objective="RMSE", alpha=None, metric="RMSE"):
    """CatBoost: the decorrelated second family (ordered boosting, target-statistic
    encoding of the categoricals instead of LightGBM/XGBoost's partition splits).
    GPU when available (Colab T4), else CPU -- CPU on the full 2M-row frame is
    hours, not minutes, so run this on Colab."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    from catboost.utils import get_gpu_device_count

    def pool(A, label=None):
        A = A.copy()
        for c in cats:  # pandas category over int codes -> plain ints for CatBoost
            A[c] = A[c].astype("int32")
        return Pool(A, label=label, cat_features=cats)

    is_clf = objective == "Logloss"
    loss = objective if alpha is None or objective != "Huber" else f"Huber:delta={alpha}"
    # max_ctr_complexity=1: no categorical feature combinations. With 14
    # categoricals (several with thousands of levels) the default combination
    # search made each GPU iteration ~0.5 s on the 1.7M-row fit (~70 min/fit).
    params = dict(loss_function=loss, eval_metric=metric, learning_rate=eta, iterations=rounds,
                  depth=8, border_count=127, max_ctr_complexity=1, random_seed=seed,
                  allow_writing_files=False, verbose=500)
    dtrain = pool(X, y)
    fit_kw = {}
    if valid is not None and es:
        fit_kw = dict(eval_set=pool(valid[0], valid[1]), early_stopping_rounds=es,
                      use_best_model=True)
    cls = CatBoostClassifier if is_clf else CatBoostRegressor
    m = None
    n_gpu = get_gpu_device_count()
    print(f"catboost: {'GPU' if n_gpu > 0 else 'CPU (no GPU visible)'}  "
          f"loss={loss}  lr={eta}  rounds={rounds}", flush=True)
    if n_gpu > 0:
        try:
            m = cls(**params, task_type="GPU")
            m.fit(dtrain, **fit_kw)
        except Exception as e:  # a loss/metric without GPU support -> CPU
            print(f"catboost GPU failed ({e}); falling back to CPU")
            m = None
    if m is None:
        m = cls(**params, task_type="CPU", thread_count=-1)
        m.fit(dtrain, **fit_kw)
    best = (m.get_best_iteration() or rounds - 1) + 1

    def predict(M, A):
        P = pool(A)
        return M.predict_proba(P)[:, 1] if is_clf else M.predict(P)

    return m, predict, best


ENGINES = {"lgb": _fit_lgb, "xgb": _fit_xgb, "cat": _fit_cat}


def _mem(stage: str) -> None:
    """RSS checkpoint, printed only when PRC_MEMLOG=1 (diagnosing Colab's ~12.7 GB cap)."""
    if not os.environ.get("PRC_MEMLOG"):
        return
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
    except ImportError:
        rss = float("nan")
    print(f"[mem] {rss:5.2f} GB  {stage}", flush=True)


def _reconstruct_taxi(offset: np.ndarray, taxi_model_raw: np.ndarray,
                       echo_prob: np.ndarray, use_prior: np.ndarray,
                       prior_taxi: np.ndarray, ceil=CEIL) -> np.ndarray:
    """taxi_model_raw (the regressor's own taxi estimate -- `offset - d_hat`
    for the flip target, or the direct taxi prediction as-is) clipped to
    [FLOOR, ceil] (scalar CEIL=10800 by default; the caller passes a
    per-row array with LIRF_RAW_CEIL for ADEP_mvt=="LIRF" rows -- see
    LIRF_RAW_CEIL) -- blended toward the echo hypothesis taxi=offset (uncapped
    up to ECHO_WIDE_CEIL), weighted by `echo_prob`, an out-of-sample
    P(BLOCK_TIME ~= SCHED_TIME) estimate from a dedicated classifier (see
    fit_echo_classifier / PROGRESS.md).

    Target-agnostic: the caller computes `taxi_model_raw` from whatever the
    regressor predicts (see `target` in run()); this function only clips and
    blends, it never subtracts anything itself.

    Supersedes the earlier has_aobt3-gated op_echo_rate heuristic: that
    version only ever fired on the ~1.5% of rows missing an NM off-block time,
    but LIRF alone echoes on ~18% of its rows (reports/lirf_investigation.md)
    -- most of its echoes still have an AOBT_3_flt value and were sailing
    straight through the ordinary clip(taxi_model_raw, FLOOR, CEIL) path
    unblended. The classifier sees the full feature set (missing AOBT_3/
    EOBT_1/FLIGHT_ID, IOBT delay, hour, operator, stand, ...) instead of a
    single proxy, so it should have much higher recall on the true echo
    population without needing a hard population gate.

    For NM-unmatched-but-not-echo rows (`use_prior`, caller-computed as
    has_aobt3=False AND ADEP_mvt != LIRF -- see PROGRESS.md), fall back to
    `prior_taxi` (median_taxi_prior) instead of trusting the regressor, which
    has no operator/EOBT/IOBT signal for this lane. A flat median fallback
    helps every airport (LFPG -20s, EDDF -60s, EHAM -49s, ... on the holdout)
    EXCEPT LIRF specifically -- its NM-unmatched rows skew toward
    larger-than-typical offsets, so a flat median undershoots there. An
    offset-anchored alternative (clip(offset - airport_mean_d, ...)) was
    also tried for LIRF and was worse than the flat median everywhere, so
    LIRF is just excluded via `use_prior` rather than given its own fallback.
    """
    taxi_model = np.clip(taxi_model_raw, FLOOR, ceil)
    taxi_echo = np.clip(offset, 0, ECHO_WIDE_CEIL)
    w_prior = (1 - echo_prob) * use_prior
    w_model = (1 - echo_prob) * ~use_prior
    return echo_prob * taxi_echo + w_prior * prior_taxi + w_model * taxi_model


def fit_echo_classifier(engine: str, X_mat, d_raw: np.ndarray, tr_mask: np.ndarray,
                        va_mask: np.ndarray, cats: list[str], *, rounds=CLF_ROUNDS,
                        eta=CLF_ETA, es=CLF_EARLY_STOP, seed=42):
    """Fit P(is_echo) = P(|d| < _ECHO_ABS_D) on the same feature matrix used
    for the d-regressor. `tr_mask`/`va_mask` follow the same convention as the
    main fit in run() (`keep & ~iv` / `keep & iv`) but must NOT be filtered by
    the [LABEL_LO, LABEL_HI] taxi-range `keep` mask -- echoes are exactly the
    rows that mask tends to exclude (an echo's `d` is near zero, often below
    LABEL_LO), so filtering by it would starve the classifier of positives.

    Returns (model, predict_fn, best_iteration, is_echo) where `is_echo` is
    the 0/1 array used as the training label, handy for the caller to report
    a quick base-rate/recall sanity check.
    """
    fitter = ENGINES[engine]
    is_echo = (np.abs(d_raw) < _ECHO_ABS_D).astype(np.float64)
    model, predict_fn, best = fitter(
        X_mat[tr_mask], is_echo[tr_mask], cats,
        eta=eta, rounds=rounds, es=es, valid=(X_mat[va_mask], is_echo[va_mask]),
        objective=CLF_OBJECTIVE[engine], metric=CLF_METRIC[engine], seed=seed)
    return model, predict_fn, best, is_echo


# ------------------------------------------------------------------ utilities
def _matrix(feats, categories=None):
    """Model matrix as a pandas frame. Categorical columns carry a pandas
    `category` dtype with a fixed [-1, 0..K-1] category set (K from `categories`)
    so LightGBM and XGBoost both see them as categorical and the codes line up
    across the train / holdout / ranking frames."""
    import pandas as pd

    X, names, cats, categories = feature_matrix(feats, categories)
    pdf = X.with_columns([pl.col(c).fill_null(-1) for c in cats]).to_pandas()
    for c in cats:
        pdf[c] = pd.Categorical(pdf[c], categories=[-1, *range(len(categories[c]))])
    return pdf, names, cats, categories


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _report(df: pl.DataFrame, pred="pred", true="taxi"):
    e = (pl.col(pred).cast(float) - pl.col(true).cast(float))
    print(f"OVERALL RMSE {df.select(e.pow(2).mean()).item() ** 0.5:.1f} s  (n={df.height:,})")
    for key in ("ADEP_mvt", "ym", "has_aobt3", "echo_pred"):
        if key not in df.columns:
            continue
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
    s = merged["TAXITIME_SEC_mvt"]
    print(f"wrote {name}.parquet  n={merged.height:,}  median={s.median()}")
    return dict(rows=merged.height, min=int(s.min()), p10=float(s.quantile(0.1)),
                median=float(s.median()), mean=float(s.mean()),
                p90=float(s.quantile(0.9)), max=int(s.max()),
                at_floor=int((s <= FLOOR).sum()), at_ceil=int((s >= CEIL).sum()))


def _md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _write_report(ev: pl.DataFrame, name: str, engine: str, sub: dict | None = None,
                  meta: dict | None = None):
    """Markdown holdout report -> reports/eval/<name>.md (CLAUDE.md Stage 5 slices)."""
    import datetime as dt

    e = pl.col("pred").cast(float) - pl.col("taxi").cast(float)
    n = ev.height
    overall = float(ev.select(e.pow(2).mean()).item() ** 0.5)
    mae = float(ev.select(e.abs().mean()).item())
    bias = float(ev.select(e.mean()).item())

    m = meta or {}
    loss_bit = m.get("loss", "l2")
    if loss_bit == "huber":
        loss_bit += f" (alpha={m.get('huber_alpha', HUBER_ALPHA)})"
    train_line = (f"- target {m.get('target', 'flip')}  |  loss {loss_bit}  |  "
                  f"eta {m.get('eta', ETA)}  |  "
                  f"rounds ceiling {m.get('rounds', ROUNDS)}  |  "
                  f"early-stopped at {m.get('best_iter', '?')} (inner-valid "
                  f"{m.get('valid_month', VALID_MONTH)})  |  full-refit rounds "
                  f"{m.get('full_rounds', '?')}")
    L = [
        f"# Holdout eval — {name}", "",
        f"- engine: **{engine}**   |   generated: {dt.datetime.now():%Y-%m-%d %H:%M}",
        train_line,
        f"- holdout = {', '.join(HOLDOUT_MONTHS)} (fit on the other 10 months of 2025)",
        f"- rows scored: {n:,} (all true taxi >= 0; no upper-bound exclusion --"
        f" the real board scores these rows too, see 'By AOBT_3_flt lane' below)",
        f"- target d clipped to [{LABEL_LO}, {LABEL_HI}] s in training; "
        f"taxi predictions clipped to [{FLOOR}, {CEIL}] s", "",
        "## Overall", "",
        _md_table([(f"{overall:.1f}", f"{mae:.1f}", f"{bias:+.1f}")],
                  ["RMSE (s)", "MAE (s)", "mean error (pred−true, s)"]), "",
    ]

    g = (ev.group_by("ADEP_mvt").agg(
            e.pow(2).mean().sqrt().alias("rmse"), e.mean().alias("bias"),
            pl.col("taxi").mean().alias("mt"), pl.len().alias("n"))
         .sort("rmse", descending=True))
    L += ["## Per airport", "",
          _md_table([(r["ADEP_mvt"], f"{r['rmse']:.1f}", f"{r['bias']:+.1f}",
                      f"{r['mt']:.0f}", f"{r['n']:,}") for r in g.iter_rows(named=True)],
                    ["airport", "rmse", "bias", "mean taxi", "n"]), ""]

    g = ev.group_by("ym").agg(
        e.pow(2).mean().sqrt().alias("rmse"), pl.len().alias("n")).sort("ym")
    L += ["## Per month (both must improve, not just the pooled number)", "",
          _md_table([(r["ym"], f"{r['rmse']:.1f}", f"{r['n']:,}")
                     for r in g.iter_rows(named=True)], ["month", "rmse", "n"]), ""]

    if "has_aobt3" in ev.columns:
        g = (ev.group_by("has_aobt3").agg(
                e.pow(2).mean().sqrt().alias("rmse"), e.pow(2).mean().alias("sqerr"),
                pl.len().alias("n"))
             .with_columns((pl.col("sqerr") * pl.col("n")).alias("sq_total"))
             .sort("has_aobt3", descending=True))
        total_sq = g["sq_total"].sum()
        L += ["## By AOBT_3_flt lane", "",
              "Rows missing an NM off-block time are a small, high-leverage "
              "population (echoes + schedule-default poison labels) that "
              "behaves very differently from the rest -- see PROGRESS.md.", "",
              _md_table([(bool(r["has_aobt3"]), f"{r['rmse']:.1f}", f"{r['n']:,}",
                          f"{100*r['sq_total']/total_sq:.1f}%")
                         for r in g.iter_rows(named=True)],
                        ["has_aobt3", "rmse", "n", "% of total squared error"]), ""]

    if "echo_pred" in ev.columns:
        g = (ev.group_by("echo_pred").agg(
                e.pow(2).mean().sqrt().alias("rmse"), e.pow(2).mean().alias("sqerr"),
                pl.len().alias("n"))
             .with_columns((pl.col("sqerr") * pl.col("n")).alias("sq_total"))
             .sort("echo_pred", descending=True))
        total_sq = g["sq_total"].sum()
        L += ["## By echo classifier verdict (P(echo) > 0.5)", "",
              "Supersedes the has_aobt3 lane above as the primary echo-lane "
              "diagnostic -- see fit_echo_classifier / PROGRESS.md.", "",
              _md_table([(bool(r["echo_pred"]), f"{r['rmse']:.1f}", f"{r['n']:,}",
                          f"{100*r['sq_total']/total_sq:.1f}%")
                         for r in g.iter_rows(named=True)],
                        ["echo_pred", "rmse", "n", "% of total squared error"]), ""]

    dec = ev.with_columns(((pl.col("taxi").rank("ordinal") - 1) * 10 // pl.len()).alias("dq"))
    g = (dec.group_by("dq").agg(
            e.pow(2).mean().sqrt().alias("rmse"), e.mean().alias("bias"),
            pl.col("taxi").min().alias("lo"), pl.col("taxi").max().alias("hi"),
            pl.len().alias("n")).sort("dq"))
    L += ["## Per true-taxi decile", "",
          _md_table([(f"d{r['dq']}", f"{r['lo']}–{r['hi']}", f"{r['rmse']:.1f}",
                      f"{r['bias']:+.1f}", f"{r['n']:,}") for r in g.iter_rows(named=True)],
                    ["decile", "taxi range (s)", "rmse", "bias", "n"]), ""]

    q = ev.select(
        pl.col("pred").min().alias("mn"), pl.col("pred").quantile(0.1).alias("p10"),
        pl.col("pred").median().alias("p50"), pl.col("pred").quantile(0.9).alias("p90"),
        pl.col("pred").max().alias("mx"),
        (pl.col("pred") <= FLOOR).sum().alias("f"),
        (pl.col("pred") >= CEIL).sum().alias("c")).row(0)
    L += ["## Predicted taxi distribution (holdout)", "",
          _md_table([tuple(f"{v:.0f}" for v in q[:5]) + (q[5], q[6])],
                    ["min", "p10", "p50", "p90", "max", "n@floor", "n@ceil"]), ""]

    if sub:
        L += ["## Submission file (ranking, Jan+Jul 2026)", "",
              _md_table([(f"{sub['rows']:,}", sub["min"], f"{sub['p10']:.0f}",
                          f"{sub['median']:.0f}", f"{sub['mean']:.0f}", f"{sub['p90']:.0f}",
                          sub["max"], sub["at_floor"], sub["at_ceil"])],
                        ["rows", "min", "p10", "median", "mean", "p90", "max",
                         "n@floor", "n@ceil"]), ""]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


# objective name per engine for each loss choice; alpha is each engine's delta/
# slope parameter for the robust (huber) loss, in seconds of residual -- within
# alpha, huber behaves like L2 (same as before); beyond it, the gradient is
# capped instead of growing with residual size, so a handful of huge-residual
# rows (LIRF, the d9 tail) can no longer pull splits toward fitting them exactly.
LOSS_OBJECTIVE = {
    "l2": {"lgb": "regression", "xgb": "reg:squarederror", "cat": "RMSE"},
    "huber": {"lgb": "huber", "xgb": "reg:pseudohubererror", "cat": "Huber"},
}
# alpha=800 beat plain L2 on the gap-realistic holdout (303.1s vs 309.2s overall,
# improved on 6/10 airports incl. LIRF, tail d9 667.7 vs 677.0s, no airport
# regressed by more than ~5s) -- see reports/eval/lgb_colab_huber_a800.md.
# alpha=400 was tried first and was worse (383.9s undertrained at a 2500-round
# ceiling, 315.6s even at the full 8000-round ceiling) -- it never converged and
# capped too much of the ordinary error range, not just genuine outliers.
HUBER_ALPHA = 800.0


# ------------------------------------------------------------------------ run
def run(engine: str = "lgb", feat_dir: Path = FEAT_DIR, name: str | None = None,
        *, rounds: int = ROUNDS, eta: float = ETA, es: int = EARLY_STOP,
        valid_month: str = VALID_MONTH, refit_scale: float = 1.1,
        loss: str = "huber", huber_alpha: float = HUBER_ALPHA, submit: bool = True,
        target: str = "flip", seed: int = 42, ev_out: str | Path | None = None):
    """target="flip" fits the regressor on pushback delay `d` and reconstructs
    taxi = offset - d_hat. target="direct" fits the regressor on `taxi`
    itself, skipping the reconstruction subtraction. target="mixed" fits BOTH
    (each still pooled across all 10 airports, exactly like the other two --
    NOT a per-airport model) and picks per-row by airport at reconstruction
    time: flip for LIRF, direct everywhere else. On the holdout, flip beats
    direct at LIRF by ~6% (857s vs 908s) while direct beats flip on every
    other airport (e.g. EDDF 231->193, EHAM 231->200) -- see PROGRESS.md.
    The echo blend in _reconstruct_taxi applies identically regardless of
    target."""
    if target not in ("flip", "direct", "mixed"):
        raise ValueError(f"target must be 'flip', 'direct', or 'mixed', got {target!r}")
    targets_to_fit = ("flip", "direct") if target == "mixed" else (target,)
    t0 = time.time()
    fitter = ENGINES[engine]
    name = name or f"{engine}_colab"
    objective = LOSS_OBJECTIVE[loss][engine]
    alpha = huber_alpha if loss == "huber" else None

    feats = pl.read_parquet(feat_dir / "train2025.parquet")
    feats_ho = pl.read_parquet(feat_dir / "holdout_gap2025.parquet")
    lab = pl.read_parquet(feat_dir / "labels2025.parquet")
    off_ho = feats_ho.select("MVT_ID_mvt", "sched_takeoff_offset")
    _mem("read feature frames")

    # the group encoders key on operator, which lives only in the feature frame
    lab = lab.join(feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"),
                   on="MVT_ID_mvt", how="left")

    is_ho = pl.col("ym").is_in(HOLDOUT_MONTHS)
    tr_lab, ho_lab = lab.filter(~is_ho), lab.filter(is_ho)

    # Holdout features come from the isolated Jan+Jul frame (export_model_inputs.py),
    # not a slice of the continuous train2025 frame -- that mirrors ranking.parquet's
    # real Jan->Jul gap instead of hiding it behind a continuous 12-month timeline,
    # which is what let a real bug (mins_since_cfg_change bridging the gap) hide
    # behind a good holdout RMSE.
    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    f_ho = apply_priors(feats_ho.join(ho_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    ho_lab = f_ho.select("MVT_ID_mvt").join(ho_lab, on="MVT_ID_mvt")

    # echo-rate / mean-d group encodings: OOF (leave-one-month-out) for the
    # training rows, full training-split fit for the holdout.
    f_tr = add_group_encodings_oof(f_tr, tr_lab)
    f_ho = apply_group_encodings(f_ho, fit_group_encodings(tr_lab))
    _mem("priors + group encodings (f_tr, f_ho)")

    Xtr, names, cats, categories = _matrix(f_tr)
    _mem("Xtr pandas matrix")
    d_tr = tr_lab["d"].to_numpy()
    taxi_tr = tr_lab["taxi"].to_numpy()
    ym_tr = tr_lab["ym"].to_numpy()
    keep = (taxi_tr >= LABEL_LO) & (taxi_tr <= LABEL_HI)
    iv = ym_tr == valid_month
    if not iv.any():
        raise ValueError(f"valid_month {valid_month!r} not in training months")
    tr_mask, va_mask = keep & ~iv, keep & iv
    fits = {}
    for t in targets_to_fit:
        y_tr = d_tr if t == "flip" else taxi_tr
        _mem(f"before holdout fit ({t})")
        m, pf, bi = fitter(
            Xtr[tr_mask], y_tr[tr_mask], cats,
            eta=eta, rounds=rounds, es=es, valid=(Xtr[va_mask], y_tr[va_mask]),
            objective=objective, alpha=alpha, seed=seed)
        print(f"holdout fit {time.time() - t0:.0f}s  best_iter={bi}  loss={loss}  target={t}")
        _mem(f"after holdout fit ({t})")
        fits[t] = (m, pf, bi)

    # Echo classifier: same X matrix as the d-regressor, NOT filtered by `keep`
    # (see fit_echo_classifier). valid_d excludes the rare null-d rows.
    valid_d = ~np.isnan(d_tr)
    clf_tr_mask, clf_va_mask = valid_d & ~iv, valid_d & iv
    clf_model, clf_pred, clf_best, is_echo_tr = fit_echo_classifier(
        engine, Xtr, d_tr, clf_tr_mask, clf_va_mask, cats)
    print(f"echo clf fit {time.time() - t0:.0f}s  best_iter={clf_best}  "
          f"base_rate={is_echo_tr[valid_d].mean():.3f}")
    _mem("after echo clf")

    Xho, _, _, _ = _matrix(f_ho, categories)
    ho_off = ho_lab.join(off_ho, on="MVT_ID_mvt")["sched_takeoff_offset"].to_numpy()
    echo_prob_ho = clf_pred(clf_model, Xho[names])
    nm_unmatched_ho = f_ho["aobt3_taxi"].is_null().to_numpy()
    not_lirf_ho = (f_ho["ADEP_mvt"] != "LIRF").to_numpy()
    prior_taxi_ho = np.nan_to_num(f_ho["median_taxi_prior"].to_numpy())
    use_prior_ho = nm_unmatched_ho & not_lirf_ho & ~np.isnan(prior_taxi_ho)
    ceil_ho = np.where(not_lirf_ho, CEIL, LIRF_RAW_CEIL)

    def _recon_ho(t):
        m, pf, _ = fits[t]
        raw = pf(m, Xho[names])
        raw_taxi = (ho_off - raw) if t == "flip" else raw
        return (_reconstruct_taxi(ho_off, raw_taxi, echo_prob_ho, use_prior_ho,
                                   prior_taxi_ho, ceil_ho), raw_taxi)

    if target == "mixed":
        is_lirf_ho = (f_ho["ADEP_mvt"] == "LIRF").to_numpy()
        recon_flip, raw_flip = _recon_ho("flip")
        recon_direct, raw_direct = _recon_ho("direct")
        taxi_hat = np.where(is_lirf_ho, recon_flip, recon_direct)
        taxi_model_raw_diag = np.where(is_lirf_ho, raw_flip, raw_direct)
    else:
        taxi_hat, taxi_model_raw_diag = _recon_ho(target)

    # Classifier sanity check BEFORE trusting the blend: precision/recall
    # against the true is_echo label (ho_lab row order matches f_ho/Xho since
    # ho_lab was re-joined onto f_ho's MVT_ID_mvt above).
    is_echo_ho = np.abs(ho_lab["d"].to_numpy()) < _ECHO_ABS_D
    pred_echo = echo_prob_ho > 0.5
    tp, fp = int((pred_echo & is_echo_ho).sum()), int((pred_echo & ~is_echo_ho).sum())
    fn = int((~pred_echo & is_echo_ho).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"echo clf holdout  precision={prec:.3f}  recall={rec:.3f}  "
          f"n_true_echo={int(is_echo_ho.sum())}  n_pred_echo={int(pred_echo.sum())}")

    # No upper-bound filter on true taxi here anymore -- a `taxi < 4h` cutoff used
    # to hide the handful of rows (echoes with a huge sched-to-takeoff gap) that
    # the real board almost certainly scores anyway. Only drop rows with a
    # physically impossible negative/null true label.
    has_aobt3 = f_ho.select("MVT_ID_mvt", has_aobt3=pl.col("aobt3_taxi").is_not_null())
    ev = (
        ho_lab.join(off_ho, on="MVT_ID_mvt")
        .join(has_aobt3, on="MVT_ID_mvt")
        .with_columns(
            pred=pl.Series(taxi_hat),
            echo_pred=pl.Series(pred_echo),
            echo_prob=pl.Series(echo_prob_ho),
            is_echo=pl.Series(is_echo_ho),
            taxi_model_raw=pl.Series(taxi_model_raw_diag),
            use_prior=pl.Series(use_prior_ho),
        )
        .filter(pl.col("taxi") >= 0)
    )
    _report(ev)
    if ev_out is not None:  # persist the holdout predictions before the long refit phase
        Path(ev_out).parent.mkdir(parents=True, exist_ok=True)
        ev.write_parquet(ev_out)
        print(f"wrote holdout predictions -> {ev_out}", flush=True)

    best_iter_report = "/".join(f"{t}={fits[t][2]}" for t in targets_to_fit)
    if not submit:
        # holdout-only: skip the refit-on-all-2025 + ranking-predict + write
        # phase, which is the expensive half of a full run and pointless when
        # we're only validating a change against the holdout, not submitting.
        _write_report(ev, name, engine, None, meta=dict(
            eta=eta, rounds=rounds, best_iter=best_iter_report, valid_month=valid_month,
            loss=loss, huber_alpha=huber_alpha, target=target))
        print(f"total {time.time() - t0:.0f}s (holdout-only, no submission)")
        return fits, ev

    # refit on all 2025 + ranking submission, no early stopping — reuse the
    # iteration count early stopping found, nudged up for the larger data.
    priors_a = fit_priors(lab)
    f_all = apply_priors(feats, priors_a)
    lab_a = f_all.select("MVT_ID_mvt").join(lab, on="MVT_ID_mvt")
    f_all = add_group_encodings_oof(f_all, lab_a)
    genc_a = fit_group_encodings(lab_a)
    Xall, names_a, cats_a, cats_map = _matrix(f_all)
    _mem("refit: Xall matrix")
    keep = lab_a["taxi"].is_between(LABEL_LO, LABEL_HI).to_numpy()
    d_all_lab = lab_a["d"].to_numpy()
    taxi_all_lab = lab_a["taxi"].to_numpy()
    fits_a = {}
    for t in targets_to_fit:
        full_rounds = max(200, int(round(fits[t][2] * refit_scale)))
        y_all = d_all_lab if t == "flip" else taxi_all_lab
        m_a, p_a, _ = fitter(Xall[keep], y_all[keep], cats_a,
                             eta=eta, rounds=full_rounds, es=0, valid=None,
                             objective=objective, alpha=alpha, seed=seed)
        fits_a[t] = (m_a, p_a)
        _mem(f"refit: after fit ({t})")

    # echo classifier, refit on all 2025 labelled rows (not filtered by `keep`
    # -- see fit_echo_classifier). No early stopping, same convention as model_a.
    clf_full_rounds = max(200, int(round(clf_best * refit_scale)))
    d_all = lab_a["d"].to_numpy()
    valid_d_all = ~np.isnan(d_all)
    is_echo_all = (np.abs(d_all) < _ECHO_ABS_D).astype(np.float64)
    clf_model_a, clf_pred_a, _ = fitter(
        Xall[valid_d_all], is_echo_all[valid_d_all], cats_a,
        eta=CLF_ETA, rounds=clf_full_rounds, es=0, valid=None,
        objective=CLF_OBJECTIVE[engine], metric=CLF_METRIC[engine])

    f_r = apply_priors(pl.read_parquet(feat_dir / "ranking.parquet"), priors_a)
    f_r = apply_group_encodings(f_r, genc_a)
    Xr, _, _, _ = _matrix(f_r, cats_map)
    r_off = f_r["sched_takeoff_offset"].to_numpy()
    echo_prob_r = clf_pred_a(clf_model_a, Xr[names_a])
    nm_unmatched_r = f_r["aobt3_taxi"].is_null().to_numpy()
    not_lirf_r = (f_r["ADEP_mvt"] != "LIRF").to_numpy()
    prior_taxi_r = np.nan_to_num(f_r["median_taxi_prior"].to_numpy())
    use_prior_r = nm_unmatched_r & not_lirf_r & ~np.isnan(prior_taxi_r)
    ceil_r = np.where(not_lirf_r, CEIL, LIRF_RAW_CEIL)

    def _recon_r(t):
        m_a, p_a = fits_a[t]
        raw = p_a(m_a, Xr[names_a])
        raw_taxi = (r_off - raw) if t == "flip" else raw
        return _reconstruct_taxi(r_off, raw_taxi, echo_prob_r, use_prior_r, prior_taxi_r, ceil_r)

    if target == "mixed":
        is_lirf_r = (f_r["ADEP_mvt"] == "LIRF").to_numpy()
        taxi_r = np.where(is_lirf_r, _recon_r("flip"), _recon_r("direct"))
    else:
        taxi_r = _recon_r(target)

    sub = _write_submission(f_r["MVT_ID_mvt"].to_list(), taxi_r, name)
    full_rounds_report = "/".join(
        f"{t}={max(200, int(round(fits[t][2] * refit_scale)))}" for t in targets_to_fit)
    _write_report(ev, name, engine, sub, meta=dict(
        eta=eta, rounds=rounds, best_iter=best_iter_report, full_rounds=full_rounds_report,
        valid_month=valid_month, loss=loss, huber_alpha=huber_alpha, target=target))
    print(f"total {time.time() - t0:.0f}s")
    return fits_a, ev


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "lgb")
