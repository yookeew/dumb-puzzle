"""Pre-registered: OOF residual corrector on CatBoost.

reports/oof_corrector_preregistration.md, steps 2 (fold QA), 3 (corrector)
and 4 (holdout evaluation). Needs:
  cache/oof/cat_mixed/fold=<m>.parquet   (models.fit.run_oof, on Colab)
  cache/eval/cat_mixed_holdout_ev.parquet (run(engine="cat", ev_out=...))
  cache/features/                         (export_model_inputs.py)
Optional (stack diagnostic): cache/eval/lgb_mixed_holdout_ev.parquet.

Run:  .venv/Scripts/python.exe tests/oof_corrector_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402
from features.encode import (  # noqa: E402
    add_group_encodings_oof,
    apply_group_encodings,
    apply_priors,
    fit_group_encodings,
    fit_priors,
)
from models.fit import (  # noqa: E402
    HOLDOUT_MONTHS,
    LABEL_HI,
    LABEL_LO,
    TRAIN_MONTHS,
    _matrix,
)

FEAT = ROOT / "cache" / "features"
EVAL = ROOT / "cache" / "eval"
# overrides exist only for the pipeline smoke test on throwaway inputs
OOF = Path(os.environ.get("PRC_OOF_DIR", ROOT / "cache" / "oof" / "cat_mixed"))
CAT_EV = Path(os.environ.get("PRC_CAT_EV", EVAL / "cat_mixed_holdout_ev.parquet"))
JAN, JUL = HOLDOUT_MONTHS
MONSTER_S = 5 * 3600
N_MONSTER_EXPECTED = 31
ES_MONTH = "2025-06"
CORR_PARAMS = dict(
    objective="regression", metric="rmse", num_leaves=15, max_depth=4,
    min_data_in_leaf=2000, lambda_l2=10.0, learning_rate=0.05, feature_fraction=0.8,
    max_bin=127, deterministic=True, force_row_wise=True, seed=42, num_threads=0,
    verbose=-1,
)
CORR_ROUNDS, CORR_ES = 2000, 100


def rmse(x, t, m=None):
    return float(np.sqrt(np.mean((x - t) ** 2 if m is None else (x[m] - t[m]) ** 2)))


def load_oof() -> pl.DataFrame:
    files = sorted(OOF.glob("fold=*.parquet"))
    months = [f.stem.split("=", 1)[1] for f in files]
    if sorted(months) != sorted(TRAIN_MONTHS):
        raise SystemExit(f"OOF folds present {months}, need {list(TRAIN_MONTHS)}")
    oof = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    if oof["MVT_ID_mvt"].n_unique() != oof.height:
        raise SystemExit("an MVT_ID_mvt appears in more than one OOF fold")
    return oof


def build_inputs():
    """Base feature matrices exactly as run() builds them for the holdout fit:
    priors on the 10 training months, OOF-by-month group encodings for the
    training rows, full training fit for the holdout rows."""
    feats = pl.read_parquet(FEAT / "train2025.parquet")
    feats_ho = pl.read_parquet(FEAT / "holdout_gap2025.parquet")
    lab = pl.read_parquet(FEAT / "labels2025.parquet")
    lab = lab.join(feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"), on="MVT_ID_mvt", how="left")
    tr_lab = lab.filter(pl.col("ym").is_in(TRAIN_MONTHS))
    ho_lab = lab.filter(pl.col("ym").is_in(HOLDOUT_MONTHS))
    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    f_ho = apply_priors(feats_ho.join(ho_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    f_tr = add_group_encodings_oof(f_tr, tr_lab)
    f_ho = apply_group_encodings(f_ho, fit_group_encodings(tr_lab))
    feats_ho_day = feats_ho.select("MVT_ID_mvt", day=pl.col("T").dt.date())
    return f_tr, f_ho, lab, feats_ho_day


def fit_corrector(X, y, ym, objective="regression", alpha=None):
    import lightgbm as lgb

    params = dict(CORR_PARAMS, objective=objective)
    if alpha is not None:
        params["alpha"] = alpha
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    va = ym == ES_MONTH
    ds = lgb.Dataset(X[~va], label=y[~va], categorical_feature=cats, free_raw_data=False)
    vds = lgb.Dataset(X[va], label=y[va], reference=ds, categorical_feature=cats, free_raw_data=False)
    m = lgb.train(params, ds, num_boost_round=CORR_ROUNDS, valid_sets=[vds],
                  callbacks=[lgb.early_stopping(CORR_ES, verbose=False)])
    best = m.best_iteration or CORR_ROUNDS
    es_rmse_before = float(np.sqrt(np.mean(y[va] ** 2)))
    es_rmse_after = float(np.sqrt(np.mean((y[va] - m.predict(X[va], num_iteration=best)) ** 2)))
    full_rounds = max(50, int(round(1.1 * best)))
    ds_all = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
    m_all = lgb.train(params, ds_all, num_boost_round=full_rounds)
    print(f"  corrector[{objective}] best_iter={best} full_rounds={full_rounds}  "
          f"{ES_MONTH} residual RMSE {es_rmse_before:.2f} -> {es_rmse_after:.2f} (label range)")
    return m_all


def main() -> None:
    oof = load_oof()
    ev = pl.read_parquet(CAT_EV).select(
        "MVT_ID_mvt", "ADEP_mvt", pl.col("ym").cast(pl.Utf8), "taxi",
        "pred", "echo_prob")
    f_tr, f_ho, lab, ho_day = build_inputs()

    # ---------------------------------------------------------- fold QA
    q = oof.join(lab.select("MVT_ID_mvt", "taxi"), on="MVT_ID_mvt", how="left")
    tr_ids = lab.filter(pl.col("ym").is_in(TRAIN_MONTHS) & (pl.col("taxi") >= 0))
    missing = tr_ids.join(oof.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").height
    print(f"OOF rows {oof.height:,}; labelled training rows without an OOF prediction: {missing}")
    print("fold QA (month m predicted by a fit on the other 9 training months):")
    for m in TRAIN_MONTHS:
        f = q.filter((pl.col("ym") == m) & (pl.col("taxi") >= 0))
        t, p = f["taxi"].to_numpy(), f["pred"].to_numpy()
        keep = t <= MONSTER_S
        r = f.row(0, named=True)
        print(f"  {m}  n={f.height:7,}  full={rmse(p, t):7.2f}  trimmed={rmse(p, t, keep):7.2f}  "
              f"valid={r['valid_month']}  iters flip/direct/clf="
              f"{r.get('best_iter_flip')}/{r.get('best_iter_direct')}/{r['best_iter_clf']}")
    th, ph = ev["taxi"].to_numpy(), ev["pred"].to_numpy()
    print(f"  holdout CatBoost (10-month fit): full={rmse(ph, th):.2f}  "
          f"trimmed={rmse(ph, th, th <= MONSTER_S):.2f}")

    # ---------------------------------------------------------- corrector inputs
    tr = (f_tr.select("MVT_ID_mvt").with_row_index("_i")
          .join(oof.select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt")
          .join(lab.select("MVT_ID_mvt", "taxi", "ym"), on="MVT_ID_mvt")
          .filter(pl.col("taxi").is_between(LABEL_LO, LABEL_HI)))
    ho = (f_ho.select("MVT_ID_mvt").with_row_index("_i")
          .join(ev.select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt"))
    if ho.height != ev.height:
        raise SystemExit(f"holdout rows: features {ho.height} vs cat ev {ev.height}")
    Xtr_all, names, cats, categories = _matrix(f_tr)
    Xho_all, _, _, _ = _matrix(f_ho, categories)
    Xtr = Xtr_all.iloc[tr["_i"].to_numpy()].reset_index(drop=True)
    Xho = Xho_all.iloc[ho["_i"].to_numpy()].reset_index(drop=True)
    del Xtr_all, Xho_all
    Xtr["base_pred"], Xtr["base_echo_prob"] = tr["pred"].to_numpy(), tr["echo_prob"].to_numpy()
    Xho["base_pred"], Xho["base_echo_prob"] = ho["pred"].to_numpy(), ho["echo_prob"].to_numpy()
    y = tr["taxi"].to_numpy() - tr["pred"].to_numpy()
    ym = tr["ym"].to_numpy()
    print(f"\ncorrector training rows {len(y):,} (training months, {LABEL_LO} <= taxi <= {LABEL_HI})")

    m_l2 = fit_corrector(Xtr, y, ym)
    m_hub = fit_corrector(Xtr, y, ym, objective="huber", alpha=800.0)

    # ---------------------------------------------------------- holdout
    df = (ho.select("MVT_ID_mvt", "pred").join(ev.select("MVT_ID_mvt", "ADEP_mvt", "ym", "taxi"), on="MVT_ID_mvt")
          .join(ho_day, on="MVT_ID_mvt", how="left")
          .with_columns(corr=pl.Series(np.maximum(0.0, ho["pred"].to_numpy() + m_l2.predict(Xho))),
                        corr_huber=pl.Series(np.maximum(0.0, ho["pred"].to_numpy() + m_hub.predict(Xho)))))
    t = df["taxi"].to_numpy()
    keep = t <= MONSTER_S
    n_monster = int((~keep).sum())
    print(f"\nholdout rows {df.height:,}; monster labels (> 5 h): {n_monster}")
    if n_monster != N_MONSTER_EXPECTED:
        raise SystemExit(f"expected {N_MONSTER_EXPECTED} monster rows, got {n_monster}")
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()

    def score(treat: str, mask: np.ndarray, label: str):
        d = df.with_columns(se_b=(pl.col("pred") - pl.col("taxi")) ** 2,
                            se_t=(pl.col(treat) - pl.col("taxi")) ** 2).filter(pl.Series(mask))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), 3000, 0)
        print(f"  {label:<24} base={np.sqrt(d['se_b'].mean()):7.2f}  corr={np.sqrt(d['se_t'].mean()):7.2f}  "
              f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    print("\nPRIMARY (L2 corrector) vs uncorrected CatBoost")
    print(" TRIMMED (decides):")
    d_jul, _ = score("corr", jul & keep, "Jul trimmed")
    d_jan, _ = score("corr", jan & keep, "Jan trimmed")
    d_pool, p_pool = score("corr", keep, "POOLED trimmed")
    print(" FULL (guard):")
    score("corr", jul, "Jul full")
    score("corr", jan, "Jan full")
    _, p_full = score("corr", np.ones(df.height, bool), "POOLED full")
    adopt = d_pool < 0 and p_pool < 0.05 and d_jan < 0 and d_jul < 0 and p_full < 0.9
    print(f"\nrule: pooled trimmed delta<0 & P<0.05 [{d_pool:+.2f}, {p_pool:.3f}], "
          f"both months trimmed<0 [Jan {d_jan:+.2f}, Jul {d_jul:+.2f}], full guard P<0.9 [{p_full:.3f}]"
          f"\n  -> {'ADOPT' if adopt else 'REJECT'}")

    # ---------------------------------------------------------- diagnostics
    print("\nDIAGNOSTICS (not decisive)")
    not_lirf = (df["ADEP_mvt"] != "LIRF").to_numpy()
    score("corr", keep & not_lirf, "pooled trimmed, no LIRF")
    print(" Huber variant:")
    score("corr_huber", keep, "POOLED trimmed (huber)")
    score("corr_huber", np.ones(df.height, bool), "POOLED full (huber)")

    gain = (df["pred"].to_numpy() - t) ** 2 - (df["corr"].to_numpy() - t) ** 2
    g = gain[keep]
    top = np.sort(g)[::-1][:100].sum()
    print(f" trimmed SSE reduction {g.sum():.4g}; top-100 rows contribute {top:.4g} "
          f"({100 * top / g.sum() if g.sum() > 0 else float('nan'):.1f}%)")

    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=2, tbl_rows=20):
        dk = df.filter(pl.Series(keep))
        agg = lambda c: ((pl.col(c) - pl.col("taxi")) ** 2).mean().sqrt()  # noqa: E731
        print("\nper airport, trimmed:")
        print(dk.group_by("ADEP_mvt").agg(agg("pred").alias("base"), agg("corr").alias("corr"), pl.len())
              .with_columns(delta=pl.col("corr") - pl.col("base")).sort("ADEP_mvt"))
        print("per airport, full:")
        print(df.group_by("ADEP_mvt").agg(agg("pred").alias("base"), agg("corr").alias("corr"), pl.len())
              .with_columns(delta=pl.col("corr") - pl.col("base")).sort("ADEP_mvt"))
        print("per decile of true taxi (all rows):")
        print(df.with_columns(dec=pl.col("taxi").qcut(10, labels=[str(i) for i in range(1, 11)]))
              .group_by("dec").agg(pl.col("taxi").max().alias("taxi_max"), agg("pred").alias("base"),
                                   agg("corr").alias("corr"), pl.len())
              .with_columns(delta=pl.col("corr") - pl.col("base")).sort("taxi_max"))
        print("mean residual (taxi - pred) per airport, labels in range: OOF training months vs holdout")
        oof_res = tr.join(lab.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt").group_by("ADEP_mvt").agg(
            (pl.col("taxi") - pl.col("pred")).mean().alias("oof_mean_res"))
        ho_res = df.filter(pl.col("taxi").is_between(LABEL_LO, LABEL_HI)).group_by("ADEP_mvt").agg(
            (pl.col("taxi") - pl.col("pred")).mean().alias("holdout_mean_res"))
        print(oof_res.join(ho_res, on="ADEP_mvt", how="full", coalesce=True).sort("ADEP_mvt"))

    lgb_path = EVAL / "lgb_mixed_holdout_ev.parquet"
    if lgb_path.exists():
        from scipy.optimize import nnls

        s = df.join(pl.read_parquet(lgb_path).select("MVT_ID_mvt", pl.col("pred").alias("lgb")),
                    on="MVT_ID_mvt")

        def cross(cols):
            out = np.empty(s.height)
            for fm, am in ((JAN, JUL), (JUL, JAN)):
                f = s.filter(pl.col("ym") == fm)
                w, _ = nnls(f.select(cols).to_numpy(), f["taxi"].to_numpy())
                mm = (s["ym"] == am).to_numpy()
                out[mm] = s.filter(pl.col("ym") == am).select(cols).to_numpy() @ w
            return out

        ts = s["taxi"].to_numpy()
        ks = ts <= MONSTER_S
        a, b = cross(["lgb", "pred"]), cross(["lgb", "corr"])
        print(f"\nstack lgb+cat vs lgb+cat_corrected (cross-fit NNLS): full {rmse(a, ts):.2f} -> {rmse(b, ts):.2f}"
              f"  trimmed {rmse(a, ts, ks):.2f} -> {rmse(b, ts, ks):.2f}")
    else:
        print(f"\n(stack diagnostic skipped: {lgb_path} not present)")


if __name__ == "__main__":
    main()
