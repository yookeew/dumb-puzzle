"""Pre-registered holdout gate: OOF correctors on both engines, 20k-round cap.

reports/oof_corrector_lgb_preregistration.md, as amended 2026-09-29 (before
results). Each corrector is the v2 design (tests/oof_corrector_v2_test.py:
Huber 800, rows and application selected on the base prediction <= 7,200 s,
NM-unmatched carrier echo rate), with the round cap raised to 20,000.

  lgbcorr20  corrector on the stack's LightGBM
  catcorr20  corrector on the rerun CatBoost (the one v21 uses)
  joint      Arm B: one corrector on base_B = NNLS(lgb, cat) with weights fit on
             the training-month OOF predictions; inputs carry both engines'
             pred and echo_prob

  control  NNLS(lgb, catcorr)            the v21 stack
  Arm A    NNLS(lgbcorr20, catcorr20)    primary
  Arm B    joint, used as is (no stack)  secondary

Stacks are cross-fit (weights fit on Jan score Jul, and the reverse). Decision
order (amendment item 5): A vs control; B replaces A only if it passes vs control
and vs A; if A fails, B only if it passes vs control.

Needs:
  cache/oof/lgb_mixed/fold=<m>.parquet          run_oof(engine="lgb", target="mixed")
  cache/oof/cat_mixed/fold=<m>.parquet          run_oof(engine="cat", target="mixed", eta=0.05)
  cache/eval/lgb_mixed_holdout_ev.parquet       the stack's LightGBM
  cache/eval/cat_mixed_rerun_holdout_ev.parquet the v21 CatBoost rerun
  cache/eval/catcorr_mixed_holdout_ev.parquet   v21's corrected CatBoost (control)
Writes cache/eval/{lgbcorr20,catcorr20,joint20}_mixed_holdout_ev.parquet, the
fitted 10-month correctors (cache/oof/corrector20_<name>_m10.txt, reused on a
rerun) and cache/oof/corrector20_meta.json (round counts, Arm B weights) for the
v22 ranking refit.

Run:  .venv/Scripts/python.exe tests/oof_corrector_lgb_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oof_corrector_test as v1  # noqa: E402
import oof_corrector_v2_test as v2  # noqa: E402
from _harness import cluster_bootstrap  # noqa: E402
from features.encode import NMU_COLS  # noqa: E402
from models.fit import HOLDOUT_MONTHS, TRAIN_MONTHS, _matrix  # noqa: E402
from stack_r20k_test import cross  # noqa: E402

ROOT = v1.ROOT
EVAL = ROOT / "cache" / "eval"
OOF_DIR = ROOT / "cache" / "oof"
JAN, JUL = HOLDOUT_MONTHS
CAP, ES_PATIENCE = 20_000, 100
BASE_EV = {"lgb": EVAL / "lgb_mixed_holdout_ev.parquet",
           "cat": EVAL / "cat_mixed_rerun_holdout_ev.parquet"}
CATCORR_EV = EVAL / "catcorr_mixed_holdout_ev.parquet"
META = OOF_DIR / "corrector20_meta.json"


def load_oof(engine: str) -> pl.DataFrame:
    files = sorted((OOF_DIR / f"{engine}_mixed").glob("fold=*.parquet"))
    months = [f.stem.split("=", 1)[1] for f in files]
    if sorted(months) != sorted(TRAIN_MONTHS):
        raise SystemExit(f"{engine} OOF folds present {months}, need {list(TRAIN_MONTHS)}")
    oof = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    if oof["MVT_ID_mvt"].n_unique() != oof.height:
        raise SystemExit(f"{engine}: an MVT_ID_mvt appears in more than one OOF fold")
    return oof.select("MVT_ID_mvt", pl.col("pred").alias(f"{engine}_pred"),
                      pl.col("echo_prob").alias(f"{engine}_echo_prob"))


def load_ev(engine: str) -> pl.DataFrame:
    return pl.read_parquet(BASE_EV[engine]).select(
        "MVT_ID_mvt", pl.col("pred").alias(f"{engine}_pred"), pl.col("echo_prob").alias(f"{engine}_echo_prob"))


def fit_corrector(name: str, X, y, ym):
    """v2's fit with the 20k cap: early stopping on June (a training month), then a
    refit on all 10 training months for round(1.1 x best_iter). Cached by name."""
    import lightgbm as lgb

    path = OOF_DIR / f"corrector20_{name}_m10.txt"
    info_path = path.with_suffix(".json")
    if path.exists() and info_path.exists():
        print(f"  {name}: loading cached corrector {path.name}")
        return lgb.Booster(model_file=str(path)), json.loads(info_path.read_text())
    params = dict(v1.CORR_PARAMS, objective="huber", alpha=v2.HUBER_ALPHA)
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    va = ym == v1.ES_MONTH
    ds = lgb.Dataset(X[~va], label=y[~va], categorical_feature=cats, free_raw_data=False)
    vds = lgb.Dataset(X[va], label=y[va], reference=ds, categorical_feature=cats, free_raw_data=False)
    m = lgb.train(params, ds, num_boost_round=CAP, valid_sets=[vds],
                  callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    best = m.best_iteration or CAP
    r0 = float(np.sqrt(np.mean(y[va] ** 2)))
    r1 = float(np.sqrt(np.mean((y[va] - m.predict(X[va], num_iteration=best)) ** 2)))
    full_rounds = max(50, int(round(1.1 * best)))
    m_all = lgb.train(params, lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False),
                      num_boost_round=full_rounds)
    m_all.save_model(str(path))
    info = dict(best_iter=best, full_rounds=full_rounds, hit_cap=best >= CAP)
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  {name}: best_iter={best} of cap {CAP}{'  (HIT THE CAP)' if info['hit_cap'] else ''} "
          f"full_rounds={full_rounds}  {v1.ES_MONTH} residual RMSE {r0:.2f} -> {r1:.2f}")
    return m_all, info


def run_corrector(name, Xtr_all, Xho_all, tr, ho, base_col, extra_cols):
    """Fit on training rows with base <= PRED_MAX, apply to holdout rows with
    base <= PRED_MAX; the rest keep the base. Returns (corrected holdout, info)."""
    trf = tr.filter((pl.col("taxi") >= 0) & (pl.col(base_col) <= v2.PRED_MAX))
    Xtr = Xtr_all.iloc[trf["_i"].to_numpy()].reset_index(drop=True)
    Xho = Xho_all.iloc[ho["_i"].to_numpy()].reset_index(drop=True)
    for X, f in ((Xtr, trf), (Xho, ho)):
        for c in (*extra_cols, *NMU_COLS):
            X[c] = f[c].cast(pl.Float64).to_numpy()
    y = trf["taxi"].to_numpy() - trf[base_col].to_numpy()
    print(f"\n{name}: training rows {len(y):,} (base {base_col} <= {v2.PRED_MAX:.0f})")
    model, info = fit_corrector(name, Xtr, y, trf["ym"].to_numpy())
    base = ho[base_col].to_numpy()
    mask = base <= v2.PRED_MAX
    out = base.copy()
    out[mask] = np.maximum(0.0, base[mask] + model.predict(Xho[mask]))
    info["applied_frac"] = float(mask.mean())
    return out, info


def main() -> None:
    oof = load_oof("lgb").join(load_oof("cat"), on="MVT_ID_mvt", how="inner")
    ev = load_ev("lgb").join(load_ev("cat"), on="MVT_ID_mvt", how="inner")
    f_tr, f_ho, lab, ho_day = v1.build_inputs()
    nmu_tr, nmu_ho = v2.nmu_inputs(f_tr, f_ho, lab)

    tr = (f_tr.select("MVT_ID_mvt").with_row_index("_i")
          .join(oof, on="MVT_ID_mvt")
          .join(lab.select("MVT_ID_mvt", "taxi", "ym"), on="MVT_ID_mvt")
          .join(nmu_tr, on="MVT_ID_mvt", how="left"))
    ho = (f_ho.select("MVT_ID_mvt").with_row_index("_i")
          .join(ev, on="MVT_ID_mvt")
          .join(nmu_ho, on="MVT_ID_mvt", how="left"))
    n_ev = pl.read_parquet(BASE_EV["lgb"]).height
    if ho.height != n_ev:
        raise SystemExit(f"holdout rows: features x both engines {ho.height} vs lgb ev {n_ev}")
    print(f"OOF rows with both engines: {tr.height:,}; holdout rows: {ho.height:,}")

    Xtr_all, _, _, categories = _matrix(f_tr)
    Xho_all, _, _, _ = _matrix(f_ho, categories)
    Xho_all = Xho_all[list(Xtr_all.columns)]  # align by name, as run() does

    # Arm B base: NNLS on the training-month OOF predictions only
    fit_rows = tr.filter(pl.col("taxi") >= 0)
    w_b, _ = nnls(fit_rows.select("lgb_pred", "cat_pred").to_numpy(), fit_rows["taxi"].to_numpy())
    print(f"Arm B base weights (NNLS on OOF): lgb {w_b[0]:.3f}  cat {w_b[1]:.3f}")
    tr = tr.with_columns(base_B=w_b[0] * pl.col("lgb_pred") + w_b[1] * pl.col("cat_pred"))
    ho = ho.with_columns(base_B=w_b[0] * pl.col("lgb_pred") + w_b[1] * pl.col("cat_pred"))

    both = ("lgb_pred", "lgb_echo_prob", "cat_pred", "cat_echo_prob")
    specs = {  # name -> (base column, extra inputs as (frame column, matrix column))
        "lgbcorr20": ("lgb_pred", {"lgb_pred": "base_pred", "lgb_echo_prob": "base_echo_prob"}),
        "catcorr20": ("cat_pred", {"cat_pred": "base_pred", "cat_echo_prob": "base_echo_prob"}),
        "joint20": ("base_B", {c: c for c in both}),
    }
    preds, infos = {}, {}
    for name, (base_col, extra) in specs.items():
        trn = tr.rename({k: v for k, v in extra.items() if k != v})
        hon = ho.rename({k: v for k, v in extra.items() if k != v})
        base_col_n = extra.get(base_col, base_col)
        preds[name], infos[name] = run_corrector(name, Xtr_all, Xho_all, trn, hon, base_col_n,
                                                 list(extra.values()))
    del Xtr_all, Xho_all

    META.write_text(json.dumps(dict(cap=CAP, pred_max=v2.PRED_MAX, huber_alpha=v2.HUBER_ALPHA,
                                    arm_b_weights=dict(lgb=float(w_b[0]), cat=float(w_b[1])),
                                    correctors=infos), indent=2))
    lab_ho = pl.read_parquet(BASE_EV["lgb"]).select("MVT_ID_mvt", "ADEP_mvt", pl.col("ym").cast(pl.Utf8), "taxi")
    catcorr = pl.read_parquet(CATCORR_EV).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias("catcorr"))
    df = (ho.select("MVT_ID_mvt", pl.col("lgb_pred").alias("lgb"), pl.col("cat_pred").alias("cat"))
          .with_columns(**{k: pl.Series(v) for k, v in preds.items()})
          .join(lab_ho, on="MVT_ID_mvt").join(ho_day, on="MVT_ID_mvt", how="left")
          .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
          .join(catcorr, on="MVT_ID_mvt"))
    if df.height != ho.height:
        raise SystemExit(f"holdout join lost rows: {ho.height} -> {df.height}")
    for name in preds:
        df.select("MVT_ID_mvt", "ADEP_mvt", "ym", "taxi", pl.col(name).alias("pred")).write_parquet(
            EVAL / f"{name}_mixed_holdout_ev.parquet")
    t = df["taxi"].to_numpy()
    keep = t <= v1.MONSTER_S
    n_monster = int((~keep).sum())
    print(f"\nholdout rows {df.height:,}; monster labels (> 5 h): {n_monster}")
    if n_monster != v1.N_MONSTER_EXPECTED:
        raise SystemExit(f"expected {v1.N_MONSTER_EXPECTED} monster rows, got {n_monster}")
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()
    alln = np.ones(df.height, bool)

    control, w_c = cross(df, ["lgb", "catcorr"])
    arm_a, w_a = cross(df, ["lgbcorr20", "catcorr20"])
    arm_b = df["joint20"].to_numpy()
    print(f"stack weights  control {w_c}\n               Arm A   {w_a}")
    for name, info in infos.items():
        print(f"  {name}: {info}")

    def score(a: np.ndarray, b: np.ndarray, mask: np.ndarray, label: str):
        d = df.with_columns(se_b=pl.Series((a - t) ** 2), se_t=pl.Series((b - t) ** 2)).filter(pl.Series(mask))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), 3000, 0)
        print(f"  {label:<28} base={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
              f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    def gate(a: np.ndarray, b: np.ndarray, title: str) -> bool:
        print(f"\nGATE: {title}")
        print(" TRIMMED (decides):")
        d_jul, _ = score(a, b, jul & keep, "Jul trimmed (fit Jan)")
        d_jan, _ = score(a, b, jan & keep, "Jan trimmed (fit Jul)")
        d_pool, p_pool = score(a, b, keep, "POOLED trimmed")
        print(" FULL (guard):")
        score(a, b, jul, "Jul full")
        score(a, b, jan, "Jan full")
        _, p_full = score(a, b, alln, "POOLED full")
        ok = d_pool < 0 and p_pool < 0.05 and d_jan < 0 and d_jul < 0 and p_full < 0.9
        print(f" rule: pooled trimmed delta<0 & P<0.05 [{d_pool:+.2f}, {p_pool:.3f}], both months trimmed<0 "
              f"[Jan {d_jan:+.2f}, Jul {d_jul:+.2f}], full guard P<0.9 [{p_full:.3f}] -> {'PASS' if ok else 'FAIL'}")
        return ok

    a_ok = gate(control, arm_a, "Arm A  NNLS(lgbcorr20, catcorr20) vs control NNLS(lgb, catcorr)")
    b_ok = gate(control, arm_b, "Arm B  joint20 vs control NNLS(lgb, catcorr)")
    b_vs_a = gate(arm_a, arm_b, "Arm B  joint20 vs Arm A")
    if a_ok:
        choice = "Arm B" if (b_ok and b_vs_a) else "Arm A"
    else:
        choice = "Arm B" if b_ok else "none (v21 stays)"
    print(f"\nDECISION (amendment item 5): A vs control {'PASS' if a_ok else 'FAIL'}; "
          f"B vs control {'PASS' if b_ok else 'FAIL'}; B vs A {'PASS' if b_vs_a else 'FAIL'}"
          f"\n  -> candidate for v22: {choice}")

    print("\nDIAGNOSTICS (not decisive)")
    for cols, label in ((["lgbcorr20", "catcorr"], "NNLS(lgbcorr20, catcorr): lgb corrector alone"),
                        (["lgb", "catcorr20"], "NNLS(lgb, catcorr20): 20k cap on cat alone")):
        p, w = cross(df, cols)
        print(f" {label}  weights {w}")
        score(control, p, keep, "pooled trimmed vs control")
        score(control, p, alln, "pooled full vs control")
    print(" single models:")
    score(df["lgb"].to_numpy(), df["lgbcorr20"].to_numpy(), keep, "lgbcorr20 vs lgb, trimmed")
    score(df["catcorr"].to_numpy(), df["catcorr20"].to_numpy(), keep, "catcorr20 vs catcorr, trim")
    print(" no LIRF:")
    no_lirf = keep & (df["ADEP_mvt"] != "LIRF").to_numpy()
    score(control, arm_a, no_lirf, "Arm A, trimmed")
    score(control, arm_b, no_lirf, "Arm B, trimmed")
    d2 = df.with_columns(control=pl.Series(control), arm_a=pl.Series(arm_a), arm_b=pl.Series(arm_b))
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=2, tbl_rows=20):
        agg = lambda c: ((pl.col(c) - pl.col("taxi")) ** 2).mean().sqrt()  # noqa: E731
        for lbl, frame in (("trimmed", d2.filter(pl.Series(keep))), ("full", d2)):
            print(f"per airport, {lbl}:")
            print(frame.group_by("ADEP_mvt").agg(agg("control"), agg("arm_a"), agg("arm_b"), pl.len())
                  .with_columns(dA=pl.col("arm_a") - pl.col("control"), dB=pl.col("arm_b") - pl.col("control"))
                  .sort("ADEP_mvt"))
        print("per decile of true taxi (all rows):")
        print(d2.with_columns(dec=pl.col("taxi").qcut(10, labels=[str(i) for i in range(1, 11)]))
              .group_by("dec").agg(pl.col("taxi").max().alias("taxi_max"), agg("control"), agg("arm_a"),
                                   agg("arm_b"), pl.len())
              .with_columns(dA=pl.col("arm_a") - pl.col("control"), dB=pl.col("arm_b") - pl.col("control"))
              .sort("taxi_max"))


if __name__ == "__main__":
    main()
