"""Build and honestly validate the tail router, following up on the oracle
ceiling in tests/tail_model_oracle_test.py (+3.63% overall, real tail-model
predictions, oracle-gated routing). This test replaces the oracle gate with
a REAL classifier (P(tail), mirroring fit_echo_classifier's structure and
discipline exactly, just retargeted at `is_tail = taxi > LABEL_HI` instead
of `is_echo`), trained only on information available pre-hoc (never the
blanked BLOCK_TIME/taxi).

Free pre-check (already run, not repeated here): `sched_takeoff_offset`
alone has AUC=0.956 for predicting tail membership on the holdout -- far
more separable than anything found in the null-lane echo investigation
(ceiling there was AUC 0.726). Strong prior this router can work.

Design: keep the existing echo blend completely untouched (it stays
exactly as production computes it -- `pred_main`). Add ONE new outer
blend: `final = P(tail)*pred_tail + (1-P(tail))*pred_main`, where
`pred_tail` is the tail-only model's own reconstruction (offset - d_tail,
clipped [0,200000], no echo blend of its own -- mirrors
tail_model_oracle_test.py exactly). This is the realistic, honest version
of the oracle-gated test: same tail model, same main model, but now
P(tail) is a REAL classifier's output, not a peek at the true label.

Run:  .venv/Scripts/python.exe tests/tail_router_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.encode import (  # noqa: E402
    _ECHO_ABS_D,
    add_group_encodings_oof,
    apply_group_encodings,
    apply_priors,
    fit_group_encodings,
    fit_priors,
)
from models.fit import (  # noqa: E402
    CEIL,
    CLF_EARLY_STOP,
    CLF_ETA,
    CLF_METRIC,
    CLF_OBJECTIVE,
    CLF_ROUNDS,
    EARLY_STOP,
    ENGINES,
    ETA,
    HOLDOUT_MONTHS,
    HUBER_ALPHA,
    LABEL_HI,
    LABEL_LO,
    LIRF_RAW_CEIL,
    LOSS_OBJECTIVE,
    ROUNDS,
    VALID_MONTH,
    FEAT_DIR,
    _matrix,
    _reconstruct_taxi,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RULE = "=" * 78
TAIL_ETA = 0.05
TAIL_ROUNDS = 2000
TAIL_ES = 30


def main() -> None:
    t0 = time.time()
    print("loading pooled all-2025 features/labels")
    feats = pl.read_parquet(FEAT_DIR / "train2025.parquet")
    feats_ho = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet")
    lab = pl.read_parquet(FEAT_DIR / "labels2025.parquet")
    lab = lab.join(feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"),
                   on="MVT_ID_mvt", how="left")

    is_ho = pl.col("ym").is_in(HOLDOUT_MONTHS)
    tr_lab, ho_lab = lab.filter(~is_ho), lab.filter(is_ho)

    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    f_ho = apply_priors(feats_ho.join(ho_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    ho_lab = f_ho.select("MVT_ID_mvt").join(ho_lab, on="MVT_ID_mvt")

    f_tr = add_group_encodings_oof(f_tr, tr_lab)
    f_ho = apply_group_encodings(f_ho, fit_group_encodings(tr_lab))

    Xtr, names, cats, categories = _matrix(f_tr)
    Xho, _, _, _ = _matrix(f_ho, categories)
    d_tr = tr_lab["d"].to_numpy()
    taxi_tr = tr_lab["taxi"].to_numpy()
    ym_tr = tr_lab["ym"].to_numpy()
    iv = ym_tr == VALID_MONTH
    fitter = ENGINES["lgb"]
    objective = LOSS_OBJECTIVE["huber"]["lgb"]
    print(f"[{time.time()-t0:.0f}s] priors/encodings ready  "
          f"train n={Xtr.shape[0]:,}  holdout n={Xho.shape[0]:,}")

    # ---------------------------------------------------- echo classifier
    valid_d = ~np.isnan(d_tr)
    clf_tr_mask, clf_va_mask = valid_d & ~iv, valid_d & iv
    is_echo_tr = (np.abs(d_tr) < _ECHO_ABS_D).astype(np.float64)
    echo_model, echo_pred, echo_best = fitter(
        Xtr[clf_tr_mask], is_echo_tr[clf_tr_mask], cats, eta=CLF_ETA,
        rounds=CLF_ROUNDS, es=CLF_EARLY_STOP,
        valid=(Xtr[clf_va_mask], is_echo_tr[clf_va_mask]),
        objective=CLF_OBJECTIVE["lgb"], metric=CLF_METRIC["lgb"], seed=SEED)
    echo_prob_ho = echo_pred(echo_model, Xho[names])
    print(f"[{time.time()-t0:.0f}s] echo classifier fit, best_iter={echo_best}")

    # ---------------------------------------------------- NEW: tail classifier
    # Same discipline as fit_echo_classifier: NOT filtered by the [LABEL_LO,
    # LABEL_HI] `keep` mask (tail rows ARE exactly what that mask excludes,
    # so filtering by it would remove every positive example).
    is_tail_tr = (taxi_tr > LABEL_HI).astype(np.float64)
    tail_clf_model, tail_clf_pred, tail_clf_best = fitter(
        Xtr[clf_tr_mask], is_tail_tr[clf_tr_mask], cats, eta=CLF_ETA,
        rounds=CLF_ROUNDS, es=CLF_EARLY_STOP,
        valid=(Xtr[clf_va_mask], is_tail_tr[clf_va_mask]),
        objective=CLF_OBJECTIVE["lgb"], metric=CLF_METRIC["lgb"], seed=SEED)
    p_tail_ho = tail_clf_pred(tail_clf_model, Xho[names])
    print(f"[{time.time()-t0:.0f}s] tail classifier fit, best_iter={tail_clf_best}  "
          f"base_rate={is_tail_tr[valid_d].mean():.5f}")

    is_tail_ho_true = (ho_lab["taxi"].to_numpy() > LABEL_HI)
    pred_tail_flag = p_tail_ho > 0.5
    tp = int((pred_tail_flag & is_tail_ho_true).sum())
    fp = int((pred_tail_flag & ~is_tail_ho_true).sum())
    fn = int((~pred_tail_flag & is_tail_ho_true).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"tail clf holdout @P>0.5: precision={prec:.3f}  recall={rec:.3f}  "
          f"n_true_tail={int(is_tail_ho_true.sum())}  n_pred_tail={int(pred_tail_flag.sum())}")

    # ------------------------------------------------------------ main model
    keep = (taxi_tr >= LABEL_LO) & (taxi_tr <= LABEL_HI)
    tr_mask, va_mask = keep & ~iv, keep & iv
    m_main, pf_main, bi_main = fitter(
        Xtr[tr_mask], d_tr[tr_mask], cats, eta=ETA, rounds=ROUNDS, es=EARLY_STOP,
        valid=(Xtr[va_mask], d_tr[va_mask]), objective=objective,
        alpha=HUBER_ALPHA, seed=SEED)
    print(f"[{time.time()-t0:.0f}s] main model fit, best_iter={bi_main}")

    off_ho = f_ho["sched_takeoff_offset"].to_numpy()
    is_lirf_ho = (f_ho["ADEP_mvt"] == "LIRF").to_numpy()
    not_lirf_ho = ~is_lirf_ho
    nm_unmatched_ho = f_ho["aobt3_taxi"].is_null().to_numpy()
    prior_taxi_ho = np.nan_to_num(f_ho["median_taxi_prior"].to_numpy())
    use_prior_ho = nm_unmatched_ho & not_lirf_ho & ~np.isnan(prior_taxi_ho)
    ceil_ho = np.where(not_lirf_ho, CEIL, LIRF_RAW_CEIL)
    true_taxi = ho_lab["taxi"].to_numpy()
    adep_ho = f_ho["ADEP_mvt"].to_numpy()
    ym_ho = ho_lab["ym"].to_numpy()

    raw_main = pf_main(m_main, Xho[names])
    raw_taxi_main = off_ho - raw_main
    pred_main = _reconstruct_taxi(off_ho, raw_taxi_main, echo_prob_ho, use_prior_ho,
                                  prior_taxi_ho, ceil_ho)
    base_rmse = float(((pred_main - true_taxi) ** 2).mean() ** 0.5)
    print(f"[{time.time()-t0:.0f}s] main-model-only (no tail router) overall "
          f"RMSE = {base_rmse:.2f}s")

    # ------------------------------------------------------------ tail model
    tail_tr_mask = taxi_tr > LABEL_HI
    n_tail = int(tail_tr_mask.sum())
    rng = np.random.default_rng(SEED)
    tail_idx = np.where(tail_tr_mask)[0]
    rng.shuffle(tail_idx)
    n_val = max(20, int(0.2 * n_tail))
    tail_va_idx, tail_tr_idx = tail_idx[:n_val], tail_idx[n_val:]
    m_tail, pf_tail, bi_tail = fitter(
        Xtr.iloc[tail_tr_idx], d_tr[tail_tr_idx], cats, eta=TAIL_ETA, rounds=TAIL_ROUNDS,
        es=TAIL_ES, valid=(Xtr.iloc[tail_va_idx], d_tr[tail_va_idx]), objective=objective,
        alpha=HUBER_ALPHA, seed=SEED)
    print(f"[{time.time()-t0:.0f}s] tail model fit, best_iter={bi_tail} "
          f"(n_train={len(tail_tr_idx)})")

    raw_tail = pf_tail(m_tail, Xho[names])
    pred_tail = np.clip(off_ho - raw_tail, 0, 200000)

    # ------------------------------------------------------ REAL blend, no cheat
    final_pred = p_tail_ho * pred_tail + (1 - p_tail_ho) * pred_main
    final_rmse = float(((final_pred - true_taxi) ** 2).mean() ** 0.5)

    pl.DataFrame({
        "MVT_ID_mvt": f_ho["MVT_ID_mvt"], "ADEP_mvt": adep_ho, "ym": ym_ho,
        "true_taxi": true_taxi, "pred_main": pred_main, "pred_tail": pred_tail,
        "p_tail": p_tail_ho, "echo_prob": echo_prob_ho,
    }).write_parquet(ROOT / "cache" / "tail_router_ev.parquet")
    print(f"[{time.time()-t0:.0f}s] saved cache/tail_router_ev.parquet for "
          f"threshold sweeps without retraining")

    print(f"\n{RULE}\nHonest (non-oracle) router result\n{RULE}")
    print(f"  main-model-only:        {base_rmse:.2f}s")
    print(f"  main + REAL P(tail) router: {final_rmse:.2f}s  "
          f"(delta={final_rmse-base_rmse:+.2f}s, "
          f"{(base_rmse-final_rmse)/base_rmse*100:+.2f}%)")
    print(f"  (for reference, tests/tail_model_oracle_test.py's oracle-gated "
          f"ceiling was -12.44s/+3.63% from 342.19s)")

    print("\nper-airport:")
    for ap in sorted(set(adep_ho)):
        m = adep_ho == ap
        a = ((pred_main[m] - true_taxi[m]) ** 2).mean() ** 0.5
        b = ((final_pred[m] - true_taxi[m]) ** 2).mean() ** 0.5
        print(f"  {ap:<6} main={a:8.1f}  +router={b:8.1f}  delta={b-a:+7.2f}  n={m.sum():,}")

    print("\nper-month:")
    for ym in HOLDOUT_MONTHS:
        m = ym_ho == ym
        a = ((pred_main[m] - true_taxi[m]) ** 2).mean() ** 0.5
        b = ((final_pred[m] - true_taxi[m]) ** 2).mean() ** 0.5
        print(f"  {ym}: main={a:8.2f}  +router={b:8.2f}  delta={b-a:+7.2f}")

    print(f"\n{RULE}\nCluster bootstrap (airport, day)\n{RULE}")
    day = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = pl.DataFrame({
        "MVT_ID_mvt": f_ho["MVT_ID_mvt"], "ADEP_mvt": adep_ho,
        "se_base": (pred_main - true_taxi) ** 2, "se_treat": (final_pred - true_taxi) ** 2,
    }).join(day, on="MVT_ID_mvt", how="left")
    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("sb"), pl.col("se_treat").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, pw = cluster_bootstrap(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                                         cl["n"].to_numpy(), 3000, 0)
    print(f"paired RMSE delta={point:+.2f}s  95% CI=[{lo:+.2f}, {hi:+.2f}]  "
          f"P(worse)={pw:.3f}")

    # worst-case check: false positives -- rows routed toward pred_tail that
    # aren't actually tail rows, do they get badly hurt?
    routed = p_tail_ho > 0.5
    fp_mask = routed & ~is_tail_ho_true
    if fp_mask.sum():
        a = ((pred_main[fp_mask] - true_taxi[fp_mask]) ** 2).mean() ** 0.5
        b = ((final_pred[fp_mask] - true_taxi[fp_mask]) ** 2).mean() ** 0.5
        print(f"\nfalse-positive rows (routed but not actually tail, n={fp_mask.sum()}): "
              f"main={a:.1f}  +router={b:.1f}  delta={b-a:+.1f}")

    print(f"\ntotal wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
