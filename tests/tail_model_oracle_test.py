"""Oracle-gated ceiling for a SEPARATE tail model, per review guidance after
tests/label_hi_test.py: raising LABEL_HI broadly regressed EDDF/EHAM/LEBL
(airports with essentially zero extreme rows) because widening the shared
regressor's label window changes split selection and leaf values
everywhere, not just on the tail. A separate tail model keeps the main
regressor untouched at LABEL_HI=7200 (so those airports pay nothing) and
tries to capture the narrow win (rows 1-3-style: -12.7% on the 3 most
extreme LIRF rows) on its own.

The routing problem (detecting a tail row at prediction time) is the same
one that killed the SS32 echo-blend fix -- rows 1-3's signature depends on
BLOCK_TIME_UTC_mvt, exactly the blanked field. So measure the tail model's
VALUE FIRST with an oracle gate (perfect routing, using the true holdout
label to decide who gets the tail model's prediction) before building any
real router. If the oracle ceiling is small, the gate question is moot.

Population: true taxi > LABEL_HI=7200, pooled across all 10 airports (not
LIRF-only -- 584 total 2025 rows, 480 LIRF / 37 LFPG / 37 EGLL / 21 LTFM /
6 LSZH / 3 EHAM; 435 training / 149 holdout). Small enough that the tail
model must be deliberately low-capacity to avoid the overfitting failure
mode already documented once in this project (stage3a_resid.md's residual
heads, num_leaves=63 on ~150k rows/airport -- here we have <450 rows total).

Run:  .venv/Scripts/python.exe tests/tail_model_oracle_test.py
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

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RULE = "=" * 78

TAIL_ETA = 0.05
TAIL_ROUNDS = 2000
TAIL_ES = 30
# _fit_lgb hardcodes num_leaves=255, min_data_in_leaf=100 -- no override param
# exists. With ~350 tail training rows, min_data_in_leaf=100 alone already
# caps the tree at a handful of leaves, which is the regularization this
# population needs; not worth modifying the shared fitter for one test.


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

    # -------------------------------------------------------- shared classifier
    valid_d = ~np.isnan(d_tr)
    clf_tr_mask, clf_va_mask = valid_d & ~iv, valid_d & iv
    is_echo_tr = (np.abs(d_tr) < _ECHO_ABS_D).astype(np.float64)
    clf_model, clf_pred, clf_best = fitter(
        Xtr[clf_tr_mask], is_echo_tr[clf_tr_mask], cats, eta=CLF_ETA,
        rounds=CLF_ROUNDS, es=CLF_EARLY_STOP,
        valid=(Xtr[clf_va_mask], is_echo_tr[clf_va_mask]),
        objective=CLF_OBJECTIVE["lgb"], metric=CLF_METRIC["lgb"], seed=SEED)
    echo_prob_ho = clf_pred(clf_model, Xho[names])
    print(f"[{time.time()-t0:.0f}s] echo classifier fit, best_iter={clf_best}")

    # ------------------------------------------------- baseline (status quo)
    keep = (taxi_tr >= LABEL_LO) & (taxi_tr <= LABEL_HI)
    tr_mask, va_mask = keep & ~iv, keep & iv
    m_base, pf_base, bi_base = fitter(
        Xtr[tr_mask], d_tr[tr_mask], cats, eta=ETA, rounds=ROUNDS, es=EARLY_STOP,
        valid=(Xtr[va_mask], d_tr[va_mask]), objective=objective,
        alpha=HUBER_ALPHA, seed=SEED)
    print(f"[{time.time()-t0:.0f}s] baseline (LABEL_HI={LABEL_HI}) fit, "
          f"best_iter={bi_base}")

    off_ho = f_ho["sched_takeoff_offset"].to_numpy()
    is_lirf_ho = (f_ho["ADEP_mvt"] == "LIRF").to_numpy()
    not_lirf_ho = ~is_lirf_ho
    nm_unmatched_ho = f_ho["aobt3_taxi"].is_null().to_numpy()
    prior_taxi_ho = np.nan_to_num(f_ho["median_taxi_prior"].to_numpy())
    use_prior_ho = nm_unmatched_ho & not_lirf_ho & ~np.isnan(prior_taxi_ho)
    ceil_ho = np.where(not_lirf_ho, CEIL, LIRF_RAW_CEIL)
    true_taxi = ho_lab["taxi"].to_numpy()

    raw_base = pf_base(m_base, Xho[names])
    raw_taxi_base = off_ho - raw_base
    pred_base = _reconstruct_taxi(off_ho, raw_taxi_base, echo_prob_ho, use_prior_ho,
                                  prior_taxi_ho, ceil_ho)
    base_se_all = (pred_base - true_taxi) ** 2
    base_rmse_all = float(base_se_all.mean() ** 0.5)
    print(f"[{time.time()-t0:.0f}s] baseline overall RMSE = {base_rmse_all:.2f}s "
          f"(sanity check vs production)")

    # ------------------------------------------------------------- tail model
    tail_tr_mask = taxi_tr > LABEL_HI
    n_tail = int(tail_tr_mask.sum())
    print(f"\ntail training population: n={n_tail}")
    # small held-out slice for early stopping -- deterministic, not random,
    # so this is reproducible
    rng = np.random.default_rng(SEED)
    tail_idx = np.where(tail_tr_mask)[0]
    rng.shuffle(tail_idx)
    n_val = max(20, int(0.2 * n_tail))
    tail_va_idx, tail_tr_idx = tail_idx[:n_val], tail_idx[n_val:]
    print(f"tail train/valid split: {len(tail_tr_idx)}/{len(tail_va_idx)}")

    m_tail, pf_tail, bi_tail = fitter(
        Xtr.iloc[tail_tr_idx], d_tr[tail_tr_idx], cats, eta=TAIL_ETA, rounds=TAIL_ROUNDS,
        es=TAIL_ES, valid=(Xtr.iloc[tail_va_idx], d_tr[tail_va_idx]), objective=objective,
        alpha=HUBER_ALPHA, seed=SEED)
    print(f"[{time.time()-t0:.0f}s] tail model fit, best_iter={bi_tail}")

    raw_tail = pf_tail(m_tail, Xho[names])
    raw_taxi_tail = off_ho - raw_tail  # flip-style reconstruction, no blend/ceil
    raw_taxi_tail_clipped = np.clip(raw_taxi_tail, 0, 200000)

    # -------------------------------------------------------------- evaluate
    true_tail_mask_ho = true_taxi > LABEL_HI
    n_tail_ho = int(true_tail_mask_ho.sum())
    print(f"\n{RULE}\nOracle-gated evaluation on the {n_tail_ho} true holdout "
          f"tail rows (taxi>{LABEL_HI})\n{RULE}")

    naive_offset = np.clip(off_ho, 0, 200000)

    for label, pred in [
        ("status quo (baseline model + full echo blend)", pred_base),
        ("naive offset-only (predict = sched_takeoff_offset)", naive_offset),
        ("tail model, raw (no blend, clipped [0,200000])", raw_taxi_tail_clipped),
    ]:
        se_tail = (pred[true_tail_mask_ho] - true_taxi[true_tail_mask_ho]) ** 2
        print(f"  {label}: rmse={se_tail.mean()**0.5:9.1f}s  "
              f"(n={true_tail_mask_ho.sum()})")

    # oracle-gated overall RMSE: tail model's prediction ONLY on true-tail
    # holdout rows (using the true label to route -- the "cheat"), baseline
    # prediction everywhere else
    oracle_pred = np.where(true_tail_mask_ho, raw_taxi_tail_clipped, pred_base)
    oracle_se = (oracle_pred - true_taxi) ** 2
    oracle_rmse = float(oracle_se.mean() ** 0.5)
    print(f"\noracle-gated overall RMSE: {oracle_rmse:.2f}s  "
          f"(baseline-only: {base_rmse_all:.2f}s, "
          f"delta={oracle_rmse-base_rmse_all:+.2f}s, "
          f"{(base_rmse_all-oracle_rmse)/base_rmse_all*100:+.2f}%)")

    print("\nper-airport breakdown of the true-tail population:")
    adep_ho = f_ho["ADEP_mvt"].to_numpy()
    for ap in sorted(set(adep_ho[true_tail_mask_ho])):
        m = true_tail_mask_ho & (adep_ho == ap)
        if m.sum() == 0:
            continue
        se_b = ((pred_base[m] - true_taxi[m]) ** 2).mean() ** 0.5
        se_t = ((raw_taxi_tail_clipped[m] - true_taxi[m]) ** 2).mean() ** 0.5
        print(f"  {ap:<6} n={m.sum():>3}  baseline={se_b:9.1f}  tail_model={se_t:9.1f}")

    print(f"\ntotal wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
