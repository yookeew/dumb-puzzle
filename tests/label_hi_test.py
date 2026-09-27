"""Test LABEL_HI itself -- the training-label truncation at 7200s that the
LIRF investigation (reports/lirf_investigation.md SS4-SS10) treated as a
structural ceiling. It isn't: it's a hyperparameter, last widened
(reports/stage3a_resid.md, 2026-09-02) under plain L2 with no Huber, no
echo classifier, no CEIL fix, no flip/direct split -- an architecture that
no longer exists. Huber's gradient is capped at `alpha` for any residual
beyond it, unlike L2's linear scaling, so the original failure mode
(extreme-label rows dominating shared tree splits) may not recur.

Design: fit `target="flip"` POOLED across all 10 airports (not LIRF-only --
the question is whether raising LABEL_HI helps LIRF's `d`-supervision
without hurting the other 9, exactly what regressed under L2 last time).
Priors, group encodings, and the echo classifier don't depend on LABEL_HI
at all (confirmed: `fit_priors` filters taxi.is_between(60,5400) for its
own unimpeded-percentile calc; the echo classifier is explicitly NOT
filtered by the `keep` mask at all -- see fit.py's fit_echo_classifier
docstring). So both are fit ONCE and shared between arms -- isolates the
comparison to exactly the one line in question, `keep = taxi.is_between(
LABEL_LO, LABEL_HI)` for the `d`-regressor's training mask.

Run:  .venv/Scripts/python.exe tests/label_hi_test.py
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
BASELINE_HI = 7200
RAISED_HI = 200000   # above the true observed max (131,167s) -- effectively unbounded
SEED = 42
RULE = "=" * 78


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

    print(f"pooled train rows={Xtr.shape[0]:,}  holdout rows={Xho.shape[0]:,}")
    print(f"[{time.time()-t0:.0f}s] priors/encodings ready")

    # --- echo classifier: NOT filtered by taxi keep, identical either way ---
    valid_d = ~np.isnan(d_tr)
    clf_tr_mask, clf_va_mask = valid_d & ~iv, valid_d & iv
    fitter = ENGINES["lgb"]
    is_echo_tr = (np.abs(d_tr) < _ECHO_ABS_D).astype(np.float64)
    clf_model, clf_pred, clf_best = fitter(
        Xtr[clf_tr_mask], is_echo_tr[clf_tr_mask], cats, eta=CLF_ETA,
        rounds=CLF_ROUNDS, es=CLF_EARLY_STOP,
        valid=(Xtr[clf_va_mask], is_echo_tr[clf_va_mask]),
        objective=CLF_OBJECTIVE["lgb"], metric=CLF_METRIC["lgb"], seed=SEED)
    echo_prob_ho = clf_pred(clf_model, Xho[names])
    print(f"[{time.time()-t0:.0f}s] echo classifier fit (shared), best_iter={clf_best}")

    off_ho = f_ho["sched_takeoff_offset"].to_numpy()
    is_lirf_ho = (f_ho["ADEP_mvt"] == "LIRF").to_numpy()
    not_lirf_ho = ~is_lirf_ho
    nm_unmatched_ho = f_ho["aobt3_taxi"].is_null().to_numpy()
    prior_taxi_ho = np.nan_to_num(f_ho["median_taxi_prior"].to_numpy())
    use_prior_ho = nm_unmatched_ho & not_lirf_ho & ~np.isnan(prior_taxi_ho)
    ceil_ho = np.where(not_lirf_ho, CEIL, LIRF_RAW_CEIL)
    true_taxi = ho_lab["taxi"].to_numpy()
    ym_ho = ho_lab["ym"].to_numpy()
    adep_ho = f_ho["ADEP_mvt"].to_numpy()

    objective = LOSS_OBJECTIVE["huber"]["lgb"]
    results = {}
    for label, hi in [("baseline (LABEL_HI=7200)", BASELINE_HI),
                      ("raised (LABEL_HI=200000)", RAISED_HI)]:
        keep = (taxi_tr >= LABEL_LO) & (taxi_tr <= hi)
        tr_mask, va_mask = keep & ~iv, keep & iv
        print(f"\n[{time.time()-t0:.0f}s] fitting {label}  "
              f"n_train={tr_mask.sum():,} n_valid={va_mask.sum():,}")
        m, pf, bi = fitter(
            Xtr[tr_mask], d_tr[tr_mask], cats, eta=ETA, rounds=ROUNDS, es=EARLY_STOP,
            valid=(Xtr[va_mask], d_tr[va_mask]), objective=objective,
            alpha=HUBER_ALPHA, seed=SEED)
        print(f"[{time.time()-t0:.0f}s] {label} done  best_iter={bi}")

        raw = pf(m, Xho[names])
        raw_taxi = off_ho - raw  # flip reconstruction, pre-blend
        recon = _reconstruct_taxi(off_ho, raw_taxi, echo_prob_ho, use_prior_ho,
                                  prior_taxi_ho, ceil_ho)
        se = (recon - true_taxi) ** 2
        results[label] = dict(pred=recon, raw_taxi=raw_taxi, se=se, best_iter=bi)

    print(f"\n{RULE}\nOverall + per-airport + per-month\n{RULE}")
    keys = list(results.keys())
    for label in keys:
        se = results[label]["se"]
        overall = float(se.mean() ** 0.5)
        print(f"\n{label}  (best_iter={results[label]['best_iter']})")
        print(f"  overall RMSE = {overall:.2f}s")
        for ap in sorted(set(adep_ho)):
            m = adep_ho == ap
            print(f"    {ap:<6} rmse={se[m].mean()**0.5:7.1f}  n={m.sum():,}")
        for ym in HOLDOUT_MONTHS:
            m = ym_ho == ym
            print(f"    {ym}: rmse={se[m].mean()**0.5:7.1f}")

    print(f"\n{RULE}\nPaired comparison (cluster bootstrap, (airport,day))\n{RULE}")
    day = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = pl.DataFrame({
        "MVT_ID_mvt": f_ho["MVT_ID_mvt"], "ADEP_mvt": adep_ho, "ym": ym_ho,
        "se_base": results[keys[0]]["se"], "se_treat": results[keys[1]]["se"],
    }).join(day, on="MVT_ID_mvt", how="left")

    rb, rt = ev["se_base"].mean() ** 0.5, ev["se_treat"].mean() ** 0.5
    print(f"overall: baseline={rb:.2f}  raised={rt:.2f}  delta={rt-rb:+.2f}")
    for ym in HOLDOUT_MONTHS:
        s = ev.filter(pl.col("ym") == ym)
        a, b = s["se_base"].mean() ** 0.5, s["se_treat"].mean() ** 0.5
        print(f"  {ym}: baseline={a:.2f}  raised={b:.2f}  delta={b-a:+.2f}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("sb"), pl.col("se_treat").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi_ci, pw = cluster_bootstrap(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                                             cl["n"].to_numpy(), 3000, 0)
    print(f"\ncluster bootstrap: paired RMSE delta={point:+.2f}s  "
          f"95% CI=[{lo:+.2f}, {hi_ci:+.2f}]  P(worse)={pw:.3f}")

    lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF")
    a, b = lirf["se_base"].mean() ** 0.5, lirf["se_treat"].mean() ** 0.5
    print(f"\nLIRF only: baseline={a:.2f}  raised={b:.2f}  delta={b-a:+.2f}")

    # rows 1-3-style extreme rows specifically: did raising LABEL_HI help them?
    extreme = lirf.filter(pl.col("se_base") > 5e8)  # matches the top-3 rows' SE scale
    if extreme.height:
        a2, b2 = extreme["se_base"].mean() ** 0.5, extreme["se_treat"].mean() ** 0.5
        print(f"LIRF extreme rows (n={extreme.height}, matches rows-1-3 scale): "
              f"baseline={a2:.2f}  raised={b2:.2f}  delta={b2-a2:+.2f}")


if __name__ == "__main__":
    main()
