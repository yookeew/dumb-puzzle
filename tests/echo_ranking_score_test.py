"""Score the production echo classifier (fit on all-2025 data, matching the
real submission pipeline exactly) on ranking.parquet, to convert the free
offset-based proxy count (tests/lirf_echo_gate_scoping_test.py part 3) into
a real echo_prob-based count.

Mirrors run()'s all-2025 refit phase in src/models/fit.py, but fits ONLY
the echo classifier (skip both d-regressors -- not needed for this count,
saves most of the wall time).

Run:  .venv/Scripts/python.exe tests/echo_ranking_score_test.py
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
    CLF_ETA,
    CLF_EARLY_STOP,
    CLF_METRIC,
    CLF_OBJECTIVE,
    CLF_ROUNDS,
    ENGINES,
    HOLDOUT_MONTHS,
    VALID_MONTH,
    FEAT_DIR,
    _matrix,
)

ROOT = Path(__file__).resolve().parents[1]
FLT_NULL_COLS = ["CALLSIGN_flt", "AIRCRAFT_TYPE_flt", "MARKET_SEGMENT_flt",
                 "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt"]
RULE = "=" * 78


def main() -> None:
    t0 = time.time()
    print("loading all-2025 features + labels")
    feats = pl.read_parquet(FEAT_DIR / "train2025.parquet")
    lab = pl.read_parquet(FEAT_DIR / "labels2025.parquet")
    lab = lab.join(feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"),
                   on="MVT_ID_mvt", how="left")

    # -- step 1: holdout-only classifier fit, just to get a real best_iter
    # (matches run()'s convention -- don't hardcode a number from an old log)
    is_ho = pl.col("ym").is_in(HOLDOUT_MONTHS)
    tr_lab, ho_lab = lab.filter(~is_ho), lab.filter(is_ho)
    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    f_tr = add_group_encodings_oof(f_tr, tr_lab)
    Xtr, names, cats, categories = _matrix(f_tr)
    d_tr = tr_lab["d"].to_numpy()
    ym_tr = tr_lab["ym"].to_numpy()
    valid_d = ~np.isnan(d_tr)
    iv = ym_tr == VALID_MONTH
    clf_tr_mask, clf_va_mask = valid_d & ~iv, valid_d & iv
    is_echo_tr = (np.abs(d_tr) < _ECHO_ABS_D).astype(np.float64)

    fitter = ENGINES["lgb"]
    _, _, clf_best = fitter(
        Xtr[clf_tr_mask], is_echo_tr[clf_tr_mask], cats, eta=CLF_ETA,
        rounds=CLF_ROUNDS, es=CLF_EARLY_STOP,
        valid=(Xtr[clf_va_mask], is_echo_tr[clf_va_mask]),
        objective=CLF_OBJECTIVE["lgb"], metric=CLF_METRIC["lgb"], seed=42)
    print(f"holdout classifier fit {time.time()-t0:.0f}s  best_iter={clf_best}")

    # -- step 2: all-2025 refit (the classifier that actually scores ranking)
    priors_a = fit_priors(lab)
    f_all = apply_priors(feats, priors_a)
    lab_a = f_all.select("MVT_ID_mvt").join(lab, on="MVT_ID_mvt")
    f_all = add_group_encodings_oof(f_all, lab_a)
    genc_a = fit_group_encodings(lab_a)
    Xall, names_a, cats_a, cats_map = _matrix(f_all)
    d_all = lab_a["d"].to_numpy()
    valid_d_all = ~np.isnan(d_all)
    is_echo_all = (np.abs(d_all) < _ECHO_ABS_D).astype(np.float64)

    clf_full_rounds = max(200, int(round(clf_best * 1.1)))
    clf_model_a, clf_pred_a, _ = fitter(
        Xall[valid_d_all], is_echo_all[valid_d_all], cats_a, eta=CLF_ETA,
        rounds=clf_full_rounds, es=0, valid=None,
        objective=CLF_OBJECTIVE["lgb"], metric=CLF_METRIC["lgb"], seed=42)
    print(f"all-2025 classifier refit done {time.time()-t0:.0f}s  "
          f"rounds={clf_full_rounds}")

    # -- step 3: score ranking
    f_r = apply_priors(pl.read_parquet(FEAT_DIR / "ranking.parquet"), priors_a)
    f_r = apply_group_encodings(f_r, genc_a)
    Xr, _, _, _ = _matrix(f_r, cats_map)
    echo_prob_r = clf_pred_a(clf_model_a, Xr[names_a])
    print(f"ranking scored {time.time()-t0:.0f}s  n={len(echo_prob_r):,}")

    scored = pl.DataFrame({
        "MVT_ID_mvt": f_r["MVT_ID_mvt"],
        "ADEP_mvt": f_r["ADEP_mvt"],
        "echo_prob_r": echo_prob_r,
    })
    scored.write_parquet(ROOT / "cache" / "ranking_echo_prob.parquet")
    print("saved cache/ranking_echo_prob.parquet")

    # -- step 4: join with the null-lane + offset population identified earlier
    rk = pl.scan_parquet(ROOT / "data" / "ranking" / "ranking.parquet").filter(
        pl.col("PHASE_mvt") == "DEP"
    ).select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
              "AOBT_3_flt", *FLT_NULL_COLS).collect()
    rk = rk.with_columns(
        has_aobt3=pl.col("AOBT_3_flt").is_not_null(),
        offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
        n_flt_null=pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in FLT_NULL_COLS]),
    ).join(scored.select("MVT_ID_mvt", "echo_prob_r"), on="MVT_ID_mvt", how="left")

    null_lane = rk.filter(~pl.col("has_aobt3")).filter(
        pl.col("n_flt_null") >= len(FLT_NULL_COLS) - 1)

    print(f"\n{RULE}\nReal echo_prob count on ranking (null lane, n={null_lane.height:,})\n{RULE}")
    print("  echo_prob_r distribution in the null lane:")
    ep = null_lane["echo_prob_r"].drop_nulls().to_numpy()
    print(f"    median={np.median(ep):.3f}  p10={np.percentile(ep,10):.3f}  "
          f"p90={np.percentile(ep,90):.3f}")
    for thr in (0.3, 0.5, 0.7):
        n = (ep > thr).sum()
        print(f"    echo_prob_r > {thr}: n={n} ({n/len(ep)*100:.1f}% of null lane, "
              f"{n/rk.height*100:.3f}% of all ranking DEP)")

    print("\n  by airport (echo_prob_r > 0.5 & null lane):")
    hi = null_lane.filter(pl.col("echo_prob_r") > 0.5)
    print(hi.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True))

    print("\n  cross-tab: echo_prob_r > 0.5 & offset > 3600s (excludes offset>30000 "
          "extreme-scale rows, kept separate):")
    combo = null_lane.filter((pl.col("echo_prob_r") > 0.5) &
                             (pl.col("offset") > 3600) & (pl.col("offset") <= 30000))
    print(f"    n={combo.height:,} ({combo.height/rk.height*100:.3f}% of all ranking DEP)")
    print(combo.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True))


if __name__ == "__main__":
    main()
