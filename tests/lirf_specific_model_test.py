"""Q3: is a LIRF-specific model better than the pooled model, post-fixes?

The original attempt (reports/stage3a_resid.md, 2026-09-02) failed, but for
two now-understood, now-fixed reasons: a widened label window fed into the
*shared* pooled model (not this experiment's design at all), and residual
heads that overfit a crude 2-fold OOF. Neither applies to a from-scratch
LIRF-only model using the current pipeline (Huber, current feature set,
current priors/group-encoding machinery). This has not been re-tested since
the CEIL fix (PROGRESS.md SS12) and echo classifier landed.

Design: fit a `flip`-target `d`-regressor on LIRF-only 2025 training rows
(own priors, own OOF group-encodings -- a genuinely separate model, not a
filtered slice of the pooled one), evaluate on LIRF-only holdout rows.
To isolate specifically "does the base regressor benefit from pooling or
specialization" (not confound it with a fresh echo-classifier fit on much
less data, or a fresh choice of blend logic), reuse the ALREADY-VALIDATED
pooled model's echo_prob / sched_takeoff_offset from
`cache/lirf_ceilfix_mixed_ev.parquet` (production v12 -- current production
is ~7.6s lower overall due to weather+minute_of_day fixes since, but LIRF
itself is untouched by both: S26 weather LIRF delta -4.9s general, minfix
LIRF delta -2.5s, both from features unrelated to this comparison's design;
noted as a caveat, not expected to change the direction of the result) --
same _reconstruct_taxi call, same ceiling (LIRF_RAW_CEIL), only the base
regressor's raw prediction changes between the two arms. `use_prior` is
always False for ADEP_mvt=="LIRF" by construction (see fit.py), so
prior_taxi never enters this comparison either way.

Run:  .venv/Scripts/python.exe tests/lirf_specific_model_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.encode import (  # noqa: E402
    add_group_encodings_oof,
    apply_group_encodings,
    apply_priors,
    fit_group_encodings,
    fit_priors,
)
from models.fit import (  # noqa: E402
    CEIL,
    ENGINES,
    HOLDOUT_MONTHS,
    LABEL_LO,
    LABEL_HI,
    LIRF_RAW_CEIL,
    LOSS_OBJECTIVE,
    ROUNDS,
    ETA,
    EARLY_STOP,
    VALID_MONTH,
    FEAT_DIR,
    _matrix,
    _reconstruct_taxi,
    _rmse,
)

ROOT = Path(__file__).resolve().parents[1]
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"
SEED = 42
RULE = "=" * 78


def main() -> None:
    t0 = time.time()
    print("loading cached features (LIRF-only slice)")
    feats = pl.read_parquet(FEAT_DIR / "train2025.parquet").filter(pl.col("ADEP_mvt") == "LIRF")
    feats_ho = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").filter(pl.col("ADEP_mvt") == "LIRF")
    lab = pl.read_parquet(FEAT_DIR / "labels2025.parquet").filter(pl.col("ADEP_mvt") == "LIRF")
    lab = lab.join(feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"), on="MVT_ID_mvt", how="left")

    is_ho = pl.col("ym").is_in(HOLDOUT_MONTHS)
    tr_lab, ho_lab = lab.filter(~is_ho), lab.filter(is_ho)
    print(f"LIRF-only train rows={tr_lab.height:,}  holdout rows={ho_lab.height:,}")

    # LIRF-only priors and group encodings -- a genuinely separate model, not
    # a filtered slice of the pooled one's fitted parameters.
    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    f_ho = apply_priors(feats_ho.join(ho_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    ho_lab = f_ho.select("MVT_ID_mvt").join(ho_lab, on="MVT_ID_mvt")

    f_tr = add_group_encodings_oof(f_tr, tr_lab)
    f_ho = apply_group_encodings(f_ho, fit_group_encodings(tr_lab))

    Xtr, names, cats, categories = _matrix(f_tr)
    d_tr = tr_lab["d"].to_numpy()
    taxi_tr = tr_lab["taxi"].to_numpy()
    ym_tr = tr_lab["ym"].to_numpy()
    keep = (taxi_tr >= LABEL_LO) & (taxi_tr <= LABEL_HI)
    iv = ym_tr == VALID_MONTH
    tr_mask, va_mask = keep & ~iv, keep & iv
    print(f"train_mask n={tr_mask.sum():,}  valid_mask n={va_mask.sum():,}")

    fitter = ENGINES["lgb"]
    m, pf, best_iter = fitter(
        Xtr[tr_mask], d_tr[tr_mask], cats, eta=ETA, rounds=ROUNDS, es=EARLY_STOP,
        valid=(Xtr[va_mask], d_tr[va_mask]), objective=LOSS_OBJECTIVE["huber"]["lgb"],
        alpha=800.0, seed=SEED)
    print(f"LIRF-only flip fit done {time.time()-t0:.0f}s  best_iter={best_iter}")

    Xho, _, _, _ = _matrix(f_ho, categories)
    off_ho = f_ho["sched_takeoff_offset"].to_numpy()
    raw_d = pf(m, Xho[names])
    raw_taxi_lirf_only = off_ho - raw_d  # flip reconstruction, pre-blend

    ev_lirf_only = pl.DataFrame({
        "MVT_ID_mvt": f_ho["MVT_ID_mvt"], "taxi": ho_lab.join(
            f_ho.select("MVT_ID_mvt"), on="MVT_ID_mvt")["taxi"],
        "raw_taxi_lirf_only": raw_taxi_lirf_only,
    })

    print("\nloading pooled production ev (cache/lirf_ceilfix_mixed_ev.parquet) "
          "for shared echo_prob / offset / use_prior")
    prod = pl.read_parquet(PROD_EV).filter(pl.col("ADEP_mvt") == "LIRF").select(
        "MVT_ID_mvt", "taxi", "sched_takeoff_offset", "echo_prob", "taxi_model_raw",
        pred_pooled="pred")

    joined = prod.join(ev_lirf_only.select("MVT_ID_mvt", "raw_taxi_lirf_only"),
                       on="MVT_ID_mvt", how="inner")
    print(f"matched rows between pooled ev and LIRF-only holdout: {joined.height:,} "
          f"(pooled ev has {prod.height:,}, LIRF-only holdout has {ev_lirf_only.height:,})")

    taxi = joined["taxi"].to_numpy()
    offset = joined["sched_takeoff_offset"].to_numpy()
    echo_prob = joined["echo_prob"].to_numpy()
    zeros_bool = np.zeros(len(taxi), dtype=bool)
    zeros_f = np.zeros(len(taxi))
    ceil_arr = np.full(len(taxi), LIRF_RAW_CEIL)

    recon_pooled = _reconstruct_taxi(offset, joined["taxi_model_raw"].to_numpy(),
                                     echo_prob, zeros_bool, zeros_f, ceil_arr)
    recon_lirf_only = _reconstruct_taxi(offset, joined["raw_taxi_lirf_only"].to_numpy(),
                                        echo_prob, zeros_bool, zeros_f, ceil_arr)

    rmse_pooled_cached = _rmse(joined["pred_pooled"].to_numpy(), taxi)
    rmse_pooled_recon = _rmse(recon_pooled, taxi)
    rmse_lirf_only = _rmse(recon_lirf_only, taxi)

    print(f"\n{RULE}\nQ3 result: LIRF-only regressor vs pooled regressor, "
          f"SAME echo blend / offset / ceiling\n{RULE}")
    print(f"  pooled model, cached production 'pred' column   : {rmse_pooled_cached:7.1f}s "
          f"(sanity check -- should equal the recon below)")
    print(f"  pooled regressor + shared blend (recomputed)     : {rmse_pooled_recon:7.1f}s")
    print(f"  LIRF-only regressor + SAME shared blend          : {rmse_lirf_only:7.1f}s")
    print(f"  delta (LIRF-only - pooled)                       : {rmse_lirf_only - rmse_pooled_recon:+7.1f}s "
          f"({(rmse_lirf_only - rmse_pooled_recon) / rmse_pooled_recon * 100:+.1f}%)")
    print(f"\n  caveat: pooled arm is production v12 (2026-09-17), current production "
          f"is ~7.6s lower overall from weather+minute_of_day fixes since -- LIRF's own "
          f"share of that is small (~-7.4s per PROGRESS.md SS26/SS28) and unrelated to "
          f"pooling vs specialization, the question this test isolates.")

    # also report raw (pre-blend) regressor comparison -- isolates the pure
    # `d`-model fit quality, before the echo/ceiling logic touches anything
    rmse_raw_pooled = _rmse(joined["taxi_model_raw"].to_numpy(), taxi)
    rmse_raw_lirf_only = _rmse(joined["raw_taxi_lirf_only"].to_numpy(), taxi)
    print(f"\n  pre-blend raw regressor only:")
    print(f"    pooled   raw RMSE = {rmse_raw_pooled:7.1f}s")
    print(f"    LIRF-only raw RMSE = {rmse_raw_lirf_only:7.1f}s")


if __name__ == "__main__":
    main()
