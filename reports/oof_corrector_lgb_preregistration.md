# OOF residual corrector on LightGBM — pre-registration

Written 2026-09-29, before any LightGBM corrector fit, holdout score or v22
submission existed. Changes made after results exist go under Amendments,
dated. Extends the adopted CatBoost corrector v2
(`reports/oof_corrector_v2_preregistration.md`, PROGRESS.md §49; v21 = 270.2)
to the other half of the lgb + catcorr stack.

**Timing note.** The 10 LightGBM OOF folds (`run_oof(engine="lgb",
target="mixed", seed=42)` → `cache/oof/lgb_mixed/`, log
`logs/oof_lgb_mixed.log`) were started just before this file was written.
They use only the 10 training months (Jan/Jul 2025 never enter a fold, as a
fit or as an early-stopping month), so no holdout number existed when this
was written.

## What is already known (the holdout is partly spent)

The corrector design was tuned on this holdout (v1 → v2, §48–§49), and the
v2 gate passed on CatBoost. This test copies that design to LightGBM
unchanged. **So a holdout pass here is weak evidence; the holdout is a gate
and the leaderboard A/B is the final test.**

## Design: v2 with LightGBM as the base (and nothing else changed)

- **Base model:** the stack's LightGBM, `run(engine='lgb', target='mixed',
  seed=42)`, all other settings default (handoff.md).
  - Holdout baseline: `cache/eval/lgb_mixed_holdout_ev.parquet` (335.6).
  - OOF folds: `cache/oof/lgb_mixed/fold=*.parquet`, same fold scheme as
    CatBoost (fold m fits on the other 9 training months; early stopping
    on June, or May for the June fold).
- **Corrector:** exactly v2 (`tests/oof_corrector_v2_test.py`):
  - Huber loss, alpha 800;
  - training rows: training-month departures with `taxi >= 0` and OOF base
    prediction `<= 7,200 s`; applied only to rows with base prediction
    `<= 7,200 s`, all others unchanged;
  - inputs: base features, `base_pred`, `base_echo_prob` (from the
    LightGBM run), `nmu_op_echo_rate`, `nmu_op_n` (same OOF-by-month
    scheme);
  - LightGBM, `num_leaves=15`, `max_depth=4`, `min_data_in_leaf=2000`,
    `lambda_l2=10`, `learning_rate=0.05`, `feature_fraction=0.8`,
    `max_bin=127`, deterministic, seed 42;
  - 5,000-round cap, early stopping on June with patience 100, then refit
    on all 10 training months for `round(1.1 × best_iter)` rounds;
  - output `max(0, pred + r_hat)`.
- The round cap stays at 5,000 even though v2 hit it. Raising the cap is a
  separate test, so the two effects stay separable.

No other variant is computed (no three-model stack, no corrector on the
stacked prediction).

## Holdout gate (decided on the stack)

- **Control:** the cross-fit NNLS stack of `lgb` + `catcorr`
  (`cache/eval/lgb_mixed_holdout_ev.parquet` +
  `cache/eval/catcorr_mixed_holdout_ev.parquet`, the v21 stack).
- **Treatment:** the cross-fit NNLS stack of `lgbcorr` + `catcorr`.
- Cross-fit as in `tests/stack_r20k_test.py`: weights fit on Jan score Jul,
  and the reverse.
- Trimmed set: holdout labels > 5 h excluded, expected 31 rows (the script
  stops if not).

**Pass iff all of:**
1. Pooled trimmed RMSE delta < 0 with P(worse) < 0.05.
2. Trimmed RMSE point delta < 0 in both Jan and Jul.
3. Full RMSE guard: pooled P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap in
`tests/_harness.py`, 3,000 resamples, seed 0.

**Reported, not decisive:**
- `lgbcorr` alone vs `lgb` alone (the v2-style single-model comparison),
  trimmed and full;
- stack weights in each cross-fit direction;
- per airport (trimmed and full) and per decile of true taxi;
- pooled trimmed delta without LIRF;
- corrector best_iter (does it hit the 5,000 cap?), fraction of rows
  touched, mean correction for NM-unmatched vs matched rows.

**If the gate fails: stop.** No rescue variants. Record the result in
PROGRESS.md as a finding; v21 stays the submission.

## If the gate passes: v22 for the leaderboard

**LightGBM rerun (local).** The corrector needs the ranking rows' `pred`,
`echo_prob`, `taxi_model_raw` and `use_prior`, which `lgb_mixed.parquet`
doesn't hold. Rerun `run(engine='lgb', target='mixed', seed=42,
name='lgb_mixed', submit=True, rank_out=...)`. LightGBM is deterministic,
so the rerun's holdout predictions are compared with
`lgb_mixed_holdout_ev.parquet`:
- **RMS difference ≤ 1 s:** the existing files are used.
- **Otherwise:** the rerun's holdout and ranking outputs are used
  throughout, the corrector is re-applied to the rerun's holdout
  predictions, and the difference is reported. No re-gating.

**Ranking corrector.** Refit on all 12 months (training-month OOF rows plus
the holdout rows with LightGBM's 10-month-fit predictions), inputs exactly
as in the gate, same row filter and parameters, a fixed round count equal
to the gate's 10-month refit.

**Stack and ADS-B stages.** `lgbcorr` + `catcorr` NNLS stack on the full
holdout, then the ADS-B blend and partial-track stages refit on top, as in
v21 (`src/post/stack_submit.py`). The v21 `catcorr` files are reused
unchanged.

**Checks before upload:** ids and order match the template, no nulls, no
negative values; report the RMS difference from v21 overall and per airport.

**Leaderboard rule: keep v22 iff its score < 270.2** (v21). Otherwise v21
stays.

## Amendments

(none)
