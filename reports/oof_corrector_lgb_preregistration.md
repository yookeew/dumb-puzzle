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

### 2026-09-29 — round cap, CatBoost corrector refit, Arm B (before results)

Written while the LightGBM OOF folds were still running. No corrector had
been fit on them and no holdout number existed. Supersedes the sections
above where they conflict.

**1. Round cap 20,000 (was 5,000).** v2 hit its 5,000 cap, so the cap is
raised here instead of in a later separate test. The learning rate stays
0.05, so only one thing changes. Everything else is unchanged: early
stopping on June (a training month, never Jan/Jul 2025) with patience 100,
the 10-month refit at `round(1.1 × best_iter)`, and the 12-month ranking
refit at that same fixed count (no early stopping on any scored month).
- The cap applies to every corrector below. Each reports its best_iter.
  If one still hits 20,000, that is reported, not re-run.

**2. The CatBoost corrector is refit at the same cap (`catcorr20`).** Same
v2 design and inputs, fit on `cache/oof/cat_mixed/`. As in v21, the rerun
CatBoost is used throughout (`cat_mixed_rerun_holdout_ev.parquet`,
`cat_mixed_rank.parquet`).

**3. Arm A (primary):** NNLS(`lgbcorr20`, `catcorr20`), cross-fit. It is
compared with the v21 control NNLS(`lgb`, `catcorr`) under the gate rule
above. Diagnostics (not decisive) separate the parts:
- NNLS(`lgbcorr20`, `catcorr`) vs control: the LightGBM corrector alone;
- NNLS(`lgb`, `catcorr20`) vs control: the cap alone, on CatBoost.

**4. Arm B (secondary): one joint corrector on both engines.**
- Base: `base_B = w_l · lgb + w_c · cat`, with NNLS weights (no
  intercept) fit on the **training-month OOF predictions** of both
  engines. Holdout rows use the same weights on the engines' holdout
  predictions. No holdout data enters the weights.
- Target: `taxi − base_B`. Inputs: base features, both engines' `pred`
  and `echo_prob`, `nmu_op_echo_rate`, `nmu_op_n`.
- Rows: training rows with `taxi >= 0` and `base_B <= 7,200 s`. Applied
  where `base_B <= 7,200 s`, and elsewhere `base_B` is kept unchanged.
- Model, loss, early stopping and cap as in item 1. Output
  `max(0, base_B + r_hat)`. It is used as the final model prediction,
  with no further stack.

**5. Decision (two arms on a partly spent holdout, so a fixed order):**
- **A passes the gate vs control** → A is the candidate.
- **B replaces A** only if B passes the gate vs control **and** passes the
  same rule (pooled trimmed delta < 0 with P(worse) < 0.05, both months
  trimmed < 0, full guard P(worse) < 0.9) **vs A**.
- **If A fails,** B can be the candidate only if it passes vs control.
- **If neither passes,** v21 stays.
- One candidate goes to the leaderboard as v22, built as described above
  (ADS-B stages refit on top). Keep it iff its score < 270.2.

**Inputs this needs on top of the original list:** `cache/oof/cat_mixed/`
(10 folds), `cache/eval/cat_mixed_rerun_holdout_ev.parquet` and
`cache/eval/cat_mixed_rank.parquet`, all from the v21 build.

## Results (2026-09-29) — all arms FAIL; v21 stays

Log: `logs/oof_corrector_lgb_test.log`. Run after the amendment was
committed (`64765e4`).

**Correctors (20k cap).** None reached the cap:
- lgbcorr20: best_iter 4,585
- catcorr20: best_iter 5,378
- joint20: best_iter 516

Arm B's base weights (NNLS on OOF): lgb 0.462 / cat 0.544. Each corrector
was applied to 99.97% of holdout rows.

**Gates (trimmed decides; the full guard is P(worse) < 0.9):**

| comparison | Jul trimmed | Jan trimmed | pooled trimmed (P worse) | pooled full (P worse) | result |
|---|---|---|---|---|---|
| A vs control | +0.44 | -2.16 | -0.56 (0.228) | -0.99 (0.053) | FAIL: pooled not significant, Jul > 0 |
| B vs control | +0.44 | -4.90 | -1.60 (0.000) | -1.26 (0.014) | FAIL: Jul > 0 |
| B vs A | +0.00 | -2.74 | -1.04 (0.005) | -0.27 (0.290) | FAIL: Jul not < 0 |

Decision per item 5: **no candidate; v21 stays.** Per the pre-registration,
no rescue variants.

**Diagnostics (not decisive):**
- **The LightGBM corrector works alone but not in the stack.** lgbcorr20 vs
  lgb is -5.25 trimmed (P 0.000). Inside the stack it's only -0.58
  (P 0.230). NNLS(lgb, catcorr) already captures most of what the
  corrector adds.
- **The cap doesn't matter.** catcorr20 vs catcorr is -0.03; the stack
  with catcorr20 is -0.04. So the "hit the 5,000 cap" question is closed:
  the extra rounds add nothing.
- **All of Arm B's gain is in January.** July is flat in every
  comparison. Arm B improves deciles 1-8 by 4-14 s and loses 7.2 s on the
  top decile. By airport, 8/10 improve (EDDF, EGLL, LEMD and LFPG by
  2-3 s); EDDM is +0.15.
- **The control's stack weights swing between cross-fit directions**
  (lgb 0.515 fit on Jan vs 0.368 fit on Jul), so the stack itself is noisy
  at this level.
