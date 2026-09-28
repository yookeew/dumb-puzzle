# OOF residual corrector v2 (CatBoost) — pre-registration

Written and committed 2026-09-28, before any v2 corrector fit, v2 holdout
score or v21 submission existed. Changes made after results exist go under
Amendments, dated. Follows v1
(`reports/oof_corrector_preregistration.md`, PROGRESS.md §48), which was
rejected.

## What is already known (the holdout is partly spent)

v1 was scored on the same Jan + Jul 2025 holdout. We have already seen
that:
- the L2 corrector trained on labels in 30–7,200 s helped 9 airports and
  hurt LIRF and the top decile;
- its Huber diagnostic scored pooled trimmed -3.10, full -1.13.

v2 was designed after seeing those numbers. So a v2 holdout pass is weaker
evidence than a fresh pre-registered pass. **The holdout is a gate; the
leaderboard A/B is the final test.**

## Changes from v1 (and nothing else)

1. **Loss:** Huber, alpha 800 (the base models' default, §3).
2. **Row selection on the base prediction, not the label.**
   - Training rows: training-month departures with a valid label
     (`taxi >= 0`, the evaluation's own filter) and OOF base prediction
     `pred_oof <= 7,200 s`. No filter on the true taxi beyond validity.
   - Application: the corrector is applied only to rows with base
     `pred <= 7,200 s`. Other rows keep the base prediction unchanged.
   - Why: v1's filter on the true label removed exactly the rows where a
     large base prediction was right. That plausibly taught the corrector
     to pull every large prediction down.
3. **New input: carrier echo rate among NM-unmatched rows** (Task 2,
   PROGRESS.md §46 item 2).
   - Definition: per (`ADEP_mvt`, `AIRCRAFT_OPERATOR_flt`), the share of
     echo labels (`|d| < 30 s`, `_ECHO_ABS_D`) among labelled NM-unmatched
     departures (`aobt3_taxi` null). Smoothed toward the airport's
     NM-unmatched echo rate with pseudo-count 30 (`_ENC_SMOOTH`). A count
     column comes with it. Unseen or null operators get the airport rate
     (and n = 0).
   - Columns: `nmu_op_echo_rate`, `nmu_op_n`, given on every row (matched
     rows too; `aobt3_taxi` already tells the corrector which rows are
     unmatched).
   - Out of fold: a training row in month m gets the rate fit on the other
     training months (never Jan/Jul 2025). Holdout rows get the rate fit on
     the 10 training months. For the ranking refit (below), every labelled
     row gets the rate from the other 11 months of 2025, and ranking rows get
     the rate fit on all 12.
4. **Round cap raised to 5,000** (v1 hit its 2,000 cap: best_iter 2000 /
   1996). Early stopping still uses patience 100.

Everything else is exactly v1:
- the OOF folds in `cache/oof/cat_mixed/`;
- the base feature inputs built as `run()` builds them;
- `base_pred` and `base_echo_prob` as inputs;
- LightGBM with `num_leaves=15`, `max_depth=4`, `min_data_in_leaf=2000`,
  `lambda_l2=10`, `learning_rate=0.05`, `feature_fraction=0.8`,
  `max_bin=127`, deterministic, seed 42;
- early stopping on June (fit on the other 9 months), then a refit on all
  10 training months for `round(1.1 × best_iter)` rounds;
- output `max(0, pred + r_hat)` on the rows it applies to.

No other variant is computed.

## Holdout gate

- Baseline: `cache/eval/cat_mixed_holdout_ev.parquet` (uncorrected
  CatBoost).
- Treatment: v2 applied to it.
- Trimmed set: labels > 5 h, expected 31 rows (the script stops if not).

**Pass iff all of:**
1. Pooled trimmed RMSE delta < 0 with P(worse) < 0.05.
2. Trimmed RMSE point delta < 0 in both Jan and Jul.
3. Full RMSE guard: pooled P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap,
`tests/_harness.py`, 3,000 resamples, seed 0.

**Reported, not decisive:**
- per airport (trimmed and full) and per decile;
- pooled trimmed delta without LIRF;
- per-month full RMSE;
- the fraction of holdout rows the corrector touches;
- mean correction for NM-unmatched vs matched rows.

**If the gate fails: stop.** No rescue variants. Record the result in
PROGRESS.md as a finding.

## If the gate passes: v21 for the leaderboard

**CatBoost rerun (Colab).** `run(engine="cat", target="mixed", eta=0.05,
submit=True)`, also saving the ranking rows' `pred`, `echo_prob`,
`taxi_model_raw` and `use_prior`. The rerun's holdout predictions are
compared with the existing `cat_mixed_holdout_ev.parquet`:
- **RMS difference ≤ 1 s:** the existing holdout file is used for
  everything below.
- **Otherwise:** the rerun's holdout and ranking outputs are used
  throughout, the corrector is re-applied to the rerun's holdout
  predictions, and the difference is reported. No re-gating.

**Ranking corrector.**
- Refit on all 12 months: the training-month OOF rows plus the holdout
  rows (CatBoost's 10-month-fit holdout predictions), with inputs exactly
  as in the gate.
- Same row filter and parameters. A fixed round count equal to the gate's
  10-month refit (no early stopping).
- Applied to the rerun's ranking predictions. Ranking inputs: priors and
  group encodings fit on all 12 months (as `run()`'s refit builds them),
  `nmu_op_echo_rate` fit on all 12 months, and the rerun's `pred` and
  `echo_prob`.

**Integration.** Corrected CatBoost replaces CatBoost in
`src/post/stack_submit.py`; LightGBM is unchanged.
- The NNLS stack weights are refit on the full holdout, using the lgb and
  corrected-cat holdout predictions.
- The ADS-B blend, quality factors and partial-track stage are refit on
  that stacked holdout prediction exactly as for v19.

Output: `stack_lgb_catcorr_adsbq_partial.parquet`, submitted as
`smart-jigsaw_v21`.

**Leaderboard rule.** Keep v21 iff its score is lower than v19's 274.25.
Otherwise keep v19 and record v21 as a finding. Row-level differences
between v21 and v19 (count changed, mean |shift|, per airport) are reported
before upload.

## Amendments

(none)

## Gate result (2026-09-28, `tests/oof_corrector_v2_test.py`, log `logs/oof_corrector_v2_test.log`) — PASS

| | base | corrected | delta | CI | P(worse) |
|---|---|---|---|---|---|
| Jul trimmed | 299.27 | 296.13 | -3.14 | [-4.54, -1.68] | 0.000 |
| Jan trimmed | 238.45 | 234.97 | -3.48 | [-5.86, -1.30] | 0.000 |
| **pooled trimmed** | 273.80 | 270.55 | **-3.25** | [-4.52, -2.00] | 0.000 |
| pooled full (guard) | 337.48 | 334.72 | -2.76 | [-3.91, -1.77] | 0.000 |

All three clauses hold, so the gate passes and v21 gets built.

- **Every airport improves**, on both trimmed and full RMSE: LFPG -6.8,
  EHAM -6.2, EDDM -5.5, LSZH -3.8, LIRF -3.6 (trimmed) / -3.1 (full), EGLL
  -3.2, LEMD -3.0.
- **Deciles 1-9** improve by 2.5-10.3 s; the top decile is +2.8 (v1:
  +49).
- The corrector touches 99.97% of rows. Mean correction is +69.8 s on
  NM-unmatched rows (mean |.| 190 s, n = 5,323) and -4.5 s on matched rows
  (mean |.| 30 s).
- The corrector reached the 5,000-round cap (best_iter 5000, refit 5,500
  rounds). The cap was fixed in advance, so it stays.
- Caveat, as stated up front: v2 was designed after seeing v1 on this
  holdout. The leaderboard A/B (v21) is the final test.
