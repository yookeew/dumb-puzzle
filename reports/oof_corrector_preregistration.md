# Out-of-fold (OOF) residual corrector on CatBoost — pre-registration

Written and committed 2026-09-28, before any OOF prediction, corrector fit
or corrected holdout score existed. Changes made after results exist go
under Amendments, dated. Motivation: PROGRESS.md §46 item 1 (the competitors'
chat: "a corrector trained on out-of-fold base predictions, folds by month,
did more than new features"). Decision metric follows the §47 convention.

## Question

Does a second-stage model, trained on month-wise OOF CatBoost predictions to
predict the residual `taxi - pred`, lower CatBoost's holdout RMSE when it is
applied to CatBoost's ordinary holdout predictions?

This test covers **CatBoost alone**. How a corrector would be integrated
with the lgb+cat stack and the ADS-B stages is a separate pre-registration,
written only if this one adopts.

## Base model (unchanged from production CatBoost)

`engine="cat"`, `target="mixed"`, Huber (800), `eta=0.05`, `rounds=8000`
cap, early stopping 150, seed 42, `max_ctr_complexity=1`, the engine's own
echo classifier, `_reconstruct_taxi` with the echo blend, the `use_prior`
fallback and the LIRF raw ceiling. The per-row output is the same `pred`
that `run()` writes.

## Step 1 — refactor, with an equivalence check

A new function in `src/models/fit.py` trains on an arbitrary set of labelled
months and predicts an arbitrary set of months, returning `run()`'s per-row
reconstruction (`pred`, `echo_prob`, `taxi_model_raw`, `use_prior`). `run()`
keeps its behaviour.

**Equivalence check (local, before any Colab run):** with `engine="lgb"`
and a small round cap (100), the new function with train = the 10 training
months, valid = June, predict = Jan + Jul from `holdout_gap2025.parquet`
must reproduce `run(engine="lgb", rounds=100, target="mixed",
submit=False)`'s holdout `pred` to within 1e-6 s on every row (joined on
`MVT_ID_mvt`). If it doesn't, the refactor is fixed before anything else
runs.

## Step 2 — OOF folds

Training months M = Feb–Jun, Aug–Dec 2025. For each held-out month m in M,
one fold:

- **Fit months:** M \ {m} (9 months). Jan and Jul 2025 never enter any
  fold's fit, priors, encodings or early stopping.
- **Priors:** `fit_priors` on the fit months' labels only.
- **Group encodings:** leave-one-month-out (`add_group_encodings_oof`)
  across the 9 fit months for the fit rows; `fit_group_encodings` on the 9
  fit months, applied to month m.
- **Early stopping** (both regressors and the echo classifier): June,
  carved out of the fit months; May when m = June.
- **Predicted rows:** month m's departures, features from
  `train2025.parquet` (the continuous frame; see Known mismatches).
- **Asserts in code:** the `ym` set of every label frame passed to
  `fit_priors`, the encoders and the fitter equals the fold's fit months;
  m is not in it; neither holdout month is in it.
- **Output**, written as each fold finishes (Colab sessions drop):
  `cache/oof/cat_mixed/fold=<m>.parquet` with `MVT_ID_mvt`, `ym`, `pred`,
  `echo_prob`, `taxi_model_raw`, `use_prior`, and the fold's best
  iterations. The run skips folds whose file already exists.

Expected cost: 10 folds × 3 fits (flip, direct, echo classifier) at ~6
min per fit on a T4, about 3 hours.

**Fold QA (reported before the corrector is trained, not decisive):** per
fold, trimmed and full RMSE of `pred` on month m, row count, and best
iterations. All 10 folds must be present and cover every labelled training
row exactly once. Holdout CatBoost's RMSE is shown alongside for scale.

## Step 3 — the corrector

- **Rows:** training-month departures with an OOF prediction and
  `LABEL_LO <= taxi <= LABEL_HI` (30–7,200 s, the base models' label
  range; keeps monster labels out of an L2 fit).
- **Target:** `r = taxi - pred_oof`.
- **Inputs:** the base feature matrix exactly as `run()` builds it for the
  training rows (priors fit on the 10 training months, OOF-by-month group
  encodings, same categoricals), plus `pred_oof` and `echo_prob_oof`.
- **Model:** LightGBM, L2 objective, `num_leaves=15`, `max_depth=4`,
  `min_data_in_leaf=2000`, `lambda_l2=10`, `learning_rate=0.05`,
  `feature_fraction=0.8`, `max_bin=127`, cap 2,000 rounds, early stopping
  100 on June (fit on the other 9 training months), `deterministic=true`,
  `force_row_wise=true`, seed 42. Then refit on all 10 training months for
  `round(1.1 × best_iter)` rounds, the base models' convention.
- **Application:** `pred_corr = max(0, pred + r_hat)`, on every row,
  including rows outside the label range.

No other corrector variants are decisive. One diagnostic variant is
reported: the same model with Huber loss (alpha 800).

## Step 4 — holdout evaluation

- **Baseline:** CatBoost's existing holdout predictions
  (`cache/eval/cat_mixed_holdout_ev.parquet`, fit on the 10 training
  months).
- **Treatment:** the corrector (fit only on training-month OOF rows)
  applied to those same predictions, with its inputs built from the
  holdout features exactly as `run()` builds them (`holdout_gap2025`,
  priors and encodings fit on the 10 training months) plus that file's
  `pred` and `echo_prob`.
- The corrector never sees a holdout label, so no Jan↔Jul cross-fit is
  needed; each month is scored directly.
- **Trimmed set:** holdout rows with label > 5 h (18,000 s), after the
  file's `taxi >= 0` filter. Expected 31 rows; the script stops if the
  count differs.

## Rule

Adopt the corrector iff all of:

1. **Trimmed RMSE, pooled:** delta < 0 with P(worse) < 0.05.
2. **Trimmed RMSE, per month:** point delta < 0 in both Jan and Jul.
3. **Full RMSE guard:** pooled P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap in
`tests/_harness.py` (3,000 resamples, seed 0).

## Reported, not decisive

- Trimmed and full RMSE per month and per airport, and per decile of true
  taxi.
- The pooled trimmed delta with LIRF excluded.
- Share of the trimmed SSE reduction coming from the 100 rows with the
  largest individual improvement (a gain made of a few rows is noise).
- Mean residual per airport, OOF (training months) vs holdout (baseline):
  the corrector assumes they're the same kind of prediction.
- The Huber diagnostic variant.
- Stack `lgb + cat_corrected` vs `lgb + cat`, cross-fit NNLS as in
  `tests/stack_test.py`. It informs the integration pre-registration; it
  doesn't decide this one.

## Known mismatches (accepted up front)

- **Frames:** OOF month m's features come from the continuous
  `train2025` frame; holdout features come from the isolated
  `holdout_gap2025` frame (which mirrors the ranking set's Jan→Jul gap).
  Context-dependent features can differ between the two. The corrector is
  trained on the first kind and applied to the second. Rebuilding
  isolated per-month frames isn't feasible before the freeze.
- **Training size:** OOF folds fit on 9 months; the holdout predictions
  come from a 10-month fit. The ranking predictions come from a 12-month
  refit.
- **Priors on corrector inputs:** for training rows the priors are fit on
  months including the row's own month (in-sample, the base models'
  existing convention); for holdout rows they are out-of-sample.

## If adopted: ranking (pre-registered separately)

Per HANDOFF.md: refit the corrector on all 12 months (training-month OOF
predictions plus the holdout predictions) and apply it to the all-2025
CatBoost refit's ranking predictions. How it enters the lgb+cat stack and
the ADS-B stages (`src/post/stack_submit.py`) is decided in the follow-up
pre-registration, not here.

## Amendments

(none)
