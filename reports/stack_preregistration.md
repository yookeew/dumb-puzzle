# Second model family + stack — pre-registration

Written and committed 2026-09-27, before any XGBoost or CatBoost holdout
prediction on the current feature set existed. Changes made after results
exist go under Amendments, dated.

## Models

All three use the same feature frames (`cache/features/`), the same
`target="mixed"`, Huber loss (alpha/delta/slope 800), seed 42, and the same
echo classifier and reconstruction, via `models.fit.run`:

- `lgb`: LightGBM, current production (already cached:
  `cache/eval/prod_mixed_holdout_ev.parquet`).
- `xgb`: XGBoost `hist`, depth 10 (the existing engine).
- `cat`: CatBoost, depth 8, ordered target-statistic categoricals (new
  engine, `_fit_cat`).

Each model's echo classifier comes from its own engine. The stack combines
final taxi predictions only.

## Stack

Non-negative least squares on the holdout predictions, **no intercept**, one
global weight per model (no per-airport weights).

- **Cross-fit:** direction A fits on Jan 2025 and scores Jul 2025;
  direction B fits on Jul and scores Jan.
- **Primary:** the stack of every model that finished (`lgb+xgb+cat`; if
  CatBoost can't run, `lgb+xgb`, stated as such).
- **Diagnostics only (not decisive):** each single model, and the 2-model
  subsets. No picking the best subset after the fact.

## Rule

Adopt iff both months improve against `lgb` alone (point delta < 0 in A and
B) and the pooled (airport, day) cluster-bootstrap P(worse) < 0.05 (3000
resamples, seed 0).

## ADS-B on top

If the stack is adopted, the §39 ADS-B blend (per-airport lag + weights,
same fitting procedure, LIRF excluded) is refit on the **stacked**
prediction with the same month cross-fit. It's kept on top iff
stack + ADS-B beats stack alone in both months. The blend was already
adopted; this only checks it still adds something to a different base.

## Ranking application

Stack weights refit on the full holdout (Jan + Jul) are applied to each
engine's all-2025 refit submission (`run(..., submit=True)`).

## Amendments

(none)
