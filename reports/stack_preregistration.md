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

**2026-09-27 — XGBoost dropped, before any XGBoost or CatBoost holdout result
existed.** On free Colab (about 12.7 GB of RAM), the XGBoost GPU fit
exhausts memory roughly 7 minutes in: the local CPU profile peaks at
8.3 GB, of which about +3.5 GB is XGBoost's own transient during the fit.
This held even after the group-encoding memory fix (`ffde2e1`). The primary
stack is therefore **`lgb + cat`**. XGBoost isn't run, so there are no
`xgb` diagnostics. The rule is otherwise unchanged. `lgb` holdout predictions
come from the existing local cache (`prod_mixed_holdout_ev.parquet`); its
submission comes from a local `run(engine="lgb", target="mixed",
submit=True)`.

**2026-09-27 — CatBoost settings changed, before any CatBoost holdout result
existed.** The first CatBoost run on a T4 took about 70 minutes per fit
(~0.5 s per iteration), and its holdout predictions would only have been
written after a ~5-hour refit. It was stopped after one inner fit; only
June inner-validation values had been seen, no holdout score. New settings:
`max_ctr_complexity=1` (no categorical feature combinations) and learning
rate 0.05 for the CatBoost regressors (`run(eta=0.05)`). Everything else is
unchanged. The run is holdout-only first (`submit=False`); the submission
refit happens only if the stack is adopted.
