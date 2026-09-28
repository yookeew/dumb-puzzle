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

## Results (2026-09-27, `tests/stack_test.py`, log `logs/stack_test.log`)

CatBoost alone scored 336.72 on the holdout (LightGBM 335.63). It's better at
EDDF, EGLL, EHAM, LEBL, LFPG and LSZH, and worse at LIRF (670 vs 653).

**Primary stack lgb+cat: ADOPT.**

| | lgb | stack | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 319.76 | 316.11 | −3.65 | [−5.48, −1.53] | 0.000 |
| B: Jan (fit Jul) | 354.34 | 353.01 | −1.33 | [−5.44, +3.28] | 0.251 |
| pooled | 335.63 | 333.08 | −2.55 | [−4.77, −0.15] | 0.021 |

- Every airport improves (LIRF −5.6, LFPG −3.5, EGLL −3.1).
- January's gain alone isn't significant; the rule (both point deltas < 0,
  pooled P(worse) < 0.05) is met.
- Weights: lgb 0.61 / cat 0.40 (fit on Jan), 0.48 / 0.54 (fit on Jul);
  full-holdout 0.544 / 0.468.

**ADS-B on top: kept.** Stack + ADS-B against stack alone: Jul −2.09, Jan
−4.27 (both P(worse) = 0.000). Against lgb alone, pooled: 335.63 → 329.97.

**Leaderboard:** v15 (stack) 296, v16 (stack + ADS-B) **277**, from 302.

## Follow-up: CatBoost with a 20,000-round cap (pre-registered 2026-09-28, before results)

The §40 CatBoost `flip` fit hit the 8,000-round cap while still improving
(~−2 per 500 rounds at lr 0.05). New run: identical settings except
`rounds=20000`, with the same early stopping (150 rounds on the June inner
validation) and the same refit rule (1.1 × best iteration). The echo
classifier is unchanged. Output: `cat_mixed_r20k` (holdout predictions and
submission).

**Rule:** the new CatBoost replaces the old one iff the cross-fit NNLS stack
`lgb + cat_r20k` beats `lgb + cat` (the current stack) in **both** months
(point delta < 0), with pooled cluster-bootstrap P(worse) < 0.05 for that
comparison. Otherwise the current stack stays.

**Result (2026-09-28, `tests/stack_r20k_test.py`, log `logs/stack_r20k_test.log`):
KEEP the current stack.** Full RMSE (the pre-registered metric): Jul +0.04,
Jan −0.28, pooled −0.11 (P(worse) 0.274), which fails the rule. Trimmed RMSE
(31 holdout labels > 5 h excluded; reported, not decisive): Jul −0.26
(P(worse) 0.030), Jan −0.54 (0.045), pooled −0.37 (0.007). So there's a
small real gain on ordinary rows, masked by LIRF's monster rows (LIRF
+0.70). Not adopted: the rule is followed, and ~0.4 s on the holdout
isn't worth a pipeline change this close to the freeze.
