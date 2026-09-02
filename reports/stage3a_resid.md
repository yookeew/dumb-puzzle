# Per-airport residual heads + widened label window — REGRESSION, abandoned

Run 2026-09-02 (`lgbm_v2`). Two changes tested together:
1. Raise the training label window from [60, 5400] s to [0, 14400] s so LIRF's
   long tail is learned.
2. Global LightGBM on d + a per-airport LightGBM on (d − out-of-fold global d̂),
   trained on 2 month-folds, added back at predict time.

## Holdout (Jan+Jul 2025)

| model | overall | EDDF | EHAM | LIRF | LEMD |
|---|---|---|---|---|---|
| lgbm_v1 (baseline, [60,5400], no heads) | **301** | 262 | 262 | 601 | 72* |
| v2 global only ([0,14400]) | 313 | 349 | 305 | 563 | 198 |
| v2 global + residual heads | 332 | 403 | 352 | 577 | 198 |

\* lgbm_v1 per-airport numbers were on a slightly different scoring subset; the
directional comparison holds.

## Why it failed

- **Widening the label window hurt the 9 well-behaved airports.** L2 on d with
  LIRF-scale outliers in the training set pulls the shared trees toward the tail:
  EDDF 262 → 349, EHAM 262 → 305. LIRF itself only improved 601 → 563.
- **The residual heads overfit.** num_leaves 63, 400 rounds, ~150 k rows/airport,
  full ~50-feature set → they memorised the 2-fold out-of-fold residuals, which
  are seasonal and did not transfer to held-out Jan/Jul. Every airport got worse
  (overall 313 → 332).

## Kept / reverted

- **Kept**: prediction floor lowered to 0 (no positive floor).
- **Reverted**: label window to [30, 7200] (a middle ground). Prediction ceiling
  10800 s.
- **Reverted**: residual heads removed. `src/models/train_lgbm.py` and the
  portable `src/models/fit.py` are back to a single global model.

### Result of the reverted config (`lgbm_v3` / `lgb_colab`, portable fit.py)

Holdout **308 s** overall; LIRF **570** (vs lgbm_v1 601, lgbm_v2 563). Per airport:
EDDF 306, EDDM 200, EGLL 313, EHAM 295, LEBL 261, LEMD 197, LFPG 309, LSZH 219,
LTFM 289. The [30,7200]+floor-0 change trades ~7 s overall for a small LIRF gain —
not clearly better than lgbm_v1. **lgbm_v1 (301) remains the submission to beat.**
Portable pipeline wall time: 245 s (vs 25 min for the v2 run) — confirms the
Colab split is the right call for iteration.

## What to try instead (on Colab — local fit is ~4 min, too slow)

1. **Robust loss on d** — Huber / fair, or `objective="quantile", alpha=0.5`, so
   the LIRF tail informs without dominating the L2 gradient.
2. **Out-of-fold echo features** — per (airport, operator) and (airport, stand)
   historical P(|d| < 30 s) and mean d, computed with a proper time-fold, fed to
   the single global model. Gives it the LIRF-specific structure without a
   separate model (echo rate ranges 1.7 %–54 % by operator — strong signal).
2. **Sample weights** — down-weight the extreme tail rather than clip it.
3. If per-airport capacity is still needed: heavily-regularised heads on a
   *small* feature subset (stand, runway, hour, operator, offset), or
   `is_unbalance`-style class handling for the echo/no-echo split.
