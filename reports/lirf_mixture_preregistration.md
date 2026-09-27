# LIRF echo-aware ADS-B mixture — pre-registration

Written and committed 2026-09-27, before any result of this design. The
earlier diagnosis (reports/adsb_blend_preregistration.md, H_echo) is the only
prior look at LIRF ADS-B rows: on Jul 2025 non-echo rows, ADS-B RMSE was 529 s
against the model's 751 s; echo rows had ADS-B 3,118 s.

## Why a separate design

Every production engine's prediction is already a mixture
(`src/models/fit.py::_reconstruct_taxi`; LIRF never takes the prior branch):
`pred_i = e_i · offset + (1 − e_i) · m_i`, where `e_i` is engine i's
out-of-sample echo probability, `offset = T − SOBT`, and
`m_i = clip(taxi_model_raw_i, 0, 140000)`. ADS-B measures the physical
pushback, which is the right target for the **non-echo** component only. So
the nudge is applied to `m_i`, not to the whole prediction.

## Definition

- **Rows:** LIRF, `adsb_tier ∈ {appear, dwell}`.
- `adsb_c = adsb_taxi − lag`, with
  `lag = median(adsb_taxi − taxi)` over the fit rows with `is_echo = 0`
  (the true label is used only when fitting).
- `c = Σ_i s_i · (1 − e_i) · (adsb_c − m_i)` over engines i ∈ {lgb, cat},
  where `s_i` are the stack weights used for that row.
- `new = base + w · c`, with w by least squares on the fit rows, clipped to
  [0, 1].
- **Base:** the current production pipeline on the holdout (cross-fit stack
  → §39 ADS-B blend → §41 partial tracks). LIRF rows are unchanged by §39
  and §41, so LIRF's base is the cross-fit stack.

## Evaluation — the deviation from the standing rule

Jan 2025 has **zero** LIRF ADS-B coverage, so the Jan↔Jul month cross-fit is
impossible. Instead, a **within-July day split**: fit on odd July days, score
even days, and the reverse. Stack weights for July rows come from the Jan fit,
as in production cross-fit.

- **Metric:** RMSE over all July holdout rows, paired against the base,
  (airport, day) cluster bootstrap, 3000 resamples, seed 0.
- **Rule:** adopt iff **both** halves (odd→even, even→odd) improve (point
  delta < 0) **and** pooled P(worse) < 0.05.
- **Robustness:** the gain must survive dropping the 3 July days with the
  largest LIRF gain (point delta still < 0). LIRF's error is tail-dominated,
  and a gain resting on a few days would not transfer.
- **Reported:** LIRF RMSE split by true `is_echo`, and by `e` bands
  (< 0.1, 0.1–0.5, ≥ 0.5).

## Ranking application

Fitted on all of July 2025 (the only month with data), with stack weights
from the full holdout. It applies to Jan 2026 LIRF rows with appear/dwell
(Jul 2026 has no LIRF coverage).

## Amendments

(none)
