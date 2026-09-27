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

## Results (2026-09-27, `tests/lirf_mixture_test.py`, log `logs/lirf_mixture_test.log`)

**REJECT.** The decomposition check was exact for both engines (max error
0.000000). Fits: odd days lag −173 s, w 0.276; even days lag −183 s,
w 0.146.

| | base | new | delta | P(worse) |
|---|---|---|---|---|
| even days (fit odd) | 320.77 | 320.77 | −0.00 | 0.455 |
| odd days (fit even) | 306.49 | 306.42 | −0.07 | 0.188 |
| pooled July | 313.49 | 313.46 | −0.04 | 0.338 |
| minus best 3 days | 310.41 | 310.46 | +0.05 | 0.623 |

Why it fails:
- Non-echo LIRF rows improve (706 → 688 s), but echo rows get worse
  (876 → 953 s). The classifier can't separate them (recall ~9%).
- By echo-probability band, the model already beats raw ADS-B on the
  low-echo majority (< 0.1: model 395 s vs ADS-B 538 s). ADS-B helps only
  where the model was already bad (0.1–0.5: 806 vs 761 s).
- LIRF's error is a *labelling* problem (real block time vs schedule echo),
  not a pushback-timing problem.
