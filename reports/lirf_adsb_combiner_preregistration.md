# Rome (LIRF) ADS-B combiner — pre-registration

Written 2026-10-03, before any LIRF combiner was fit or scored. Changes
made after results exist go under Amendments, dated.

## Why

- LIRF is ~36% of the estimated 2026 squared error (§63), and every ADS-B
  stage so far excludes it. In §37 the raw ADS-B taxi at LIRF was worse
  than the NM off-block, and labelled LIRF ADS-B was almost nonexistent:
  Jan 2025 has none.
- **Scoping on the 2025 OOF months** (training months only, no holdout):
  on appear/dwell rows, after LIRF's median lag, the label agrees with
  ADS-B within ±120 s for 45% of rows (other airports 53–88%). 8% are off
  by > 600 s (others ~1%), and 16% are echoes (others 3–6%).
  - So ADS-B is informative at LIRF but noisier, and the label is
    sometimes not the physical pushback.
  - A learned model can weigh this by delay, echo propensity and
    detection quality.
  - Two alternative label hypotheses were rejected on the same data: the
    label as the last restart after a stop, and as an assigned or rounded
    value.
- **Data now available:**
  - the 115 thin-day-filtered Feb–Dec 2025 days (§65), of which LIRF
    coverage starts in May;
  - Jul 2025 holdout: 11,294 LIRF departures with ADS-B information;
  - Jan 2026 ranking: 7,476 (Jul 2026: 49).

## Treatment (`tests/lirf_adsb_combiner_test.py`)

- **Rows:** LIRF departures with ADS-B information (`adsb_matched`, or a
  fallback tier), from `cache/adsb_pushback_v4/`.
- **Training:** the OOF-month days kept by the §64/§65 thin-day rule.
  - Labels ≤ 5 h.
  - Stack `s` = the OOF (lgb, cat) stack, as in
    `tests/adsb_combiner_12m_test.py` with all ten months.
- **Model:** LightGBM, L2, on `taxi − s`.
  - **Inputs:** the combiner's FEATS (`tests/adsb_combiner_test.py`) plus
    `offset = T − SOBT`, `nm_unmatched` (AOBT_3_flt null), and
    `inbound_echo_day`. The last is the share of that operator's LIRF
    arrivals that day with |block − sched| < 30 s, computed from arrivals
    only (§63).
  - **Hyperparameters:** num_leaves 15, lr 0.03, min_data_in_leaf 200,
    lambda_l2 10, feature_fraction 0.9, max_bin 127, seed 42,
    deterministic, force_row_wise.
  - **Training:** early stopping (patience 200, cap 5,000) on training
    days with day-of-month divisible by 5, then a refit at the best
    iteration.
- **Scored on:** Jul 2025 holdout LIRF rows with ADS-B information.
  - `s` = NNLS(lgb, catcorr) weights fit on Jan 2025, as in the v27
    cross-fit.
  - Prediction = `s + model`, clipped to [0, 140,000].
  - Every other holdout row keeps the base value.
- **Base:** v27's method cross-fit (`tests/adsb_combiner_fullyear_test.py`
  treatment). LIRF rows keep the v23 value.

Jan 2025 has no LIRF ADS-B, so the evaluation is Jul 2025 only.

## Rule

On the Jul 2025 holdout (all airports, so the delta is the LIRF effect).
Adopt iff all of:
1. trimmed RMSE delta < 0 with P(worse) < 0.05;
2. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- LIRF-only RMSE before and after (trimmed and full);
- by delay band (≤ 30 min, 30–60 min, > 60 min);
- best iteration;
- training rows;
- top feature gains.

**If it passes:** build v28 = v27 with the LIRF rows that have ADS-B
information replaced. The model is fit on the OOF-month rows plus the Jul
2025 LIRF rows (`s` from the production stack weights) and applied to the
2026 ranking. **Keep iff the board < 262.13 (v27).** LIRF label rules have
failed on the board three times (v20, v22, §42), so the board is the
final arbiter.

**If it fails: stop.** No other inputs, hyperparameters or row filters.

## Amendments

(none)
