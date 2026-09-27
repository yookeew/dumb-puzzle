# ADS-B pushback blend — pre-registration

Written and committed 2026-09-27, **before** any blend result on the 124-day
pull was computed. The commit hash of this file is the timestamp. Any change
after results exist is recorded as a dated amendment below, never an edit.

## Why a blend, not a model feature

ADS-B exists only for Jan + Jul 2025/2026 (plus two single days in Sep/Nov
2025). The production model trains on the 10 non-holdout months, where the
feature would be null for every row, so it could never learn to use it. ADS-B
therefore enters as a second stage on top of the frozen model's prediction,
fit and scored on the Jan+Jul 2025 holdout, and applied to Jan+Jul 2026
ranking rows.

## Inputs

- `pred`: production model (`target="mixed"`, seed 42) holdout prediction,
  `cache/eval/prod_mixed_holdout_ev.parquet`.
- `adsb_taxi = MVT_TIME_UTC_mvt − adsb_pushback_ts` from
  `cache/adsb_pushback/` (`src/link/adsb_pushback.py`; reads only takeoff time
  and stand).
- Eligible rows: `adsb_tier ∈ {appear, dwell}`. `pass` is dropped (validated
  useless, §37). All other rows keep `pred` unchanged.

## Primary blend (tier-only weights + per-airport lag)

1. **Lag correction, per airport.** On the fit set's eligible rows,
   `lag_a = median(adsb_taxi − taxi)` at airport a; airports with < 100 fit
   rows use the pooled median. `adsb_c = adsb_taxi − lag_a`. This is a
   physical offset (transponder switched on after pushback starts; EGLL/LEMD
   ≈ +140–180 s), not a fitted weight.
2. **Tier weights.** `final = pred + w_t · (adsb_c − pred)`, with one weight
   per tier (appear, dwell), fit by least squares on the fit set and clipped
   to [0, 1]. That's two weights plus up to 9 medians.
3. **LIRF excluded** from the primary blend, pending the diagnosis below.

## Evaluation

- **Cross-fit across months.** Direction A fits on Jan 2025 and applies to
  Jul 2025; direction B fits on Jul and applies to Jan. Each month is scored
  only with parameters fit on the other.
- **Metric.** RMSE over **all** holdout DEP rows, blend vs model alone,
  paired. Uncertainty from an (airport, day) cluster bootstrap, 3000
  resamples, seed 0 (`tests/_harness.py`).
- **Reported alongside:** per airport, per tier, per month, and EDDM with its
  3 worst de-icing-suspect days (largest model error) shown separately.

## Adoption rule (standing)

Adopt the primary blend iff **both** months improve (point delta < 0 in A
and in B) **and** the pooled P(worse) < 0.05.

## Secondary variant — per-airport weights

`w_{a,t}` for airport-tier cells with ≥ 200 fit rows (else the tier weight).
It is adopted over the primary blend **only if** it beats the primary blend in
**both** cross-fit directions. Otherwise it's reported and not used.

## The summer question — decided now

The one summer day available so far (2025-07-15) showed no benefit from
combining, and EHAM got slightly worse. If July shows nothing while January
shows a gain, the standing both-months rule rejects the blend. A
**winter-only** blend (applied to January rows only; July rows keep `pred`)
is accepted **only if all** of the following hold:

- **Definitions.** "Jan shows a gain": direction B point delta < 0 with
  month-level P(worse) < 0.05. "Jul shows nothing": direction A point delta
  ≥ −0.5 s or P(worse) ≥ 0.10.
- **W1.** A within-January day-split cross-fit (fit on odd days → score even
  days, and the reverse) also gives P(worse) < 0.05. This removes the
  Jul→Jan transfer as the reason for the gain.
- **W2.** The January gain survives dropping the 3 January days with the
  largest per-day gain (point delta still < 0). This guards against a
  de-icing-day artefact, the 2025-01-15 EDDM lesson.
- **W3.** For ranking, only Jan 2026 rows get the blend, with parameters fit
  on Jan 2025. July 2026 rows get `pred`.

If W1, W2 or W3 fails, nothing is adopted. A significantly *worse* July
(P(worse) > 0.9) is recorded as evidence that the mechanism is winter-specific
and does not by itself block a winter-only blend that passes W1–W3.

## LIRF — diagnose before excluding

On LIRF eligible rows of Jan+Jul 2025, split by the true echo label
(`is_echo`: |d| < the echo threshold used in `src/models/fit.py`), report the
ADS-B error of each split against true block time.

- **H_echo.** The non-echo split's ADS-B RMSE is ≤ 1.5 × the pooled non-LIRF
  ADS-B RMSE, **and** the echo split carries the majority of LIRF's ADS-B
  squared error.
- **If H_echo holds:** variant L adds LIRF rows gated by the classifier's
  `echo_prob < 0.5` (available at ranking time; threshold fixed here, not
  tuned), with the same lag + tier-weight procedure. Adopted iff it improves
  LIRF RMSE in both cross-fit directions without breaking the overall rule.
- **If H_echo fails:** LIRF stays excluded, with the measured reason recorded.

## Data check run alongside (not a decision rule)

The wider fetch boxes (±0.10°/±0.15°, against the old ±0.06°/±0.09°) add
points. Detector outputs on the four days present in both extracts
(2025-01-15, 2025-07-15, 2026-01-15, 2026-07-15) are compared old box vs new
box. Recovered rows should be a near-superset with near-identical pushback
times. Material divergence blocks the evaluation until explained.

## Amendments

(none)
