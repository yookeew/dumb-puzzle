# ADS-B partial-track estimate — pre-registration

Written and committed 2026-09-27, before any holdout (Jan/Jul 2025) result.
The design was explored only on 2025-09-15 and 2025-11-15, which are
training-month days, outside the holdout.

## What it adds

Departures matched to an ADS-B track that never shows pushback (tier null or
`pass`) are currently unused: about 100k ranking rows, more than the 94.5k
already blended. Such a track still shows most of the taxi:
`L = MVT_TIME − adsb_first_ts`. The unseen part is the stretch from the stand
to the first sighting.

Exploration (training days, n=3,148): the unseen part's spread (IQR) is
236–275 s when first seen within 600 m of the stand, against about 450 s for
taxi itself. Beyond ~1 km it is as uncertain as the whole taxi. As a
standalone estimate it is heavy-tailed (RMSE 429 s vs taxi sd 364 s;
median abs error 179 s). So it enters as a weighted nudge, not a replacement.

## Definition

- **Eligible ("partial"):** `adsb_matched`, tier not in {appear, dwell},
  `adsb_first_own_m` present and **< 1000 m**, airport ≠ LIRF.
- **Estimate:** `est = L + a_airport + b · adsb_first_own_m / 1000`.
  - `a_airport` is per airport (airports with < 100 fit rows share a pooled
    intercept); `b` is one global slope.
  - Fit by OLS on the fit rows with `unseen = taxi − L` clipped to its fit-set
    1st–99th percentiles.
- **Blend:** `final = base + w · (est − base)`, with w fit by least squares on
  the fit rows and clipped to [0, 1].

## Base and evaluation

- **Base:** the current production pipeline on the holdout, cross-fit exactly
  as in `tests/stack_test.py`: the lgb+cat NNLS stack, then the §39 ADS-B
  blend (per-airport-tier weights, LIRF excluded), each with parameters from
  the other month.
- **Evaluation:** direction A fits the partial parameters on Jan and scores
  Jul; direction B the reverse. RMSE over **all** holdout rows, paired
  against the base, (airport, day) cluster bootstrap, 3000 resamples,
  seed 0.
- **Rule:** adopt iff both months' point deltas < 0 and pooled P(worse) <
  0.05.
- **Secondary:** two weights, by distance band (< 500 m, 500–1000 m).
  Adopted over the primary only if it beats it in both directions.
- **Reported:** per airport, per distance band, the fitted `a`, `b` and `w`.

## Amendments

(none)
