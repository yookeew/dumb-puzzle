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

(none; the rules above were applied unchanged)

## Results (2026-09-27, `tests/adsb_partial_test.py`, log `logs/adsb_partial_test.log`)

**Primary: ADOPT.** Base = stack + ADS-B, cross-fit, 329.97.

| | base | + partial | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 314.02 | 313.61 | −0.41 | [−0.62, −0.21] | 0.000 |
| B: Jan (fit Jul) | 348.75 | 348.25 | −0.50 | [−0.87, −0.32] | 0.000 |
| pooled | 329.97 | 329.52 | −0.45 | [−0.62, −0.32] | 0.000 |

- Fitted values: b ≈ 240 s/km; w 0.17 / 0.16. There are 38,225 eligible
  holdout rows.
- Per airport: LEBL −3.1, EDDM −2.1, LSZH −1.7, EDDF −1.6, EHAM −0.4; none
  worse.
- The first-sighting-under-500 m rows go 180.9 → 168.4 s. The 500–1000 m
  rows go 205.6 → 206.2 s under the single weight.

**Secondary (two distance-band weights): beats the primary in both
directions** (Jul 313.61 → 313.49, Jan 348.25 → 348.07), so it's adopted.
Band weights: near 0.28 / 0.26, far 0.07 / 0.09.

**Known flaw, negligible here.** The "pooled" intercept for airports with
< 100 fit rows is a column that is all zeros when every airport in the fit
month has ≥ 100 rows. That happened when fitting on January, so July's LEMD
(121 rows) and LFPG (68 rows) got an intercept of 0. Their per-airport
effect rounds to 0.0 s. In the final Jan+Jul fit only LFPG (68 rows) uses
it, fitted on its own rows (356 s).
