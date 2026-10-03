# ADS-B combiner retrained with all of 2025 — pre-registration

Written 2026-10-03, after v26 (Sep–Dec, board 262.70) and **before any
Feb–Aug 2025 ADS-B day was normalised, detected or used**. Seven such days
(04-01, 05-01, 06-01, 06-02, 08-01, 08-02, 08-03) arrived early. They sit
untouched in `external-data/adsb-later/`. Changes after results go under
Amendments, dated.

## Why

Adding Sep–Dec training rows to the combiner gave −1.22 trimmed on the
holdout and −1.69 on the board (§64), with no airport worse. The fits also
became more stable. The other six months (Feb–Jun, Aug) have the same
honest month-wise OOF base predictions, so their ADS-B is the obvious
next increment.

## Data

- **Days:** the same sample as v26, applied to Feb, Mar, Apr, May, Jun and
  Aug 2025: days 01, 02, 03, 04, 07, 10, 13, 16, 19, 22, 25 and 28, as many
  as the Colab pull delivers.
- **Pipeline:** fetched into `adsb_restofyear/` by the existing Colab cell.
  The `adsb-later/` days are moved there too. Then
  `src/ingest/normalise_adsb.py` and `src/link/adsb_pushback.py --v4`.
- **The same thin-day rule** (fourth amendment of
  `reports/adsb_combiner_12m_preregistration.md`): status ok/tolerant and
  ≥ 50% of the month's median points.
- **The months, the day sample and the rule are fixed now.** The gate runs
  once, on whatever the pull has delivered when it's run. The number of
  days per month is reported.

## Treatment and base (`tests/adsb_combiner_fullyear_test.py`)

- **Base = v26's method, cross-fit:** the restricted combiner trained on
  the other holdout month plus the Sep–Dec days.
- **Treatment:** identical, but trained on the other holdout month plus
  Feb–Jun, Aug and Sep–Dec days.
- **Same for both:** hyperparameters, inputs, early stopping, OOF stack
  and restricted application.

## Rule

Trimmed RMSE decides (labels ≤ 5 h, the fixed 31-row set). Adopt iff all
of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- days kept per month;
- training rows;
- best iterations;
- per airport.

**If it passes:** build v27 = `src/post/adsb_combiner.py` trained on both
holdout months plus all the kept days. **Keep iff the board < 262.70
(v26).**

**If it fails: stop.** No month subsets or weighting.

## Amendments

(none)
