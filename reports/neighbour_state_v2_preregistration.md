# Neighbour-state stage v2 (within-day only) — pre-registration

Written 2026-10-03, after v29's board result (262.33, rejected) and
before any holdout run of this variant. Changes made after results exist
go under Amendments, dated.

## Why

- **v29 is directionally right but too strong.** From the two board
  scores, its 2026 changes align with the true errors, but the
  SSE-optimal scale was ~0.49 (`reports/neighbour_state_preregistration.md`,
  Board).
- **The diagnosis is level drift.** The neighbour and own-flight inputs
  carry period-level NM-bias and delay levels that move more from 2025
  to 2026 than between Jan and Jul 2025. v29 shifted whole airports by
  10-20 s.
- **The shared-state signal lives within the day.** Only ~1.4% of
  residual variance is at the airport-day level (§20). On the 2025 OOF
  months, removing the airport-day mean of the part-(b) correction keeps
  most of its gain: −1.94% / −2.15% residual RMS vs −2.06% / −2.63%. By
  construction it can't move an airport's level.
- **Part (a)** (neighbours inside the ADS-B combiner) didn't help on the
  holdout (+0.29) and is dropped.

## Treatment (`tests/neighbour_state_v2_test.py`)

- Rows with ADS-B information keep the base (v27 method) value.
- **Every other row:** s + c̃, where:
  - c = the part-(b) neighbour corrector of
    `tests/neighbour_state_test.py`, with the same inputs, training rows,
    hyperparameters and fit;
  - c̃ = c minus the mean of c over the rows it's applied to in the same
    (airport, UTC day).
- Clipped to [0, 140,000].

**Base:** v27's method cross-fit, as in the first neighbour gate.

## Rule

Trimmed RMSE (labels ≤ 5 h, the fixed 31-row set) decides. Adopt iff all
of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- per airport;
- the undemeaned (b)-only arm, for reference;
- the 2026 ranking correction's RMS per airport, compared with the
  holdout's (label-free).

**If it passes:** build v30 with `src/post/neighbour_state.py --v2`
(production fit as v29's part (b), then the same demeaning on the 2026
rows). **Keep iff the board < 262.13 (v27).**

**If it fails: stop.** No scale factors, no window or feature variants on
the holdout.

## Amendments

(none)
