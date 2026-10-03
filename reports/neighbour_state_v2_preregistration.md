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

## Results (2026-10-03, `tests/neighbour_state_v2_test.py`, log `logs/neighbour_state_v2_test.log`; run after the pre-registration commit)

**Primary: ADOPT.**
- **Fits:** best_iter 696 / 1,059. Applied to 100,895 (Jul) / 83,993
  (Jan) holdout rows.
- **Correction RMS:** Jul 76.9 raw -> 74.3 demeaned; Jan 64.8 -> 60.0.

| trimmed RMSE | base (v27 method) | v2 | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 286.82 | 282.72 | −4.10 | [−6.88, −1.90] | 0.000 |
| B: Jan (fit Jul) | 214.90 | 212.66 | −2.24 | [−3.95, −0.59] | 0.003 |
| pooled | 257.22 | 253.85 | −3.37 | [−5.24, −1.88] | 0.000 |
| guard: pooled full | 322.65 | 320.72 | −1.93 | [−3.94, −0.28] | 0.010 |

- **Per airport (trimmed):** LTFM −10.74, EGLL −5.82, LIRF −3.68, LEMD
  −2.95, EDDM −1.44, LSZH −1.38, LEBL −1.14, EDDF −0.83, LFPG −0.75, EHAM
  +0.04.
- **Reported:** (b) only, not demeaned: pooled −4.08, Jan −3.71, Jul
  −4.37, full −2.44.
- **Caveat recorded before the board:**
  - Airport-day levels are only 12% of v29's 2026 change energy (airport
    levels 2%), so level drift explains at most part of v29's ~2×
    overshoot.
  - The within-day part of v29 had an implied optimal scale of ~0.55-0.69.
  - Expected v30 board vs v27: roughly −0.7 to −2.7, *assuming* part (b)
    aligns with 2026 as well as v29's changes did on average.
  - The holdout keeps overstating this family by ~2×. The most likely
    remaining cause is that the corrector is trained against 10-month and
    OOF base predictions but applied to the 12-month production stack.

**Board: v30 = 261.40 (v27 262.13, −0.73). ADOPTED; new best.**
- From the two scores: Σc² = 1.28e9, ΔSSE = −1.31e8, so Σc·r = 7.0e8. The
  SSE-optimal scale is ~0.55 (≈ 260.0 at that scale).
- So v30 is still ~2× too strong. The overshoot is in the within-day
  signal, not the airport-day level.
