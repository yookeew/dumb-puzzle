# ADS-B v5: runway-ending tracks as matches, through the restricted combiner — pre-registration

Written 2026-10-02, before any holdout result with v5 detections. The design
was explored only on 2025-09-15 / 2025-11-15 (training-month days) and on
2026 days without labels. Changes after results go under Amendments.

## Why

At EDDF, EGLL, LSZH, LEBL and LEMD, 60–95% of departures with no matched
track have a surface run ending on a runway within [T − 300 s, T + 30 s].
The climb-out wasn't received, so the takeoff-time match (gs ≥ 40 within
120 s) drops them. In 2026 that's ~130–180 departures a day at EDDF.

On the non-holdout days the v5 pass adds 568 / 308 matches:
- 95 of the new rows get an at-stand pushback (appear/dwell), 69% within
  ±120 s of the label;
- the rest add first-sighting information. That's weak alone
  (corr(taxi − base, L − base) = 0.03, vs 0.09 for existing untiered
  matches).

## What changed (`src/link/adsb_pushback.py --v5` → `cache/adsb_pushback_v5/`)

Split-blind (takeoff time, stand, adsb.lol points, runway geometry). After
the existing takeoff-time match and before the fallback:
- **Candidate runs:** for departures still unmatched, surface runs (split
  at gaps > 20 min) that aren't takeoff runs and whose last sample is
  within 200 m of a runway centreline segment
  (`data/external/runways.csv`).
- **Time window:** last sample in [T − 300 s, T + 30 s].
- **Matching:** runs visiting the own stand first, then
  |last − (T − 30 s)|; 1:1 greedy.
- **Downstream:** the matched row gets the normal pushback tiering and
  first-sighting fields, plus a new flag `adsb_rwy_end`. The v4 fallback
  then runs as before.
- Existing takeoff-time matches are unchanged by construction.

## Treatment and base (`tests/adsb_v5_test.py`)

**Base = v25's method, cross-fit:** the v24 combiner
(`tests/adsb_combiner_test.py`, v24 hyperparameters) on v4 detections,
applied only to rows with ADS-B information (matched or a fallback tier),
v23 pipeline value elsewhere. LIRF keeps the v23 value.

**Treatment:** the same restricted combiner on v5 detections:
- an added input `rwy_end` (the flag);
- fit and applied where v5 has ADS-B information (now including the
  runway-ending matches);
- everything else identical: hyperparameters, early stopping, cross-fit.

## Rule

Trimmed RMSE decides (labels ≤ 5 h, the fixed 31-row set). Adopt iff all
of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- per airport;
- the newly matched rows' RMSE, base vs treatment;
- v5 row counts per month;
- best iterations.

**If it passes:** build v26 = the v25 build on v5 detections with the
flag (`src/post/adsb_combiner.py`). **Keep iff the board < 264.39.**

**If it fails: stop.** No other window, distance or matching rule.

## Amendments

(none)

## Results (2026-10-02, `tests/adsb_v5_test.py`, log `logs/adsb_v5_test.log`; run after commit 3b29a3d)

**Primary: REJECT.** v5 adds 26,456 runway-ending matches on the holdout
(rows with ADS-B information 159,451 → 178,537).

| trimmed RMSE | base (v25 method) | v5 | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 288.41 | 288.44 | +0.03 | [−0.12, +0.18] | 0.645 |
| B: Jan (fit Jul) | 216.84 | 216.90 | +0.05 | [−0.23, +0.36] | 0.637 |
| pooled | 258.93 | 258.97 | +0.04 | [−0.10, +0.18] | 0.678 |
| guard: pooled full | 324.01 | 324.04 | +0.03 | | 0.695 |

- **Newly covered rows** (19,085): 272.1 → 273.1. **Rows covered in both:**
  239.7 → 239.7.
- **Per airport:** EDDF −1.08, but EHAM +1.18 and LSZH +0.80.
- Best iterations 3,523 / 428 (v24: 4,614 / 1,008).
- Per the rule: stop, with no other window, distance or matching rule.
  v25 stays.
