# LIRF long-delay hedge recalibration — pre-registration

Written 2026-10-02, before any recalibrated prediction was computed. Changes
made after results exist go under Amendments, dated.

## Why

- The leaderboard gap (v21 270.2 vs the leader's ~224) is 7.9e9 squared
  seconds over 344,841 ranking rows. That equals one row off by 89,000 s,
  or ~5 rows off by 40,000 s; closing it uniformly would need −31% MSE on
  every row. On our holdout, the top 5 rows carry 35% of the squared error
  (327.5 → 265.1 without them). Our base model is not behind: on the same
  holdout cuts arnavhm13 reports Jan 357 / Jul 333 / Jan ≤ 1 h 218; v21 is
  347 / 311 / 205 (PROGRESS.md §54).
- So the gap most likely sits in how a few dozen huge labels are hedged,
  mostly LIRF delayed departures whose label is either a normal taxi or
  the schedule echo (≈ T − SOBT).
- v21 never recalibrates these rows. The OOF corrector skips base
  predictions > 7,200 s, and every test since §47 has been decided on
  trimmed RMSE, which excludes them.

**Disclosure — what was already seen.** In the §53/§54 session these
holdout statistics were looked at for LIRF by (delay band, NM-unmatched):
echo rate, median prediction/offset, SSE share. Example: 1–3 h NM-unmatched
has echo rate 0.28, offset median 6,358 s, prediction median 3,278 s (a
squared-loss hedge would sit nearer ~2,600 s). The cells and the shrinkage
constant below were chosen after seeing those, but no recalibrated
prediction or RMSE has been computed.

## The change

On top of v21's final prediction (stack + quality ADS-B blend + partial;
none of the ADS-B stages touch LIRF):

- **Population:** LIRF departures with `offset = T − SOBT` in (1 h, 6 h].
- **Cells:** delay band {(1 h, 3 h], (3 h, 6 h]} × NM-unmatched
  (`AOBT_3_flt` null) = 4 cells.
- **Shift:** `pred' = pred + s_cell`, with
  `s_cell = n/(n + 50) · mean(taxi − pred)` over the fit month's cell rows
  (the squared-loss-optimal constant, shrunk toward 0). Then clip to
  [0, 140,000] as in production.
- **Excluded on purpose:** offsets > 6 h. Their labels include the "+24 h"
  rows, and v20 showed a holdout gain there turning into a board loss.
  It's reported as a variant, not decisive.

## Evaluation

`tests/lirf_hedge_test.py`. Base = v21 cross-fit on the Jan+Jul 2025
holdout exactly as in `tests/ltfm_egll_anatomy.py::v21_pred`. Direction A
fits the shifts on Jan and scores Jul; B the reverse.

**Decision metric is full RMSE** (all holdout rows), not trimmed. That's
the deliberate exception to the §47 convention: this change targets exactly
the rows trimming removes.

Adopt iff all of:
1. full-RMSE delta < 0 in both Jan and Jul;
2. pooled full-RMSE P(worse) < 0.05;
3. guard: trimmed RMSE (labels ≤ 5 h, the fixed 31-row set) P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0 (`tests/_harness.py`).

Reported, not decisive: the fitted shifts and cell counts per direction;
per-cell RMSE; the variant that adds an (6 h, ∞) band.

**If it passes:** build v22 = v21 + shifts fitted on Jan+Jul together,
applied to the 2026 LIRF ranking rows in the population (4,185 rows with
offset > 1 h, of which 348 NM-unmatched, before the 6 h cut). **Keep iff
the board score < 270.2.** Holdout gains on these rows have failed to carry
to the board before (v20), so the board is the final arbiter.

**If it fails: stop.** No other cell layout, shrinkage or band edges.

## Amendments

(none)

## Results (2026-10-02, `tests/lirf_hedge_test.py`, log `logs/lirf_hedge_test.log`; run after commit 684eb91)

**Primary: ADOPT.** Base = v21 cross-fit, full RMSE 327.52. Population
4,124 holdout rows.

| | base | new | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 310.95 | 309.92 | −1.03 | [−1.94, −0.21] | 0.006 |
| B: Jan (fit Jul) | 346.98 | 346.51 | −0.47 | [−1.32, +0.05] | 0.045 |
| pooled | 327.52 | 326.75 | −0.77 | [−1.39, −0.27] | 0.002 |
| guard: pooled trimmed | 263.30 | 262.32 | −0.99 | [−1.68, −0.36] | 0.001 |

- **The gain comes from the NM-unmatched cells.** Their shift is negative
  and similar in both directions: 1–3 h −637 (fit Jan) / −625 (fit Jul).
  1–3 h NM-unmatched RMSE 2,377 → 2,232.
- **The NM-matched shifts flip sign** between months (1–3 h: −29 fit on
  Jan, +108 fit on Jul) and make their cells slightly worse (768 → 774,
  920 → 926). Per the rule, all four cells are kept. If v22 disappoints,
  this is the first suspect, but no rescue variant is allowed.
- **Variant with a 6 h+ band** (not decisive): pooled −1.50, both months
  better. Its 6 h+ NM-unmatched cell has only 5 (Jan) and 16 (Jul) fit
  rows. Not used, as pre-registered.

**v22 build** (`src/post/lirf_hedge.py`, log `logs/lirf_hedge_build.log`).
- Shifts fitted on Jan+Jul cross-fit v21 predictions: 1–3 h matched +84,
  unmatched −720; 3–6 h matched +100, unmatched −425.
- Applied to the uploaded `smart-jigsaw_v21.parquet`: 4,142 ranking rows
  change, RMS diff vs v21 22.7 s.
- Board A/B pending: keep iff < 270.2.

**Board: v22 = 270.86 (v21 270.2, +0.66). REJECTED; v21 stays.** Per the
rule, no rescue variant (e.g. NM-unmatched cells only).
