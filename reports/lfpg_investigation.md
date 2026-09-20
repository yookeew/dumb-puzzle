# LFPG investigation (2026-09-18)

Context: after the LIRF CEIL fix shipped (`smart-jigsaw_v12`, 337.6s local /
302s actual), LFPG became the single worst airport (581s RMSE, next-highest
is EGLL at 289s) with no dedicated investigation — flagged as the top
priority in `PROGRESS.md` §12's next steps. This doc follows the same
methodology as `lirf_investigation.md`: check before assuming, quantify
before building.

Diagnostic scripts: `lfpg_diag.py` (repo root, ad hoc, not committed to
`src/`), reading the already-cached `cache/lirf_ceilfix_mixed_ev.parquet`
(the shipped v12 holdout eval frame) — no retraining needed for this
investigation.

## 1. Headline finding: LFPG's 581s RMSE is an outlier-concentration
   illusion, not a systemic problem

Per-row squared-error concentration, full LFPG holdout population (n=39,590):

| exclude top-K worst rows | RMSE | 
|---|---|
| none (baseline) | 581.1 |
| top 1 | 403.1 |
| top 2 | **282.7** |
| top 10 | 265.0 |

**Excluding a single row drops RMSE by 31%; excluding two rows lands LFPG
right in the middle of the well-behaved airport band (200-290s).** This is
not a <100-row-noise footnote — it is *the entire story* for this airport.

Cross-checked against every other airport to confirm this is unusual and
not just "how holdout RMSE always looks":

| airport | top-1 SSE share | top-10 SSE share |
|---|---|---|
| **LFPG** | **51.9%** | **79.2%** |
| LIRF (post-fix) | 9.6% | 28.5% |
| LSZH | 7.8% | 29.2% |
| EHAM | 7.2% | 18.8% |
| EGLL | 2.9% | 14.6% |
| EDDM | 3.6% | 13.0% |
| LTFM | 1.3% | 6.9% |
| EDDF | 1.2% | 6.8% |
| LEMD | 0.7% | 4.4% |
| LEBL | 0.5% | 3.4% |

No other airport comes close. LFPG is a genuine outlier in *outlier
concentration itself* — the other nine airports (LIRF included, post-fix)
have broadly distributed error; LFPG's is a statistical spike from
essentially 1-2 rows in a 39,590-row holdout.

## 2. Checked for the LIRF mechanism (CEIL discarding a correct raw
   prediction) — absent here

LIRF's fix worked because the `flip` target's raw `offset - d̂` naturally
tracked the true value for large-offset rows, and `CEIL=10800` was
discarding an already-correct answer. LFPG uses `direct` under the
`target="mixed"` production config, whose raw output structurally cannot
exceed ~6-7k seconds (trained on labels clipped to `LABEL_HI`) — so there
is no CEIL-discard mechanism available for `direct` at all (already known
from the LIRF doc: confirmed 0/26,528 LIRF rows exceed CEIL raw for
`direct`; same structural limit applies to LFPG). Verified directly:

| | LFPG `use_prior` lane (n=877), RMSE |
|---|---|
| current fallback (`median_taxi_prior`) | 3465.3 |
| raw model (`taxi_model_raw`, uncapped) | 3455.3 |

Raw and fallback are statistically the same (both wrong by the same
magnitude) — there is no hidden correct answer being thrown away. The LIRF
fix's mechanism does not transfer.

## 3. Root cause of the 2 catastrophic rows: NM-unmatched + physically
   anomalous, not a learnable pattern

The two worst rows:

| MVT_ID | true taxi | true d (AOBT-SOBT) | offset (T-SOBT) | operator | stand | type |
|---|---|---|---|---|---|---|
| 182378789 | 84,240s (23.4h) | -82,500s | 1,740s | **null** | C03 | A319 |
| 182377662 | 58,206s (16.2h) | -56,163s | 2,043s | **null** | D06 | A320 |

Both are missing not just `AOBT_3_flt` but the *entire* NM flight match
(`AIRCRAFT_OPERATOR_flt` is null) — these aren't "AOBT_3 individually
missing," they're movements with no NM flight-list counterpart at all,
exactly the "movement/flight reconciliation limitations" CLAUDE.md's Note 2
warns about. `sched_takeoff_offset` (small, 29-34 min) is unremarkable —
nothing about the takeoff timing signals anything unusual; the anomaly is
entirely in the actual off-block time, which is 15-23 hours *before*
schedule.

**Full-year check (239,552 LFPG departures, all of 2025, not just
holdout):** only **7 rows** all year have `d < -20,000s`, and only **3**
have `taxi > 20,000s`. One of those 7 extreme-`d` rows has **`taxi = -7s`**
— a physically impossible negative taxi time, which proves at least some
of these records are corrupted (wrong date/timestamp on `BLOCK_TIME_UTC_mvt`
or a bad movement/flight match), not genuine rare operational events. The
outcomes for the other extreme-`d` rows are also wildly inconsistent
(`taxi` ranging from 1 to 84,240s for similarly-extreme `d` values) — there
is no consistent physical story linking "off-block recorded far before
schedule" to any predictable taxi outcome.

**No feature discriminates these rows in advance.** Checked operator (null
for all), stand (C03, D06 — no shared cluster), aircraft type (A319, A320 —
common types, not a rare-fleet signal). Chasing a targeted fix here would
be building a model for 2-7 rows a year with inconsistent, sometimes
physically-impossible labels — precisely the overfitting trap the CEIL-raise
regression (`PROGRESS.md` §6) and the contestants' "reject gains concentrated
in <100 rows" rule (§5) both warn against, except more extreme (n=2, not
n<100).

## 4. Theoretical ceiling, for the record

If those exact 2 rows were predicted perfectly (an unreachable oracle
bound), overall holdout RMSE would drop **337.6s → 290.4s (-14%)** — a
startling number on paper for 2 rows out of 344,339. This is *why* LFPG's
headline number looked like the obvious next lever. But the bound is not
achievable: there is no available signal, and one of the handful of extreme
rows in the full-year data is provably a corrupted record, not a modelable
event. Reported here so a future session doesn't re-derive this and get
tempted by the same 14%-on-paper number.

## 5. One small, real, but low-impact fix found

For the **other** 875 NM-fully-unmatched LFPG rows (excluding the 2
catastrophic ones), the raw model prediction beats the current flat
`median_taxi_prior` fallback:

| | RMSE (n=875, catastrophic 2 excluded) |
|---|---|
| current fallback (`median_taxi_prior`) | 610.5 |
| raw model (`taxi_model_raw`, clipped [0,10800]) | **536.0** |

Swapping the fallback to the raw model for LFPG's `use_prior` rows only:
overall holdout RMSE 337.56 → 337.30 (-0.26s), LFPG-only 581.1 → 579.8,
**both months improve slightly** (Jan 356.1→355.9, Jul 321.8→321.5). Real
and correctly-signed, but the population is only 0.25% of all rows, so the
aggregate effect is negligible. Not worth a dedicated submission on its
own; a candidate to fold into a future batch of small fixes, not a
priority in isolation. (This doesn't contradict the `use_prior` fallback's
original justification — that was measured in aggregate across *all*
airports, where the flat median helps EDDF/EHAM/etc.; this is a
narrower, LFPG-specific observation about the non-catastrophic subset only,
found by decomposing the population the outliers had been drowning out.)

## Conclusion / redirect

**LFPG is not the next LIRF.** The earlier assumption (this airport's high
RMSE = a systemic, fixable problem worth prioritizing) was wrong — it was
based on the pooled RMSE number alone, before decomposing it. Once
decomposed, LFPG is an ordinary, well-modeled airport with two
irreducible-by-available-data outlier rows sitting on top. Recommend:

1. **Do not build a targeted LFPG fix.** No safe, generalizable mechanism
   exists; the addressable population is 2 rows/year with inconsistent,
   partly-corrupted labels.
2. **Re-run this same top-K/SSE-share check as a standard diagnostic**
   before treating any airport's RMSE as a priority target — it would have
   caught this in minutes instead of after a full LIRF-style investigation
   pass. LSZH (29.2% top-10 share) and EHAM (18.8%) are the next
   highest-concentration airports and worth the same 5-minute check before
   assuming their gaps are broad-based.
3. The genuinely broad-based remaining gap (LIRF still at 660s even
   excluding its own top rows, EGLL/LTFM at 260-290s baseline) is where any
   further modeling effort has real addressable room — LFPG isn't it.

## 6. Follow-up: same check run on LSZH and EHAM (2026-09-18) — cleared,
   no LFPG-style pathology

Both flagged above as the next-highest top-10 SSE shares. Neither
reproduces LFPG's pattern:

| airport | top-1 share | top-10 share | decay shape |
|---|---|---|---|
| LFPG | 51.9% | 79.2% | cliff — 2 rows explain almost everything |
| LSZH | 7.8% | 29.2% | gradual — RMSE 212→166 even excluding top 50 |
| EHAM | 7.2% | 18.8% | gradual — RMSE 200→168 even excluding top 50 |

Two structural differences from LFPG confirm this is the already-known
tail problem, not a new bug:

- Their worst rows are mostly **`has_aobt3=True`** (fully NM-matched,
  normal flights) — e.g. LSZH's #2 worst row: taxi=9,085s, pred=1,910s,
  offset=17,106s, matched and unremarkable except for a large genuine delay
  the model underpredicts. This is the same regression-to-mean
  large-delay underprediction already documented project-wide (the d9
  decile pattern), just moderately concentrated at these two airports —
  not a data-quality artifact.
- Their `use_prior` (NM-unmatched) lanes are proportionally elevated
  (LSZH 668.6 RMSE / 21.3% SSE on 477 rows, EHAM 655.1 / 21.1% on 808 rows)
  but at a normal magnitude matching LFPG's *non-catastrophic* 875 rows
  (610.5) — not LFPG's 3,465.

**No action needed for LSZH/EHAM specifically.** Their gap is the same
broad, already-understood large-delay tail every airport has, moderately
elevated — not a hidden bug worth a dedicated investigation. This closes
out the outlier-concentration audit: LFPG really was the only anomaly.
