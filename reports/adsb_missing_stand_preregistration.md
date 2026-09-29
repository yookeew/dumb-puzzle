# Pre-registration: ADS-B detector with empirical positions for stands missing from Gateway

Written 2026-09-29, committed **before** the gate is run. PROGRESS.md §51 Part B
(exploration: `tests/step51_partB_stands_and_parked.py`,
`tests/step51_partB3b_missing_only.py`).

## Change

The stand-appearance detector (`src/link/adsb_pushback.py`) resolves `STAND_mvt`
to X-Plane Scenery Gateway coordinates. Stands with no Gateway coordinates get
no appear/dwell tier. The change supplies **empirical** coordinates for exactly
the **19 stands below**, which are absent from Gateway, and leaves every Gateway
position untouched. Positions come from `data/derived/stands_empirical.csv`
(commit f5432d3, sha256 `23f4dc4d...d40b7ed3`), rows with `keep` and not
`gateway_resolved`.

| airport | STAND_mvt | parked runs n | spread m | lat | lon | nearest Gateway stand (m) |
|---|---|---|---|---|---|---|
| EDDF | V151 | 14 | 4.5 | 50.041700 | 8.561349 | V157 (231) |
| EDDF | V153 | 24 | 1.2 | 50.041571 | 8.560753 | V157 (186) |
| EDDF | V322 | 12 | 13.1 | 50.027305 | 8.569788 | S310 (11) |
| LEMD | 259 | 44 | 5.3 | 40.461880 | -3.560382 | 221 (167) |
| LEMD | 263 | 35 | 1.8 | 40.461873 | -3.561661 | 221 (153) |
| LEMD | T20 | 11 | 8.9 | 40.470316 | -3.567505 | 16 (65) |
| LSZH | 107 | 10 | 0.7 | 47.454312 | 8.572945 | GAC 107 (4) |
| LSZH | 501 | 49 | 2.6 | 47.454966 | 8.569799 | Stand 507 (28) |
| LSZH | F70 | 92 | 3.4 | 47.455044 | 8.567811 | Stand F 75 (37) |
| LSZH | F71 | 126 | 2.8 | 47.455130 | 8.566695 | Stand F 74 (33) |
| LSZH | GA6 | 149 | 33.9 | 47.457667 | 8.573833 | GAC S4 E (62) |
| LSZH | HAN | 20 | 1.3 | 47.453862 | 8.573685 | GAC 101 (5) |
| LSZH | I01 | 153 | 2.6 | 47.455863 | 8.557457 | Stand 101 (8) |
| LSZH | I02 | 196 | 2.9 | 47.455898 | 8.556871 | Stand 102 (9) |
| LSZH | I03 | 152 | 2.6 | 47.455959 | 8.555971 | Stand 103 (9) |
| LSZH | I04 | 137 | 1.6 | 47.455994 | 8.555374 | Stand 104 (9) |
| LSZH | I05 | 155 | 2.7 | 47.456050 | 8.554779 | Stand 105 (7) |
| LSZH | MFG | 35 | 6.0 | 47.454525 | 8.573799 | GAC 304 (57) |
| LSZH | REG | 23 | 3.2 | 47.457935 | 8.572787 | REGA Center (74) |

**How the positions were derived (label-free, all 126 ADS-B days, including the
Jan/Jul 2025 holdout days and the Jan/Jul 2026 ranking days).**
- Take every departure matched to a takeoff-ending ADS-B surface run, using the
  detector's own matching.
- Keep runs whose first sample lies in a stationary segment (gs <= 1 kt,
  >= 180 s, starting within 30 s of the run's first sample).
- The stand position is the median lat/lon of those segments per
  (airport, STAND_mvt). Keep stands with n >= 10 and spread (median distance to
  the median point) <= 40 m.
- Inputs are ADS-B points, STAND_mvt and MVT_TIME only. No label (block time,
  taxi time) is read. Using holdout and ranking days is therefore not label
  leakage, but it is recorded here.

## Deviation from the Part B stop bar

Part B set 5,000 moved ranking rows as the bar for pursuing a change. This one
moves ~2,063 ranking rows into appear/dwell (LSZH 1,678, LEMD 251, EDDF 134;
Jan/Jul 2025: ~2,229). It proceeds anyway because the mechanism is clear (IDs
missing from Gateway, mostly an LSZH renumbering) and the only remaining cost is
one gate run. Expected leaderboard effect is small (~-0.4 by scaling v16's ADS-B
gain).

## Step 1 -- detector re-run and identity check (label-free)

Re-run the detector on all 126 days with the override
(`cache/adsb_pushback_ms/`).
- **Rows at the 19 stands** (normalised STAND_mvt key in the override list, same
  airport) may change freely.
- **Every other row** must be identical in every column to the shipped output:
  `cache/adsb_pushback/` for the 124 Jan/Jul days; for 2025-09-15 and
  2025-11-15, the unchanged detector re-run on the same re-fetched points (§51
  B0).
- **One exception, declared in advance.** The detector matches runs to movements
  greedily and 1:1, preferring runs that visit the movement's own stand. A
  movement at a newly resolved stand can therefore take a run that previously
  went to a neighbour. Exploration (B3b) showed ~19 such rows per year moving out
  of appear/dwell. A non-override row may differ **only if** its previously
  matched run (identified by airport, `adsb_first_ts`, `adsb_n_pts`) is now
  matched to a movement at one of the 19 stands. These rows are counted and
  listed.
- **Stop rule:** any other difference stops the test.

## Step 2 -- gate

- **Arms.** Both arms use the v21 stack base: NNLS of LightGBM + OOF-corrected
  CatBoost (`cache/eval/{lgb,catcorr}_mixed_holdout_ev.parquet`), cross-fit
  Jan<->Jul, identical in both arms. On top, the three ADS-B stages exactly as
  `src/post/stack_submit.py --cat-corrected` fits them: `adsb_blend.fit`
  (per-airport cells), `adsb_blend.fit_quality`, and `adsb_partial.fit` on the
  quality-modulated base. Each stage is fitted on one holdout month and applied
  to the other, in both directions.
  - Baseline arm: shipped detector output.
  - Treatment arm: override detector output.
  - The only difference between the arms is the detector output.
- **Metric.** RMSE on the Jan+Jul 2025 holdout, paired (airport, day) cluster
  bootstrap, 3,000 resamples, seed 0.
  - Trimmed = holdout labels <= 5 h (the fixed set of 31 monster rows excluded;
    assert 31).
  - Full = all rows.
- **Decision rule (§47).** ADOPT iff all of:
  - pooled trimmed delta < 0 with P(worse) < 0.05;
  - Jan trimmed delta < 0 and Jul trimmed delta < 0;
  - full-RMSE guard P(worse) < 0.9.
- **Secondary (not decisive):**
  - the moved rows on their own (~2,229), with both arms' RMSE, per airport;
  - per-airport deltas;
  - how much the refit parameters (lags, cell and tier weights, q factors,
    partial intercepts, slope and band weights) move between arms, both when
    fitted on the full holdout and per cross-fit month.

## If it passes

- Do **not** submit it on its own.
- Stage it: `src/link/adsb_pushback.py` gains an opt-in to write the override
  output, and `src/post/stack_submit.py` gains an option to read it, so the next
  build (v22, together with the LightGBM corrector once that is gated) includes
  it.
- PROGRESS.md records which of v22's changes are included and each one's holdout
  gain, so the leaderboard change can be attributed approximately.

## If it fails

Nothing is staged. The override code stays opt-in and off; the result is
recorded in §51 Part B.
