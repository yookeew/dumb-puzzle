# ADS-B combiner v2 (ADS-B rows only, regularised) — pre-registration

Written 2026-10-02, after the v24 combiner (holdout −2.63 trimmed, board
265.66) and before any v2 fit. Changes made after results exist go under
Amendments, dated.

## Why

The v24 gate showed two things:
- **Harm where there's no ADS-B.** Rows without ADS-B information got
  worse (274.4 → 276.4). The combiner learned month-specific airport/hour
  biases there that don't transfer (§58).
- **Unstable fits.** Best iterations differed 4.6× between directions
  (4,614 fitting on Jan, 1,008 fitting on Jul). That's a sign of
  overfitting month-specific structure.

## The change (one variant)

Same inputs, target, early-stopping split and cross-fit as v24
(`tests/adsb_combiner_test.py`), with two changes:
1. **Population:** fit and apply only on rows with ADS-B information
   (`adsb_matched`, or tier `fb_dwell` / `fb_appear`). All other rows keep
   the v23 pipeline value. LIRF is excluded as before.
2. **Regularisation:** num_leaves 15 (was 31), min_data_in_leaf 2,000
   (500), lambda_l2 50 (10), feature_fraction 0.8 (0.9). Everything else
   is unchanged: lr 0.03, max_bin 127, seed 42, deterministic, patience
   200, cap 5,000.

## Base and evaluation (`tests/adsb_combiner_v2_test.py`)

**Base = v24's combiner, restricted the same way:** the cross-fit v24
combiner on rows with ADS-B information, the v23 pipeline value elsewhere.
Both arms then agree on every row without ADS-B. The comparison is only
over ADS-B rows, whose v2 outcome is unknown. This avoids re-scoring the
"no ADS-B" group, which was already seen.

Rule (trimmed RMSE decides, labels ≤ 5 h):
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- per airport;
- best iterations per direction (is the 4.6× gap smaller?);
- top feature gains;
- v2 vs the unrestricted v24 combiner.

**If it passes:** build v26 = v2 fit on Jan+Jul 2025 and applied to 2026
rows with ADS-B information (v23 value elsewhere, LIRF too). **Keep iff
the board beats the best score at build time** (v24 265.66, or v25 if
better).

**If it fails: stop.** No further combiner hyperparameter search.

## Amendments

(none)

## Results (2026-10-02, `tests/adsb_combiner_v2_test.py`, log `logs/adsb_combiner_v2_test.log`; run after commit e1fe6a9)

**Primary: ADOPT, narrowly.** v2 vs the v24 combiner restricted to ADS-B
rows (159,451 holdout rows with ADS-B information):

| trimmed RMSE | base | v2 | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 288.41 | 288.23 | −0.19 | [−0.35, −0.03] | 0.010 |
| B: Jan (fit Jul) | 216.84 | 216.82 | −0.02 | [−0.14, +0.10] | 0.376 |
| pooled | 258.93 | 258.81 | −0.12 | [−0.23, −0.02] | 0.013 |
| guard: pooled full | 324.01 | 323.92 | −0.10 | [−0.19, −0.01] | 0.014 |

- **Per airport:** EDDF −0.79, EGLL −0.48, EHAM −0.46, LFPG −0.22,
  LEMD −0.07; EDDM +0.86, LSZH +0.36, LEBL +0.11.
- **Best iterations 4,999 / 3,014.** Regularisation narrowed the gap
  (v24: 4,614 / 1,008) but the Jan fit now hits the cap.
- **Reported:** v2 vs the unrestricted v24 combiner is pooled −1.24, most
  of it from the restriction (§58, v25).

**v26 build** (`src/post/adsb_combiner.py --v2`, log
`logs/adsb_combiner_v2_build.log`).
- Fit on 148,157 Jan+Jul ADS-B rows; best_iter 5,000 (the cap).
- `data/submissions/smart-jigsaw_v26.parquet`: 196,621 rows differ from
  v25, RMS 21.8 s.
- Keep iff the board < 264.39 (v25).
