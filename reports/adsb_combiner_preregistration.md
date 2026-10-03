# Learned ADS-B combiner — pre-registration

Written 2026-10-02, before any combiner was fitted or scored. Changes made
after results exist go under Amendments, dated.

## Why

ADS-B enters v23 through four hand-built linear stages:
- the per-airport-tier blend (§39);
- quality factors (§43);
- the partial-track estimate (§41);
- the gated fallback (§56).

Each uses fixed weights and hand-set cut-offs (1 km, 500 m, 300 s gate).
The board has rewarded every ADS-B improvement (v16, v17, v19, v23).

A small gradient-boosted model can learn, from the same inputs:
- how much to trust an observed pushback given its quality and its
  disagreement with the model;
- the shape of the partial-track relation by distance and speed;
- the gate itself.

That last point lets it also use the v4 detections. v4 adds a fallback for
matched rows that have no appear/dwell tier (§57), with events required to
precede the matched run. On the non-holdout days that gave 656 extra
rows; 177 passed the 300 s gate (58% within ±120 s).

## Definition (`tests/adsb_combiner_test.py`)

- **Detections:** `cache/adsb_pushback_v4/` (v3 plus the matched-row
  fallback).
- **Stack prediction `s`:** NNLS(lgb, catcorr), weights from the fit
  month, as in v21/v23.
- **Target:** `taxi − s`. Train rows: fit month, labels ≤ 5 h, LIRF
  excluded.
- **Inputs:**
  - airport (categorical), hour, `s`;
  - ADS-B tier (categorical: none/appear/dwell/pass/fb_dwell/fb_appear),
    matched flag, fallback flag, day-coverage flag;
  - `adsb_taxi − s`, `adsb_taxi`;
  - pushback distance, speed, gap;
  - `L − s` and `L` (L = T − first sighting);
  - first-sighting distance and speed, closest approach to the stand,
    number of points, takeoff gap.
- **Model:** LightGBM, L2 objective, 31 leaves, lr 0.03,
  min_data_in_leaf 500, lambda_l2 10, feature_fraction 0.9, max_bin 127,
  seed 42, deterministic, force_row_wise.
  - Early stopping (patience 200, cap 5,000) on the fit month's days with
    day-of-month divisible by 5.
  - Then a refit on all fit-month rows at the best iteration.
- **Prediction:** `s + model` on non-LIRF rows. LIRF rows keep the v23
  pipeline value.
- **Cross-fit:** direction A fits on Jan and scores Jul; B the reverse.

**Base:** v23's pipeline cross-fit on the holdout, exactly
`tests/adsb_v3_test.py::pipeline` on v3 detections with the gated
fallback stage.

## Rule

Decision metric: **trimmed RMSE** (labels ≤ 5 h, the fixed 31-row set).
Adopt iff all of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- per airport;
- by ADS-B tier group;
- best iterations;
- feature importance (gain);
- the same combiner on v3 detections (the v4 rows' contribution).

**If it passes:** fit on Jan+Jul 2025 together, rounds set by the same
day-of-month inner split, and apply to the 2026 ranking rows with v4
detections and the production stack. LIRF rows keep v23's value. Then
clip as in production. **Keep iff the board < 269.50 (v23).**

ADS-B coverage moved between 2025 and 2026 (EGLL tiers ~3% → 22–75%). The
inputs are movement-type, never receiver-geography, but the board is the
arbiter.

**If it fails: stop.** No other hyperparameters or input sets.

## Amendments

(none)

## Results (2026-10-02, `tests/adsb_combiner_test.py`, log `logs/adsb_combiner_test.log`; run after commit cb77fe7)

**Primary: ADOPT.**

| trimmed RMSE | base (v23) | combiner | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 291.76 | 289.82 | −1.94 | [−3.06, −0.93] | 0.000 |
| B: Jan (fit Jul) | 221.34 | 217.53 | −3.82 | [−5.17, −2.51] | 0.000 |
| pooled | 262.68 | 260.05 | −2.63 | [−3.49, −1.86] | 0.000 |
| guard: pooled full | 327.02 | 324.94 | −2.08 | [−2.98, −1.35] | 0.000 |

- **Per airport (trimmed):**
  - better: EHAM −16.55, LSZH −10.27, LEBL −9.29, EDDM −9.23,
    EDDF −6.72, LEMD −3.79, EGLL −2.86;
  - **worse: LTFM +8.04, LFPG +0.91** (airports with little or no
    ADS-B).
- **By ADS-B group (trimmed RMSE, v23 → combiner):**

  | group | v23 | combiner |
  |---|---|---|
  | appear/dwell | 162.5 | 146.2 |
  | unmatched fallback | 244.0 | 228.6 |
  | matched fallback (v4) | 258.4 | 250.9 |
  | matched, no pushback | 308.7 | 304.2 |
  | **no ADS-B** | **274.4** | **276.4 (worse)** |

- **Observed flaw (post-hoc, not acted on):** the combiner also sees
  airport, hour and `s`. It learns month-specific biases (LTFM: −39 s in
  Jan vs +46 s in Jul, §53) that don't transfer between months, and
  applies them to rows with no ADS-B. Restricting it to rows with ADS-B
  information is the obvious refinement. Because it is motivated by
  these results, it needs its own pre-registration and board A/B.
- Best iterations 4,614 / 1,008. Top gain: `a_rel` (0.23–0.24), `L_rel`
  (0.17), `s`, first-sighting distance, pushback distance.
- **v3 vs v4 (reported):** the same combiner on v3 detections gives
  pooled −2.49, so the v4 rows add ≈ −0.14.

**v24 build** (`src/post/adsb_combiner.py`, log `logs/adsb_combiner_build.log`).
- Fit on Jan+Jul 2025 (317,809 rows), best_iter 2,873.
- Applied to 2026 with the production stack. LIRF keeps v23's value.
- `data/submissions/smart-jigsaw_v24.parquet`: RMS 75.6 s from v23.
  Per-airport RMS diff 36 s (LTFM) to 121 s (EHAM).
- Keep iff the board < 269.50.

**Board: v24 = 265.66 (v23 269.50, −3.84). ADOPTED; new best.**
