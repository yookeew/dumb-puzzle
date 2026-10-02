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
