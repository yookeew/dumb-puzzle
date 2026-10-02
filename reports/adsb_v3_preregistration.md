# ADS-B v3: inferred stands + fallback match — pre-registration

Written 2026-10-02, before any holdout (Jan/Jul 2025) result with the v3
detections. The design was explored only on 2025-09-15 and 2025-11-15
(training-month days, outside the holdout) and on 2026 ranking days without
labels. Changes after results go under Amendments, dated.

## Why

- Every 2026 day has ADS-B coverage at EDDF, EGLL, LEMD and LSZH, yet 42%,
  22%, 32% and 15% of their departures match no track (PROGRESS.md §56).
  The match needs the takeoff roll in the track (a surface run followed by
  gs >= 40 kt within 120 s, takeoff within ±180 s of MVT_TIME). Many tracks
  lose the aircraft before that, even when it was seen at its stand.
- 9,455 ranking rows have a matched track but no stand coordinates, so no
  pushback can be read.
- Observed-pushback rows score far better than the rest (holdout RMSE
  90–140 vs 240–320 at the same airports), and ADS-B stages have carried
  to the board at 2–4× their holdout gain (v16, v17, v19).

## What changed (`src/link/adsb_pushback.py --v3` → `cache/adsb_pushback_v3/`)

Split-blind like the production detector: it reads only takeoff time,
STAND_mvt and adsb.lol points.

1. **Inferred stands** (`src/ingest/stand_infer.py` →
   `data/external/stands_inferred.csv`): for stands missing from the
   Gateway table, the median position of matched runs' stationary first
   samples (n >= 3, median spread <= 30 m; LFPG excluded).
   - Validation on 848 Gateway stands: median 7 m from the Gateway
     position, 97% within 100 m.
   - 50 stands are added (LSZH 22, LEMD 13, EDDF 11, EGLL 3, LEBL 1).
   - Their rows get ordinary `appear`/`dwell` tiers.
2. **Fallback match** for departures with no matched run: the most recent
   at-stand event of any aircraft in [T − 90 min, T − 2 min].
   - `fb_dwell`: last stationary at-stand sample, then the next sample.
   - `fb_appear`: the first sample after a > 20 min silence is at the
     stand.
   - The aircraft must then be seen leaving. Events are used once and must
     follow the previous departure from that stand.
   - Tiers `fb_dwell` / `fb_appear`, flag `adsb_fallback`.

Existing detections are unchanged except 34 of 159,138 (run reassignment at
inferred stands).

**Scoping (non-holdout days, honest OOF base, `tests/adsb_v3_scope.py`):**
- Inferred-stand rows (n=74): ADS-B RMSE 153 vs base 157; in-sample LS
  weight 0.52 (157 → 133).
- Fallback rows (n=391): median error ~0, but heavy-tailed (RMSE ~1,200;
  wrong events). Wrong events have implied taxi far from the base
  prediction.
- With the gate |ADS-B taxi − lag − base| <= 300 s: n=191, 65–69% within
  ±120 s of the label, base 142 → 123 at w ≈ 0.5. All rows: −0.25 s.
- The 300 s gate was chosen on these two days.

## Treatment and base

**Base:** v21 cross-fit on the holdout, exactly
`tests/ltfm_egll_anatomy.py::v21_pred` (production detector cache).

**Treatment:** the same pipeline on `cache/adsb_pushback_v3/`:
NNLS(lgb, catcorr) → quality-modulated ADS-B blend (`appear`/`dwell`,
which now include the inferred-stand rows) → partial-track estimate. Then
one new stage:
- **Fallback blend:** on rows with tier `fb_dwell` / `fb_appear`, LIRF
  excluded, and `|c − base| <= 300 s` where `c = adsb_taxi − lag_airport`.
  - `lag_airport` is the main blend's lag from the same fit month.
  - `final = base + w_tier · (c − base)`, with `w_tier` fit by least
    squares per fallback tier on the fit month, clipped to [0, 1].
- Every parameter is fit on the other month (Jan → Jul, Jul → Jan).

## Rule

Decision metric: **trimmed RMSE** (holdout labels <= 5 h, the fixed
31-row set), per the §47 convention. Adopt iff all of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- The two parts separately: inferred stands only; fallback only.
- Fallback without the gate.
- Fitted weights and lags; per-airport deltas; row counts per tier and
  month.

**If it passes:** build the next submission = v21's pipeline on v3
detections plus the fallback stage, parameters fit on Jan+Jul together.
**Keep iff the board < 270.2.**

**If it fails: stop.** No other gate width, window or tier split.

## Amendments

(none)

## Results (2026-10-02, `tests/adsb_v3_test.py`, log `logs/adsb_v3_test.log`; run after commit 81ff1a0)

**Primary: ADOPT.**

| trimmed RMSE | base (v21) | v3 | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 292.24 | 291.76 | −0.48 | [−0.64, −0.36] | 0.000 |
| B: Jan (fit Jul) | 222.22 | 221.34 | −0.88 | [−1.09, −0.68] | 0.000 |
| pooled | 263.30 | 262.68 | −0.62 | [−0.75, −0.51] | 0.000 |
| guard: pooled full | 327.52 | 327.02 | −0.50 | [−0.66, −0.38] | 0.000 |

- **Per airport (trimmed):** LSZH −6.44, EGLL −1.26, EDDF −1.18,
  LEMD −0.76, LEBL −0.23, EDDM −0.08, EHAM +0.02. LFPG, LIRF and LTFM
  are unchanged.
- **Fallback weights:** fb_dwell 0.36 / 0.47, fb_appear 0.76 / 0.79 (fit
  Jan / Jul). About 4,150 gated rows applied per month.
- **The parts, reported:**
  - inferred stands only: −0.30 (P 0.000);
  - fallback only, gated: −0.35 (P 0.000);
  - fallback ungated: weights ≈ 0.01, total −0.31, the same as inferred
    stands alone. The gate carries all of the fallback's value.

**Build** (`src/post/stack_submit.py --cat-corrected --adsb-v3`).
- The default path still reproduces the uploaded v21 byte for byte
  (checked before the change).
- Fallback weights fit on Jan+Jul: fb_dwell 0.41, fb_appear 0.77.
- 2026: 96,877 blended rows, plus 5,730 gated fallback rows (EGLL 1,870,
  LEMD 1,718, LSZH 830, EDDF 646, LEBL 513).
- `data/submissions/smart-jigsaw_v23.parquet`: 89,591 rows differ from
  v21, RMS 19.4 s.
- Board A/B pending: keep iff < 270.2.

**Board: v23 = 269.50 (v21 270.2, −0.70). ADOPTED; new best.**
