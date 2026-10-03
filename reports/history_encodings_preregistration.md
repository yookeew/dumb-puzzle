# LightGBM history encodings (interaction taxi stats) — pre-registration

Written 2026-10-03, before any of these encodings was computed or fit.
Changes made after results exist go under Amendments, dated.

## Why

- **The tail is unreachable.** The untrimmed decomposition (§68) puts
  36.5% of holdout squared error in 31 rows. 28 points of that are two
  unpredictable LFPG monsters, and the scan for their mechanism failed
  validation (§68).
- **What a model can still move:** NM-matched rows (48.5% of SSE) and the
  base model at LIRF, LTFM and LFPG, the airports with little or no ADS-B
  (~66% of the estimated 2026 error).
- **The base model's label-derived inputs cover only main effects:**
  - in-sample taxi percentiles per stand, per stand-group × runway and per
    runway (`fit_priors`);
  - leave-month-out echo rate and mean delay per (airport, operator) and
    (airport, stand) (`GROUP_ENC_KEYS`).
- **No input carries a historical taxi level for the interactions** that
  set route length and queueing: stand × runway, stand × runway × hour,
  operator × stand, runway × hour. Trees can learn these from the raw
  categoricals in principle, but stand has hundreds of levels per airport
  and the pairs are sparse.
- **Prior from §60:** residual group means by flight number, callsign,
  stand and operator barely moved. These interaction keys weren't tested
  there. Honest expectation: 0.5–2 board points, possibly 0.

## Encodings (`src/features/encode.py`, new block; same discipline as `GROUP_ENC_KEYS`)

- **Groups**, each within the airport:

  | name | keys |
  |---|---|
  | `sr` | stand, runway |
  | `srh` | stand, runway, hour-of-day of T (UTC, 3-hour buckets) |
  | `os` | operator, stand |
  | `rh` | runway, hour-of-day of T (UTC, 1-hour buckets) |

- **Per group:** `{g}_taxi` = mean taxi of normal rows (60 ≤ taxi ≤ 5,400
  s), shrunk toward the airport-runway mean with pseudo-count 30, plus
  `{g}_n` (row count).
- **Fitting is split-blind:**
  - training rows get leave-month-out values (each month from the other
    months, as in `add_group_encodings_oof`);
  - holdout and ranking rows get the full fit-month table;
  - unseen groups fall back to the airport-runway mean with n = 0.
- **Labels never touch a row's own month.** Holdout months never enter a
  fit, and the existing leak guards in `fit_predict_months` hold.

## Stage A — screen (no model fit; decides whether stage B runs)

`tests/history_enc_screen.py`, on the 10 OOF training months only (no
holdout):
- `s` = the OOF stack, NNLS(lgb_mixed, cat_mixed) from `cache/oof/`.
- Residual r = taxi − s, on rows with label ≤ 5 h.
- For each group, fit the shrunk mean residual (pseudo-count 30) on even
  months and apply to odd, then the reverse. Also all four groups
  additively, fitted in sequence.

**Pass iff** the four-group combination lowers residual RMS by ≥ 0.5% in
**both** directions (even → odd and odd → even). Otherwise **stop**: no
fits, no submission.

Reported, not decisive: each group alone; per airport (LIRF, LTFM, LFPG
separately).

## Stage B — holdout gate (only if A passes)

`tests/history_enc_test.py`:
- **Treatment:** LightGBM `target="mixed"` with the eight new columns
  added, through `fit_predict_months` on the 10 training months; predict
  the Jan+Jul 2025 holdout.
- **Base:** the current lgb_mixed holdout predictions.
- **Both arms:** same seed and settings; stacked with the existing catcorr
  via NNLS, cross-fit Jan ↔ Jul (weights fit on one month, scored on the
  other).
- **Scored at the stack level, before ADS-B stages.** The combiner is
  retrained only in stage C.

**Adopt iff all of:**
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

Reported, not decisive: per airport; best iterations; gain share of the
new columns.

## Stage C — build and board (only if B passes)

- `run_oof(engine="lgb", target="mixed")` with the new features →
  `cache/oof/lgb_mixed_hist/`.
- Production fit on all 2025.
- Rebuild v27's chain with the new lgb in place of the old: NNLS stack,
  v23 ADS-B stages, full-year combiner retrained on the new OOF stack.
  This is v29.
- **Keep iff the board < 262.13 (v27).**
- One submission. If it fails, no variants (no subsets of groups, no
  re-tuned pseudo-counts).

CatBoost isn't changed here; cat_ctr2 stays separate, pending the L4.

## Amendments

(none)
