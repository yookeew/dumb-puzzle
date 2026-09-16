# Progress log — 2026-09-15

Context: local holdout RMSE was consistently ~200s lower than the actual
ranking RMSE returned on submission (309s local vs 508s actual on
`smart-jigsaw_v5`). This session chased that gap, fixed one real bug along
the way (which turned out not to be the cause), ruled out two more
hypotheses, landed a real (if modest) win from a robust loss, then found a
second real bug that looks much more consequential than the first.

## 1. Bug found and fixed: holdout couldn't see the Jan→Jul gap

`ranking.parquet` bundles Jan 2026 + Jul 2026 with a 5-month hole in
between — no Feb–Jun 2026 data exists anywhere. Two places assumed a
continuous timeline instead:

- **`mins_since_cfg_change`** (`src/features/build_features.py`) detected a
  runway-config change by comparing each 5-min bin to the previous one. Fed
  the whole `ranking.parquet` frame at once, the last bin of January and the
  first bin of July become "adjacent," so if the config string happened to
  match across the gap, no change was detected and the feature measured
  minutes-since-change back into January — values that can't occur anywhere
  in a continuous 12-month training file.
  **Fix:** `_chg` now also fires whenever the bin-to-bin gap exceeds
  `CFG_GAP_RESET_MIN` (24h), regardless of whether the config string changed.

- **The holdout eval itself** (`src/models/train_lgbm.py`, `src/models/fit.py`)
  built features on the *continuous* 12-month `train2025.parquet` frame and
  only sliced out Jan+Jul *after* the fact — so the held-out rows never saw
  a gap at all, meaning this whole class of bug was invisible to local
  validation no matter how good the holdout RMSE looked.
  **Fix:** `export_model_inputs.py` now also emits an isolated
  `cache/features/holdout_gap2025.parquet` (just Jan+Jul 2025 raw rows,
  features built on that two-month frame alone), and both training scripts
  use it for `f_ho` instead of slicing `train2025.parquet`.

**Result: this was real and worth fixing, but it was not the cause of the
gap.** Realistic holdout RMSE barely moved (300.9s → 308.7s in
`train_lgbm.py`; 308.9s → 309.2s in `fit.py`), and submitting the fixed
model (`smart-jigsaw_v5`) still scored 508s — same ~200s gap as before the
fix.

## 2. Two more hypotheses checked and ruled out

- **Categorical vocabulary shift.** Checked whether stands/operators/runways
  in `ranking.parquet` (2026) exist in the 2025 training vocabulary (unseen
  categories fall back to LightGBM's missing-value handling). Unseen rows:
  `STAND_mvt` 0.08%, `AIRCRAFT_OPERATOR_flt` 0.05%, `RUNWAY_mvt` 0.00%,
  `AIRCRAFT_TYPE_mvt` 0.01%. Negligible — not the cause.
- **Planned/actual NM timestamp fields.** `EOBT_1_flt`, `IOBT_flt`,
  `LOBT_flt`, `AOBT_3_flt` feed 4 of the top-6 features by gain
  (`eobt_delay`, `iobt_delay`, `lobt_delay`, `aobt3_vs_eobt`, `aobt3_taxi`).
  Checked null rates and medians between train (2025) and ranking (2026):
  null rates 1.08% vs 1.53%, medians nearly identical (e.g. `aobt3_taxi`
  958s vs 978s). No meaningful mismatch — not the cause.

## 3. Robust loss experiment: Huber on `d`

Motivation: plain L2 lets a handful of huge-residual training rows (LIRF,
the d9 tail) pull tree splits toward fitting them exactly. Huber caps the
gradient contribution beyond a residual threshold `alpha` (seconds),
behaving exactly like L2 below it.

| attempt | alpha | rounds ceiling | best_iter | overall RMSE | outcome |
|---|---|---|---|---|---|
| baseline (L2) | — | 8000 | 2192 | 309.2s | reference |
| huber v1 | 400 | 2500 | 2500 (hit ceiling) | 383.9s | **FAILED** — undertrained, worse on every airport/decile incl. the tail |
| huber v2 | 400 | 8000 | 8000 (hit ceiling) | 315.6s | **FAILED** — still never converged; tail unchanged (679.8 vs 677.0), EDDF regressed +67s |
| huber v3 | 800 | 8000 | 7998 (~converged) | **303.1s** | **SUCCEEDED** — adopted as new default |

`alpha=800` improved 6/10 airports (EHAM -15.2s, LIRF -13.7s, EDDF -13.3s,
LTFM -11.3s, EGLL -7.1s, LSZH -4.6s) with only small regressions on the
other 4 (all ≤5.2s), and the tail improved too (d9: 667.7 vs 677.0s). Not a
tail-vs-everything-else tradeoff — a broad, modest, genuine win (-6.1s
overall, ~2%). Note it still hadn't fully plateaued at the 8000-round local
ceiling (`best_iter=7998/8000`) — worth pushing further on Colab/GPU where
iteration is cheap.

Submitted as `smart-jigsaw_v6`: **303.1s local → 501s actual.** Gap barely
moved from the L2 baseline's 309.2s local → 508s actual. A 2% local gain is
real but nowhere near enough to close a ~200s gap on its own.

## 4. Falsified: the gap is not a fixed "noise floor"

After two submissions both landing ~200s above their local number, tried
modeling it as `actual² = local² + B²` (constant hidden variance B,
independent of model quality):

- v5: local 309.2 → actual 508 → implied B = 403.1s
- v6: local 303.1 → actual 501 → implied B = 398.9s

That's a suspiciously tight match (within 1%) across two structurally
different models (different loss functions, measurably different tail
behavior). **But this theory is falsified by the leaderboard itself**: the
current #1 score is ~245s. If B were a real, model-independent floor, no
submission could score below B (since `actual = sqrt(local² + B²) ≥ B`
always), and 245 < 400. So whatever produces this gap is not immutable
noise in the 2026 labels — it's specific to blind spots in *our current
feature set / approach*, which a fundamentally different approach can
clearly get past. Corrected conclusion: closing a gap this size needs a
difference in *kind* of information or model, not incremental feature
tuning (weather, OSM distance) — those are proven single-digit-percent
levers (see §3), not 2x levers.

## 5. Input from other contestants (via forum/Discord, read 2026-09-15)

Two contestants posted detailed writeups of the exact same
CV-improves-but-board-doesn't-follow experience. Their findings, folded
into our own process:

- **Check Jan and Jul separately, require both to improve** — a pooled
  number hides month-specific regressions. (One of them lost -7.6s on the
  board from a change their CV said was +43s on July at p<0.001 — a stark
  warning against trusting CV signal alone, however strong it looks.)
- **Split error by `AOBT_3_flt`-present vs missing lane** — the missing
  lane is a small population that carries a disproportionate share of
  squared error and behaves differently across years. This directly
  cross-validated our own finding in §6 below before we'd even run the
  numbers ourselves.
- **Statistical significance, not point estimates** — require multi-seed
  confirmation or a permutation control before trusting any gain.
  Their hyperparameter-tuning "wins" (apparent 12-21s from single-seed
  search) turned out p=0.06-0.18 under multi-seed testing — not real.
- **Reject gains concentrated in <100 rows** — usually noise, not signal.
- **Compare every new feature's 2025-holdout vs 2026-ranking distribution
  before training on it.** They lost 3s once to a count feature whose scale
  depended on how many months of history a row had — the same class of bug
  as our `mins_since_cfg_change` fix (§1), and they credit computing
  rolling/congestion counts *inside `(airport, month)` partitions* for
  keeping their 52 features stable (<0.25 sd shift) between 2025 and 2026.
- **Ensemble blend weights fit on a holdout backfired 4/4 times** — equal
  weights won every time. Relevant for our own not-yet-built ensemble step.
- **Hyperparameter tuning was not significant** for LightGBM, CatBoost, or
  XGBoost once tested across multiple seeds.
- **Weighting training rows toward the scored (airport, month) cells** and
  **adding a third blend member (XGBoost)** both did nothing for them.
- **A real bug they found and fixed:** a weather feature was silently null
  for 19 submissions because the join key was timezone-aware on one side
  and naive on the other, and the join exception never surfaced. Fixed for
  a 3.4s gain. Directly informs our own upcoming weather ingest: add an
  explicit null-rate/variance assertion right after any new join, and
  watch tz-aware-vs-naive datetime mismatches specifically.
- **Temperature/dewpoint spread beat a binary snow flag by a distance** —
  most de-icing happens on clear frosty mornings with no precipitation
  code. Prioritize this feature shape when we build weather features.

**Integrity flag, not adopted:** one poster described a mechanism where a
tiny number of high-leverage rows (see §6) let a team extract exact hidden
labels via differential submissions (change one row, read the label off
the score delta; a handful of submissions a day resolves all of them
within days), and attributes the board's "flat plateau then one big jump"
pattern to this. **We are not pursuing this.** It also means some fraction
of the gap to top leaderboard scores may reflect this rather than
legitimate modeling — worth keeping in mind so we're chasing a realistic
target, not necessarily parity with a score that may be partly gamed.

## 6. Found: our own eval was hiding a real, fixable failure mode

Prompted by the contestants' "check the `AOBT_3_flt` lane separately"
lesson, checked our own data directly instead of just taking the general
advice on faith.

**Bug 1 — our local eval silently excludes the rows that matter most.**
`fit.py`/`train_lgbm.py`'s holdout scoring filtered
`taxi.is_between(0, 4*3600)` before computing RMSE. Recomputing the Jan+Jul
2025 holdout *without* that exclusion (scoring the excluded rows at
whatever our CEIL clip would produce): **local RMSE goes from 303.1s to
385.5s from just 45 rows out of 344,419 (0.013%).** The real board almost
certainly has no such exclusion, so our reported local number has been
optimistic in a way that has nothing to do with model quality.

**Bug 2 — those rows aren't random, and the "fix" already existed in our
architecture, just clipped away.** 43 of those 45 rows (95.6%) are in the
`AOBT_3_flt`-missing lane, which is only 1.56% of all rows but has a naive
RMSE of 3,987s vs 484s for the rest (8x). Investigating *why*: 101 of 119
such rows (85%, year-wide) have `BLOCK_TIME_UTC_mvt ≈ SCHED_TIME_UTC_mvt`
(within 30s) — the same "echo" signature as the known LIRF phenomenon
(`reports/lirf_investigation.md`). Since `taxi = MVT_TIME − BLOCK_TIME` by
definition, an echo row's true taxi ≈ `MVT_TIME − SCHED_TIME`, i.e. exactly
our existing `sched_takeoff_offset` feature — fully observable, never
blanked, and already the #1 feature by gain. Our architecture already
predicts `taxi_hat = sched_takeoff_offset − d_hat`; if the model recognizes
an echo (it has `echo_rate` group-encoding features for this) and predicts
`d_hat ≈ 0`, the reconstruction is correct *however large*
`sched_takeoff_offset` is — one row in the data had taxi = 84,240s from
exactly this mechanism. **The `CEIL=10800` clip on the final prediction was
throwing this correct answer away**, capping an 84,000s reconstruction down
to 10,800s. This is a pure own-goal, not a modeling limitation.

**Fix, part 1 (kept):**
- Removed the `taxi < 4h` exclusion from holdout scoring in both training
  scripts (score all rows with `taxi >= 0`; only drop physically impossible
  negative/null labels).
- Added a `has_aobt3` lane breakdown to `_report`/`_write_report`
  (`fit.py`) and to the shared `report()` (`src/eval/holdout.py`), showing
  RMSE and % of total squared error per lane.

**Fix, part 2 (tried, FAILED, reverted):** raised `CEIL` from 10800s to
140000s on the theory that this only changes behavior for rows that were
being clipped anyway. **Wrong — this was a net regression.** Validated
(`lgb_colab_ceilfix`, huber alpha=800 + both fixes): overall holdout RMSE
went **303.1s → 560.7s**, and the damage wasn't confined to the intended
lane — several majority-lane airports got wrecked (EDDF 290s → 1036s, LFPG
→ 709s, EHAM → 670s), while the `has_aobt3=False` lane itself only reached
3365.6s (barely better than a naive per-lane median baseline of 3987s).
Only **1** row actually hit the new 140000 ceiling in the whole holdout —
the regression came from a small number of *unrelated* rows getting a
wildly-extrapolated `d_hat` that used to be safely capped at 10800.
Conclusion: `CEIL=10800` was quietly acting as a general safety backstop
against LightGBM's cumulative-tree extrapolation instability for far more
rows than just the genuine echo cases, not a targeted block on a specific
mechanism. **Reverted `CEIL` to 10800 in both files.** A real fix for the
echo lane needs to flag likely-echo rows specifically (e.g. high
`echo_rate` + `has_aobt3=False`) and only relax the ceiling for those, not
raise it globally — not yet built.

**Result — this is the headline finding of the session.** Honest baseline
(`lgb_colab_honest`, eval-filter fix kept, CEIL back at safe 10800):
**overall holdout RMSE = 522.6s.** Compare to `smart-jigsaw_v6`'s actual
board score of 501s — the two are now essentially the same number (the
~20s difference is well within run-to-run noise for a ~5,300-row lane).

**The ~200s "local-vs-actual" gap that drove this entire session was
almost entirely our own eval hiding real error, not a real 2025-vs-2026
generalization problem.** Every prior theory this session entertained to
explain it (§4's hidden noise floor, tail/LIRF year-over-year drift,
adaptive overfitting to the fixed holdout) was an attempt to explain a gap
that mostly didn't need a exotic explanation — once the eval stopped lying
to us, local and actual converged on their own. `has_aobt3=False` now
shows **71.1% of total squared error from 1.5% of rows** (even more
concentrated than the flawed ceiling-raise run showed); LIRF's honest RMSE
is 1509s and LFPG's is 602s — both were quietly hidden by the exclusion
bug the whole time, not "well-behaved" or "genuinely resistant to modeling"
as earlier reports implied.

This reframes the project: we now have a trustworthy local metric, and the
single highest-leverage, best-quantified target going forward is a
**targeted** (not blanket) fix for the `has_aobt3=False` lane — flag
likely-echo rows specifically (high `echo_rate` + `has_aobt3=False`) and
only widen the prediction range for those rows, leaving `CEIL=10800` in
place as the safety net for everyone else.

## 7. Targeted echo-lane fix — built and validated (net win)

Implemented `_reconstruct_taxi()` (`fit.py`): for the 98.5% of rows with
`AOBT_3_flt` present, behavior is byte-for-byte identical to before
(`clip(offset - d_hat, 0, 10800)`). Only for rows missing `AOBT_3_flt` does
it blend toward the echo hypothesis `taxi = offset` (uncapped up to a
140000s sanity bound), weighted by that row's operator's historical
`op_echo_rate` — a row from an operator that echoes 80% of the time gets
pulled 80% of the way to the offset reconstruction; a row from an operator
that never echoes stays ~100% on the ordinary model prediction. This is
deliberately soft (no hard threshold) and gated so narrowly that it cannot
reproduce the earlier blanket-CEIL disaster.

Validated holdout-only (`run(..., submit=False)`, same isolated Jan+Jul
frame, same huber alpha=800 model) against the honest baseline (§6, 522.6s):

| | overall | `has_aobt3=True` lane | `has_aobt3=False` lane | LIRF |
|---|---|---|---|---|
| honest baseline | 522.6s | 283.1s | 3544.9s | 1509.0s |
| **+ echo-lane blend** | **484.6s** | **283.1s (unchanged)** | 3176.0s | 1320.1s |

**Net -38s (~7.3%) overall, majority lane exactly untouched, no airport
blew up.** A few airports with few affected rows (EDDF, EHAM, LEBL) ticked
up ~10-12s each — their echo-rate estimates apparently don't discriminate
as well as LIRF's does — but nowhere near enough to offset the net win.
Predicted max landed at 25,998s, well under the 140000 sanity bound,
confirming the blend behaves conservatively rather than swinging wildly.

**Status:** running the full pipeline now (`submit=True`) to produce a real
submission and confirm this transfers to the actual board — local now
tracks actual closely (§6), so this should be a trustworthy signal.

## Code changes

- `src/features/build_features.py` — `CFG_GAP_RESET_MIN` (24h) gap-aware
  reset in `_runway_config`'s `mins_since_cfg_change`.
- `src/features/export_model_inputs.py` — emits
  `cache/features/holdout_gap2025.parquet` (isolated Jan+Jul 2025 frame).
- `src/models/train_lgbm.py`:
  - Holdout built from the isolated frame instead of sliced from the
    continuous one. (Simpler/faster local sanity-check path; `fit.py` is
    the actual production pipeline.)
  - Holdout eval no longer excludes `taxi > 4h`; added `has_aobt3` lane
    column. `CEIL` briefly raised to 140000, reverted back to 10800 (§6 —
    net regression).
- `src/models/fit.py`:
  - Same isolated-holdout-frame fix as `train_lgbm.py`.
  - `loss` / `huber_alpha` params threaded through `run()`, `_fit_lgb`,
    `_fit_xgb` — training objective switches between L2 and Huber per
    engine; `metric` stays `rmse` always, so early stopping and reporting
    are unaffected by the objective choice.
  - Defaults: `loss="huber"`, `huber_alpha=800.0`, `ROUNDS=8000` (raised
    back from a stray `2500` that would have undertrained Huber badly —
    see huber v1 in §3).
  - Holdout eval no longer excludes `taxi > 4h`; added `has_aobt3` lane to
    `_report`/`_write_report` (RMSE + % of total squared error per lane).
    `CEIL` briefly raised to 140000, reverted back to 10800 (§6 — net
    regression).
  - `_reconstruct_taxi()` (§7): targeted echo-lane blend, gated to rows
    missing `AOBT_3_flt`, weighted by `op_echo_rate`. Validated net win
    (-38s holdout, majority lane untouched).
- `src/eval/holdout.py` — shared `report()` now also breaks down by
  `has_aobt3` when present, with % of total squared error per lane.
- `notebooks/colab_train.py` — noted the Huber default and that it's worth
  raising `ROUNDS` further on Colab/GPU.
- `run()` (`fit.py`) gained a `submit: bool` flag — `submit=False` skips the
  refit-on-all-2025 + ranking-predict + write-submission phase, so a
  holdout-only validation run takes ~15 min instead of ~40.

## Submissions this session

- `smart-jigsaw_v5` — gap-fix only (L2). 309.2s local (old, filtered metric)
  / **508s actual**.
- `smart-jigsaw_v6` — Huber alpha=800. 303.1s local (old, filtered metric)
  / **501s actual**.
- `lgb_colab_ceilfix` — Huber alpha=800 + eval-filter fix + CEIL raised to
  140000. **Not submitted** — 560.7s local, a regression (§6).
- `lgb_colab_honest` (holdout-only, no submission file produced) — Huber
  alpha=800 + eval-filter fix, CEIL back at 10800. **522.6s local — matches
  `smart-jigsaw_v6`'s 501s actual almost exactly.** This is the number to
  compare all future changes against.
- `lgb_colab_echolane` — targeted echo-lane blend (§7). **484.6s local**
  (matches the earlier holdout-only validation exactly — deterministic).
  Full pipeline completed, submission file written:
  `data/submissions/lgb_colab_echolane.parquet` (344,841 rows, pred range
  0-29,056s). **Not yet renamed to `smart-jigsaw_vN` or uploaded** — ready
  to go, waiting on a decision to actually submit it.

## Next steps

1. **Confirm the echo-lane fix transfers to the real board** — full pipeline
   running now (§7); submit the result once it lands and compare against
   501s. With local now tracking actual closely, this should be a reliable
   signal for the first time this session.
2. If it transfers, look at whether the operator-echo-rate blend can be
   improved for the airports that ticked up slightly (EDDF, EHAM, LEBL) —
   e.g. stand-level echo rate instead of/alongside operator-level, or a
   per-airport weight, rather than treating all `has_aobt3=False` rows
   identically.
3. Adopt the contestants' eval discipline going forward (see §5): Jan/Jul
   both must improve, `has_aobt3` lane always reported, 2025-vs-2026
   feature distribution check before trusting any new feature, multi-seed
   significance check before trusting any gain, ignore gains from <100 rows.
4. Ask on OSN Discord whether OpenSky ADS-B ground trajectories are
   permitted — only for training-side feature engineering (2025), never
   for reconstructing the blanked 2026 ranking labels (that would be
   leakage, not prediction). Potentially the one lever big enough to matter
   if the `AOBT_3_flt` fix doesn't fully close the gap.
5. Start scoping Stage 3b (queue refinement) — a structurally different
   modeling approach, not just another feature family.
6. Weather ingest (NOAA ISD) — still worth doing, but recalibrated to
   "minor lever" status (§4): expect low single digits, not a fix for the
   remaining gap. When we do it: prioritize temperature/dewpoint spread
   over a binary snow flag, and add an explicit null-rate/variance
   assertion after the join (§5).
7. OSM routed taxi distance (Future Step B) — still deprioritized.
