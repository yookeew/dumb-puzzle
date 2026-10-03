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

**Status: CONFIRMED.** Submitted (as `smart-jigsaw_v7`, by file timestamp —
generated 12:21, matches the echo-lane run exactly): **484.6s local → 460s
actual**, a real **-41s** board improvement (matching the -38s local
prediction almost exactly, slightly beating it). This is the first time all
session that a local improvement transferred to the board at close to face
value — direct confirmation that the local metric is now trustworthy (§6)
and that this fix is real, not an artifact of the holdout split.

(Note: `data/submissions/smart-jigsaw_v8.parquet` also exists, generated
14:03 — before the §8 classifier work below started. Not something this
session produced; flagging since its provenance isn't in this log.)

## 8. LIRF still dominant after the echo-lane fix — moving to a proper P(echo) classifier

Even after §7's fix, LIRF sits at **1320s RMSE vs 200-300s everywhere else**
— a 4-6x gap that dwarfs every other remaining lever. Re-read
`reports/lirf_investigation.md` (prior session's work) to ground the next
step instead of re-deriving from scratch, per the discipline the CEIL-raise
mistake in §6 taught us.

**Why the echo-lane fix barely dented LIRF specifically:** that document
found LIRF echoes (`BLOCK_TIME ≈ SCHED_TIME`) on **18.1%** of its rows —
but §7's fix only ever fires on rows *missing `AOBT_3_flt`*, which is only
~1.5% of rows overall (and a similarly small slice of LIRF). The other
~16.6% of LIRF's departures echo the schedule while still *having* a valid
`AOBT_3_flt` value, so they sail straight past the gate and get the
ordinary `clip(offset - d_hat, 0, 10800)` treatment — wrong in exactly the
way the fix was built to catch, just outside its narrow trigger condition.
The investigation doc's own long-term recommendation was: *"an explicit
echo probability feature ... and possibly a two-part model (P(echo) ×
schedule-delay-taxi + (1−P) × normal-taxi)"* — which §7 approximates in
spirit (single-signal, hard-gated) but not in the form actually recommended
(multi-signal, continuous, ungated).

**Decision (discussed with user):** build vs. a separate per-airport LIRF
model — rejected. LIRF's error is a *mixture* of two processes (schedule-echo
artifact + genuine congestion tail), and a full separate model would have to
re-learn everything (distance, aircraft-type, congestion effects) from
LIRF's own data alone, losing the cross-airport pooling the global model
gets from the other 9 airports. This was already tried once in a different
form — a per-airport residual head (`reports/stage3a_resid.md`) — and was a
regression. A classifier targeting the specific mixture is the narrower,
already-validated mechanism (§7 already works this way) extended to have
much better recall.

**Built:** `fit_echo_classifier()` (`fit.py`) — an OOF-style binary
LightGBM classifier predicting `is_echo = |d| < 30s`, trained on the *same*
feature matrix as the `d`-regressor (so it automatically has access to
every signal the investigation doc flagged as predictive — missing
AOBT_3/EOBT_1/FLIGHT_ID shows up as nulls in existing features, plus IOBT
delay, hour, operator, stand — no new feature engineering needed). Its
output replaces the `has_aobt3`-gated `op_echo_rate` weight in
`_reconstruct_taxi`: now `taxi = P(echo)·offset + (1−P(echo))·model_pred`
for **every** row, not just the missing-`AOBT_3_flt` lane.

Trained on the training split only (mirrors the main model's train/valid
split), with one important difference: **not** filtered by the
`[LABEL_LO, LABEL_HI]` taxi-range mask the `d`-regressor uses — an echo's
`d` is near zero (often *below* `LABEL_LO=30`), so that filter would starve
the classifier of positives. Refit on all-2025 data for the final ranking
submission the same way `model_a` is.

**Risk, held in mind given §6's CEIL-raise mistake:** this opens the blend
to a much larger population (≈18% of LIRF vs ≈1.5% before), so a
false-positive-prone classifier could inject real damage on genuinely
non-echo rows within that larger population. Mitigated by: (a) it's a soft
probability blend, not a hard gate, (b) added a precision/recall sanity
print and an `echo_pred` (P>0.5) lane to `_report`/`_write_report` so this
is checked *before* trusting any RMSE number, matching the same
validate-before-trust discipline that caught the CEIL regression.

**Status: VALIDATED — large, clean win.** Holdout-only run (`lgb_colab_echoclf`):

| | overall | LIRF | LFPG | d9 tail | 2025-01 | 2025-07 |
|---|---|---|---|---|---|---|
| §7 echo-lane (has_aobt3 gate) | 484.6s | 1320.1s | 605.9s | 1351.5s | 429.4s | 524.8s |
| **§8 echo classifier** | **397.3s** | **883.1s** | **602.5s** | **1010.5s** | **383.6s** | **408.0s** |
| Δ | **-87.3s (-18%)** | **-437.0s (-33%)** | -3.4s | -341.0s (-25%) | -45.8s | -116.8s |

**Every airport improved or held flat — zero regressions**, both months
improved (contestants' §5 discipline satisfied), and the tail decile (d9)
dropped 25%. This is the single largest single-change win of the session,
bigger than expected.

Classifier diagnostics: `precision=0.688, recall=0.095` at the P>0.5
threshold (n_true_echo=17,349, n_pred_echo=2,406) — it only *confidently*
flags ~14% of true echoes, but those it does flag are right 69% of the
time, and the actual blend uses the **continuous** probability (not a hard
cutoff) for every row, so partial-confidence rows below 0.5 still get
pulled partway toward the echo hypothesis — that's almost certainly why the
overall gain is so much larger than the low recall alone would suggest.
Room to improve recall further later (§ Next steps), but even at this
recall level it clearly dominates the has_aobt3-gated heuristic: the
`echo_pred=True` lane resolves at 1976s (2,406 rows) vs the old
`has_aobt3=False` lane's 3176s (5,323 rows) — a smaller, more precisely
targeted population with a much better resolved error.

Only 33s of the ~896s total run was the classifier itself (`best_iter=174`)
— cheap to include.

**Status: CONFIRMED.** Submitted: **397.3s local → 376s actual**, a real
**-84s** board improvement (predicted -87.3s — the tightest local/actual
match of the whole session, ~4% off). Board total now **460s → 376s** since
§7. Two consecutive changes have now transferred to the board within a few
seconds of their local prediction — strong evidence the eval is fully
trustworthy at this point (§6's fix was the real unlock; everything since
has just been building on a metric that finally means what it says).

## 9. Re-tested flip-vs-direct target under the honest (post-§8) eval — mixed result, NOT adopted

`fix-RMSE` (a since-abandoned branch) found direct taxi prediction beating
the flip-then-reconstruct target by ~11-12% on two separate holdouts. That
result was measured under the *old*, buggy eval (§6's `taxi < 4h` filter),
which happened to exclude almost exactly the echo population where the
flip's reconstruction gets a free structural advantage (`taxi_hat =
offset - d_hat` reproduces `taxi ≈ offset` automatically whenever
`d_hat ≈ 0`, no learning required — direct has no such shortcut). §8's
`_reconstruct_taxi` now blends toward `taxi = offset` explicitly via the
P(echo) classifier for **either** target, so the old advantage was expected
to shrink, hold, or reverse. Re-tested cleanly instead of porting the old
numbers.

**Change:** `run()` gained a `target: "flip" | "direct"` param (`fit.py`).
`target="direct"` fits the regressor on `taxi` directly instead of `d`;
`_reconstruct_taxi` was made target-agnostic (takes the model's own taxi
estimate directly, no longer assumes `offset - d_hat` internally). Also
added a `seed` param to `run()`, threaded to the main regressor fit only,
to support the significance check below — previously hardcoded to 42.

**Sanity check:** `target="flip"` reproduces `lgb_colab_echoclf` (§8)
exactly — 397.3s overall, LIRF 883.4s, LFPG 602.3s, d9 tail 1010.4s,
2025-01 383.6s, 2025-07 408.1s. Confirms the `_reconstruct_taxi` refactor
is behavior-preserving.

**Result, two seeds (42 and 7 — near-identical, so not a single-run fluke):**

| | flip (§8, seed 42) | flip (seed 7) | direct (seed 42) | direct (seed 7) |
|---|---|---|---|---|
| overall | 397.3s | 399.5s | **383.9s** | **383.7s** |
| 2025-01 | 383.6s | 384.0s | **364.0s** | **363.7s** |
| 2025-07 | 408.1s | 411.5s | **399.2s** | **399.0s** |
| LIRF | **883.4s** | **885.1s** | 938.5s | 937.9s |
| LFPG | 602.3s | 602.4s | **579.7s** | **579.4s** |
| d9 tail (taxi 1505-88132s) | **1010.4s** | — | 1053.0s | — |
| `echo_pred=True` lane (n=2,406) | **1976.5s** | **1981.2s** | 2118.2s | 2117.1s |
| holdout fit wall time | 534-469s (best_iter hit the 8000 ceiling) | | **173-154s** (best_iter 2811/2405) | |

Direct wins overall (-3.4%) and on **both** months (contestants' §5
discipline satisfied) and on 9/10 airports, and converges ~3x faster
(never hits the round ceiling, unlike flip). But it **loses, consistently
across both seeds**, on exactly the population §8 was built to fix: the
`echo_pred=True` lane is +141s/+7.2% worse (n=2,406 — not a <100-row
fluke), LIRF (the project's single largest remaining error source, current
Next-Steps item #1) is +55s/+6% worse, and the d9 tail is +43s worse. The
`has_aobt3=False` proxy lane (coarser, superseded by `echo_pred`) actually
*improves* under direct (2390.7s → 2282.1s) — so direct's regression is
specific to the high-confidence echo-classified subset, not the broader
missing-AOBT_3 population.

**Not the clean win the task was hoping for.** Overall board RMSE would
likely improve, but at the cost of the exact metric currently under active
work (§8's Next Steps item 1 is "LIRF still sits at 883s ... needs a
targeted fix") — adopting `target="direct"` now would work against that
effort, not with it. **Not adopted; no submission produced.** Flagging for
a decision rather than picking unilaterally: either (a) stay on flip and
keep chasing LIRF via classifier recall (§8's own next step), or (b) adopt
direct for the overall gain and treat LIRF/echo as a now-separate problem
to solve on top of it (e.g. a stronger echo blend specific to the direct
target). The `target` param is in `fit.py` either way, so this is a
one-line switch whenever a decision is made — no code changes needed to
revisit.

The `test_flip*`/`test_direct*` holdout reports are in `reports/eval/`;
no submission files were produced (`submit=False` throughout).

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
  - (§8) `_reconstruct_taxi()` signature changed — takes an
    explicit `echo_prob` array instead of deriving weight internally from
    `has_aobt3`/`op_echo_rate`. New `fit_echo_classifier()` fits a binary
    LightGBM/XGBoost classifier (`CLF_ROUNDS/CLF_ETA/CLF_EARLY_STOP/
    CLF_OBJECTIVE/CLF_METRIC` constants) predicting `is_echo` on the same
    feature matrix as the `d`-regressor; wired into `run()` for both the
    holdout fit and the all-2025 refit. `_fit_lgb`/`_fit_xgb` gained a
    `metric` kwarg (was hardcoded to `"rmse"`) so the classifier can train
    with `binary_logloss`/`logloss` instead. Added an `echo_pred` (P>0.5)
    lane to `_report`/`_write_report`, plus a console precision/recall
    sanity check against the true `is_echo` label on the holdout.
  - (§9) `run()` gained a `target: "flip" | "direct"` param (default
    `"flip"`, unchanged behavior) and a `seed` param (default 42, matches
    the prior hardcoded value). `_reconstruct_taxi()` made target-agnostic:
    renamed its `d_hat` arg to `taxi_model_raw` and dropped the internal
    `offset - d_hat` subtraction — `run()` now computes that estimate
    itself (`offset - raw_pred` for flip, `raw_pred` as-is for direct)
    before calling it. Echo classifier untouched (still always trained on
    `d`, independent of `target`).
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
- `lgb_colab_echolane` — targeted echo-lane blend (§7). **484.6s local
  → 460s actual** (submitted as `smart-jigsaw_v7`, confirmed transfer).
- `lgb_colab_echoclf` — P(echo) classifier, replaces the has_aobt3 gate
  (§8). **397.3s local → 376s actual.** Confirmed transfer, -84s real
  board improvement.
- `candidate_direct_v1` (§9) — `target="direct"`, full pipeline
  (`data/submissions/candidate_direct_v1.parquet`, 344,841 rows, integrity
  checks pass). **383.9s local (-3.4% vs `lgb_colab_echoclf`'s 397.3s),
  but +6% on LIRF and +7.2% on the echo lane specifically — not adopted,
  see §9.** Distribution sane (median 963s vs echoclf's 962s, mean 1036s
  vs 1042s, fewer rows pinned at the floor/ceiling: 1/38 vs 82/95).
  **Generated for comparison only, not uploaded** — pending the §9 / Next
  Steps #0 decision.

## Next steps

0. **Decision needed: flip vs direct target (§9)** — `direct` beats `flip`
   by -3.4% overall and on both months, reproducibly across two seeds, but
   costs +6% on LIRF and +7.2% on the `echo_pred=True` lane specifically —
   the exact population item 1 below is about. Not picked unilaterally;
   `target=` is a one-line switch in `fit.py` either way. If item 1's
   classifier-recall push lands well on `flip`, re-run this comparison
   after — the gap may change once LIRF/echo is less of an open wound.
1. **LIRF still sits at 883s vs 200-300s elsewhere even after §8** — the
   classifier's low recall (9.5% at P>0.5) means most true echoes are still
   only partially corrected by the continuous blend, not fully. Two
   candidate directions, not yet decided between:
   - Push the classifier harder (more rounds/capacity, or a lower-recall
     bottleneck check — is it feature-starved or just undertrained at only
     `best_iter=174`?) to catch more of the 17,349 true echoes it's
     currently missing at the 0.5 cutoff.
   - Or: the remaining LIRF gap may now be dominated by the investigation
     doc's *other*, unrelated problem — genuine congestion tail at runway
     25, July, hours 08-14 — which no echo-detection change can touch and
     needs separate treatment (congestion features, or revisit a
     per-airport residual head now that the eval is honest and the loss is
     Huber, both different from when it failed before, §6/stage3a_resid.md).
   - Cheapest next step to tell these apart: break LIRF's remaining 883s
     RMSE down by `echo_pred` lane specifically (not just airport-pooled) —
     if the `echo_pred=False` LIRF rows are still much worse than
     `echo_pred=False` rows elsewhere, that's congestion, not echo-recall.
2. Adopt the contestants' eval discipline going forward (see §5): Jan/Jul
   both must improve, echo-lane always reported, 2025-vs-2026 feature
   distribution check before trusting any new feature, multi-seed
   significance check before trusting any gain, ignore gains from <100 rows.
3. Ask on OSN Discord whether OpenSky ADS-B ground trajectories are
   permitted — only for training-side feature engineering (2025), never
   for reconstructing the blanked 2026 ranking labels (that would be
   leakage, not prediction).
4. Start scoping Stage 3b (queue refinement) — a structurally different
   modeling approach, not just another feature family.
5. Weather ingest (NOAA ISD) — still worth doing, but recalibrated to
   "minor lever" status (§4): expect low single digits, not a fix for the
   remaining gap. When we do it: prioritize temperature/dewpoint spread
   over a binary snow flag, and add an explicit null-rate/variance
   assertion after the join (§5).
6. OSM routed taxi distance (Future Step B) — still deprioritized.

## 10. Branch switch: ported the Rome fixes onto `target="direct"`, then found a better combination

Switched to `test-direct-target-with-rome` (teammate's branch, built on top of
the merged echo-classifier work via PR #5), where `run()` gained a
`target="flip"|"direct"` option: `direct` fits the regressor on `taxi`
itself instead of the `d`-flip. The branch already had the echo classifier
and target-agnostic `_reconstruct_taxi`, but not the later session-2 work
(classifier tuning, LIRF-excluded median fallback) — ported both over
(`CLF_ETA`/`CLF_ROUNDS`/`CLF_EARLY_STOP` tuning, `use_prior`/`prior_taxi`
params on `_reconstruct_taxi` gated on `has_aobt3=False AND ADEP_mvt !=
LIRF`). Verified the port is exact: `target="flip"` holdout reproduces
376.0s/LIRF 857s byte-for-byte.

**Found while comparing the teammate's own `test_flip.md`/`test_direct.md`
reports already on the branch:** `direct` beats `flip` on 8 of 10 airports
individually (e.g. EDDF 231→193, EHAM 231→200, EDDM 193→190) but is *worse*
at LIRF specifically (857s → 908s) — `flip`'s `d`-based inductive bias
still wins for LIRF's harder echo/congestion mix, `direct`'s simpler target
wins everywhere else.

**Built `target="mixed"`:** fits BOTH regressors (each still pooled across
all 10 airports — not a per-airport model, same training data both times),
shares one echo classifier (target-independent), and picks the
reconstruction per row by airport: `flip` for LIRF, `direct` everywhere
else. Validated holdout-only:

| target | overall | LIRF | everything else |
|---|---|---|---|
| flip + fixes | 376.0s | **857s** | worse |
| direct + fixes | 379.3s | 908s | **better** (8/10 airports) |
| **mixed** | **370.1s** | **857s** (= flip) | **=direct's numbers exactly** |

Confirms the per-airport split works correctly (LIRF's number matches flip
exactly, every other airport matches direct exactly) and **mixed beats both
pure approaches** — both months improved too (Jan 365s, Jul 374s).
`direct` also trains ~2-2.5x faster than `flip` (best_iter 3310 vs 7998,
~315-450s vs ~650-900s wall time for the holdout fit) since it needs far
fewer boosting rounds to plateau; `mixed` costs roughly the sum of both
(not faster than either alone) since it fits both regressors, and the full
submission-ready run will need two separate all-2025 refits (~1.5-1.8x a
single-target full run, est. 45-60 min).

**Status: holdout-validated (370.1s), full submission-ready run
(`submit=True`) intentionally held off until tomorrow per user request —
not yet run.**

## 11. Full mixed-target run submitted — new best board score

Ran the full submission-ready pipeline (`engine="lgb", target="mixed",
submit=True`, `run_mixed_full.py`) — both regressors (flip + direct) fit on
the train/valid split, echo classifier fit once, all three refit on full
2025, ranking predicted. 2738s (~46 min) wall time. Holdout reproduced
§10's number exactly (370.1s local — flip=857s LIRF, direct everywhere
else). Independently re-verified submission integrity (row count vs
`submitting.parquet`, `MVT_ID_mvt` set equality, zero nulls, zero
negatives) outside `_write_submission`'s own asserts.

Submitted as `smart-jigsaw_v11`: **370.1s local → 343s actual.** New best
board score, beating `smart-jigsaw_v9`/`lgb_colab_echoclf`'s 376s by -33s
(local predicted -27.2s from 397.3→370.1 — actual gain came in even larger
than predicted, continuing the pattern since §8 of local changes
transferring at or above face value).

**Workflow decision (adopted going forward, per user):** `target="direct"`
trains ~2-3x faster than `flip` (best_iter ~3310 vs ~7998, ~315-450s vs
~650-900s holdout fit) and is what most day-to-day feature/tuning
iteration should use for fast holdout-only signal. Reserve `target="mixed"`
(fits both regressors) for final submission-quality runs only, since that's
where flip's LIRF advantage (857s vs direct's 908s) actually pays for its
~2x extra compute cost. Iterating on `direct` alone should still report the
`echo_pred`/LIRF lanes so a direct-only change isn't silently regressing the
population `mixed`'s flip half is covering for.

**Next-step decision (2026-09-17): prioritize LIRF/LFPG over weather.**
LIRF (857s) and LFPG (581s) are 3-15x every other airport's RMSE (next
highest, EGLL, is 289s) and together are ~63% of total squared error
against a combined ~35% of rows — still the single largest lever on the
board. Weather was already downgraded to "minor lever, low single digits"
in §4 (falsified-noise-floor finding) and stays deprioritized until the
Rome/Paris gap closes further. Concrete next move (cheapest diagnostic,
per existing Next Steps #1): split LIRF's 857s by `echo_pred` lane
specifically — if `echo_pred=False` LIRF rows are still much worse than
`echo_pred=False` rows elsewhere, that's the unaddressed congestion tail
(runway 25, July, hours 08-14, per `reports/lirf_investigation.md`), not an
echo-recall problem; LFPG has no equivalent investigation doc yet and needs
one started from scratch.

## 12. LIRF fix found and shipped: a targeted CEIL relaxation — new best
    board score (2026-09-17)

Followed up on §11's LIRF/LFPG priority call. Full diagnostic trail is in
`reports/lirf_investigation.md` §3-5 (row-level holdout `ev` saved to
`cache/lirf_diag_*.parquet` at each step, all reproducible via
`lirf_diag.py`/`lirf_diag_flip.py`/`lirf_critique_check.py`/
`lirf_naive_check.py`/`lirf_check2.py`) — summary here.

**Two dead ends, both ruled out with direct evidence, not argument:**
- Echo-classifier miscalibration — checked via a predicted-vs-actual
  calibration table, both globally and for LIRF; the classifier is
  well-calibrated, low recall (9.6%) reflects genuinely low per-row echo
  probability, not systematic bias. Not pursued.
- An initial "LIRF is a broad regression-to-mean compression, worst in the
  tail" framing (bias -820s in the top true-taxi decile, `offset` correlating
  0.90 with taxi there) turned out to be **mostly a binning artifact**
  (caught by review before any fix was built): binning by the *prediction*
  instead of the outcome showed bias was small and flat everywhere
  (-17.8s in the top decile, not -820s), and `offset ≡ taxi + d` exactly, so
  selecting on the outcome mechanically inflates that correlation. Re-run
  properly (binned by offset, not taxi): real correlation in the extreme
  bin was 0.74, bias -47s — a real but much smaller effect. Error budget:
  bias² was only 10.2% of LIRF's total MSE; d9 alone was 78.2% of it, and
  the top 10 worst rows alone were 58.4% — pointing at a small set of
  extreme rows, not a fixable broad bias. This reframed the whole
  investigation from "fix a broad compression" to "find out what's special
  about the ~20 worst rows."

**The actual mechanism, found by testing (not assuming) where the real
lever was:**
- A naive `taxi = offset` reconstruction *loses* to the production blend on
  every fair (pre-hoc-selected, non-circular) population tested — so
  offset-anchoring alone isn't the answer, dead end for that idea.
- But a **learned** `d̂` (the `target="flip"` regressor's actual raw
  output, `offset - d̂`, pre-clip/pre-blend) *beats* the full production
  pipeline on every population tested (offset>7200: 3536 vs 3928; top-50-
  by-offset: 10942 vs 14768; all LIRF: 816 vs 857) — meaning the raw flip
  model already has the right answer for large-offset LIRF rows and
  something downstream is throwing part of it away.
- Isolated the cause: `CEIL=10800` (3h), applied to the raw model output
  before blending. For the `direct` target this never binds (raw taxi
  never exceeds 6410s anywhere in LIRF, since it was never trained on any
  label above `LABEL_HI=7200`) — but for `flip`, `raw = offset - d̂`
  naturally scales with the always-known `offset` input regardless of
  `d̂`'s own bounded training range, so raw **can** and does exceed 10800
  (22/26,528 LIRF holdout rows, 0.08%) — and those 22 rows alone carried
  **54.5% of LIRF's total squared error** under the old cap.
- **Critical safety check before touching anything**: is "raw > CEIL" a
  good signal everywhere, or just at LIRF? Checked every airport with any
  over-CEIL rows. Everywhere else (EDDF, EDDM, EGLL, EHAM, LEBL, LFPG),
  raw was **8-84x worse** than the capped/blended prediction on those rows
  — i.e. tree/d̂ instability, exactly what CEIL is supposed to catch, and
  exactly the mechanism that caused the §6 blanket-CEIL-raise regression.
  LIRF was the *only* airport where raw beat the blend (16190 vs 21975,
  26% better) — the only airport with genuine 10-24h+ ground delays where
  a confident large raw value usually means something real.
- Also tested a non-airport-named alternative gate (`ceil = offset +
  margin`, "taxi can't exceed offset by more than a plausible early-
  pushback margin"): checked directly and it doesn't discriminate — `raw <
  offset` in literally every over-CEIL row, at every airport (max
  raw-offset = -2823s), so an offset-based margin would loosen the cap
  identically for LIRF and the harmful other-airport rows alike. Not used.

**Fix shipped**: `_reconstruct_taxi()` (`fit.py`) gained a `ceil` parameter
(array or scalar, defaults to the old scalar `CEIL=10800`); both call
sites (`_recon_ho` for holdout, `_recon_r` for ranking) now pass a
per-row ceiling — `10800` normally, `LIRF_RAW_CEIL=140000` where
`ADEP_mvt=="LIRF"` — built with a plain `np.where` on the same
`ADEP_mvt` mask already used for `use_prior`. No threshold-tuning
(self-selects purely on the model's own `raw > CEIL`, not on offset or any
hand-picked cutoff), so overfitting risk on the small n=22 is limited.
Added `taxi_model_raw`/`echo_prob`/`is_echo`/`use_prior` diagnostic
columns to the holdout `ev` frame (reporting-only — built from `ho_lab`
joins, never touches `Xtr`/`Xho`/`Xr`/the feature matrix, so this can't
leak into the model or the submission file; verified explicitly).

**Validated exhaustively before shipping** (`lirf_ceilfix_validate.py`,
holdout-only `target="mixed"` re-run):
- **Non-LIRF rows: exactly byte-identical**, not just similar RMSE —
  diffed the full prediction arrays against a pre-fix run, max abs
  diff = 0.0 across all 317,811 non-LIRF holdout rows.
- **LIRF: 857.4s → 660.6s** (realized, via exact formula re-simulation,
  not an estimate). All 22 affected rows improved, zero worsened.
- Paired bootstrap (2000 resamples, same indices both arms): **P(new ≥
  old) = 0.0000** — never once did the fix do worse.
- Robust across slices: offset>7200 excluding the top 10 by error still
  improves (2008→1917); LIRF excluding its top 10 worst rows entirely
  still improves, if modestly (580.0→569.3) — the win is concentrated in
  a handful of rows but isn't *purely* an artifact of them.
- Both months improve, July (the harder ranking month) by more: Jan
  549.2→464.1 (-15.5%), Jul 1028.5→775.1 (-24.6%).

**Full submission-ready run** (`run_mixed_ceilfix_full.py`, `target=
"mixed", submit=True`) reproduced the validated holdout number exactly
(337.6s overall, LIRF 661s) and confirmed the fix genuinely engages on
the real 2026 ranking set, not just holdout: LIRF ranking rows with
offset>30,000s now get large offset-tracking predictions (e.g.
offset=111,654s → predicted 110,367s) instead of being flattened to
≤10,800s as before. Noted for the record, not treated as a red flag: 4 of
27 extreme-offset LIRF ranking rows get a *small* predicted taxi despite
huge offset (700-2900s on 32-34k offsets) — plausible (model believes
those pushed back close to takeoff despite a large schedule delay, exactly
what the flip target is designed to distinguish) but unverifiable since
the true label is blanked; worth rechecking once the real board score is
back.

Submitted as `smart-jigsaw_v12`: **337.6s local → 302s actual** (down from
`smart-jigsaw_v11`'s 370.1s local / 343s actual, -8.8% local / -11.9%
actual, the single largest jump of the whole LIRF investigation) from a
2-line, airport-gated, thoroughly-validated change. Integrity re-verified
independently (row count, MVT_ID_mvt set equality, zero nulls, zero
negatives). Actual again beat the local prediction (as with every
transfer since §8), continuing the pattern that this eval is trustworthy
and, if anything, conservative. Current leaderboard lead is ~241s — gap
narrowed from ~40% to ~25%.

**Decision made, not yet acted on**: whether to switch LIRF from `flip` to
`direct` in the `mixed` target (raised by user re: `direct`'s ~2-3x faster
training). **Answer: no.** Before this fix the flip-vs-direct gap on LIRF
was small (857 vs 908, 6%) — arguably close enough to trade for speed.
This fix only works for `flip` (structural: `direct`'s raw output can
never exceed `CEIL` in the first place, confirmed 0/26,528 rows, so there
is nothing to relax) — post-fix the gap is 660.6 vs 908, **27%**, and
switching to `direct` would forfeit this section's entire win. `mixed`
stays as the production target.

**Next steps, updated:**
1. ~~LFPG still has no dedicated investigation doc — highest-remaining-
   leverage item~~ **DONE, 2026-09-18, see `reports/lfpg_investigation.md`
   — REVERSED the priority call.** LFPG's 581s RMSE is not a systemic
   problem: excluding its single worst row drops RMSE to 403s, excluding
   the worst 2 drops it to 283s (in line with every other airport).
   Top-10-rows SSE share is 79.2% for LFPG vs 3-29% for every other
   airport including LIRF post-fix — a unique, extreme outlier-concentration
   pattern, not a broad modeling gap. The 2 dominant rows are fully
   NM-unmatched (operator null) with physically anomalous true `d`
   (-56k to -82k seconds); a full-year check found only 7 such rows/year
   at LFPG, one with an outright impossible negative taxi label — data
   corruption, not a learnable pattern. Checked and ruled out the LIRF
   mechanism (raw prediction discarded by CEIL) — doesn't apply, `direct`
   target's raw output structurally can't reach these magnitudes either.
   Theoretical ceiling if solved: -14% overall (337.6→290.4s) — tempting on
   paper, unreachable in practice, flagged so nobody re-derives and chases
   it. One minor, low-priority finding: raw model beats the flat
   `median_taxi_prior` fallback for LFPG's *non-catastrophic* NM-unmatched
   rows (536s vs 610s RMSE, n=875) — real, both-months-positive, but only
   -0.26s overall impact; not worth a submission alone. **Recommend
   applying the same top-K/SSE-share check to LSZH (29.2% top-10 share)
   and EHAM (18.8%) before assuming either is a broad-based priority.**
2. The AOBT_3-availability lever found during this investigation (within
   `offset>7200`, rows with `aobt3_taxi` present resolve at 1372s RMSE vs
   the model's own 1827s on that exact subset — non-circular, ~5% LIRF
   ceiling) is still unbuilt; smaller than the CEIL fix but real and
   evidence-based. Candidate for a future session.
3. Re-run the flip-vs-direct-vs-mixed comparison (§9/§10) is now stale for
   LIRF specifically since flip's LIRF number moved a lot (857→661) —
   doesn't change the "keep mixed" conclusion above but worth noting the
   comparison table in §10 is now out of date for LIRF's row.
4. Bounding `d̂` directly (e.g. near the existing `D_CLIP=(-3600,10800)`
   used for group encodings) instead of bounding the reconstructed taxi
   was flagged as a possible airport-agnostic generalization of this fix
   during review, but not built — the implied-d̂ magnitude differs sharply
   between LIRF's good over-CEIL rows (≤12.9k) and other airports' bad
   ones (up to 91.2k), suggesting it might generalize where the
   offset-margin idea didn't, but this needs its own validation pass.

## 13. LFPG audit (2026-09-18) — reversed the "LFPG is the next lever"
    assumption; echo-classifier hour-encoding tried and rejected

Full detail in `reports/lfpg_investigation.md`; summary here.

**LFPG audit.** Board score for `smart-jigsaw_v12` confirmed: **337.6s
local -> 302s actual** (leaderboard lead ~241s at the time, gap ~25%).
LFPG's 581s RMSE (worst airport after LIRF) was assumed to be the next
LIRF-style lever. It wasn't: excluding LFPG's single worst holdout row
drops its RMSE 581->403s; excluding the worst 2 drops it to 283s, in line
with every other airport. Top-10-row SSE share is 79.2% for LFPG vs 3-29%
for every other airport (LIRF post-fix included) — a unique,
outlier-concentration pathology, not a broad modeling gap. The two
dominant rows are fully NM-unmatched with physically anomalous true `d`
(-56k to -82k seconds, i.e. actual off-block 15-23h before schedule); a
full-year check found only 7 such rows/year at LFPG, one with an
impossible negative taxi label — data corruption, not a learnable
pattern. Checked and ruled out the LIRF CEIL-discard mechanism (doesn't
apply: raw model RMSE 3455 vs fallback's 3465 on the affected lane,
statistically identical). Theoretical ceiling if solved: -14% overall
(337.6->290.4s) — unreachable in practice, flagged so it isn't re-chased.
Follow-up check on LSZH (29.2% top-10 share) and EHAM (18.8%), the next
two highest-concentration airports: both cleared — gradual decay curves,
worst rows are ordinary NM-matched large-delay underprediction (the
already-known d9 tail pattern), not an LFPG-style cliff. **No airport
currently has an LFPG-style hidden bug; the outlier-concentration audit
is closed.**

**Echo-classifier recall — hour-of-day group encoding tried, REJECTED.**
Per PROGRESS.md's own next-steps, classifier recall (9.5-9.6% at P>0.5,
unchanged since S8 despite the eta/rounds tuning noted at `fit.py`
CLF_ROUNDS/CLF_ETA) was flagged as the more tractable of the two
remaining structural levers (the other being unstarted Stage 3b). Free
pre-check (no training, plain groupby on `labels2025.parquet`) showed
real-looking signal: echo rate varies 3.5x by hour-of-day at LIRF
(14.0%-48.5% vs an 18.1% airport average) and 2-3x at most other
airports relative to their own baseline rate — a plausible new,
orthogonal signal beyond the existing `op_echo_rate`/`stand_echo_rate`
group encodings (`encode.py` `GROUP_ENC_KEYS`).

Built: added `"hour": ["ADEP_mvt", "hour"]` to `GROUP_ENC_KEYS`
(`src/features/encode.py`), joined `hour` into the `lab` table in
`run()` (`src/models/fit.py`) alongside the existing
`AIRCRAFT_OPERATOR_flt` join so the OOF group-encoding machinery could
compute `hour_echo_rate`/`hour_mean_d`/`hour_n` the same way it already
does for operator and stand — no other code changes needed, the
mechanism is fully generic over `GROUP_ENC_KEYS`.

Tested cleanly: reran the `target="direct"` holdout baseline fresh
(`echoclf_baseline_direct.{log,md}`, reproduces the existing
`lirf_diag.md` numbers exactly) rather than reusing a cached eval, then
one retrain with the new feature (`echoclf_hour_test.md`), same seed/
loss/rounds settings both times.

| | baseline (no hour enc) | + hour_echo_rate |
|---|---|---|
| overall | 379.3 | **386.1 (worse)** |
| LIRF | 908.0 | **945.2 (worse, +4.1%)** |
| 2025-01 | 366.6 | 364.1 (slightly better) |
| 2025-07 | 389.2 | **403.0 (worse, +3.6%)** |
| echo clf recall @P>0.5 | 0.096 | 0.096 (unchanged) |
| echo clf precision @P>0.5 | 0.694 | 0.690 (unchanged) |
| d9 tail | 1037.1 | 1060.7 (worse) |

**Rejected.** Overall regressed (+1.8%), July regressed (+3.6%, fails the
both-months-must-improve rule), LIRF regressed most (+4.1%) despite
being the airport where the pre-check signal was strongest, and the
echo classifier's recall/precision didn't move at all (2406 vs 2417
flagged rows, statistically the same) — the new feature wasn't even
picked up as decision-relevant for classification, yet still hurt the
main regressor. Likely mechanism: `hour` is probably heavily confounded
with existing `op_echo_rate`/`stand_echo_rate` (an operator's schedule
concentrates in certain hours already), so `hour_echo_rate` added
mostly redundant, noisier information — extra tree-split surface with
sparser per-cell statistics (up to 240 airport x hour cells) that hurt
generalization, especially at LIRF where the model already runs closest
to its tail-fitting limits. **Reverted** (`git checkout` on
`src/features/encode.py` and `src/models/fit.py`) — not shipped.

**Lesson for next time, matching the community's own reported
experience (S5):** a promising-looking raw statistic (the hour-rate
spread) does not reliably survive being turned into a model feature —
verify via an actual retrain, not just the descriptive check, before
trusting a "this should help" intuition. The pre-check was directionally
right about where variance exists but wrong about whether the model
could exploit it without cost.

**Next steps, updated:**
1. Echo-classifier recall is still open. The hour-of-day angle is now
   ruled out; untried remaining ideas (from the original investigation
   doc): richer interaction keys (e.g. operator x stand) with adequate
   smoothing, or features that flag "no NM match at all" more directly
   (LFPG's S13 finding that `AIRCRAFT_OPERATOR_flt is null` for the
   catastrophic rows suggests a specific "fully unmatched" indicator,
   distinct from the softer `has_aobt3` flag, might separate a cleaner
   subpopulation).
2. Stage 3b (queue refinement) remains completely unstarted — still the
   other open structural lever, bigger scope than a feature tweak.

## 14. Stage 3b (queue refinement) scoped, killed cheaply before the
    expensive part was built (2026-09-18)

**Mechanism check before building anything.** The plan's Stage 3b calls
for "N(t)-style features from out-of-fold predicted off-blocks" — i.e. a
queue/congestion count anchored on pushback (AOBT) time instead of the
existing takeoff-anchored family 2. AOBT is exactly what's blanked for
every departure in `ranking.parquet`, so a true AOBT-anchored feature
needs a two-pass OOF architecture (fit a base model -> get predicted
off-blocks -> build queue features from those -> refit) to be
deployable at all. Before building that: checked whether an *oracle*
version (built from the true, un-blanked AOBT available in training/
holdout) would even help, and separately whether it's trustworthy.

**Second review caught a real risk before any code was written:**
running the oracle arm directly is not a clean feasibility test. The
model already has near-exact clock-time information — `hour`/
`minute_of_day` (`build_features.py` lines 256-261) are derived from
`T` (takeoff, never blanked) and fed as raw features, and
`sched_takeoff_offset = T - SOBT` is exact and already the #1 feature by
gain. That means `SOBT`'s clock time is already approximately
recoverable too. A new feature correlated with AOBT's clock time (which
a true-AOBT-anchored queue count would be) hands an 8000-round,
255-leaf ensemble a third timing reference to triangulate AOBT between
two already-known anchors — i.e. a path to partially reconstruct `d`/
`taxi` from timing alone, not genuine congestion modelling. An oracle-arm
win could be entirely this leak, invisible on today's holdout, and would
evaporate once real OOF-predicted (noisy) off-blocks replace the true
ones at deployment. Same class of failure as the echo mechanism (S6-S8),
just a new mechanism for it.

**So the oracle arm was skipped.** Built the safe version instead: a
schedule-anchored (`SCHED_TIME_UTC_mvt` = SOBT) queue/pressure family --
`n_dep_sched_30m_prev`, `n_dep_sched_60m_prev`, `n_dep_sched_30m_next`,
`sched_dep_pressure` (`build_features.py`, new `_sched_congestion()`,
`_roll_count()` generalized to take an `index_col` param). SOBT carries
none of the AOBT-leak risk -- it's already fully exposed via
`sched_takeoff_offset` -- so this tests "does published-schedule density
around a flight's own slot add anything over realized-traffic
(takeoff-anchored) congestion" with zero new architecture and zero
leak exposure. Deployable today if it had worked; no two-pass OOF
machinery needed.

**Evaluation upgraded per the same review's second point**, after the S13
hour-encoding experiment got called out for scalar-diffing single-run
RMSE numbers with no idea what their run-to-run noise looks like (the
echo lane's implied standard error alone is ~29s on n=2,406, before
heavy tails). This time: paired per-row squared-error deltas between a
freshly-reproduced baseline and the treatment on the *same* holdout
rows, cluster-bootstrapped at the (airport, day) level rather than
resampled by row -- congestion errors are correlated within an
airport-day window, so row-level resampling would treat them as
independent and understate the interval on exactly the features where
that matters. 620 clusters, 3000 resamples. Primary metric
pre-registered before the fresh run: paired delta in overall RMSE.

Also caught and fixed a real bug in the process (unrelated to the
feature itself): a housekeeping hook in this repo auto-relocates
loose root-level `.py`/`.log` files into `tests/`/`logs/`. A first
attempt at this retrain silently no-opped -- the script it invoked had
already been relocated mid-session, `python` errored immediately, but a
trailing `ls` on a stale report file from an *earlier* experiment made
the shell command's own exit code look like success, and the "results"
read back were just that stale file. Caught by noticing the file's
`generated:` timestamp inside the report matched the previous run
exactly. Re-ran from the relocated path (`tests/echoclf_hour_test.py`)
and confirmed a genuinely new `best_iter`/timing before trusting the
numbers.

| | baseline (fresh reproduction) | + sched_dep_pressure family |
|---|---|---|
| overall RMSE (point estimate) | 379.3 | 382.4 |
| LIRF | 908.0 | 927 |
| 2025-01 | 366.6 | 363 |
| 2025-07 | 389.2 | 397 |
| echo clf recall / precision @P>0.5 | 0.096 / 0.694 | 0.096 / 0.691 |

**Cluster-bootstrap (airport, day), 3000 resamples, on paired overall
RMSE delta (treatment - baseline):** point estimate **+3.14s**, 95% CI
**[-3.42, +12.17]** -- includes zero, so not significant at 95%, but
clearly skewed toward harm: P(worse) = 0.75, P(better by >=2s) = 0.089.
Against the pre-registered rule (ship only if the CI shows a clear
improvement), this is an unambiguous **reject** -- there's no evident
signal, and if anything the point estimate leans negative, consistent
with July regressing on top of a small January gain (fails the
both-months rule too).

**Reverted** (`git checkout` on `build_features.py`; feature caches
regenerated back to the pre-experiment state).

**Stage 3b implication.** The critique's own logic applies directly:
if the leak-free, zero-architecture SOBT version doesn't move the
needle, the harder AOBT-anchored version -- same underlying congestion
signal, plus the leak risk above, plus a full two-pass OOF build -- is
very unlikely to be worth it. **Stage 3b is deprioritized on this
evidence**, not just "still unstarted." Total cost to learn this: two
retrains (~20 min) and one feature-family build that was safe to write
regardless of outcome, versus building the full OOF architecture first
and finding out the same thing afterward.

**Next steps, updated:**
1. Stage 3b: deprioritized per above. Correction (2026-09-18, see S15):
   the claim "SOBT-anchored schedule pressure showed no signal" should
   NOT be generalized to "surface occupancy in general is uninformative"
   -- `AOBT_3_flt` (98.5% populated in `ranking.parquet` DEP rows, a
   *different*, already-used column from `BLOCK_TIME_UTC_mvt`, which
   *is* the one fully blanked) means a genuinely realized queue feature
   is buildable today from neighbors' true off-blocks, no OOF machinery
   needed -- untried, and a better test of the underlying hypothesis
   than the SOBT version was.
2. Echo-classifier recall remains the only open item from S13 --
   operator x stand interaction or a "fully NM-unmatched" flag, per the
   LFPG finding that the catastrophic rows had `AIRCRAFT_OPERATOR_flt`
   null entirely. See S15 for a deeper, mixed-result look at this exact
   lever before building it.
3. Methodological note to carry forward: the paired-cluster-bootstrap
   harness built here (`(airport, day)` clusters, pre-registered primary
   metric) is reusable for any future feature test on this holdout --
   worth formalizing into a small helper rather than rewriting ad hoc
   each time.

## 15. Run manifests built; echo/offset routing ceiling investigated and
    found smaller and harder to cash in than it first looked (2026-09-18)

**Manifest system built** (`src/eval/manifest.py`), directly motivated by
the S14 silent-no-op incident. `build_manifest()`/`write_manifest()` wired
into `fit.py`'s `_write_report()` -- every holdout report now gets a
sibling `<name>.manifest.json` with git SHA + dirty flag, a sha256 of the
three core pipeline files (`build_features.py`, `encode.py`, `fit.py` --
catches uncommitted edits that a bare git SHA would miss), a hash of the
sorted feature-name list, row count, seed, `best_iter`/`clf_best_iter`,
and wall time. `assert_comparable(path_a, path_b)` raises if two runs
have identical `src_hash` AND `feature_hash` -- the no-op signature.
Verified with a smoke test (two fast 50-round runs, identical code):
manifests came back byte-identical on both hashes, and the comparison
helper correctly refused to diff them. Caveat, for the record: this
catches "ran twice, nothing changed" -- it does NOT by itself catch
"script silently failed to run at all and a stale report file sat
there" (S14's actual failure mode), since a truly-never-regenerated
report's manifest is just old, not necessarily identical to the
baseline's. Real protection against that needs the comparison tool to
also check the manifest's own freshness, not just diff two hashes --
not built, flagged as a gap.

**Oracle-routing ceiling, re-investigated after a review challenged the
96.4s figure from earlier the same day.** The review's specific guess
(near-tie coin-flips inflating the oracle) was checked directly and
**refuted**: decomposing the SSE gain by `|taxi_echo - taxi_model|`, rows
with gap <30s contribute 0.1% of total gain; gap >=500s contributes
93.7%. The gain is maximally concentrated in huge, unambiguous margins
(median gap ~4,450s in the top 0.1% of rows), not coin-flips.

**But a bigger, different flaw was found instead: the ceiling had been
measured on the wrong baseline.** The original 96.4s figure came from
`echoclf_baseline_direct_ev.parquet` (a `direct`-only run), not the
actual production `mixed` model. `direct`'s raw output is structurally
capped near the training label range and can never represent a large
true delay -- so "oracle beats current" there was partly rediscovering
that structural gap, not new information. 367/500 top-gain rows were
LIRF. Redone on the real production frame (`lirf_ceilfix_mixed_ev.parquet`,
correct per-row CEIL):

| | overall RMSE |
|---|---|
| current (production) | 337.6 |
| oracle, all rows | 285.8 -- ceiling **51.7s** |
| oracle, **excluding LIRF** | 318.4 -- ceiling **19.1s** |

LIRF still dominates the full-population ceiling (360/500 top rows) even
post-CEIL-fix -- but that's the same population `lirf_investigation.md`
S4/S6 already exhausted (naive offset-anchoring loses on every fair,
non-circular population tested there; a hindsight oracle will always
look good on a population like this regardless of whether any real
router could ever capture it). Excluding it, the honest, not-yet-
investigated ceiling across the other 9 airports is **19.1s** -- real,
but only ~2.5x the width of the paired-bootstrap noise floor measured
in S14 (15.6s), not the 6x-clear-signal the original 96.4s number
implied.

**Discriminability check (non-circular): does the existing classifier
output actually separate the winner, out of sample?** AUC of the
production `echo_prob` against the true "which candidate wins" label,
ex-LIRF, non-`use_prior` rows, split by month (train-independent, since
`echo_prob` was itself fit OOF on the other 10 months already):

| | AUC |
|---|---|
| all rows | 0.825 |
| gap >=500s only (87.5% of the ceiling lives here) | 0.803 |
| Jan only | 0.808 |
| Jul only | 0.838 |

Strong, real, and generalizing across months -- not chance-level. This
looked like a clean "go" signal for building something on top of it.

**Cheapest possible thing to build on it -- an isotonic recalibration of
the blend weight, no retrain, using only already-cached predictions --
backfired in both directions.** Fit an isotonic map from `echo_prob` to
the per-row optimal blend weight `w* = clip((taxi-taxi_model)/(taxi_echo-
taxi_model), 0, 1)`, sample-weighted by the squared gap (so near-ties
can't dominate the fit), training on one month and testing on the other:

| | current production RMSE | recalibrated-weight RMSE |
|---|---|---|
| fit Jan -> test Jul | 232.1 | **241.8 (worse)** |
| fit Jul -> test Jan | 207.1 | **216.6 (worse)** |

So: a real, generalizing discriminating signal exists (AUC ~0.82), but
the most direct way to cash it in made things worse both ways. Likely
mechanism: `w*` is a noisy, single-observation target (derived from one
realized `taxi` value per row), and even weighting by squared gap
doesn't stop the isotonic fit from overfitting to month-specific
idiosyncrasies in the *rare*, high-weight tail it's forced to prioritize.
Good discrimination on a binary "who wins" label doesn't imply a usable
continuous weight estimate.

**Not adopted -- no code changes from this investigation.** Three
independent "this should obviously work" hypotheses today (the original
96.4s ceiling, the AUC-implies-actionable-signal read, and the isotonic
recalibration) each needed a further layer of scrutiny to fail correctly
instead of shipping something broken. The honest position: real
discriminating signal exists in the ~19s ex-LIRF ceiling, but nothing
tried so far captures it, and a full stacker (regularized GBM on
out-of-fold candidate predictions plus real features, not a 1-D
recalibration of an existing scalar) is a bigger, harder build than the
ceiling size alone justifies committing to right now.

**Next steps, updated:**
1. Manifest freshness gap (above) is a small, cheap follow-up if another
   silent-no-op class of bug shows up again.
2. Realized (not scheduled) `AOBT_3_flt`-neighbor queue feature, per S14's
   correction -- still the most promising untried version of the
   original Stage 3b hypothesis, and unlike the SOBT version, it doesn't
   share the LIRF/direct-target confound this section had to correct for.
3. Echo/offset routing via a structural fix (see S16) rather than
   post-hoc blending -- the remaining real lever.

## 16. Echo/offset routing lever closed out cleanly -- three post-hoc
    attempts, all negative, with a tight enough CI to call it dead
    (2026-09-18)

Continuation of S15. A second review pushed back further, correctly
predicted the isotonic failure's real mechanism, and derived the
squared-error-optimal closed form: with `gap = taxi_echo - taxi_model`
and `resid = taxi - taxi_model`, the optimal per-row blend weight
satisfies `w* = E[resid*gap | x] / E[gap^2 | x]`, which falls out for
free from an ordinary least-squares regression of `resid` on
`[gap, gap*echo_prob]` (no intercept) -- automatically weighted by
`gap^2`, so near-ties can't dominate the fit the way they could an
unweighted target. Also correctly predicted that blending changes are
**pure post-processing on already-cached predictions** (no GBM retrain,
no bagging/feature-subsample stochasticity), so a paired cluster
bootstrap on fixed per-row predictions gives a much tighter interval
than the retrain-vs-retrain comparisons in S14 -- confirmed: the S14
noise floor was +-7.8s half-width; the post-processing-only bootstrap
below resolved effects at +-0.06s.

**Attempt 1 (2-parameter closed-form fit, ex-LIRF non-`use_prior` lane,
fit Jan -> test Jul and reverse): essentially flat.** RMSE unchanged to
2 decimals (232.11 -> 232.12 one direction, 207.09 -> 207.04 the other).
Diagnosed why, not just accepted the null: on this exact lane, forcing
`w=0` (ignore the offset hypothesis entirely) gives RMSE 221.14 vs
current production's 221.19 -- **the existing echo_prob blend is
already doing almost nothing here**, so there was never much room for a
linear-in-`echo_prob` reweighting to find. The earlier apparent
"blending helps a lot" evidence (302.9 vs 409.0, S15) came from the
direct-only baseline including LIRF and its structural range-cap issue,
not from this specific ex-LIRF population.

**Per-airport breakdown of the remaining oracle gap on this lane:**
gaps of 20-48s per airport, roughly proportional to each airport's own
baseline RMSE (EDDF 20s on 188s baseline, LFPG 48s on 271s baseline,
etc.) -- spread evenly, not concentrated in a subgroup `echo_prob` (or
any airport interaction) would obviously catch. Consistent with generic
model-error variance rather than a systematic, routable pattern.

**Attempt 2 (shrinkage sweep, `w_final = w_prod + lambda*(w_new -
w_prod)`, lambda in [0,1]):** several lambda values (0.1-0.7) technically
improved both directions vs lambda=0, but by 0.01-0.05s -- three orders
of magnitude below anything meaningful.

**Attempt 3 (formal check, not eyeballing): paired (airport, day)
cluster bootstrap on the lambda=0.5 treatment vs production, 3000
resamples.** Point estimate **-0.038s**, 95% CI **[-0.100, +0.024]**.
Tight, centered on zero. **Confirmed dead, not just unmeasured** -- the
tightness of this interval (vs S14's +-7.8s) is itself informative: a
19-34s effect would have been trivially visible here if it existed and
were reachable via `echo_prob`; it isn't.

**Status: closed.** Three independent post-hoc attempts (isotonic
recalibration, closed-form linear blend, shrinkage sweep) on the same
underlying signal (`echo_prob`) all failed, the last two cleanly rather
than ambiguously. The remaining oracle-vs-production gap is real
(19-34s depending on lane) but not capturable via post-processing on
this model's current outputs with the features currently available to
the classifier.

**The one lever left with real (non-oracle-bounded) upside**, per the
same review: fix the cause, not the symptom. `direct`'s raw output is
structurally capped near its training label range and can't represent
large delays regardless of routing; predicting `log(taxi)`, a
multiplicative factor over `offset`, or regressing the residual against
`offset` directly would change what the *base* model can represent,
which no amount of post-hoc blending of a capped output can fix. This
is a retrain, not post-processing, so it's not free to evaluate the way
S15/S16's attempts were -- flagged as the next real candidate, not
started this session.

**Next steps:**
1. ~~Structural range fix for `direct` (log target / multiplicative /
   residual-on-offset)~~ **TRIED AND REJECTED, same day, see S17.**
2. Echo/offset routing via post-processing: closed, do not reopen
   without new features for the classifier (not just a different
   functional form of the existing `echo_prob`).
3. `AOBT_3_flt`-neighbor realized queue feature (S15) and the manifest
   freshness gap (S15) remain open, lower-priority items.

## 17. Structural range fix for `direct` (target="direct_log") -- tried
    same day, rejected before it was even a fair test

Built `target="direct_log"` (`fit.py`): trains on `log(taxi)` instead of
`taxi` directly, with the training label window widened from
`[30, 7200]` to `[30, 21600]` (`LABEL_HI_LOG`) -- the theory being that
`LABEL_HI=7200` is *why* `direct`'s raw output structurally can't
represent a large true delay (the model has never seen a training
example above it), and log-compression should let the window widen
without reintroducing the outlier-domination problem `LABEL_HI` exists
to prevent. Reconstructed via `exp(clip(raw, -5, 13))`. Deliberately
left the *output* ceiling at the existing `CEIL=10800` for this first
test, to isolate the training-side effect safely before touching
anything with the same blast radius as the S6 blanket-CEIL-raise
regression.

**Checked the addressable population size before trusting any result.**
Non-LIRF rows with true taxi > 7200s (what the widened window targets):
**36 out of 317,811 (0.01%)** -- but those 36 rows carry **41.9%** of
all non-LIRF squared error; a perfect prediction on just them would cut
non-LIRF RMSE from 295.0 to 224.8 (-24%). More importantly: rows that
actually *cross the still-unchanged output ceiling* (10800s) are just
**6 in the entire 317,811-row population** -- 3 LFPG, 2 EHAM, 1 EGLL.
The 3 LFPG rows almost certainly overlap with the exact catastrophic,
NM-completely-unmatched, partly-data-corrupted rows the LFPG audit (S13)
already investigated and closed out. This is the same "tiny population,
mostly irreducible or already-dead-end" pattern as LFPG, just spread
thinner across more airports -- recognized *before* the retrain
finished, which correctly predicted the outcome.

**Result confirmed the prediction: essentially flat.** 380.6s overall
vs the existing `direct` (huber) baseline's 379.3s -- a wash, arguably
a hair worse (though loss also differed, l2 vs huber, so not a fully
clean single-variable comparison; the direction is what matters here).
Every airport moved by low single digits in either direction, LIRF and
LFPG unchanged (expected -- LIRF isn't targeted by this, and LFPG's
predictions still hit the same unchanged ceiling). **Not adopted --
reverted** (`git checkout` on `fit.py`).

**Why this was correctly predictable in advance, for next time:** the
experiment widened the *training* window but left the *output* ceiling
untouched, so it could only ever improve calibration on the ordinary
bulk of rows -- it structurally could not touch the population that
actually carries the error (bounded by the output ceiling, not the
training window). Relaxing the output ceiling too would require the
exact same per-row, per-airport validation rigor as the LIRF CEIL fix
(S12) -- and for a population of 6 rows, several of which are already
known dead ends, that's a lot of careful work for a very small, likely
mostly-irreducible target. **Not worth pursuing further** unless a much
larger genuinely-addressable population turns up elsewhere.

**Next steps:**
1. No further work on `direct`'s range cap -- closed, same reasoning as
   LFPG (S13) and the echo-routing dead end (S16): a handful of extreme
   rows dominating squared error is not automatically a lever, and this
   session has now hit that pattern three times.
2. Remaining open items unchanged from S15/S16: `AOBT_3_flt`-neighbor
   realized queue feature, manifest freshness gap, echo classifier with
   genuinely new features (not just a new function of existing
   `echo_prob`).

## 18. Per-airport specialization revisited (retrospective on
    `stage3a_resid.md`) -- killed cheaply before any build, same day
    (2026-09-18)

Revisited the 2026-09-02 per-airport residual-head regression
(`reports/stage3a_resid.md`) to see whether a better-designed version
(post-Huber, post-mixed-target, post-CEIL-fix, with the OOF
group-encoding machinery now available) could succeed where it failed
before. Diagnosed the original failure as two bundled mistakes: (a) a
widened label window fed into the *shared* pooled model, pulling every
airport's trees toward LIRF's tail (EDDF 262->349, EHAM 262->305); (b)
residual heads at full capacity (num_leaves=63, ~50 features) on a
crude 2-fold OOF, which memorised seasonal fold-specific noise instead
of a transferable bias.

A review pushed back before any build started: the S13/S16 outlier-
concentration audit shows EGLL (14.6% top-10 SSE share) and LTFM (6.9%)
have *broad-based*, not outlier-driven, elevated RMSE (289s and 268s
vs 180-220s for the well-behaved airports) -- but broad-based error is
exactly what irreducible per-airport noise looks like too, and nothing
in that audit distinguishes "systematically wrong in a learnable way"
from "this airport is intrinsically noisier." The free, decisive check:
does the pooled model's residual show any out-of-sample-transferable
structure at all (mean residual by hour/runway/operator/stand,
Jan-derived pattern checked against Jul)?

**Result: no transferable structure -- actively anti-transferable.**
Jan-vs-Jul correlation of mean residual by group:

| | by hour | by runway | by operator | by stand |
|---|---|---|---|---|
| EGLL | -0.315 | -0.779 | -0.153 | +0.143 (noise, small n) |
| LTFM | -0.505 | -0.916 | -0.294 | +0.066 (noise, small n) |

Negative correlations mean whatever direction the model is wrong in
January, it tends to be wrong the *opposite* direction in July for the
same hour/runway -- the exact signature of month-specific noise (a
weather event, an operational quirk), not a stable bias. This kills not
just the residual-head idea but the cheaper alternatives too (row
upweighting, airport-interaction features) -- all three assume the same
transferable structure that isn't there.

**Not built -- killed for the cost of one diagnostic query instead of
an afternoon.** This is the fourth "looks systematic, isn't fixable"
result today (LFPG S13, echo-routing S16, `direct` range cap S17, now
per-airport specialization) -- every angle that re-slices the existing
feature set differently has now been tried and closed. Reinforces the
S16-era conclusion: remaining upside looks like it needs genuinely new
information (the airport-wide arrival-delay disruption proxy, or
external weather/ATFM data), not further feature engineering on what's
already in hand.

**Next steps:**
1. Per-airport specialization: closed, do not revisit without new
   out-of-sample-transferable structure showing up somewhere first.
2. ~~Airport-wide arrival-delay disruption proxy~~ **TRIED, same day,
   see S19 -- null result, closed.**
3. Weather (NOAA ISD) and ATFM/ground-stop regulation data (permission
   + licensing status unresolved) -- the two genuinely-new-information
   candidates left.

## 19. Airport-wide arrival-disruption feature -- the one idea that
    survived every free check, built, and still came back null
    (2026-09-18)

Follow-up to S18. The per-airport residual-head idea died on a free
check (no transferable structure by hour/runway/operator at EGLL/LTFM).
But the *raw* relationship between a proposed new signal -- system-wide
mean arrival delay in the trailing 60 min, `arr_delay_60m`, a
delay-*magnitude* axis distinct from family 2's volume counts and
family 4's single-flight rotation -- and taxi showed something the
other checks didn't: a stable, monotonic, always-positive relationship
at EGLL across **all 12 months of 2025** (slopes +82.7 to +271.1, no
sign flips), versus an erratic, sign-flipping pattern at LTFM (a wild
+499.5 in February alongside single digits elsewhere). Correlation with
existing congestion features was weak everywhere (|r|<0.22,
strongest `n_arr_60m_around` at -0.22) -- not a rediscovery of
something already named differently.

**Caught before over-trusting it:** a review pointed out the raw check
uses *raw taxi*, not model *residual* -- the quantity that actually
determines whether a new feature can help is what's left over after
existing features already explain what they explain, and the model
doesn't need to relearn what it already knows. Re-ran the same 12-month
check on OOF residuals instead: since the real production GBM's OOF
residuals aren't available for all 12 months without an expensive
12-fold full retrain, used a fast proxy -- leave-one-month-out Ridge
regression on the *existing* congestion/rotation feature set (RMSE 409s
at EGLL, vs the real GBM's ~289s) as a stand-in baseline. **EGLL's
residual-based slope was ALSO positive in all 12 months** (35.0 to
148.0) -- passing the harder version of the test. LTFM remained
erratic.

**Flagged, not resolved, before building:** the Ridge baseline
under-captures what the full GBM already knows by ~120s worth of RMSE
-- a gap the GBM likely closes partly through *interactions* (congestion
count x hour x runway approximating "disrupted right now" without
naming it), which a linear proxy structurally cannot detect. This is
exactly the kind of gap that can't be closed by a better diagnostic --
only by the real build. Pre-registered before running it: primary =
paired cluster-bootstrap on overall RMSE; secondary/noisy = EGLL
specifically; success = both months improve; **and the failure
interpretation, in advance**: if neither EGLL nor overall move, the GBM
already had this via interactions and the Ridge proxy explains the
whole earlier disagreement.

**Built:** `_arrival_disruption()` (`build_features.py`) -- mean
`BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` across all ARR rows at the
airport in the trailing 60 min, joined onto every departure via
`join_asof` (backward). Uses only ARR-side `BLOCK_TIME_UTC_mvt`, never
blanked, no leak-by-construction risk. Fed to all 10 airports uniformly
(not gated to EGLL), letting the pooled model use it or not per airport.

**Result: clean null, matching the pre-registered failure case exactly.**

| | overall RMSE | EGLL RMSE |
|---|---|---|
| point estimate (treatment - baseline) | +0.53s | +0.39s |
| 95% cluster-bootstrap CI | [-4.98, +6.74] | **[-0.17, +0.93]** |
| P(worse) | 0.560 | **0.911** |

Both CIs include zero (not significant), but EGLL's is tight and
centered on "no better, if anything slightly worse" -- a genuinely
informative null, not just an unmeasured one. Jan improved slightly
(366.6->364.2), Jul got worse (389.2->392.0) -- fails both-months too.

**Interpretation, as pre-registered: the production GBM already had
this information via existing feature interactions.** This retroactively
resolves the gap flagged when the raw-taxi and residual checks first
disagreed -- the Ridge proxy's weaker baseline was the entire
explanation, not evidence of a real, reachable gap. **Not adopted --
reverted** (`git checkout` on `build_features.py`, feature caches
regenerated back to baseline).

**Assessment of the day's arc, not just this item:** this was the only
idea all day to survive every free check that could have killed it
early (raw stability, correlation-with-existing-features, and a genuine
residual-based 12-month check) -- a meaningfully different outcome from
the four things that failed the moment a more careful version ran
(LFPG, echo-routing, `direct` range cap, per-airport heads). That it
*still* came back null on the actual build is the honest final word on
today's reslice-the-existing-data approach: even the best-vetted
candidate available from internal data alone didn't move the needle.
Reinforces, with about as much confidence as a single day of work can
provide, that weather and ATFM data are where the real remaining
information is, if it exists at all within current rules.

**Next steps:**
1. Airport-wide disruption proxy: closed. Do not rebuild without a
   genuinely new signal (not a repackaging of existing congestion/
   rotation columns) to feed it.
2. Weather (NOAA ISD) and ATFM/ground-stop regulation data remain the
   only unexplored, genuinely-new-information candidates. Recommended
   next real step for a future session, not attempted today.
3. Stopping point for 2026-09-18: `smart-jigsaw_v12` (302 actual) is
   still the standing submission. Five investigations today (LFPG,
   echo-routing x3, `direct` range cap, per-airport heads, arrival
   disruption) all closed without a new submission -- a legitimate,
   well-evidenced negative-results day, not a wasted one.

## 20. EUROCONTROL ATFM data ingested, built, tested -- null, and then
    measured *why* it had to be null. A granularity ceiling, not a
    tuning failure (2026-09-19)

S19 named ATFM data as one of the two genuinely-new-information
candidates left. It has now been fetched, built, tested, and closed --
but the useful output of this session is not the null result, it's the
reusable bound that explains it.

**Built.** `src/features/atfm.py` + family 5 in `build_features.py`,
from three EUROCONTROL Aviation Intelligence Unit datasets
(`atfm_slot_adherence`, `apt_dly`, `airport_traffic`; 2025 + 2026
vintages, licensing in `DATA_SOURCES.md` / `external-data/LICENSE`,
fetch in `src/ingest/fetch_atfm_data.py`). All three are keyed
(APT_ICAO, FLT_DATE). Joined twice per departure -- on `ADEP_mvt` (shifts
the whole day) and on `ADES_mvt` (varies flight to flight) -- yielding 19
columns: regulated-departure share, slot compliance, arrival ATFM delay
per flight, weather/staffing cause shares, traffic counts, plus
`ades_in_panel` flagging destinations outside the 377-ICAO
`airport_traffic` panel so the model sees missingness rather than an
imputed zero. Both 2026 ranking months are covered, so this was
deployable, not training-only.

**Tested** (`tests/atfm_v1_test.py`), paired on identical holdout rows,
`(airport, day)` cluster bootstrap, 3000 resamples. Baseline arm
reproduced the established `direct` numbers exactly (379.3s overall,
LIRF 908.0, Jan 366.6, Jul 389.2 -- byte-for-byte vs S13/S17), so the
harness is sound.

| | baseline | + family 5 |
|---|---|---|
| overall | 379.3 | 378.9 |
| 2025-01 | 366.6 | 361.1 |
| 2025-07 | 389.2 | 392.6 |

Paired RMSE delta **-0.38s, 95% CI [-7.98, +7.98], P(worse)=0.453**.
Fails the both-months rule (July, the harder scored month, regressed).

**The first diagnosis was a story, not a measurement** -- "raw counts
drowned out the cause-shares" -- and acting on it (an ablation) would
have skipped the checks that actually explain the result. Ran four
checks first (`tests/atfm_diagnose.py`, `tests/atfm_ceiling.py`), using
saved row-level eval frames from models trained *without* atfm features
(`cache/lirf_ceilfix_mixed_ev.parquet` = production v12), so no weak
proxy model was involved -- the exact trap that misled S19.

- **Data quality is fine.** Jan vs Jul, 2025 and 2026: non-null rates
  74-77%, traffic 100%, 373-375 airports each. July genuinely carries
  3.3x January's regulated share (0.383 vs 0.117, mirrored in 2026:
  0.356 vs 0.125) -- real summer ATFM pressure, not a data defect. So
  the Jan/Jul split is not a provisional-data artifact. (Note the
  holdout is Jan+Jul **2025**, so 2026 revision risk never applied.)
- **Coverage and the dual-join premise both hold.** ADES join rate 67%
  (holdout) / 67% (ranking), `ades_in_panel` ~0.70. ADES-side features
  genuinely vary flight-to-flight within an airport-day (within-day sd
  / overall sd = 0.60-0.70). The design worked as intended.
- **Residual gradients exist but are worth nothing.** Mean-residual
  spread across deciles looked like 5-34s. Converted to actual MSE value:
  best single feature is **0.31s in-sample** (optimistic, fitting decile
  means on the evaluation rows themselves) and **+0.03s under Jan->Jul
  transfer** -- with most features *negative* on transfer
  (`dep_atfm_reg_share` -0.75s, traffic columns -0.37s). Same
  anti-transfer signature as S18's per-airport residual structure: these
  gradients are month-specific noise. The measured -0.4s retrain result
  sits squarely inside this range, i.e. it is exactly what the theory
  predicts, not an anomaly.

**The structural bound, which is the real finding.** Decomposing the
production model's residual variance:

| granularity | noise-corrected oracle RMSE gain |
|---|---|
| (airport, day) -- bounds ALL ADEP-side features | **2.07s** |
| (airport, day, ADES) -- bounds the FULL dual join | 8.17s |
| (airport, day, hour) -- what HOURLY data could reach | 10.82s |

**98.6% of the production model's residual variance lives WITHIN an
(airport, day) cell.** Every ADEP-side ATFM feature is constant within
such a cell by construction, so an *oracle* knowing each cell's true
mean residual could win at most 2.07s -- and no interaction, encoding, or
ablation can exceed a bound set by granularity alone. The 8.17s dual-join
figure needs its own caveat: the naive number is 53.9s, but 84% of that
is small-cell noise, and what survives is mostly *destination identity*
rather than ATFM values (part A shows the ATFM values themselves transfer
at ~0.03s).

Noise correction mattered enormously here and is worth carrying forward:
a naive oracle on 55,342 cells looks like a 53.9s opportunity and is
actually 8.2s.

**Conclusion: this is a ceiling, not a tuning problem.** EUROCONTROL does
not publish these series at sub-daily resolution, so the mismatch cannot
be engineered away. **Not adopted.** Code, docs and licensing kept -- the
ingest path is correct and reusable, and the data is properly attributed
either way.

**The genuinely useful output: a pre-screening tool.** `tests/atfm_ceiling.py`
computes a noise-corrected oracle bound at any candidate dataset's native
granularity, *before* building anything. Applied to what is left:

- **Daily external data of any kind: capped at ~2s.** Do not build it.
- **Hourly external data: ~10.8s.** Weather (NOAA ISD) is hourly and does
  NOT share this defect -- this materially raises its priority relative to
  the "low single digits" estimate it carried since S4, and it should be
  screened with this tool first.

**Incidental finding, unrelated to ATFM but surfaced by the ceiling
work:** `ADES_mvt` is not a model feature at all (`CAT_COLS` has
`ADEP_mvt` only, and `build_features` never selects it). Destination
identity is internal, free, and plausibly physical -- destination
determines SID/departure route, hence runway exit and holding point. Much
of the 8.17s dual-join bound appears to be exactly this. Untested; a
better prior than anything remaining in the ATFM direction.

**Next steps:**
1. ~~`ADES_mvt` as a categorical feature~~ **TRIED SAME DAY, REJECTED --
   see S21.**
2. Weather (NOAA ISD), screened with `tests/atfm_ceiling.py` at hourly
   granularity before any build.
3. Do not revisit daily-granularity external data. The 2.07s bound is
   structural and applies to any such dataset, not just this one.

## 21. `ADES_mvt` as a plain categorical -- rejected, and unlike S20 this
    one is significantly *harmful* (2026-09-19)

S20's incidental finding was that `ADES_mvt` had never been a model
feature at all (`CAT_COLS` carried `ADEP_mvt` only). Physically motivated
(destination sets the SID/departure route, hence runway exit and holding
point) and, unlike family 5, not bounded away in advance -- so it got a
real test rather than an argument.

**Pre-check (free).** 1,569 distinct destinations in train, 1,106 in
ranking; 633 destinations cover 99.4% of rows; only **0.05%** of ranking
rows carry a destination unseen in training. Vocabulary is stable -- the
same check S2 ran for the other categoricals. Nothing here predicted the
outcome.

**Test** (`tests/ades_cat_test.py`). Family 5 dropped from BOTH arms (S20
did not adopt it), so this isolates destination identity exactly. Same
harness as S20: paired rows, `(airport, day)` cluster bootstrap, 3000
resamples. Baseline again reproduced 379.3s / LIRF 908.0 exactly.

| | baseline | + `ADES_mvt` |
|---|---|---|
| overall | 379.3 | **394.3** |
| 2025-01 | 366.6 | 379.8 |
| 2025-07 | 389.2 | 405.6 |
| LIRF | 908.0 | **986.1** |
| best_iter | 3310 | **2165** |

Paired RMSE delta **+15.04s, 95% CI [+0.74, +31.55], P(worse)=0.979.**
Note this CI *excludes zero* -- S20 was a null, this is a measured
regression.

**Mechanism is observable, not a story.** The damage is ~entirely LIRF
(+78.1s); every other airport moved by <=1.3s. And `best_iter` collapsed
3310 -> 2165: 1,569 categorical levels hand LightGBM an enormous space of
categorical splits, it fits the inner-validation month quickly on
memorised destination subsets, early-stops sooner, and leaves the model
undertrained overall. LIRF pays most because it has the heaviest label
tail to memorise and then fail to transfer -- the same population that
has dominated every error analysis since S8.

**Observation, deliberately not called a finding:** the `echo_pred=True`
lane moved 1994 -> 1398s. That looks large, but lane *membership* changed
between runs (n_pred_echo 2406 -> 2444), so the two numbers aren't
measured on identical rows and the comparison isn't paired. Not pursued
on that basis.

**Reverted** -- `ADES_mvt` removed from `CAT_COLS` and from
`build_features`'s base select, with a comment at the `CAT_COLS` site
recording why so it isn't re-added casually. Feature caches regenerated
to the clean production set.

**Also cleaned up in this pass:** family 5 is now behind
`build_features(..., atfm=False)` rather than always-on. Left wired but
off, not deleted -- the ingest path is correct and the reason it failed
is granularity, not correctness. This also means a fresh clone without
`external-data/` no longer breaks the feature build.

**Next steps:**
1. Weather (NOAA ISD) -- now the only remaining candidate with real
   headroom. Screen with `tests/atfm_ceiling.py` at hourly granularity
   (~10.8s oracle bound) BEFORE building anything.
2. If destination is revisited at all, it needs a low-cardinality
   encoding (frequency-capped, or great-circle distance/bearing, which
   would need an airport-coordinate source and its own licensing row) --
   but note S20's oracle bound puts the whole destination dimension in
   single-digit seconds at best, against a demonstrated -15s downside.
   Low expected value; not recommended.
3. Two "obvious wins" in a row (S20 ATFM, S21 ADES) came back null and
   negative respectively, both with the mechanism measured rather than
   argued. The pre-screening habit (oracle bound at native granularity,
   noise-corrected) is now the cheapest filter available -- use it before
   the build, not after.

## 22. "Why is the leaderboard front at ~245?" -- five hypotheses tested
    and killed in one session, plus two measurements worth keeping
    (2026-09-19)

Prompted by the standing question of how the front of the board sits at
~245s while `smart-jigsaw_v12` sits at 302s. Everything below is measured;
the scripts are `tests/error_floor.py`, `tests/echo_arrival_signal.py`,
`tests/extreme_rows.py`, `tests/bounds_on_extremes.py`,
`tests/train_gap.py`, `tests/loss_l2_vs_huber.py`.

### The two measurements that should outlive this session

**A. Where the error actually is** (production holdout, 337.6s):

| offset band | rows | % of total MSE | model R^2 |
|---|---|---|---|
| [0, 1800) | 203,854 (59%) | 30.7% | 53% |
| [1800, 3600) | 95,339 (28%) | 26.5% | 58% |
| [3600, 7200) | 33,357 (10%) | 18.5% | 60% |
| [7200, 14400) | 7,718 (2%) | 13.7% | 44% |
| >= 14400 | 1,383 (0.4%) | 10.1% | 95% |

57% of MSE is in ordinary flights (offset < 1h). Error IS also highly
concentrated by row (top 1,000 rows = 58.5% of MSE) but those two facts
coexist -- the top-k view alone misleads toward "it's all a few weird
rows", which it is not. **The model explains only 44-60% of variance in
the mid bands: it is not saturated.**

**B. There is a real variance component.** Same model, three populations:
train 166.4s / inner-valid 230.7s / holdout 246.4s. **train -> holdout
+48.1%**, while valid -> holdout is only +6.8% (so it is generalization
gap, not Jun-vs-Jan/Jul seasonality). Config is `num_leaves=255`,
`min_data_in_leaf=100`, ~3,310 trees = ~844k leaf values against 1.56M
training rows. High capacity, high variance, by construction.

**Implication:** part of holdout error is reducible with NO new
information. CLAUDE.md's ensemble stage was planned and never built. S5's
warning was about *fitted* blend weights backfiring -- equal weights won
every time for those contestants -- so equal-weight seed averaging is the
untried, plan-sanctioned form. Expect 1-3% (~3-10s), not 60s.

### Five hypotheses, all killed

1. **Arrival-side echo signal.** An echo is a data-feed artifact, so it
   should show on ARR rows too -- and arrival block times are never
   blanked, so it would be computable live on the 2026 ranking set.
   Pooled correlation between arrival and departure echo rate per
   (airport, day) is r=0.860 -- **and is a between-airport confound.**
   Per airport: LEBL -0.01, LEMD -0.00, LTFM 0.07, LFPG 0.14, EDDF 0.26,
   LIRF 0.45. The decile table is decisive: arrival-echo deciles d0-d8 all
   give departure-echo 4.1-4.6% (flat), only d9 jumps to 18.2%, and d9 is
   just LIRF days. It re-derives "LIRF is different", which `ADEP_mvt` and
   the existing group encodings already carry.
2. **Stand-occupancy bounds on the extreme population.** CLAUDE.md dropped
   Stage 4 on a whole-population median width; the hypothesis was that
   `L_sec` binds where it matters even if not on average. It does not.
   On offset >= 7200s: `L_sec > 0` on 6.7% of pushed-late vs 4.1% of
   sat-at-stand rows (no discrimination, slightly wrong direction), and
   `L_sec > 0.5*offset` is **0.0% in both arms**. Clipping predictions up
   to `L_sec` moves **23 rows**, RMSE +0.0s. Physically obvious after the
   fact: if an aircraft sat at its stand for hours, the next arrival came
   after it left, so `A_next` never constrains. **Stage 4 was correctly
   dropped, for the extreme population too.**
3. **L2 vs Huber.** Genuinely promising on paper: Huber was adopted in S3
   on 303.1 vs 309.2, and S6 later showed BOTH came from the buggy eval
   that excluded `taxi > 4h` -- exactly where a robust loss hurts. Never
   re-tested since. Theory: RMSE wants the conditional mean, L2 estimates
   it, Huber pulls toward the median, conditionals here are right-skewed
   mixtures. **Result: L2 is WORSE, +0.79s, 95% CI [+0.18, +1.41],
   P(worse)=0.996.** Mechanism falsified too -- under-prediction rates are
   identical (huber/l2 by band: 46/46, 44/43, 45/44, 48/48, 47/46%).
   **Huber stays.**
4. **"The model under-predicts its worst rows" (75%) -- SELECTION
   ARTIFACT**, and the reason hypothesis 3 looked strong. Selecting rows
   *on* squared error mechanically selects under-predictions when the
   target is right-skewed: you can under-predict by 8,000s but almost
   never over-predict by 8,000s, since taxi has a floor near zero.
   Unconditionally, predictions are balanced (~45% under in every band).
   Same class of error as S12's binning artifact. **Do not diagnose from
   outcome-conditioned subsets.**
5. **Echo routing as the path to 245.** Priced with oracles: perfect
   `is_echo` -> predict offset reaches 325.1s; perfect best-of-two routing
   reaches 296.7s. Board-calibrated (~x0.895) that is ~291s and ~265s.
   **Even a perfect router lands short of 245**, so the leaders are not
   getting there by special-casing extremes. Note also that under squared
   loss an uncertain `echo_prob` is ALREADY optimally blended -- the only
   way to win is a classifier that is genuinely more certain, i.e. new
   information, which is what S16 concluded and this re-confirms.

### What this says about the 245 question

To reach ~245 board needs ~275-280 local, a ~34% MSE cut. Oracle routing
between the current model and the offset heuristic caps out at 296.7 local.
**Therefore the remaining gap is not in the extreme-row special cases this
project has been mining since S7 -- it is in the ordinary population,
where the model explains only ~53-58% of variance.** Every win since S7
(echo lane, echo classifier, LIRF ceiling) was a targeted special-case
fix, and together they took 484 -> 302. That well now looks close to dry.

**Next steps:**
1. Equal-weight seed-averaged ensemble -- the only measurement-backed
   untried lever (measurement B), and in the original plan. Cheap.
   Expect single digits.
2. Capacity/regularization sweep motivated by measurement B specifically
   (`num_leaves` 255 is high for this data size), NOT blind hyperparameter
   search, which S5's contestants found insignificant.
3. Treat the ordinary bands as the target from here. Five consecutive
   special-case hypotheses failed; the band decomposition says that is
   where the remaining mass is.

## 23. Seed ensemble, OSM geometry screen, and the realised queue -- the
    last three candidates, all closed (2026-09-19)

**Seed-averaged ensemble: real, significant, negligible.** Four seeds
(42/7/13/99) at target="direct": 379.28 / 379.07 / 379.37 / 379.26.
Equal-weight average = **378.96s**, -0.29s vs mean single seed; paired
cluster bootstrap vs the incumbent seed 42 gives **-0.32s, 95% CI
[-0.45, -0.20], P(worse)=0.000**. Every airport and both months improve.
Statistically clean and practically irrelevant.

**This invalidates S22's reading of the train/holdout gap.** Seed spread is
sd=**0.12s** -- the four models are nearly identical, so the +48.1%
train->holdout gap is NOT stochastic variance that averaging can remove.
Every seed fits the *same* non-transferable structure in the training
labels. Note also `best_iter` swings 2405-3310 across seeds for
indistinguishable RMSE, i.e. a very flat loss surface, which independently
argues capacity/round tuning has little left (consistent with S5).
**Corollary for future sessions: do not cite the train/holdout gap as
evidence that variance reduction is available. It was measured and it is
0.3s.**

**OSM routed distance (CLAUDE.md "Future Step B") -- screened and killed
without building it.** Routed distance is a deterministic function of
(airport, stand, runway), so that cell structure bounds it exactly
(method as S20, noise-corrected):

| granularity | cells | raw | corrected |
|---|---|---|---|
| airport | 10 | 0.09 | 0.09 |
| airport + runway | 59 | 0.31 | 0.28 |
| airport + stand | 2,432 | 2.88 | 1.68 |
| airport + stand + runway | 9,123 | 9.38 | **4.81** |

Capped at **4.81s**. And the specific argument for geometry -- that a
metric generalises where a categorical has thin training support -- is
worth **0.00s**: within the thin-support subset (<50 training rows per
stand+runway, 17,383 rows carrying 23.2% of MSE) the raw 64.37s bound is
*entirely* small-cell noise at ~4.3 rows/cell. Future Step B would have
cost per-airport OSM graphs, stand-coordinate matching and ODbL
compliance for at most 4.8s. **Closed. Do not build it.**

**Realised pushback-anchored queue (family 6) -- the one candidate that
could NOT be bounded in advance.** S14's own correction flagged it as the
better version of the Stage 3b hypothesis and nobody had built it.
`AOBT_3_flt` is 98.5% populated in ranking DEP rows, so a queue built from
neighbours' realised pushbacks is deployable with no OOF machinery.

Causality was the risk (a queue must contain only aircraft that pushed
back strictly BEFORE this flight; anchoring on T instead of on own
pushback would, for a long-taxi flight, count aircraft *behind* it). Built
anchored on `AOBT_3_flt` with strict `AOBT_3_j < AOBT_3_i`, and verified
rather than asserted (`tests/queue_causality_check.py`): brute-force
recomputation from the definition matches on 400 rows x 4 features, and
shifting a flight's own takeoff by +3h leaves its own features unchanged
25/25. The check also caught two real bugs pre-RMSE: `q_ahead_rwy >
q_ahead` on 39 rows (flights with takeoff before pushback break the
interval arithmetic; a `max(...,0)` clamp hid it asymmetrically between
the full and per-runway populations) and NaN-vs-null emission poisoning
distribution summaries.

| | baseline | + queue |
|---|---|---|
| overall | 379.3 | **381.1** |
| 2025-01 | 366.6 | 363.8 |
| 2025-07 | 389.2 | **394.5** |
| LIRF | 908.0 | 920.1 |
| offset >= 14,400 | 3,238.1 | **3,308.3** |

**+1.81s, 95% CI [-3.61, +8.64], P(worse)=0.642.** Fails both-months.

**The pre-registered mechanism test is the informative part.** Gain by
queue depth: -0.4 / **+2.6** / -0.9 / -1.4 for q_ahead in [0,5) / [5,10) /
[10,15) / [15,100). **No gradient** -- if this captured queue mechanics the
gain would scale with queue length. Likely explanation: for an ordinary
~15-minute taxi, family 2's takeoff-anchored 30-minute window overlaps
heavily with the pushback-anchored queue window, so the two measure nearly
the same thing. That fits the whole shape of the result -- 8/10 airports
improve by a trivial -0.1 to -2.2s, swamped by LIRF and the >=4h-offset
band where the two anchorings diverge and the feature destabilises.

**So the Stage 3b hypothesis is now ANSWERED rather than untried: the
takeoff-anchored proxy was already good enough.** Not adopted; kept behind
`build_features(..., queue=False)` since the implementation is verified
correct -- it is the hypothesis that failed, not the build.

### Session tally and the honest position

Eight candidates tested in one day, every one null or harmful: ATFM data
(S20), `ADES_mvt` (S21), arrival-echo signal, stand-occupancy bounds,
L2-vs-Huber, seed ensemble (+0.3s), OSM geometry, realised queue. Standing
submission is unchanged at `smart-jigsaw_v12`, 302s.

What is now measured rather than assumed:
* the gap to ~245 needs ~60s local, and **perfect** oracle routing between
  the current model and the offset heuristic only reaches 296.7 local
  (~265 board) -- so the remaining gap is NOT in extreme-row special cases
* 57% of MSE is in ordinary flights where the model explains 53-58% of
  variance -- it is not saturated, so the headroom is real
* every cell-granularity candidate screened is bounded small (daily 2.07s,
  stand+runway 4.81s, airport+day+hour 10.82s)
* variance reduction is measured at 0.3s, not available
* the one flight-level candidate available from internal data (realised
  queue) is redundant with existing takeoff-anchored congestion

**Next steps:**
1. The screening tools (`tests/atfm_ceiling.py`, `tests/geometry_screen.py`)
   now make a candidate cheap to kill *before* a build. Use them first;
   today they would have saved two of the eight builds.
2. Remaining unexplored classes are genuinely flight-level and external
   (e.g. OpenSky ADS-B ground trajectories, still an open permissions
   question per S13 #3). Note the integrity line: training-side feature
   engineering only, never reconstructing blanked 2026 labels.
3. Do not re-mine the extreme rows. The oracle bound above settles it.

## Dead ends — quick reference (2026-09-19)

Short list of what was tried and failed today, so nobody re-derives it.
All measured on the Jan+Jul 2025 holdout, `target="direct"` baseline
379.3s, paired (airport, day) cluster bootstrap. Detail in S20-S23.

| # | tried | result | why it failed |
|---|---|---|---|
| 1 | EUROCONTROL ATFM data (slot adherence, apt_dly, airport traffic), joined on ADEP + ADES | -0.4s, CI [-7.98, +7.98], Jul regressed | published per (airport, day); 98.6% of residual variance is WITHIN an airport-day -> ~2.07s ceiling for ANY daily feature |
| 2 | `ADES_mvt` as a plain categorical | **+15.0s**, CI [+0.74, +31.55] | 1,569 levels collapse best_iter 3310->2165; model early-stops on memorised destination splits. Damage ~all LIRF (+78s) |
| 3 | Arrival-side echo rate as a live echo predictor | no usable signal | pooled r=0.860 is a between-airport confound; per-airport r = -0.01..0.45, and arrival-echo deciles d0-d8 are flat. Re-derives "LIRF is different" |
| 4 | Stage 1 `L_sec` stand-occupancy bounds on the extreme population | 23 rows moved, +0.0s | if an aircraft sat at its stand for hours the next arrival came after it left, so `A_next` never constrains. `L_sec > 0.5*offset` is 0.0% in both arms |
| 5 | L2 instead of Huber (adopted in S3 on pre-S6 buggy-eval evidence) | +0.79s, CI [+0.18, +1.41] | Huber genuinely better. Mechanism also falsified: under-prediction rates identical (46/46, 44/43, 45/44%) |
| 6 | Equal-weight 4-seed ensemble | -0.29s (significant, negligible) | seed spread is sd=0.12s; the models are near-identical, so the +48% train/holdout gap is NOT averageable variance |
| 7 | OSM routed taxi distance (CLAUDE.md "Future Step B") | killed pre-build, 4.81s ceiling | deterministic function of (airport, stand, runway); thin-support subset bound is **0.00s** after noise correction |
| 8 | Realised pushback-anchored queue (family 6, S14's untried Stage 3b variant) | +1.81s, CI [-3.61, +8.64], Jul regressed | no gain gradient with queue depth (the pre-registered mechanism test) -- family 2's takeoff-anchored 30-min window already measures nearly the same thing for an ordinary ~15-min taxi |

Two of our own diagnostic framings were also wrong and are worth not
repeating:

* **"the model under-predicts 75% of its worst rows"** -- selection
  artifact. Conditioning on large squared error mechanically selects
  under-predictions when the target is right-skewed (you can under-predict
  by 8,000s, never over-predict by 8,000s against a floor of ~0).
  Unconditionally predictions are balanced (~45% under in every band).
  Same class as S12's binning artifact.
* **"the +48% train/holdout gap means variance is reducible"** -- measured
  at 0.3s (#6). Do not cite that gap as evidence of available headroom.

Standing submission unchanged: `smart-jigsaw_v12`, 302s.

## 24. METAR weather (family 7) -- built, tested, NOT adopted; real but
    outweighed signal, LIRF absorbs the collateral damage again
    (2026-09-20)

Prompted by a teammate's report of a null result from GHCN-h. Could not
find that test anywhere in this repo (no branch, commit, or report
mentions GHCN-h) to check what feature set/mechanism-test it actually
used, so per user direction the check was skipped and this went straight
to building METAR instead. Independently confirmed via NOAA NCEI docs that
the "GHCN-h lacks visibility/ceiling" premise was wrong regardless -- GHCN-h
v1 does carry `visibility`, `sky_cov_baseht_1-3` (ceiling), and the same
present-weather groups (`pres_wx_AU/AW/MW`) as METAR -- moot for this
session's path but worth knowing if GHCN-h comes up again.

**Data.** Hourly METAR (report_type=3, routine obs only) for the 10
challenge departure airports (LTAI excluded, zero departures) from the Iowa
Environmental Mesonet archive, full 2025 + Jan/Jul 2026: 8.4MB, ~10,250
rows/station. Source is NOAA/NWS (public domain); IEM states no additional
license/attribution terms on the dataset itself. Full row in
DATA_SOURCES.md. Fetch: `src/ingest/fetch_metar_data.py`.

**Pre-screened before building anything** (`tests/metar_ceiling.py`, reusing
the S20 noise-corrected granularity-bound method): (airport, day, hour)
oracle ceiling reproduces **10.82s exactly** on the current production
model (`lirf_ceilfix_mixed_ev.parquet`) -- confirms weather's hourly
granularity clears the ~2.07s daily bound that killed ATFM (S20), i.e. worth
building.

**Built** (`src/features/weather.py`, family 7 in `build_features.py`,
`weather=False` by default): visibility, ceiling (lowest BKN/OVC/VV layer;
null means genuinely unlimited ceiling, kept explicit via `has_ceiling`
rather than silently conflated with a missing METAR hour), temperature,
dewpoint, `temp_dewpoint_spread_c` (prioritized over a binary snow flag per
the contestants' own reported finding, PROGRESS.md S5), `below_freezing`,
`deicing_risk` (freezing + precip/snow/FZ code), present-weather flags
(snow, freezing, thunderstorm, obscuration), a standard `flight_category`
(VFR/MVFR/IFR/LIFR, 4-level categorical, added to `CAT_COLS` -- unlike
`ADES_mvt`'s 1,569 levels this is deliberately low-cardinality), wind, and
1h precipitation. Joined on (ADEP_mvt, hour bucket of `T`=takeoff time,
never blanked) -- same anchor family 2 uses. Caught and fixed a tz-aware
(T) vs tz-naive (raw METAR `valid`) join bug before it could reproduce the
other team's 19-submission silent-null incident (S5) -- explicit UTC tz on
both sides, plus a hard null-rate assertion right after the join. Coverage
after the fix: 99.99%+ non-null on train, holdout, AND ranking (both 2026
months) -- deployable, not training-only.

**Pre-registered test** (`tests/metar_weather_test.py`, same harness as
`atfm_v1_test.py`: paired per-row squared error, (airport, day) cluster
bootstrap, 3000 resamples, both-months rule, plus a mechanism test specific
to this feature -- the gain must concentrate on low-vis/freezing rows, not
spread uniformly, same logic that would have rejected the queue feature
S23):

| | baseline | + weather |
|---|---|---|
| overall | 379.3 | **384.1** |
| 2025-01 | 366.6 | 362.4 |
| 2025-07 | 389.2 | **400.8** |
| LIRF | 908.0 | **936.3** |
| best_iter | 3310 | 2669 |

Paired RMSE delta **+4.86s, 95% CI [-1.70, +12.01], P(worse)=0.917.** July
(the harder scored month) regressed; **fails the both-months rule**
outright regardless of the CI technically spanning zero.

**Per airport, the damage is almost entirely LIRF.** 9/10 airports moved
by -2.9 to +2.2s (noise level); LIRF alone moved +28.3s (908.0 -> 936.3),
and at LIRF's ~44% share of total squared error that single airport
accounts for roughly half the overall regression. `best_iter` dropped
3310 -> 2669 (19% fewer trees before early stop) -- the same undertraining
signature S21 measured for `ADES_mvt` (extra split surface lets the model
fit the inner-validation month faster and stop early, leaving it
undertrained everywhere, LIRF's heavy tail paying for it most).

**The mechanism test is genuinely interesting, unlike every other S18-S23
rejection: it partially PASSES.** The gain does NOT spread uniformly --
it concentrates exactly where the pre-registered mechanism predicted:

| regime | n | delta (treat - base RMSE) |
|---|---|---|
| low_vis (IFR/MVFR/LIFR) | 68,510 (19.9%) | **-6.3s** |
| VFR (ordinary) | 275,829 (80.1%) | +8.9s |
| below_freezing | 13,660 (4.0%) | **-8.6s** |
| above freezing | 330,679 (96.0%) | +5.3s |
| deicing_risk | 1,321 (0.4%) | **-33.5s** |
| no deicing_risk | 343,018 (99.6%) | +5.1s |

Real, correctly-signed weather signal exists in the small adverse-condition
populations (de-icing rows improve by 33.5s). It is simply outweighed in
net RMSE by (a) sheer row count -- 80-99.6% of rows are ordinary conditions
where the extra columns are just noise to fit, and (b) the LIRF-specific
undertraining collateral damage, the same failure mode that killed
hour-of-day encoding (S13) and `ADES_mvt` (S21): any added split surface
disproportionately hurts LIRF's already tail-heavy fit.

**NOT adopted under `target="direct"`.** But `direct` is only the fast
day-to-day diagnostic target, not what ships -- `target="mixed"` (flip for
LIRF, direct everywhere else) is production (S11). Re-ran the identical
test under `mixed` before writing this off. Result reverses:

| | baseline (mixed) | + weather (mixed) | delta |
|---|---|---|---|
| overall | 337.6 | **336.4** | **-1.21s** |
| 2025-01 | 356.1 | 354.5 | -1.6s |
| 2025-07 | 321.8 | 321.0 | -0.8s |
| LIRF | 660.6 | **655.7** | **-4.9s** |
| best_iter (flip / direct) | 7998 / 3310 | 7997 / 2669 | flip untouched |

**Paired cluster bootstrap: -1.21s, 95% CI [-2.48, -0.13], P(worse)=0.013.**
Both months improve. 8/10 airports improve or hold flat (<=0.9s); only LTFM
regresses, +2.2s. **LIRF improves instead of absorbing collateral damage**
-- confirms the mechanism suspected from the `direct` result: `flip`'s
`d`-based inductive bias (the same property that made it worth the CEIL fix
in S12) is far more robust to the extra weather split surface than `direct`
is. `direct`'s own `best_iter` still drops 3310->2669 (same undertraining
signature as before, unchanged by this rerun) but LIRF no longer routes
through it in `mixed`, so that collateral damage no longer reaches the
number that matters.

**CORRECTION (S25): the mechanism test does not cleanly pass, and the
paragraph originally written here overstated it.** Under `mixed`, every
regime improves (no regime went net negative, unlike the `direct` run) and
the PER-ROW concentration pattern still holds -- deicing_risk -33.5s,
below_freezing -8.6s, low_vis -2.4s vs VFR -0.7s, above-freezing -0.9s.
But the pre-registered mechanism claim was that the gain concentrates on
adverse conditions, and per-row delta is the wrong lens to judge that --
share of the AGGREGATE gain is. Since low_vis+VFR, freezing+non-freezing,
and deicing+non-deicing are each a complete partition of all 344,339
holdout rows, their contributions to the total -808.8 MSE-delta (-1.21s
RMSE) sum to ~100% and can be attributed exactly, not estimated:

| partition | adverse side | adverse share | ordinary side | ordinary share |
|---|---|---|---|---|
| low_vis / VFR | low_vis (19.9% of rows) | **59.2%** | VFR (80.1%) | 40.2% |
| freezing | below_freezing (4.0%) | 25.1% | above_freezing (96.0%) | **72.3%** |
| de-icing | deicing_risk (0.4%) | 17.8% | no deicing_risk (99.6%) | **82.7%** |

Only the broadest adverse proxy (low_vis) keeps a majority share (59.2%),
and even there 40.2% of the realized gain -- a substantial, non-trivial
fraction -- comes from ordinary VFR rows. By the narrower, more literally
"de-icing" proxies, ordinary conditions supply the clear majority of the
gain (72-83%), because even a 30x larger per-row effect can't outweigh a
population 200x bigger. **The mechanism is real and correctly directional
(adverse rows do benefit disproportionately per-row), but it is not what is
mostly driving the realized number** -- part of the win is "the model fits
marginally better across the board once LIRF isn't destabilized by the
extra split surface," a `mixed`-specific stabilization effect, not weather
explaining variance specifically in bad conditions. This is the same
gradient-flattening failure mode that would have been used to REJECT the
queue feature (S23, "no gain gradient with queue depth") if the per-row
deltas alone were misread as a clean pass -- the honest reading here is
"partially confirmed, majority-driven-by-share only for the broadest
proxy," not "passes."

Isolated-gain ceiling arithmetic (requested before any further build, now
superseded by the exact partition attribution above but kept for the
single-regime numbers): converting each regime's RMSE delta into its
MSE-weighted contribution to the overall number, holding everything else
at baseline -- deicing_risk alone is worth ~0.19s, below_freezing ~0.27s,
low_vis ~1.64s.

**Status: ADOPTABLE under `target="mixed"`, pending a decision** -- clears
every bar this project uses on the primary metric (significant paired CI
excluding zero, both months, no airport regression beyond noise). The
mechanism test is a genuine partial pass (adverse rows benefit more per
row, and the broadest adverse proxy keeps a bare majority share) but not
the clean pass the pre-registration was hoping for -- log it as a real,
mostly-`mixed`-stabilization-driven gain with a real but partial weather
component, not a demonstration that weather explains taxi variance in bad
conditions specifically. Small (-1.21s, ~0.4%) but real, and the first
EXTERNAL-data family (vs. internal resliced features) to survive this
project's screening process at all -- S18-S23 killed eight candidates in a
row before this.
`weather=False` is still the code default pending that decision; flipping
it and re-running the full submission-ready `mixed` pipeline (refit on all
2025 + predict ranking, ~45-60 min per S10) is the next step if adopted.
`cache/features_metar_tmp/` (the temporary treatment build) is under
`cache/`, already gitignored, not committed.

**A note on what a second-seed check can and can't confirm here (flagged
before running one, S25 item 5):** if part of the gain is `mixed`-specific
stabilization of LIRF's reconstruction rather than weather information
itself (per the partition attribution above), a second seed is a weaker
check than it usually is in this project. It can confirm the -1.21s
NUMBER reproduces (rules out a single-run fluke) -- it cannot confirm the
MECHANISM is what the pre-registration assumed, since a stabilization
effect and a genuine weather effect would both reproduce across seeds
identically. Treat a matching seed-7 result as "reproducible," not as
independent confirmation of "the de-icing/low-vis story is what's
happening."

**Candidate refinements, NOT built this session** (mostly superseded by the
`mixed` result, kept for completeness):
1. Whether LIRF's own regression under `direct` was concentrated in
   ordinary or adverse conditions is now moot for the adoption decision
   (LIRF uses `flip` in production and improves there) but would still be
   informative for understanding `direct`'s standalone weakness.
2. More boosting rounds/capacity for `direct` to compensate for its
   undertraining -- lower priority now that `mixed` sidesteps the problem.
3. Restricting the weather join to exclude LIRF -- unnecessary under
   `mixed`, since LIRF now improves rather than regresses.

**Decision (2026-09-20): adopted.** `weather=True` is now the default in
`export_model_inputs.py` (`atfm`/`queue` stay off, S20/S23). Feature cache
rebuilt (`train2025.parquet` 84.5MB -> 88.5MB) and the full submission-ready
`target="mixed"` pipeline run end-to-end (`run_metar_mixed_full.py`,
holdout fit + full refit on all 2025 + ranking predict, 2991s):

**Holdout: 335.9s** (flip best_iter=7997, direct best_iter=3169), matching
the isolated test's 336.4s. LIRF 655.1s (further improved from the isolated
test's 655.7s). Both months: Jan 353.9s, Jul 320.7s.

Wrote `data/submissions/smart-jigsaw_v14.parquet`. Integrity independently
re-verified outside `_write_submission`'s own asserts: row count matches
`submitting.parquet` (344,841), `MVT_ID_mvt` set equality, zero nulls, zero
negatives, sane distribution (median 959s, max 110,368s -- consistent with
the LIRF CEIL-relaxation from S12 still engaging correctly). **Not yet
uploaded** -- built and verified locally, upload is a separate explicit step
(`src/post/upload.py`) held pending user go-ahead.

(Note: `data/submissions/smart-jigsaw_v13.parquet` also exists, timestamped
10:58 the same day -- before this session's weather work started. Not
something this session produced, and it predates every file this session
touched; flagging since its provenance isn't in this log, same as the S7
note about v8.)

## 25. `target="direct"` was the wrong instrument for feature testing --
    auditing every prior `direct`-only rejection under `mixed`
    (2026-09-20)

S24's `direct`-vs-`mixed` flip (weather: rejected at +4.86s under `direct`,
adopted at -1.21s under `mixed`, LIRF swinging 33s between the two) raised
an obvious question: every candidate rejection since S18 (ATFM, `ADES_mvt`,
arrival-echo, stand-occupancy bounds, L2-vs-Huber, seed ensemble, OSM
geometry, realised queue, hour-of-day encoding) was tested under `direct`
because it's faster -- but `direct` is not what ships. `target="mixed"`
(flip for LIRF, direct everywhere else, S11) is production. A `direct`-only
result is a valid statement about `direct`; it doesn't answer the question
actually being asked ("does this belong in the submitted model").

**1. Per-airport delta comparison, direct vs mixed (no compute -- tables
already in hand from S24).** For the weather feature specifically:

| airport | direct delta | mixed delta | same? |
|---|---|---|---|
| EDDF | -0.9 | -0.9 | yes |
| EDDM | -2.9 | -2.9 | yes |
| EGLL | +0.6 | +0.6 | yes |
| EHAM | -0.7 | -0.7 | yes |
| LEBL | -0.4 | -0.4 | yes |
| LEMD | -0.5 | -0.5 | yes |
| LFPG | -2.0 | -2.0 | yes |
| **LIRF** | **+28.3** | **-4.9** | **no -- 33.2s swing** |
| LSZH | -2.0 | -2.0 | yes |
| LTFM | +2.2 | +2.2 | yes |

Not just "almost entirely LIRF" -- **exactly, structurally 100% LIRF, and
this is provable rather than empirical.** `target="mixed"` fits the direct
regressor as a fully independent, deterministic fit (same training data,
same seed, `deterministic=true`/`force_row_wise=true` per CLAUDE.md
conventions) -- `mixed` only swaps LIRF's *reconstruction* from direct to
flip (`_reconstruct_taxi`, per-airport `np.where` on `ADEP_mvt`, S10). The
other 9 airports' predictions under `mixed` ARE the direct regressor's
predictions, byte-for-byte, whether or not `mixed` also independently fits
a flip model alongside it. This generalizes directly: for any feature test,
the non-LIRF per-airport deltas from an existing `direct`-only run are
already guaranteed to reproduce under `mixed`; a rerun can only tell you
what happens to LIRF once it routes through flip instead, and to the shared
early-stopping curve from fitting both regressors together. Re-testing the
other 9 airports individually would spend compute to reconfirm a
mathematical certainty.

**2. `mixed` is now the harness default.** New `tests/_harness.py`:
`DEFAULT_TARGET = "mixed"` plus the (previously copy-pasted into every
test file) `cluster_bootstrap` function. `tests/metar_weather_test.py`,
`tests/queue_test.py`, `tests/ades_cat_test.py` now import both and default
`target` to `DEFAULT_TARGET`, with a CLI override still available for a
fast `direct` check. `metar_weather_test.py` also gained a `seed` param
(for item 5 below) and now prints each mechanism-test regime's exact SHARE
OF THE TOTAL REALIZED GAIN (the partition-attribution method from the §24
correction above), not just its per-row delta, so a future run can't
misread a flattened gradient as a clean pass again. `atfm_v1_test.py` and
`geometry_screen.py` are deliberately NOT touched -- their rejections come
from residual-structure ceilings (2.07s daily bound, 4.81s geometry bound)
that are computed directly from the production `ev` frame, not from a
fresh model fit, so they don't depend on `target` at all and re-running
them under `mixed` would just reproduce the same bound with extra steps.

**Two more ambient-cache bugs found and fixed before either rerun could
produce a number.** Both `queue_test.py` and `ades_cat_test.py` (S23, S21)
were written the same way `atfm_v1_test.py` was: treatment = `FEAT_DIR`
as-is, baseline = `FEAT_DIR` with the family's columns dropped -- valid ONLY
if the ambient shared cache already contains the family being tested. That
was true when each was originally run (someone had rebuilt the cache with
that family on first), but `cache/features/` has been rebuilt since (now
weather=True, both families still off) -- so `FEAT_DIR` contains NEITHER
queue nor `ADES_mvt` columns any more, and the "drop to get baseline" trick
silently produces a baseline and treatment that are IDENTICAL (nothing to
drop) rather than erroring.

- `queue_test.py`: crashed loudly (`ColumnNotFoundError: q_ahead`) --
  `_realised_queue` needs raw `AOBT_3_flt`/`PHASE_mvt`/`RUNWAY_mvt` that
  aren't in the already-derived cached feature parquet at all, so there was
  no column to silently keep. Fixed: treatment is now rebuilt from raw data
  (`build_features(..., weather=True, queue=True)`, matching current
  production flags plus queue), baseline is `FEAT_DIR` unchanged.
- `ades_cat_test.py`: did NOT crash -- silently would have produced a
  no-op comparison, the more dangerous failure mode, since `ADES_mvt` was
  never in `CAT_COLS`/the base feature select at all (S21), so "keep it by
  not dropping it" kept nothing. Fixed the same way: treatment rebuilds
  `ADES_mvt` from raw and registers it in `encode.CAT_COLS` for the
  duration of that one `run()` call only (via `try/finally`, restored
  immediately after, production `encode.py` untouched on disk).
- A third instance hit `metar_weather_test.py` itself once weather became
  the production default (below) -- same root cause, opposite direction.

Both fixes verified with a dry-run column check before spending compute on
a full rerun (see commit/session notes). Lesson for future sessions: a
test script that reads the shared `cache/features/` ambient state instead
of asserting what it expects to find there will silently go stale the next
time that cache is rebuilt for an unrelated reason. `metar_weather_test.py`
now auto-detects which side of the comparison the ambient cache represents
(S25) rather than assuming a fixed direction; `queue_test.py` and
`ades_cat_test.py` now always rebuild their own treatment side from raw
data rather than trusting the ambient cache for anything except the
baseline.

**3. Realised queue, rebuilt-treatment rerun under `mixed`.** Confirmed
null, no config artifact:

| | overall | 2025-01 | 2025-07 | LIRF |
|---|---|---|---|---|
| baseline | 335.9 | 353.9 | 320.7 | 655.1 |
| treatment | 336.1 | 354.0 | 321.0 | 658.4 |
| delta | +0.2 | +0.1 | +0.3 | **+3.2** |

Paired cluster bootstrap: **+0.17s, 95% CI [-0.32, +0.64], P(worse)=0.758.**
LIRF moved only +3.2s -- nowhere near weather's 33s swing -- so this is not
a repeat of the `direct`-vs-`mixed` flip; S23's "not adopted" verdict holds
under production config too. Queue-depth gradient (the original mechanism
test) is still flat: -0.0 to -1.1s across `q_ahead` bins, no monotonic
relationship. **Confirmed dead, not just under `direct`.**

**4. `ADES_mvt`, rebuilt-treatment rerun under `mixed`.** Also null overall,
but the mechanism behind the S21 rejection does NOT reproduce:

| | overall | 2025-01 | 2025-07 | LIRF |
|---|---|---|---|---|
| baseline | 335.9 | 353.9 | 320.7 | 655.1 |
| treatment | 335.9 | 354.6 | 319.9 | **651.4** |
| delta | -0.1 | +0.7 | -0.8 | **-3.7** |

Paired cluster bootstrap: **-0.07s, 95% CI [-2.99, +3.10], P(worse)=0.472**
-- indistinguishable from a coin flip. **LIRF's S21 catastrophe (+78.1s
under `direct`, `best_iter` collapsing 3310->2165) does not reproduce here**
-- LIRF slightly improves (-3.7s) instead, confirming that specific number
was a `direct`-target undertraining artifact, not something intrinsic to
`ADES_mvt` as a feature. `best_iter` for the `direct` component still drops
(3169->2485, the same undertraining signature) but LIRF routes through
`flip` under `mixed` (7997->7994, untouched) so that damage no longer lands
on the one airport that dominates total error. **But it doesn't just
vanish either** -- it resurfaces as a small, consistent regression spread
across 8 of 9 non-LIRF airports (EDDF +0.7, EDDM +0.6, EGLL +2.0, EHAM +0.7,
LEBL +0.3, LFPG +0.7, LSZH +1.3; only LTFM improves, -0.6), which roughly
cancels LIRF's improvement. Answers the pre-registered question precisely:
the original headline number WAS a config artifact, but the feature is
still dead -- for a materially different, smaller, more diffuse reason than
S21 found (1,569-level undertraining spread thinly across the pooled model,
not concentrated on one airport's tail). **Confirmed dead under `mixed`,
not adopted, no further action.**

**5. Weather, second seed (7) under `mixed`.** Also caught and fixed a
third instance of the ambient-cache bug (item 2/3/4 above) here: once
`weather=True` became the production default, `FEAT_DIR` silently already
had the 16 weather columns baked in, so "baseline (no weather)" was
actually reading weather-included data, and the treatment build then tried
to left-join weather again onto it, colliding on `flight_category` and
producing a stray `_right`-suffixed string column LightGBM can't train on
(`ValueError: pandas dtypes must be int, float or bool`). Fixed:
`metar_weather_test.py` now checks `_feat_dir_has_weather()` at runtime and
builds whichever side (baseline or treatment) the ambient cache is
currently missing, instead of assuming a fixed direction. Verified with a
dry-run column check before rerunning.

Reproduces and strengthens:

| | seed 42 (original) | seed 7 |
|---|---|---|
| overall delta | -1.21s | **-1.73s** |
| 95% CI | [-2.48, -0.13] | [-3.09, -0.61] |
| P(worse) | 0.013 | **0.001** |
| LIRF delta | -4.9s | -6.8s |
| both months | yes (-1.6/-0.8) | yes (-2.1/-1.4) |

**Number confirmed, not just seed-42 noise.** But per the pre-registered
caveat about what a second seed can and can't show (noted in the S24
correction before this ran): it confirms REPRODUCIBILITY, not attribution.
The mechanism-share split moved, if anything, further toward "ordinary
rows explain most of it" under this seed -- low_vis/VFR share is 51.1%/48.9%
(vs seed 42's 59.2%/40.2%, already a bare majority), and the
freezing/de-icing partitions are MORE ordinary-dominated (below_freezing
share 18.9% vs 25.1%; deicing_risk share 12.6% vs 17.8%). The gain itself
is robust across seeds; the "adverse conditions explain most of the gain"
reading is not -- if anything it weakens on a second look. Logged honestly:
**adopted for the RMSE gain, not as evidence the pre-registered de-icing/
low-vis mechanism is the primary driver.**

**Session tally (S25):** 3 of 3 reruns landed (queue confirmed dead, ADES
confirmed dead via a different mechanism than originally measured, weather
confirmed adopted and reproducible), plus 3 ambient-cache bugs found and
fixed in the test harness before they could corrupt a result (one loud
crash, one silent no-op that would have been worse, one crash from the
adoption decision itself changing the ground truth mid-session). No change
to the standing submission (`smart-jigsaw_v14`, weather adopted, queue and
ADES stay off) -- this session's reruns confirmed the existing decisions
were correct and better-understood, not something requiring a new
submission.

## 26. The unexplained half of the weather gain -- ablated by mechanism,
    and a free LIRF noise-floor calibration (2026-09-20)

Two follow-ups on S25's mechanism-attribution correction: (a) is the gain
weather-as-operations or a seasonal/diurnal proxy the model's existing
month/hour features encode weakly, and (b) how unusual is LIRF's
sensitivity to config changes, calibrated against pure seed noise.

**LIRF noise floor (free, from S25's existing seed42/seed7 logs -- no
compute).** Same script, same config, only the seed differs, so the spread
in LIRF's treatment-minus-baseline delta is a clean noise floor:

| | seed 42 | seed 7 | \|diff\| |
|---|---|---|---|
| baseline (no weather) LIRF | 660.6 | 662.0 | 1.4s |
| treatment (weather) LIRF | 655.7 | 655.2 | 0.5s |
| delta (treat-base) | -4.9 | -6.8 | **1.9s <- noise floor** |

Calibrated against this floor:
- Weather's `direct`->`mixed` config swing on the delta (S25 item 1): 28.3s
  -> -4.9s, a 33.2s swing = **17.5x the noise floor**. Confirms what S25
  already argued structurally (deterministic direct-regressor fit, LIRF the
  only airport whose reconstruction changes target) -- now with an
  independent quantitative check: this is not something a different seed
  could produce by chance.
- Queue's LIRF delta (+3.3s, mixed only): **1.7x** the floor --
  indistinguishable from noise. Consistent with queue's confirmed-null
  verdict (S25 item 3).
- `ADES_mvt`'s LIRF delta (-3.7s, mixed only): **1.9x** the floor -- also
  indistinguishable from noise. Consistent with its confirmed-null verdict
  (S25 item 4).

**Cheap sanity check before any fit: weather columns vs month/hour.**
Pearson correlation on the full `train2025.parquet` (N=2,085,047): all
weak, |r|<=0.27 (`dewpoint_c` vs month strongest; `vis_mi`/`ceiling_ft`/
`below_freezing`/`deicing_risk` all |r|<=0.08 against both month and hour).
Nothing in the raw linear correlations screams "this is just calendar."
But `flight_category`'s non-VFR rate has a real, large, physically sensible
swing the correlation misses (threshold/tail effect, not a linear trend):
**6.4% (June) -> 42.6% (December)** by month, **~14% (afternoon) -> 29%
(night)** by hour (radiation fog). Not decisive either way -- proceeded to
the ablation rather than skip it.

**Ablation, three arms, same seed(42)/target(mixed), built by dropping
columns from the (already weather-included) production cache**
(`tests/metar_ablation_test.py`): baseline (no weather), adverse-only
(`has_ceiling`, `vis_mi`, `ceiling_ft`, `below_freezing`, `deicing_risk`,
`wx_snow`, `wx_freezing`, `wx_tstorm`, `wx_obscuration`, `flight_category`,
`precip_1h_in` -- 11 cols), benign-only (`temp_c`, `dewpoint_c`,
`temp_dewpoint_spread_c`, `wind_dir`, `wind_kt`, `gust_kt` -- 6 cols; no
pressure column, never fetched -- see `fetch_metar_data.py`'s
`DATA_FIELDS`). The two sets partition all 17 weather columns exactly.

| arm | overall delta | 95% CI | P(worse) | LIRF delta |
|---|---|---|---|---|
| adverse-only | -0.78s | [-1.47, -0.12] | 0.008 | -1.6s |
| benign-only | **-1.04s** | [-2.08, -0.17] | 0.010 | **-3.3s** |
| combined (reference, S24/S25) | -1.21s | [-2.48, -0.13] | 0.013 | -4.9s |

**Both arms are individually significant -- this is not "weather is fake."**
But benign-only (temp/dewpoint/wind, ZERO visibility/ceiling/precipitation
information) produces **57%** of the additive sum (1.04/1.82), adverse-only
43%. Per the pre-registered decision rule, that's "most," but barely --
same bare-majority pattern as every mechanism-share number in S24/S25 (59/41,
51/49, 72/28, 83/17 depending on the partition). **Verdict: roughly half of
the realized gain is NOT the de-icing/low-visibility-procedure mechanism
this project assumed since CLAUDE.md's original feature-priority write-up
-- it's more consistent with temp/dewpoint/wind acting as a smoother,
higher-resolution seasonal signal than the coarse categorical `month`/
`hour` features already in the model.** Physically sensible: `month` is a
12-level categorical split, `temp_c` is continuous and tracks cold snaps/
warm spells that don't align to month boundaries and vary within a month
too -- exactly the "existing features encode it weakly" mechanism flagged
as the thing to look for.

Two secondary findings:
- **Sub-additive**: adverse (-0.78s) + benign (-1.04s) = -1.82s vs the
  combined set's actual -1.21s -- about 0.6s of the two subsets' gain
  overlaps (physically plausible: freezing/de-icing and temperature are
  mechanistically linked, so isolating either one partially captures the
  other's signal too).
- **LIRF's split is exactly additive** (-1.6 + -3.3 = -4.9, matching the
  combined LIRF delta exactly) despite the overall set being sub-additive --
  at these effect sizes (single-digit seconds on a 26,528-row lane) this is
  more likely coincidence than a real distinction from the general
  sub-additivity pattern; not chased further.

**Not actioned as a code change** -- `weather=True` stays adopted as a
whole family (splitting it into adverse/benign sub-families for separate
on/off toggles would be over-engineering a ~1s partition question, and the
combined set already outperforms either half alone). This is a mechanism
finding for the paper-writing / honest-reporting obligation (CLAUDE.md's
"paper framing: report the split... state plainly" instruction), not a
lever to pull. Recorded so the eventual writeup doesn't claim weather was
adopted for its de-icing/low-visibility mechanism when the ablation shows
that's at most half the story.

## 27. S26 follow-up: is benign weather's gain just a seasonal proxy for
    the coarse month/hour features? No -- it survives (2026-09-22)

Direct test of the hypothesis S26 raised but didn't check: does the benign
arm (temp/dewpoint/wind, -1.04s S26) gain because it's smoother seasonal
information than the model's existing 12-level `month` categorical /
24-level `hour` categorical, or because it carries real weather content?
`tests/doy_ablation_test.py`, same discipline as S26 (paired per-row squared
error, (airport,day) cluster bootstrap, `target="mixed"`, seed 42).

Built two calendar features from the movement timestamp alone (no external
data): `doy_sin`/`doy_cos` = sin/cos(2*pi*doy/365.25) (so 31 Dec and 1 Jan
sit next to each other, unlike raw `doy`/`month`), plus `hour_sin`/
`hour_cos` on minute-of-day (period 1440) as the optional diurnal
counterpart. Two-step test:

| step | comparison | overall delta | 95% CI | P(worse) | LIRF delta |
|---|---|---|---|---|---|
| 1 | +doy vs baseline (no weather, no doy) | +0.37s | [-0.51, +1.29] | 0.791 | +2.9s |
| 2 | +doy+benign-wx vs +doy | **-0.91s** | [-1.79, -0.18] | **0.005** | -2.5s |

**Step 1: calendar resolution alone does not help.** doy_sin/cos/hour_sin/cos
add nothing distinguishable from noise on top of the existing month/hour
categoricals (CI straddles zero, P(worse)=0.79 -- if anything a slight,
non-significant regression). Matches the S26 sanity check's own read: the
existing categorical calendar features are already doing that job
adequately, gluing on a smoother version of the same thing doesn't move
RMSE.

**Step 2: the benign-weather gain survives.** -0.91s with doy features
already present vs S26's -1.04s without them (measured on a different
control arm, not directly subtractable, but the same ballpark, ~87% of the
original effect) -- and this time it clears both the CI and the
LIRF-specific check (-2.5s, consistent with S26's -3.3s). **Verdict:
temp/dewpoint/wind carry something calendar resolution can't reproduce --
S26's "roughly half is a seasonal proxy" framing was the right instinct to
check, but it doesn't hold up under a direct test: adding actual smooth
seasonality (doy_sin/cos) has ~zero effect on its own, so the benign arm's
gain isn't substitutable by better calendar features.** Reframes S26's
finding: the benign weather signal is not equivalent to "seasonal proxy the
model lacks" (the model doesn't lack one, and adding a better one doesn't
help) -- whatever benign weather is capturing is something calendar-shaped
features structurally cannot express (e.g. within-month cold snaps, not
just where in the year it is).

**No extrapolation risk flagged for the record**: training is 2025-only and
holdout is Jan+Jul 2025 (full day-of-year coverage there), but ranking is
Jan+Jul 2026 -- same days of year, so `doy_sin`/`doy_cos` would face no
unseen values if ever adopted. Not a concern either way since step 1 showed
no reason to adopt them.

**Not actioned** -- doy/hour cyclical features are not adopted (no
measurable gain, step 1). `weather=True` stays adopted as-is; this test
just strengthens confidence in why (S26's own mechanism-attribution
question, now answered rather than left open). `tests/doy_ablation_test.py`
kept for reproducibility; its three tmp cache dirs
(`cache/features_doy_{base,doy,doywx}_tmp/`) follow the same
leave-in-place convention as S26's `features_ablate_*_tmp/`. Reports at
`reports/eval/doy_{base,doy,doywx}.md`.

**Bug found during setup, NOT fixed here, flagged for a separate
decision:** `minute_of_day` (`src/features/build_features.py:475`,
`pl.col("T").dt.hour() * 60 + pl.col("T").dt.minute()`) is computed in
Int8 arithmetic and silently wraps for every hour >= 3 (e.g. hour=20 ->
stored as -80, not 1200) -- confirmed by direct repro on a synthetic
timestamp column. It is not excluded by `NON_FEATURES`/`feature_matrix`,
so it has been feeding the model garbage values for most of the day this
whole time, for as long as this column has existed. This test's own
`hour_sin`/`hour_cos` were computed fresh from `T` with an explicit
`Int32` cast specifically to avoid inheriting this bug. Left unfixed
pending a decision, since even though it's clearly wrong (not a modeling
choice), fixing it changes a live feature's values for every existing
holdout/ranking run and needs the same distribution-shift and before/after
significance check as any other feature change (S5's discipline) before
being trusted -- not something to slip in as a side effect of an unrelated
ablation.

## 28. `minute_of_day` Int8-overflow bug fixed and validated -- adopted,
    net neutral-to-positive, zero regressions (2026-09-22)

Followed up on the S27 bug flag. Fix: `build_features.py` line 475 now
casts `dt.hour()` to Int16 before the `*60` multiply
(`pl.col("T").dt.hour().cast(pl.Int16) * 60 + pl.col("T").dt.minute()`),
so `minute_of_day` no longer wraps -- verified range is now [0, 1439]
(was [-128, 127]) and matches `hour*60+minute` exactly on the rebuilt
cache.

Rebuilt the production cache (`cache/features/{train2025,
holdout_gap2025, ranking}.parquet`) via `export_model_inputs.py` with the
fix; the pre-fix cache was copied to `cache/features_before_minfix_tmp/`
first so a clean before/after comparison was possible without re-deriving
the buggy state from scratch.

**Validated before adopting** (`tests/minute_of_day_fix_test.py`, same
discipline as every other feature change this project makes --
`target="mixed"`, seed 42, paired per-row squared error, (airport,day)
cluster bootstrap):

| | overall | 2025-01 | 2025-07 | LIRF |
|---|---|---|---|---|
| before (buggy) | 335.9 | 353.9 | 320.7 | 655.1 |
| after (fixed) | **335.6** | 354.3 | **319.8** | **652.7** |
| delta | **-0.3s** | +0.4 | -0.9 | -2.5 |

Cluster bootstrap: **delta=-0.31s, 95% CI=[-0.76, +0.13], P(worse)=0.089.**
Every other airport's delta is <=0.3s (noise-floor sized, see S26's 1.9s
LIRF noise-floor calibration for scale) -- **zero regressions of concern
anywhere.** January ticked up marginally (+0.4s) while July improved
(-0.9s), so this doesn't clear the S5 "both months must improve" bar for
a speculative feature, and the overall CI technically straddles zero, so
this would NOT clear the bar this project uses to *adopt a new feature*.

**Different bar applies here, though: this is a bug fix, not a
hypothesis.** The pre-fix `minute_of_day` was not "a feature that might
not help" -- it was mislabeled data (values outside the physically
possible 0-1439 range for ~21/24 hours of the day). The relevant question
isn't "does fixing it help enough to justify adopting a new feature," it's
"does fixing it break anything" -- and it doesn't: no airport regressed
outside noise, the overall point estimate moved in the right direction,
and LIRF (which the model already leans on time-of-day-shaped signals for
via `hour`/`dow`) improved by -2.5s, comparable in size to real adopted
wins like S26's benign-weather LIRF delta (-3.3s). **Adopted** -- kept as
the new production cache; no reason to hold a known-wrong column in place
pending a stronger significance result that a correctness fix was never
going to need.

Reports at `reports/eval/minfix_{before,after}.md`. `cache/
features_before_minfix_tmp/` kept (not deleted) in case a future session
needs to re-diff against the pre-fix state.

## 29. ADS-B pushback recovery: ceiling check, then flight-level validation
    on the one-day probe -- naive detector is currently net-negative
    (2026-09-23)

Prompted by a suggestion to sanity-check the 400 GB ADS-B download idea
before committing to it: `AOBT_3_flt` (already a feature, `aobt3_taxi`) is a
noisy read on the same off-block event as `BLOCK_TIME_UTC_mvt`
(`reports/step0_audit.md` §4: 21.0% within ±1 min, 74.6% within ±5 min,
naive taxi from it alone RMSE 384.9s vs true-taxi sd 417.5s) -- confirmed
this was already known and already a feature, not a clean leak sitting
unused. ADS-B's only possible value is denoising that channel further, not
adding a new signal. Full detail in `reports/adsb_recovery_ceiling.md`;
summary here.

**Step 1 -- best-case ceiling** (`tests/adsb_ceiling_test.py`, run against
current production `target="mixed"`, holdout RMSE 335.6s): a one-day
adsb.lol ground-coverage probe (2025-01-15, 2.1 GB, `external-data/adsb/
adsb_20250115.parquet`) found ground positions at only 6 of 10 departure
airports -- EHAM, EGLL, EDDM, EDDF, LSZH, LEBL. LFPG, LEMD, LIRF, LTFM had
**zero** aircraft with a ground position that day. Covered airports are
56.1% of ranking DEP rows but only 25.0% of holdout squared error (they're
the easier airports). Oracle ceiling (perfect recovery on covered rows,
unchanged elsewhere): 335.6s -> 290.6s, **-45.0s (-13.4%)**; sensitivity at
50%/25% assumed coverage: -21.7s (-6.5%) / -10.7s (-3.2%). LIRF (29.1% of
total squared error, the single largest contributor) has zero coverage --
structurally out of reach regardless of the rest of this analysis.

**Step 2 -- measure the coverage guess instead of assuming it**
(`tests/adsb_recovery_test.py`). Neither dataset has a callsign/tail-to-
flight key in the columns actually extracted (the ADS-B probe kept `hex,
reg, ts, lat, lon, alt, gs, airport` -- no callsign message), so matching
uses the one anchor both sides share and neither blanks: movement time.
Per (airport, hex): `alt == -1` is a categorical ADS-B surface-position
flag (not a computed altitude threshold, so segmentation is exact and
airport-elevation-independent) -- a ground run immediately followed by an
airborne run is a candidate departure, its last ground sample the takeoff
proxy. Within that run, the *last* sub-run of >=3 min with ground speed
<=2 kt is the candidate gate dwell; first sample after it ends is the
pushback estimate (no such dwell = `censored`, a takeoff seen but no usable
pushback). Candidate takeoffs matched 1:1 to true DEP movements by nearest
`MVT_TIME_UTC_mvt`, same airport, +-180s tolerance, greedy closest-pairs-first.

**Recovery: 25.5% overall** (670/2,629 true DEP movements on 2025-01-15
across the 6 covered airports), ranging 9.7% (EDDF) to 77.0% (EDDM, see
caveat below) -- this lands at the **pessimistic** end of step 1's 25-50%
sensitivity range, not the middle, and it's a real measurement now, not a
guess. Most departures at EGLL/EDDF/LEBL never show a clean >=3-minute
near-zero-speed dwell in ADS-B at all (~90% `censored`), consistent with
known gate/apron ADS-B reception problems (buildings, multipath), not a
detector bug specific to those airports.

**Accuracy on the matched-and-usable set (n=670) is worse than the feature
already in the model, on every axis measured**: median offset **+7.84 min**
(detected pushback lags true off-block, not just noisier around zero),
\|d\|<=60s only 18.2%, \|d\|<=300s only 31.8%, RMSE **2,179.6s** -- vs
`AOBT_3_flt`'s 21.0% / 74.6% / (384.9s naive-taxi RMSE) from the existing
audit. True taxi sd for this same 2,629-row population is 538.8s, so the
naive ADS-B-derived taxi estimate (`MVT_TIME - adsb_pushback`, error exactly
equal to the timing diff since MVT_TIME is exact) is **RMSE 2,179.6s --
over 4x worse than just predicting the mean**, before any model even sees
it.

**EDDM's 77% recovery is very likely a different mechanism, not better
coverage** -- flagged, not chased further this session. Its ground-row
count is ~4-5x every other covered airport, its censored rate is far the
lowest (22% vs 62-88% elsewhere), and its median offset is a large positive
+21.9 min. Plausible read: Munich in January routinely holds aircraft at a
de-icing pad for several minutes after pushback before the runway -- a
second >=3-minute near-zero-speed dwell downstream of the real pushback
that "last dwell before takeoff" picks up instead of the gate departure.
Consistent with all three symptoms at once without needing a per-airport
code difference; would need per-flight ADS-B track inspection to confirm.

**Conclusion.** This is a first-pass, unrefined detector (ground-speed
threshold + longest-dwell heuristic), not a mature pushback algorithm -- a
stand/gate-position-aware version, or one that excludes de-icing-pad
dwells, would likely do better. But two findings don't depend on refining
it: (1) coverage is the harder constraint -- 4/10 airports (LIRF included)
have zero ADS-B ground data at all, and even among the 6 covered ones only
25.5% of real departures yield any usable signal, matching the pessimistic
end of the already-modeled sensitivity range; (2) where recovered, the raw
signal is currently worse than useless (RMSE 2,179.6s vs a 538.8s
do-nothing baseline) and would need substantial detector engineering
(stand-position matching, de-icing-hold exclusion, tighter dwell criteria)
before it could even match `AOBT_3_flt`, let alone denoise it.

**Decision: do not proceed to the full 400 GB download** without first
validating that a materially better pushback detector is achievable on
this same one day of data -- the current one doesn't clear the bar the
existing feature already sets, on data that cost nothing further to check.
`tests/adsb_ceiling_test.py` and `tests/adsb_recovery_test.py` are kept for
reproducibility (`logs/adsb_ceiling_test.log`, `logs/adsb_recovery_test.log`);
`external-data/adsb/adsb_20250115.parquet` is the one day of probe data
already on disk, no further ADS-B data was pulled this session.

## 30. Four concrete LIRF follow-up checks -- distribution, runway/stand,
    LIRF-only model, and what it says about the flip/CEIL fix (2026-09-23)

Full detail in `reports/lirf_investigation.md` §7 (new); summary here.
Scripts: `tests/lirf_distribution_test.py` (Q1/Q2),
`tests/lirf_specific_model_test.py` (Q3).

**Q1 -- distribution shape: a clean three-part mixture, not bimodal.**
Full-year LIRF taxi (n=160,704): mean 1195s, sd 1332s vs 974s/414s pooled
across the other 9 airports. The raw histogram is unimodal, no visible
second mode. But decomposed by the two already-known mechanisms:
"ordinary" (81.9% of rows, sd **438** -- statistically indistinguishable
from every other airport's pooled sd of 414), "echo" (\|d\|<30s, 17.8% of
rows, sd 942), "extreme" (taxi>7200s, 0.3% of rows, sd 16,504, 3.9% of
total taxi-mass). **LIRF's 3.2x variance is not intrinsic to typical LIRF
taxiing -- it's entirely the two known contaminating mechanisms**, not
quantified this cleanly before.

**Q2 -- runway/stand pattern: real, mostly already visible, one untested
candidate.** Runway 25 carries ~90% of traffic every month (no seasonal
config shift -- LIRF is single-runway-dominant year-round, not switching).
Mean taxi does vary by runway in the "core" (non-echo, non-extreme)
population (16L 1735s down to 25's 1064s), but `RUNWAY_mvt` is already a
raw categorical feature. The (runway=25, July, hour 8-14) slice flagged in
§1 shows a real diurnal effect surviving echo/extreme exclusion (823s at
hour 0 -> 1259-1514s plateau hours 8-14) -- isolated more cleanly than the
earlier "existing congestion features are flat" finding, which tested this
on the whole contaminated distribution. Not yet tested whether the
existing congestion features discriminate this specific slice directly --
flagged as a candidate, not chased further.

**Q3 -- LIRF-specific model vs pooled, post-fixes: no, slightly worse
(+2.6%), a clean negative result.** The 2026-09-02 attempt
(`stage3a_resid.md`) failed for two bundled reasons (widened label window
polluting the *shared* pooled model; residual heads overfitting a crude
2-fold OOF) that don't apply to this design (a from-scratch LIRF-only base
regressor, own priors/group-encodings, evaluated through the *same*
already-validated echo blend as production). Result: pooled 660.6s vs
LIRF-only 677.9s (+17.3s). Interesting nuance: LIRF-only's raw pre-blend
regressor is very slightly *better* in isolation (809.1s vs 816.2s) but
composes worse with a blend calibrated for the pooled model's residual
structure -- not chased further since the net (submittable) effect is
unambiguous either way. **Cross-airport pooling still helps LIRF**, even
with 134,176 of its own training rows and every current fix in place --
confirms the original architectural call (PROGRESS.md §8) directly rather
than just arguing it from first principles.

**Q4 -- are the flip/CEIL dynamics (§12) symptomatic of something deeper?
Yes -- same finding as Q1, from the model's side.** `flip`'s
`offset - d̂` reconstruction inherits the exact, always-known `offset` term
for both of Q1's contaminating segments; `direct` has no such path and is
range-capped by `LABEL_HI=7200`. Q1's decomposition and the flip/CEIL
story are one fact seen twice: LIRF is an ordinary airport (core sd≈438)
plus two identifiable contaminating mechanisms, and `offset`-aware
inductive bias (flip) handles that contamination structurally better than
`direct`. Also explains Q3: pooling helps the ordinary 82% (shared
cross-airport structure); the CEIL fix and echo blend -- not model
specialization -- handle the contaminating 18%+0.3%. Two different
problems, already treated by two different, already-shipped mechanisms --
no third undiscovered LIRF problem behind either one.

**Not actioned as a code change** -- all four are diagnostic, confirming
the current design rather than surfacing a new lever. One candidate noted
for a future session: direct-test the existing congestion features on the
(runway=25, July, hour 8-14) slice specifically (Q2).

## 31. Read LIRF's top-10 worst rows individually -- found a real, targeted
    echo-blend bug, and confirmed hedging doesn't work (2026-09-23)

Follow-up to §30's Q1 (top 10 rows = 58.4% of LIRF's SSE, never actually
read). Full detail in `reports/lirf_investigation.md` §8; script
`tests/lirf_top10_test.py`.

**Scale**: top 10 LIRF holdout rows = 28.5% of LIRF's SSE, **8.4% of the
entire holdout's squared error** -- ten rows out of 344,339.

**Shared trait, not previously flagged**: all 10 have `has_aobt3=False`
and every NM-matched (`_flt`) field null -- not a random tail draw, 100%
from the small NM-completely-unmatched population.

**Two distinct, opposite-direction mechanisms**:
- Rows 1-3 (~21% of LIRF's SSE): true taxi clustered at **24.2-24.3
  hours** each -- `BLOCK_TIME_UTC_mvt` precedes scheduled departure by
  hours, actual takeoff follows ~24h later. Reads like a
  cancelled-and-reinstated-next-day flight or a record-matching artifact.
  Model under-predicts 21-33k seconds; nothing in training is anywhere
  near this magnitude -- looks like a genuine extrapolation limit, not an
  obvious bug.
- **Rows 4-9 (6 of the 10, ~6.3% of LIRF's SSE) -- a real, targeted,
  fixable bug.** True taxi is ordinary (723-1,444s). Each has a genuinely
  large gate delay (`d`=11,800-19,500s) followed by a normal taxi-out. The
  echo classifier assigns these moderate-high `echo_prob` (0.56-0.75)
  anyway, pulling the blend toward the huge `offset` and inflating a
  12-24 minute taxi into a predicted 2.8-4.7 hours. **The pre-blend raw
  regressor was already closer to truth than the blended prediction on
  all 6** -- the echo blend is actively injecting error, not fixing
  anything, specifically within `has_aobt3=False` where the classifier
  lacks the `_flt`-derived features it otherwise relies on.
- Row 10: mirror-image -- a near-perfect echo (`d`=-5s) under-confidently
  scored (`echo_prob=0.281`), causing under-prediction.

**Hedge-strategy what-ifs (override only these 10 rows)**: oracle
(predict truth exactly) recovers -14.53s (+4.3% RMSE) -- the hard ceiling,
capped by RMSE's square root, well below the ~8% a linear SSE-share read
would suggest. Every uniform hedge tested (offset-anchored, drop-the-blend,
half-hedge) recovers near-nothing or makes it worse (drop-the-blend:
+2.73s, actively worse) -- because rows 1-3/10 need pushing *up* toward
offset while rows 4-9 need pushing *down* away from it. No single global
rule serves both.

**Decision point, not yet built**: a candidate fix (suppress/recalibrate
`echo_prob` specifically when the classifier's own inputs are almost
entirely null, i.e. `has_aobt3=False`) targets exactly the rows-4-9
mechanism. Flagging rather than building unilaterally -- the echo blend
has already caused two regressions from over-eager global changes (§6,
§12), so this needs the same validate-before-ship discipline as every
prior change, and a decision on whether to prototype it now.

## 32. The rows-4-9 fix does NOT survive validation -- built out the three
    scoping checks requested, then an end-to-end simulation killed it
    (2026-09-23, same day as §31)

Full detail in `reports/lirf_investigation.md` §9; scripts
`tests/lirf_echo_gate_scoping_test.py`, `tests/echo_ranking_score_test.py`.

Three checks done before building anything (per review): (1) rows-4-9-only
oracle = **+3.18s (+0.94%)**, matching the predicted 3-5s range, not §31's
14.53s which included out-of-scope rows 1-3. (2) The real population is
far bigger than 6 rows: widening the exact gate across the whole holdout
finds **81 rows (79 LIRF) carrying 11.71% of total squared error** --
bigger than LIRF's entire top-10. (3) Fit the production echo classifier
on all-2025 data and scored it on ranking directly (not a proxy): **178
null-lane rows with `echo_prob_r>0.5`, 177 LIRF** -- matching the "two
hundred" order of magnitude predicted, not the pessimistic case.

**Then the check that mattered most: does fixing this cost more than it
saves?** The same null lane has **116 correctly-caught true echoes** at
`echo_prob>0.5`. Forcing those onto the raw-model-only path would cost
6.03 billion in squared error -- more than the 81 false positives'
combined 4.59 billion. True and false positives turned out **observationally
indistinguishable** on everything checked (offset, stand, runway, month,
even `echo_prob` itself has heavy overlap: FP's max 0.843 exceeds TP's
median 0.709). End-to-end simulation (real `_reconstruct_taxi`, not row
counts) of every monotonic recalibration -- hard thresholds at 0.5-0.85,
proportional damping, band-limited suppression -- **regressed at every
setting tested** except one noise-level exception (×0.9 damping, -0.15s).

**Conclusion: no fix, not shipped.** The mechanism (§31) is real and
correctly diagnosed, but doesn't correspond to an extractable improvement
given current features -- the null lane's ~200 rows beyond the identified
TP/FP set are already net-helped by the existing continuous blend, so any
correction broad enough to catch the 81 false positives removes more value
than it recovers. Recorded precisely so this isn't re-derived later,
matching the project's convention for negative results
(`stage3a_resid.md`, the rejected hour-encoding/quantile-objective
attempts). A real fix would need a genuinely new separating feature, not a
blend adjustment -- none found among available fields (every `_flt` field
is null by construction for this population).

Rows 1-3 (the ~24h anomalies) remain separate and unbundled, per review
guidance -- their holdout signature depends on `BLOCK_TIME_UTC_mvt`
preceding `SCHED_TIME_UTC_mvt`, which is exactly the field blanked for
ranking DEP rows, so there is no equivalent pre-hoc detector available.
Genuinely open for a future session.

## 33. Brainstormed and exhausted every available-feature angle for the
    null lane -- genuinely closed (2026-09-23, same day as §32)

Full detail in `reports/lirf_investigation.md` §10. Checked isotonic
calibration (the "right-shaped" fix vs §32's crude thresholds): honest
Jan/Jul cross-fit gives -0.18s, noise-level -- `echo_prob` is already
close to calibrated for this lane. Then swept every available feature for
TP/FP separation: `ADES_mvt` (no concentration), `AIRCRAFT_TYPE_mvt`
(**null for 100% of both groups** -- this population's own movement
record is sparse, not just NM-flight-list-unmatched), `stand_echo_rate`
(AUC 0.607 vs `echo_prob` alone's 0.726, and combining the two scores
**worse**, 0.710 -- stand info dilutes rather than adds signal).

**Conclusion: genuinely closed, not under-explored.** Every readily
available field has been checked; none beats the status quo.

**Correction, same day: ATFM is not an open lever here.** An earlier
draft named ATFM regulation records as the remaining option -- stale,
carried from the project's original planning notes ("permission/licensing
unresolved") without checking it against what actually happened. ATFM was
already ingested, tested, and closed in §20: -0.4s, P(worse)=0.453,
against a noise-corrected **2.07s structural ceiling on any (airport,
day)-granularity feature** -- a general bound from residual-variance
decomposition, not specific to this lane, so it applies here identically.
It's also the wrong shape regardless: an airport-day feature is constant
across every LIRF departure that day, so it can't separate a true echo
from a same-day long gate delay -- exactly the per-flight separability
§10/9 already showed is missing. Not a candidate; don't reopen without
new sub-daily data (EUROCONTROL doesn't publish it).

**What's actually open**: (1) rows 1-3 -- undetectable in production by
construction, signature depends on the blanked `BLOCK_TIME_UTC_mvt`,
needs a proxy or stays unreachable; (2) whether *any* signal not yet
examined separates a true echo from a long gate delay when NM fields are
null -- a feature-discovery question, not recalibration (§32 already
closed recalibration; this section's sweep closed the *currently
available* fields, but not the general question). Beyond those two, this
investigation is at a genuine stopping point.

## 34. Correction to §33/rows-1-3: `LABEL_HI=7200` is a hyperparameter,
    not a structural limit -- raising it measurably helps, but regresses
    overall (2026-09-23)

Full detail in `reports/lirf_investigation.md` §11; script
`tests/label_hi_test.py`. Every prior mention of rows 1-3 (§8-10, and the
"structural" framing accepted through §33) treated the GBM's inability to
predict past ~7200s as an architectural fact. It's a choice
(`LABEL_LO, LABEL_HI = 30, 7200` in `fit.py`), last tested
(`stage3a_resid.md`, 2026-09-02) under an architecture that no longer
exists (plain L2, no Huber, no echo classifier, no CEIL fix). Re-tested
under the current one: `target="flip"` pooled across all 10 airports,
Huber alpha=800 unchanged, `LABEL_HI=7200` vs `LABEL_HI=200000`
(effectively unbounded). Priors/encodings/classifier don't depend on
`LABEL_HI`, fit once and shared -- isolates the test to one line.

**Confirmed real: the mechanism works.** LIRF's 3 most extreme rows
improved **-12.7%** (29,036.7s -> 25,347.9s) -- Huber's gradient-capping
at `alpha=800` lets the model learn a soft, extremity-aware adjustment
once given the training signal, unlike L2 which would let those residuals
dominate. **§33's "unreachable by construction" language for rows 1-3 is
corrected** -- it described a hyperparameter's current setting, not a
hard limit.

**But it's still a net regression, now for a precisely measured reason**:
overall +5.97s (+1.7%, P(worse)=0.978, real not noise), July alone +10.46s
(fails the both-months rule), LIRF overall +36.25s worse despite its 3
best rows improving, and -- the key finding -- **EDDF/EHAM/LEBL all got
worse too**, despite having essentially zero extreme rows. Widening the
*shared* regressor's label window changes split selection and leaf values
everywhere in the pooled model, not just on the tail -- those airports pay
a tax for a label range they never use. Same failure shape as the
original L2 test, smaller in magnitude, not eliminated.

**Decision**: don't raise `LABEL_HI` globally, and don't search for a
moderate intermediate value either -- any shared-window widening still
taxes the 9 airports that don't need it, so that search is tuning one
number against a known-bad tradeoff, not a structural fix. Points to a
**separate tail model** instead (main regressor untouched at
`LABEL_HI=7200`, so the 9 unaffected airports pay nothing) -- see §35 for
the oracle-gated ceiling check on that idea, run before building any
real router (the routing problem is exactly what killed §32: rows 1-3's
signature depends on the blanked `BLOCK_TIME_UTC_mvt`).

## 35. Separate tail model -- oracle-gated ceiling is real and substantial:
    +3.63% overall (2026-09-23, same day as §34)

Full detail in `reports/lirf_investigation.md` §12; script
`tests/tail_model_oracle_test.py`. Measured the tail model's value with an
oracle GATE (perfect routing on the true label) before building any real
router, per §34's ordering -- if the ceiling is small, the routing problem
(which killed §32) is moot.

Population: true taxi>7200, pooled across all 10 airports -- 584 total 2025
rows (480 LIRF, rest spread across 5 other airports), 435 training / 149
holdout. Tail model: same `flip` target/Huber alpha=800, fit on only 348
training rows (87 held out for early stopping) -- converged in 22 rounds,
deliberately tiny given the population size.

**On the 149 true holdout tail rows: the current production blend
(10,093.2s RMSE) is worse than just guessing `offset` with zero training
(9,126.5s)** -- a free heuristic beats the shipped architecture here. The
actual tail model edges out even that (9,086.0s).

**Oracle-gated overall RMSE (real tail-model predictions, oracle only on
which rows get routed to it): 342.19s -> 329.76s, -12.44s (+3.63%).**
Larger than several wins already adopted this session (weather ~1-4s,
minute_of_day fix -0.3s) -- clears the bar for the gate to be worth
solving. Broad-based, not LIRF-only: EGLL -69% (n=20), LIRF -26% (n=113),
LSZH -28% (n=4); LFPG barely moves (-3%, n=9, matches its known
data-corruption-style outliers).

**Decision: build the router next.** The ceiling justifies the harder
problem -- detecting tail rows using only pre-hoc information (offset,
`has_aobt3`, `echo_prob`, congestion features, never the blanked
`BLOCK_TIME`/taxi), checked for the same false-positive-damage risk that
sank §32's echo-blend fix. Not yet built.

## 36. The router, built and honestly validated -- decisive failure at the
    default threshold, exactly the risk flagged going in (2026-09-23,
    same day as §35)

Full detail in `reports/lirf_investigation.md` §13; script
`tests/tail_router_test.py`. Pre-check: `offset` alone has AUC=0.956 for
predicting tail membership -- far more separable than the null-lane
investigation's 0.726 ceiling, a strong prior this could work. Built a
P(tail) classifier exactly mirroring `fit_echo_classifier`'s structure and
discipline, retargeted at `is_tail = taxi>LABEL_HI`. Final prediction:
`P(tail)*pred_tail + (1-P(tail))*pred_main`, with `pred_main` (the
existing echo blend) completely untouched.

**Result: a certain, large regression.** 342.19s -> 373.08s (+30.89s,
-9.03%), **P(worse)=1.000**. Every airport got worse, including ones with
essentially zero tail rows -- broad damage, not a false-positive hotspot.
At P>0.5, precision is only 24.2% (68 TP, **213 FP**) against a 0.025%
base rate, and those 213 false positives get pulled toward the tail
model's huge output, exploding their RMSE from 2,585 to 6,426. High AUC
on one feature doesn't guarantee usable precision at this base rate --
the asymmetric cost (a wrong tail-route is catastrophic, a right one is
merely helpful) means absolute FP count matters more than AUC.

**Not yet closed** -- default operating point tested, not necessarily the
best one. Per-row predictions saved (`cache/tail_router_ev.parquet`) for
a free threshold sweep -- checking whether a much higher-confidence
threshold or a hard gate recovers part of §35's ceiling without the
broad damage. In progress.

## 37. Stand-aware ADS-B pushback detector -- §29's null reversed on
    accuracy, coverage now measured across 6 days; bulk pull running
    (2026-09-26)

Follow-up to §29 (naive "last dwell before takeoff" detector: 25.5%
recovery, 2,179.6s RMSE, EDDM's 77% likely de-icing holds).

**Stand coordinates.** `src/ingest/fetch_stands.py` pulls apt.dat row codes
1300/1301 from the X-Plane Scenery Gateway (`xplane_airports` package, no
account) -> `data/external/stands.csv`, 2,838 stands, 151-513 per airport
(EDDF 227), all 10 packs full 3D sceneries, every stand inside its airport
box. Licence: GPLv2 (`COPYING` in each pack, kept as
`data/external/stands_LICENSE_GPLv2.txt`); DATA_SOURCES.md row added.
`src/ingest/stands.py` normalises Gateway names ("Gate A04", "202-(C)",
"F6") to STAND_mvt ("A04", "202", "F06"), with a variant-letter fallback
(115B -> 115): own-stand coords for 86-100% of 2025 DEP rows per airport.

**Diagnosis before building** (`tests/adsb_stand_diag_test.py`, 2025-01-15).
The first worrying number (3-30% of surface points within 60m of a stand)
counted every taxiway/runway point; per matched departure the picture is:
coordinates right (tracks that start before off-block begin a median
7-16m from own stand); tolerance secondary (60->150m moves own-stand hit
rate modestly, never doubles from a real base); **the dominant factor is
that tracks start late** -- only 2-38% of matched departures have any
surface point at/before true off-block, median first point +1.5 to +13
min after it. The EDDM plot shows a track *appearing* on its stand at
-0.1 min with nothing before: transponders go on at pushback. So the
signal is appearance at the own stand, not the end of a dwell -- the §29
rule discarded exactly the good cases.

**Detector** (`src/link/adsb_pushback.py`; split-blind, reads only
MVT_TIME and STAND_mvt). Surface = gs < 40 (alt == -1 agrees on 374,462
points, disagrees on ~18.6k). Surface runs ending in takeoff matched 1:1 to
DEP by takeoff time +-180s, preferring runs that visit the own stand.
Tiers: `dwell` (stationary at own stand, then moves: first sample after),
`appear` (run starts at the stand: first sample), `pass` (reaches the
stand after starting elsewhere -- useless, drop). Output
`cache/adsb_pushback/day=*.parquet` keyed on MVT_ID_mvt.

2025-01-15, same rows (n=838): ADS-B median +42s, 57.5% within +-60s,
87.8% within +-5min, **RMSE 198s vs AOBT_3_flt 654s**. appear 119s, dwell
225s, pass 465s.

**Raw extract contamination.** The Colab day files for 2025-07-15,
2025-09-15, 2025-11-15, 2026-07-15 contained 12-28% points from *other*
collected days (work directory not cleared between days). Filtering each
file to its own UTC date recovers it (2025-01-15 reproduces exactly:
823 appear+dwell). `src/ingest/normalise_adsb.py` does this and writes one
schema to `data/external/adsb/day=*/`. The replacement fetcher streams the
tar and never extracts to disk, so it can't recur.

**Coverage across 6 days** -- recovery (appear+dwell) % of DEP:

| day | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | all |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-01-15 | 12 | 81 | 4 | 54 | 32 | 0 | 0 | 0 | 25 | 20.5 |
| 2025-07-15 | 17 | 0 | 1 | 73 | 18 | 0 | 0 | 14 | 13 | 16.1 |
| 2025-09-15 | 36 | 70 | 2 | 75 | 25 | 26 | 0 | 1 | 15 | 28.6 |
| 2025-11-15 | 48 | 72 | 14 | 75 | 44 | 15 | 0 | 10 | 26 | 33.1 |
| 2026-01-15 | 8 | 76 | 19 | 71 | 34 | 25 | 0 | 11 | 32 | 29.4 |
| 2026-07-15 | 2 | 76 | 72 | 77 | 62 | 0.5 | 0 | 0 | 19 | 34.3 |

§29's "4 airports have zero coverage" was a Jan-2025 artifact for LIRF,
LEMD, LFPG -- the feeder network grew into them during 2025. But: LIRF has
**1 aircraft** with surface points on 2026-07-15 (none of the July-ranking
LIRF opportunity survives, on this day at least); LFPG tracks are too
sparse to ever reach a stand (median 4-16 points per matched run); LTFM
zero on every day (its box is right -- all 351 stands inside). Day-to-day
swings are large (EDDF 2-48%): one day per month cannot characterise it.

**Accuracy, 2025 days pooled** (`tests/adsb_multiday_test.py`): ADS-B
beats AOBT_3_flt everywhere except LIRF and EGLL. LEMD (+123s median)
and EGLL (+138-182s) show a consistent late lag (learnable offset). LIRF
is harmful: RMSE 1,142s vs AOBT_3 546s (2025-07-15: 1,445s vs model 561s)
-- unverified hypothesis: echo rows, where the label is not the physical
pushback ADS-B measures.

**The load-bearing EDDM question.** 2025-07-15 has no EDDM coverage, so
the ordinary days are 2025-09-15 / 2025-11-15 -- in the model's training
set, no fair model comparison. ADS-B RMSE there 184s / 155s (AOBT_3 347s
/ 243s) vs the model's full-holdout EDDM RMSE 187s: on normal days ADS-B
alone is ~at parity with the model at EDDM. The 430s -> 224s headline was
the de-icing day. Fair holdout comparisons, simple mean of model and
ADS-B: 2025-01-15 excl EDDM 146 -> 120s (n=532); 2025-07-15 excl LIRF 153
-> 154s (n=789; EHAM 125 -> 130); all holdout rows excl LIRF 225 -> 169s
(n=1,612). Winter helps, the one summer day doesn't. Still a feature
candidate, not a replacement, but the evidence now rests on two holdout
days.

**Next.** `src/ingest/fetch_adsb.py` (PREFERRED_RELEASES.txt first, then
prod-0 / staging-0 / prod-0tmp / staging-0tmp; corrupt replica -> next;
atomic per-day writes + `manifest.csv`, resumable). Smoke test on
2026-01-15 matches the Colab extract on 99.8% of (hex, ts, airport) keys;
MLAT is 0.1% of points, so MLAT noise is not a factor. Full Jan+Jul 2025
and 2026 pull running on Colab (2026-09-26, ~5 min/day Jan, ~10-12
min/day Jul). Then: detector over all days, holdout evaluation as a model
feature (tier + coverage flag), EDDM Jul-2025 excluded by necessity.
Data licence: ODbL 1.0 (adsb.lol), DATA_SOURCES.md row added.

## 38. Stand/runway geometry as model features -- REJECTED (2026-09-27)

Family 8 (`src/features/geometry.py`, `build_features(geometry=...)`, off by
default): stand position (metres from the airport's mean stand) and the
straight-line distance from stand to the departure-runway threshold, from
Gateway apt.dat (`data/external/runways.csv`, 72 thresholds; `RUNWAY_mvt`
names match the apt.dat ends exactly). The distance is physically real
(Spearman with taxi 0.28-0.61 by airport; resolves for 96% of rows).
Pre-registered test `tests/stand_geometry_test.py` (log
`logs/stand_geometry_test.log`):

- overall 335.6 -> 336.2 s, **+0.59 s**, CI [+0.08, +1.20], P(worse)=0.990
- Jan -0.13, **Jul +1.23** (fails the both-months rule)
- the loss is almost all LIRF (+5.2 s); 7 of the other 9 airports move
  -0.1 to -0.5 s
- mechanism check fails: rare (stand, runway) pairs (<50 training
  departures) got worse too (+0.46 vs +0.63), when geometry should help
  exactly there

`STAND_mvt`/`RUNWAY_mvt` categoricals plus the taxi priors already carry the
geometry. This also lowers the expected value of OSM routed distance
("Future step B") unless routing captures something straight-line doesn't.
Stands stay in use for the ADS-B detector only.

## 39. ADS-B pushback blend -- ADOPTED, pre-registered, both months
    (2026-09-27)

124 days (Jan+Jul 2025 and 2026) pulled with `src/ingest/fetch_adsb.py` on
Colab, all `ok`, 0 bad members, no cross-day contamination
(`external-data/adsb-fetch/manifest.csv`). Detector run over 126 days
(`logs/adsb_pushback_all.log`). The evaluation was pre-registered in
`reports/adsb_blend_preregistration.md` (committed d3ed616 before any
result); results are appended there. Script `tests/adsb_blend_test.py`,
log `logs/adsb_blend_test.log`.

**Why a blend, not a feature.** There's no ADS-B in the 10 training months,
so a feature would be null for every training row. The blend is instead
`final = pred + w_tier * (adsb_taxi - lag_airport - pred)` on appear/dwell
rows, fit on one holdout month and scored on the other.

**Result (primary, tier weights, LIRF excluded):** Jul (fit Jan) 319.76 ->
317.71 (**-2.05**, P(worse)=0.000); Jan (fit Jul) 354.34 -> 350.12
(**-4.22**, P(worse)=0.000); pooled 335.63 -> 332.56 (**-3.07**, CI
[-4.28, -2.22]). EDDM -25.7, EHAM -15.6, LEBL -8.9, LSZH -3.7, EDDF -3.6,
EGLL -0.9, nothing worse. Weights are stable across months (appear
0.70/0.63, dwell 0.49/0.35).

- **Wide-box check:** old-box and new-box detector outputs are identical on
  the overlap days.
- **EDDM ordinary days** (the load-bearing question from §37): 171.7 ->
  151.2 s excluding its 3 worst days, so the gain isn't a de-icing artefact.
- **Summer:** July improves on its own; the pre-registered winter-only
  clauses never came into play.

**Shipped choice.** Per-airport-tier weights beat the primary in both
directions, by 0.07 s (Jul) and 0.05 s (Jan). The pre-registered rule says
adopt, so they ship. The margin is noise-sized; the rule is followed rather
than argued with after the fact.

**LIRF.** The pre-registered H_echo test failed (non-echo ADS-B RMSE 529 s
vs a 301 s bar), so LIRF is excluded from this blend. But the comparator
was wrong: the model's RMSE on those same rows is 751 s. Echo rows (ADS-B
3,118 s, 83% of LIRF's ADS-B squared error) are where the label is SOBT, not
the physical pushback. The follow-up is a mixture,
`P(echo)*(T-SOBT) + (1-P(echo))*adsb_c`, with a new pre-registration.

**Coverage limits.** LEMD, LFPG and LTFM have ~0 eligible rows, and LIRF has
none in July 2026. All the gain comes from the six northern airports plus
LEBL.

**Before ranking use: a label-free drift check.** Compare
`adsb_pushback - AOBT_3_flt` per airport and tier, and the tier mix, between
2025 and 2026 (the receiver network changed). The lag medians are the part
most exposed to drift.

**Revised plan (2026-09-27; leaderboard 302 vs the top team's ~240):**
1. Second model family (XGBoost/CatBoost on the same inputs) with a
   cross-fit non-negative stack, ahead of further feature tests (feature
   record this session: 1 adopted of ~10 tested).
2. Drift check, then apply the blend to the ranking set.
3. LIRF echo mixture (new pre-registration).
4. Partial-track floor for matched-but-no-stand departures
   (movement-derived: T - first_seen plus stand-to-first-seen distance / typical
   speed); extends coverage to LFPG/LEMD.

## 40. LightGBM + CatBoost stack, then ADS-B on top -- leaderboard
    302 -> 296 -> 277 (2026-09-27)

**Second model family.** XGBoost was dropped: its GPU fit ran out of memory
on free Colab even after the group-encoding memory fix (`ffde2e1`; a narrow
frame in `add_group_encodings_oof`, output verified identical, peak memory
down ~1.8 GB). CatBoost was added as engine `cat` (`_fit_cat`).
- **Speed:** `max_ctr_complexity=1` (no categorical feature combinations)
  made it about 12x faster on a T4: 5-6 minutes per fit instead of 70.
- **Accuracy:** with lr 0.05 it also scored better on the inner validation
  (flip 493 vs 503).
- Both changes were recorded as dated amendments before any holdout result
  (`reports/stack_preregistration.md`).
- `run(ev_out=...)` now saves the holdout predictions before the long refit,
  and `PRC_MEMLOG=1` prints memory checkpoints.

**Holdout.** CatBoost alone scored 336.7 (LightGBM 335.6): better at six
airports, worse at LIRF. The cross-fit NNLS stack gives pooled 333.08
(-2.55, P(worse)=0.021): Jul -3.65, Jan -1.33 (Jan alone not significant).
Every airport improves. It was adopted under the pre-registered rule.
Weights on the full holdout: lgb 0.544, cat 0.468. The ADS-B blend, refit on
the stacked prediction, still adds -2.09 (Jul) and -4.27 (Jan); stack +
ADS-B against LightGBM alone is 335.63 -> 329.97.

**Drift check (`tests/adsb_drift_test.py`, label-free, against
AOBT_3_flt).**
- The spread of ADS-B timing and the tier mix are stable 2025 -> 2026.
- The AOBT_3 yardstick itself swings +-40-80 s month to month even where the
  truth-based lag is stable (EDDM appear: +63 vs -17 against AOBT_3, +42
  vs +44 against truth). So most year-over-year shifts are within its noise.
- The exception is EGLL: all four cells shift by -39 to -80 s. That's a
  possible receiver change; it's left uncorrected, because a correction
  can't be validated.
- Coverage rises a lot in 2026 (EGLL Jul 1% -> 75%, EDDM Jul 26% -> 74%).

**Submissions (`src/post/stack_submit.py`; blend logic in
`src/post/adsb_blend.py`, verified to reproduce the test exactly).**
- v15 = stack alone: **296**.
- v16 = stack + ADS-B: **277**. The blend changes 94,529 of 344,841
  ranking rows (27.4%); LIRF is excluded, EGLL and LEMD are as fitted.
- The ADS-B gain on the leaderboard (-19) is about 3x the coverage-scaled
  projection (-6). That's not yet understood: 2026 errors may concentrate
  where coverage grew, or the leaderboard may score a subset with a
  different mix.

**Next.** ADS-B is the strongest lever by far, so extending its coverage
comes first:
1. Partial-track floor for matched departures with no stand fix (LFPG,
   LEMD, and missed rows elsewhere).
2. The LIRF echo mixture (new pre-registration).
3. EGLL re-check.

The top team is at ~230.

## 41. ADS-B partial-track estimate -- ADOPTED, pre-registered, small
    (2026-09-27)

Top team now ~224 (we're at 277). Departures matched to an ADS-B track that
never shows pushback were unused: about 100k ranking rows, more than the
94.5k already blended. The detector now emits first-sighting fields
(`adsb_first_ts`, `adsb_first_gs`, `adsb_first_own_m`, `adsb_min_own_m`); the
existing columns were verified unchanged.

**Design.** Explored only on 2025-09-15 / 2025-11-15 (training months), so
the holdout stayed unseen. `L = MVT_TIME - first_seen`. The unseen part
(taxi - L) has IQR ~250 s when first seen < 600 m from the stand, against
~450 s for taxi. It's useless beyond ~1 km. As a standalone estimate it's
heavy-tailed (RMSE 429 vs taxi sd 364), so it goes in as a weighted nudge.
Pre-registration: `reports/adsb_partial_preregistration.md` (987073b).

**Result** (`tests/adsb_partial_test.py`), on top of cross-fit stack +
ADS-B: pooled 329.97 -> 329.52 (**-0.45**, CI [-0.62, -0.32], P(worse)=
0.000); Jul -0.41, Jan -0.50. The two-band variant (first sighting < 500 m /
500-1000 m) beats the single weight in both directions and ships. Its
weights are ~0.27 near and ~0.08 far. LEBL -3.1, EDDM -2.1, LSZH -1.7, EDDF
-1.6 per airport. Small on the holdout, but the ranking set has 47,304
eligible rows (13.7%) against 38,225 (11.1%) in the holdout, including 8,665
at LEMD, whose intercept comes from only ~120 holdout rows.

**Submission v17 candidate:** `stack_lgb_cat_adsb_partial.parquet`
(`src/post/adsb_partial.py`, verified to reproduce the test exactly).

## 42. v17 = 275; LIRF echo mixture REJECTED; EGLL drift probe built
    (2026-09-27)

**Leaderboard:** v17 (stack + ADS-B + partial tracks) **275** (v16 277). The
holdout gain was -0.45, so the leaderboard gain is ~4x, consistent with the
ranking set having more partial-eligible rows.

**LIRF echo-aware ADS-B mixture** (pre-registered,
`reports/lirf_mixture_preregistration.md`; `tests/lirf_mixture_test.py`).
- Nudges only each engine's non-echo component `m_i` towards ADS-B,
  weighted by `(1 - echo_prob)`.
- Jan 2025 has no LIRF ADS-B coverage, so it was evaluated by a within-July
  day split.
- **Result: nil.** Pooled July -0.04 (P(worse)=0.338); dropping the best 3
  days gives +0.05. Rejected.
- Non-echo rows improve (706 -> 688), but echo rows worsen (876 -> 953):
  the classifier can't find them (recall ~9%).
- On the low-echo majority the model already beats ADS-B (395 vs 538). So
  LIRF's error is about the *label* (real block time vs schedule echo), not
  pushback timing. ADS-B can't fix that, and the only remaining LIRF lever
  would be better echo identification.

**EGLL drift probe (leaderboard A/B).** The label-free drift check (§40)
showed ADS-B minus AOBT_3 moving by -39 to -80 s (mean -59) at EGLL from
2025 to 2026, in all four cells. This can't be validated offline, so it's
tested on the leaderboard directly.
- `src/post/stack_submit.py --lag-shift EGLL=59` rebuilds v17 with the
  EGLL lag moved -169 -> -110 s for 2026 rows only
  (`stack_lgb_cat_adsb_partial_lagshift.parquet`).
- It differs from v17 only on the 19,547 EGLL blended rows (-31 s mean).
- Submitted as v18. Score pending.

**EGLL probe result: v18 = 276 (v17 275).** Shifting EGLL's 2026 lag by the
AOBT_3-derived +59 s made it worse. The AOBT_3 yardstick's apparent EGLL
drift was not real (or not in that direction), consistent with its own
+-40-80 s month-to-month noise. The 2025-fitted lag stays, and v17 remains
the base. Also, a 31 s mean shift on 19.5k rows moved the leaderboard by ~1
point, so leaderboard A/B probes are sensitive enough to be informative.

## 43. Quality-modulated ADS-B blend weights -- ADOPTED (2026-09-27)

The detector now emits per-row pushback quality (`adsb_pb_gap_s`,
`adsb_pb_dist_m`, `adsb_pb_gs`). The existing fields are unchanged apart
from float noise of ~5e-10 m in the two distance fields.

**Exploration** (training days 2025-09-15 / 2025-11-15 only; ADS-B error
after removing each airport-tier's lag):
- appear: first sighting > 70 m from the stand gives 159 s (vs 39-105 s);
  already moving > 5 kt gives 188 s.
- dwell: > 40 s sampling gap after the last stationary sample gives ~204 s
  (vs 121-162 s); > 70 m from the stand gives 231 s.

**Design** (`reports/adsb_quality_preregistration.md`, 3fc6cc9): keep the
production blend weights and multiply by `q[tier, good/poor]` (4 factors,
least squares, product clipped to [0, 1]).

**Result:** Jul -0.40 (P(worse)=0.017), Jan -0.55, pooled 329.37 -> 328.90
(**-0.47**, P(worse)=0.000). Factors stable across months (good x1.2-1.7,
poor x0.62-0.75); all four groups improve. The production code
(`adsb_blend.fit_quality`, `apply(..., q=)`) reproduces the test exactly.

**v19 candidate:** `stack_lgb_cat_adsbq_partial.parquet`. Rebuilt v17 is
byte-identical to the uploaded v17. v19 changes 93,344 rows by 32 s mean
absolute. Final factors: appear 1.21 / 0.75, dwell 1.63 / 0.64.

**Leaderboard: v19 = 274.25** (v17 275, v18 276). The holdout gain was -0.47,
so the leaderboard gain is ~1. Current best.

## 44. LFPG/LIRF monster labels; LIRF "+24 h" date bug -- ADOPTED
    (2026-09-27)

**Where the error is.** LFPG (34.9%) and LIRF (29.7%) carry 65% of all
holdout squared error; ADS-B work had only touched the remaining ~18%.
- LFPG: the top 10 rows (0.03%) are 79.7% of its squared error. Without its
  top 400 rows, LFPG's RMSE is 210.
- Two LFPG rows (84,240 s and 58,206 s, both easyJet, both NM-unmatched)
  have a block time on the previous evening. They are NOT the inbound
  arrival's in-block time: across 2025, label == takeoff - inbound in-block
  for only 0.006% of rows. They look unpredictable.
- 89/91 labels > 5 h are NM-unmatched (~7,000x the matched rate).

**LIRF date bug.** All 14 labels in the 24h-10min .. 24h+90min band are
NM-unmatched (13 LIRF). For 86% of them the block date equals the scheduled
date, and block + 1 day is a median 15 min before takeoff. The feed stamps
the real pushback clock time onto the scheduled date, so the label = real
taxi + 86,400 s. Among LIRF NM-unmatched rows delayed >= 12 h in the
training months, the labels are: echo 12, +24 h 6, normal 0.

**Adjustment** (pre-registered, `reports/lirf_dayplus_preregistration.md`,
408d348). On LIRF & NM-unmatched & T-SOBT >= 12 h:
`new = (1-p) pred + p (86400 + t0)`, with p = 0.333 and t0 = 1,210 s, both
from the 10 training months only.

**Holdout:** 11 segment rows. Overall **333.08 -> 329.04 (-4.0)**; Jan
-2.9, Jul -5.1; LIRF 647 -> 620. Segment SSE falls in both months.
Adopted. It rests on the mechanism, since 11 rows can't give statistical
certainty.

**v20 candidate** (`src/post/lirf_dayplus.py`,
`stack_lgb_cat_adsbq_partial_lirf24.parquet`) changes 11 ranking rows
(3 Jan, 8 Jul) by +9,900 s mean.

## 45. Status and next steps (2026-09-28)

**Leaderboard history (lower is better):**

| version | what | score |
|---|---|---|
| (pre-§40) | LightGBM mixed | 302 |
| v15 | + CatBoost stack | 296 |
| v16 | + ADS-B pushback blend | 277 |
| v17 | + partial-track estimate | 275 |
| v18 | v17 with EGLL lag shifted (probe, rejected) | 276 |
| v19 | v17 + quality-modulated ADS-B weights | **274.25** (current best) |
| v20 | v19 + LIRF "+24 h" adjustment | pending: submit 2026-09-28 |

Top team: ~224. Freeze is **2026-10-04** (CLAUDE.md), so ~6 days remain.

**Submit first:** `data/submissions/stack_lgb_cat_adsbq_partial_lirf24.parquet`
as `smart-jigsaw_v20`. It differs from v19 on only 11 LIRF rows (+9,900 s
mean), so the score change is a clean read of the §44 mechanism.
- Better: keep it, and extend the same idea (step 1 below).
- Worse: the date bug didn't carry into 2026. Revert to v19 and drop
  step 1.

**Where the error is (holdout, current pipeline):** LFPG 34.9%, LIRF 29.7%,
LTFM 8.8%, EGLL 8.7%, the other six 17.8%. A few monster labels dominate
LFPG and LIRF. Feature and ADS-B work on the "easy" airports now buys ~1
point per step; label-structure work on LIRF/LFPG buys much more.

**Next, in priority order:**

1. **Generalise §44 to all LIRF NM-unmatched delay bands** (only if v20
   helps).
   - For LIRF NM-unmatched rows the label is one of three kinds: schedule
     echo (T - SOBT), +24 h bug, or normal taxi. Their mix depends strongly
     on the delay (T - SOBT): e.g. 3-6 h is echo 82 / +24 h 1 / normal 34.
   - The model currently mis-mixes them. In the holdout's 3 h band, rows
     with normal ~1,000 s labels get 4,000-8,000 s predictions, because the
     echo classifier says echo.
   - Replace it, for this population only, with the squared-loss-optimal
     mixture per delay band:
     `pred = p_echo * offset + p_24 * (86400 + t0) + p_norm * model_normal`,
     with the p's estimated on the 10 training months. Pre-register; holdout
     check; leaderboard A/B.
2. **LTFM and EGLL error anatomy** (8.8% + 8.7%, never examined). Same
   recipe as LFPG: top-k share of squared error, label types,
   NM-unmatched, any recording pattern. Cheap (no training). It only pays
   off if there's a pattern like the LIRF bug; LFPG's two monsters had
   none.
3. **More CatBoost rounds** (fits were still improving at 8,000; ~6
   min/fit on a T4 with `max_ctr_complexity=1`). Rerun on Colab with more
   rounds, re-test the stack (`tests/stack_test.py`), rebuild. Expect ~1
   point.
4. **Freeze-week work (don't leave it to the last day):**
   - `REPRODUCE.md`: the end-to-end path. Data fetch, stands, the ADS-B
     fetch on Colab, then normalise, detector, features, the lgb (local) and
     cat (Colab) runs, and `src/post/stack_submit.py`.
   - `DATA_SOURCES.md` check: every external source used by the final
     submission (Gateway stands/runways GPLv2, adsb.lol ODbL, METAR,
     EUROCONTROL) has a row.
   - Make the repo public on GitHub under GPLv3, as the rules require.
   - JOAS paper draft. The framing from CLAUDE.md still holds (post-ops
     reconstruction), plus the ADS-B stand-appearance detector and the
     label-recording findings (echo, +24 h bug) as contributions.

**Not worth more time:** OSM routed distance (§38: straight-line geometry
added nothing); a LIRF ADS-B mixture (§42: LIRF's problem is labels, not
timing); lag re-anchoring from AOBT_3 (v18); XGBoost on free Colab (OOM).

## 46. v20 = 278 -> LIRF +24 h adjustment REVERTED; takeaways from the
    competitors' chat; revised plan (2026-09-28)

**v20 = 278** (v19 274.25). The §44 rule didn't transfer to 2026: the loss
matches "none of the 11 adjusted rows carried the bug". The stage is
removed from `stack_submit.py` (`src/post/lirf_dayplus.py` deleted; it's in
git history). The rebuilt v19 is byte-identical to the uploaded v19.
**Lesson:** a holdout gain carried by 11 monster rows (-4.0) was the
opposite of the leaderboard (+3.75). §45 step 1 (generalise to all delay
bands) is dropped.

**From the competitors' chat (GREKI, top team; Vudueprajacu; piyush7911).
Confirmed, already doing:** Jan<->Jul validation; drift checks
(movement-type ADS-B features transfer, receiver-geography ones don't);
a second tree family averaged with the first; no external data moved the
no-ADS-B airports (IST/CDG/MAD) for anyone, their gains came from a
stronger base model.

"For NM-unmatched rows nothing beat a calibrated hedge; rules learned on
2025 rarely carried over" (GREKI) is exactly v20.

**New and actionable:**
1. **Out-of-fold corrector.** "A corrector trained on out-of-fold base
   predictions (folds by month, not by day) did more than new features."
   A second-stage model on month-wise OOF predictions of the base model,
   with the base features plus the OOF prediction, learns the base
   model's systematic errors. It needs OOF predictions for the 10 training
   months (~10-12 extra fits; CatBoost on the T4 at ~6 min/fit is
   feasible). The most-cited winning technique; top priority.
2. **Carrier echo rate conditioned on no NM match.** "The echo of the
   planned time is mostly a carrier thing when there's no NM match, and
   stable month to month." Our op_echo_rate isn't conditioned on
   NM-unmatched, and the echo classifier's recall is ~9%. Add a
   per-(airport, operator) echo rate among NM-unmatched rows (OOF by
   month). Cheap.
3. **Trimmed RMSE alongside full RMSE** in every test: "two rows were ~30%
   of our squared error, so we stopped trusting the full RMSE alone".
   Fix the monster set once (e.g. holdout labels > 5 h) and report
   RMSE both with and without it.

**Revised plan (freeze 2026-10-04):**
1. CatBoost 20k-round run (on Colab now), then the pre-registered stack
   re-test (`reports/stack_preregistration.md`).
2. OOF corrector (item 1), pre-registered, Jan<->Jul cross-fit, full +
   trimmed RMSE.
3. NM-unmatched carrier echo feature (item 2). Test inside the corrector
   or as a feature.
4. Freeze-week deliverables (REPRODUCE.md, DATA_SOURCES check, public
   GPLv3 repo, JOAS draft). Start by 2026-10-02 at the latest.

## 47. CatBoost 20k-round cap -- not adopted (full RMSE rule) (2026-09-28)

`cat_mixed_r20k` alone: full 336.56 (8k: 336.72); trimmed (labels <= 5 h)
272.48 (8k: 273.03). Cross-fit stack lgb+cat20 vs lgb+cat
(`tests/stack_r20k_test.py`):
- full: Jul +0.04, Jan -0.28, pooled -0.11 (P(worse) 0.274), so the
  pre-registered rule says **keep**.
- trimmed: pooled -0.37 (P(worse) 0.007), both months better. 8/10
  airports improve; LIRF's monsters (+0.70) mask it on full RMSE.

**Convention from now on:** pre-registrations state the decision metric
up front as trimmed RMSE (holdout labels > 5 h excluded, a fixed set of 31
rows), with the guard "full RMSE must not be significantly worse
(P(worse) < 0.9)". This follows the chat's point about monster rows and
v20's lesson.

## 48. OOF residual corrector on CatBoost -- REJECTED (L2), Huber variant
    promising (2026-09-28)

Pre-registered in `reports/oof_corrector_preregistration.md` (42124ba,
amendment db82f0f); full results there, log `logs/oof_corrector_test.log`.

- **Plumbing (kept):** `fit_predict_months()` (train on any months,
  predict any rows; `run()` now calls it, output identical on every holdout
  row) and `run_oof()` (month-wise OOF folds, resumable on Colab). 10
  CatBoost folds are in `cache/oof/cat_mixed/`.
- The local `cache/features/` was stale (pre-§28 `minute_of_day`); it has
  been rebuilt.
- **Primary (L2 corrector on residual, trained on 30 <= taxi <= 7,200 s):**
  pooled trimmed -0.83 (P(worse) 0.380), Jul trimmed +1.45, full +6.17
  (P(worse) 0.974). **Rejected.** Nine airports improve on trimmed RMSE; LIRF
  gets worse (full +50). Deciles 1-9 each improve by 7-23 s; the top decile
  gets worse by +49 s.
- **Huber diagnostic:** trimmed -3.10 (P(worse) 0.000), full -1.13 (P(worse)
  0.094).
- **Hypothesis:** selecting the corrector's training rows on the true label
  biases it against large base predictions (it never sees a large
  prediction that was right), so it drags down the genuine tail.

## 49. OOF corrector v2 -- ADOPTED; leaderboard v21 = 270.2 (2026-09-28)

Pre-registered in `reports/oof_corrector_v2_preregistration.md` (b326555,
before results); gate, build and leaderboard results there.

**Design (v1 -> v2):**
- Huber loss (alpha 800).
- Training rows and application selected on the **base prediction**
  (<= 7,200 s) instead of the true label (v1's selection bias, §48).
- New input: the per-(airport, operator) echo rate **among NM-unmatched
  rows**, OOF by month (`features.encode.*nmu_echo_rate*`; Task 2, §46
  item 2).
- 5,000-round cap. Everything else as v1: LightGBM, 15 leaves, depth 4,
  min leaf 2,000, inputs = base features + OOF CatBoost prediction + echo
  probability.

**Holdout gate: PASS.** Pooled trimmed -3.25 (P(worse) 0.000); Jan -3.48,
Jul -3.14; full -2.76. Every airport improves, including LIRF (-3.6
trimmed). The top decile of true taxi is now +2.8 s (v1: +49). On
NM-unmatched rows the mean correction is +70 s.

**v21 build.**
- The Colab CatBoost rerun differs from the gate's CatBoost by 27 s RMS
  (GPU nondeterminism). Per the rule, the rerun is used throughout.
- The corrector reproduces on it: trimmed -3.48, full -2.94.
- The ranking corrector is refit on 12 months (2.08M rows).
- Stack weights lgb 0.459 / catcorr 0.555. The ADS-B stages are refit on
  top.
- v21 vs v19 on ranking rows: RMS 40 s. A local-only control (rerun
  CatBoost, no corrector) is RMS 9.8 s from v19, so the corrector is most
  of the change.

**Leaderboard: v21 = 270.2** (v19 274.25, -4.05). Adopted; new best.

| version | what | score |
|---|---|---|
| v19 | stack + ADS-B + partial + quality weights | 274.25 |
| v20 | + LIRF "+24 h" rule (reverted) | 278 |
| **v21** | v19 with CatBoost replaced by OOF-corrected CatBoost | **270.2** |

**Rebuild v21:**
1. Colab: `run_oof(engine="cat", target="mixed", eta=0.05)` and the "CatBoost
   rerun for v21" cell in `notebooks/colab_train.py`.
2. Locally: `tests/oof_corrector_v2_test.py` (writes the round count), then
   `src/post/corrector_v2.py`, then
   `src/post/stack_submit.py --cat-corrected`.

**Lessons:**
- The competitors' tip held: the OOF corrector was worth more than any
  single feature since ADS-B.
- v1's failure was a design error (selecting rows on the outcome), not
  a dead idea. Recording the Huber diagnostic in v1 is what pointed to v2.

**Open, if time allows before the 2026-10-04 freeze:**
- The same corrector for LightGBM, which needs LightGBM OOF folds (~20
  min/fit locally, slow).
- The corrector hit its 5,000-round cap in the gate.

Both would need new pre-registrations. Task 3 (freeze deliverables) now
comes first.

## 50. Plan to 2026-10-10; LightGBM corrector started; ADS-B far-sighting idea logged (2026-09-29)

**Freeze moved to 2026-10-10.** OpenSky / Trino data is not allowed, so
ADS-B coverage can't be extended with a second source; adsb.lol stays the
only one.

**LightGBM OOF corrector (in progress).** v2's corrector with the stack's
LightGBM as the base. Pre-registered in
`reports/oof_corrector_lgb_preregistration.md`. The gate is decided on the
stack (lgbcorr + catcorr vs lgb + catcorr, cross-fit), not on LightGBM alone.
- The 10 LightGBM OOF folds are running locally (22 cores beat Colab's 2 CPUs
  for LightGBM): `run_oof(engine="lgb", target="mixed", seed=42)` ->
  `cache/oof/lgb_mixed/`, log `logs/oof_lgb_mixed.log`, ~8-10 h.
- Gate script: `tests/oof_corrector_lgb_test.py`.
- Letong's v21 files (`catcorr_mixed_holdout_ev.parquet`,
  `catcorr_mixed.parquet`) were copied here. They reproduce the v21 stack
  weights exactly (0.459 / 0.555).

**Idea, not started (Letong to pick up): use ADS-B tracks first seen
already taxiing.**
- From the competitors' chat: arnavhm13 matches ~96% of LEMD Jan 2026 and
  ~70% of Jul 2026 departures to an adsb.lol surface track. CDG is thin
  (~13%), and IST and FCO-July have nothing.
- Ours: LEMD matched 96% Jan / 45% Jul, so matching isn't the gap (July
  may be stricter than theirs). But only 26% / 12% of LEMD rows get a
  pushback tier. Untiered LEMD tracks are first seen a median ~1 km from
  the stand, already moving (~14 kt), so there's no pushback to detect.
- Across all airports, **~66k ranking rows (19%)** are matched, untiered,
  and first seen moving (`adsb_first_gs >= 5` kt,
  `adsb_first_own_m >= 200` m). §41's partial estimate only covers first
  sightings < 1 km.
- **The floor `T - adsb_first_ts` is valid only for these rows.** On the
  two non-holdout 2025 days with ADS-B (09-15, 11-15), true taxi fell
  below it by more than 30 s in 2.8% of untiered-moving rows (EHAM 16%,
  the rest 0-7%). For tiered rows the rate was 36%: the first sighting is
  at the stand before pushback, so it isn't a floor.
- The floor is loose: true taxi is a median ~540 s above it (p10 110-510 s
  by airport, EHAM negative).
- v19 predicts below the floor on only ~3% of these rows (EHAM 15%). So a
  plain clip gains little.
- **The candidate** is extending §41's partial estimate to far, moving
  sightings: `est = (T - first_ts) + a_airport + b * first_dist_km`,
  blended with a fitted weight, excluding EHAM or giving it its own lag.
  Expected gain ~1-3 board points. Needs its own pre-registration and a
  holdout check (Jan/Jul 2025 ADS-B is cached).

**Other chat notes.**
- One team found a second model family hurt them; ours helped (v15, -6).
- arnavhm13 reports holdout full RMSE Jan 357 / Jul 333, and Jan ~218 with
  rows > 1 h removed. Their trim is 1 h, ours 5 h, so the trimmed numbers
  aren't comparable.

## 51. LightGBM corrector, 20k cap and joint corrector (Arm B) -- all REJECTED (2026-09-29)

Pre-registered in `reports/oof_corrector_lgb_preregistration.md`
(`5f84a60`, amendment `64765e4`, both before results). Full results are
there; log `logs/oof_corrector_lgb_test.log`.

**All three gates fail on July, so v21 (270.2) stays.**
- Arm A, NNLS(lgbcorr20, catcorr20) vs v21's NNLS(lgb, catcorr): pooled
  trimmed -0.56 (P(worse) 0.228), Jan -2.16, Jul +0.44.
- Arm B, one joint corrector on both engines' OOF: pooled trimmed -1.60
  (P 0.000), full -1.26 (P 0.014). It fails only because Jul is +0.44
  (Jan -4.90). It also beats A (pooled -1.04, P 0.005) but ties it in July.
- Per the pre-registration, no rescue variants.

**What we learned:**
- **Correcting LightGBM on its own works (-5.25 trimmed), but the stack
  had already captured almost all of it (-0.58).** Stacking two models and
  correcting them overlap: once CatBoost is corrected and stacked, there's
  little left for the LightGBM corrector to add.
- **The corrector's round cap is closed.** At 20k, CatBoost's corrector
  stops at 5,378 (vs the 5,000 cap), and the gain is -0.03.
- **Arm B's gain is all January.** July is flat in every comparison. Like
  §47/§48, gains that show in only one month have not carried over to the
  board reliably. One more holdout month would settle it, but we don't
  have one to spare.

**Artefacts kept** (cache, not committed):
- `cache/oof/lgb_mixed/`: 10 LightGBM OOF folds, ~30-50 min each locally.
- `cache/oof/corrector20_*`: the fitted correctors.
- `cache/eval/{lgbcorr20,catcorr20,joint20}_mixed_holdout_ev.parquet`.

## 52. ADS-B far-sighting extension (§50 idea) -- killed at scoping, no holdout read (2026-10-02)

Scoping only, on the two non-holdout 2025 ADS-B days (09-15, 11-15) with an
honest base (mean of the month-wise OOF LightGBM and CatBoost predictions).
No pre-registration was written because no holdout row was read. Script
`tests/adsb_far_scope.py`, log `logs/adsb_far_scope.log`.

**Population:** matched, no pushback tier, first seen >= 1 km from the own
stand, LIRF excluded (the complement of §41). 1,577 of 11,253 rows (14%),
9% of squared error.

**Why it fails:**
- These tracks are first seen late, typically near the runway. The unseen
  part (`taxi - L`) has a median of 500-1,000 s, so it is most of the taxi,
  and it doesn't grow with the first-sighting distance (fitted slope
  ~15 s/km, against ~240 s/km under 1 km in §41).
- §41's estimate, extended: the in-sample LS blend weight is 0.04
  (corr(residual, est - base) = 0.09). The far rows go 187.2 -> 186.4 s,
  which is ~0.1 s on all rows.
- The floor `T - first_ts` is not a floor here: 10-20% of rows per airport
  have taxi < L - 30 s. Clipping the base to it makes the far rows much
  worse (187 -> 348 s).
- A flexible check: a small Huber LightGBM on (L, distance, ground speed,
  L - base, base, n_pts, min distance, airport), fit on one day and scored
  on the other. At full weight it's worse; at half weight it gains ~2 s on
  far rows, so ~0.2-0.3 s on all rows.

**Conclusion:** expected holdout gain ~0.1-0.3 s, against §50's estimate of
1-3 board points. Not worth a pre-registration and gate. §50's "floor is
valid for untiered-moving rows" doesn't hold for the >= 1 km subset.

## 53. LTFM and EGLL error anatomy (§45 item 2) -- no recording pattern, nothing to build (2026-10-02)

No training. The v21 pipeline cross-fit on the holdout (NNLS(lgb, catcorr),
quality-modulated ADS-B blend, partial estimate): RMSE 327.52, trimmed
263.30. Scripts `tests/ltfm_egll_anatomy.py`, `tests/ltfm_egll_late_mix.py`;
logs `logs/ltfm_egll_anatomy.log`, `logs/ltfm_egll_late_mix.log`.

**Share of squared error:** LTFM 8.9% (trimmed 13.7%), EGLL 8.8% (13.6%).
Both are spread out, unlike LFPG: the top-10 rows carry 7% (LTFM) and 15%
(EGLL); LFPG's carry 79%.

**LTFM (RMSE 265; Jan 227, Jul 294):**
- The error is long taxis on a few July disruption afternoons (07-26,
  07-21, 07-13: day bias +110 to +160 s). Labels of 30-60 min are
  under-predicted by +525 s on average; 10-15 min labels are over-predicted
  by -95 s, the usual regression to the mean.
- Echo rows: 186, negligible. NM-unmatched: 657 rows, 8% of LTFM's SSE.
  No ADS-B. Runway 35L carries 48%.
- No operator, stand or hour pattern beyond that. Per-group bias would
  remove at most ~0.3% of SSE in any grouping.

**EGLL (RMSE 284; Jan 252, Jul 311):**
- NM-unmatched rows: 466 (1.2%), 21% of EGLL's SSE (RMSE 1,204, bias
  +312). The worst rows are on disruption afternoons (07-30, 07-06): label
  ~ (T - SOBT) minus a few minutes, i.e. left the stand on time and waited
  2-3 h.
- Labels > 1 h: 240 rows, 34% of SSE.

**Can "pushed on time, then waited" be predicted?** (`ltfm_egll_late_mix.py`,
non-LIRF, departures more than 1 h late)
- It's rare everywhere: 0-8% of NM-matched and 0-5% of NM-unmatched rows
  (EGLL NM-unmatched 2.8%). The rate is not stable between months:
  NM-matched Jan 4.5%, Jul 2.1%.
- v21 is already calibrated by delay band: mean prediction ~ mean label in
  every NM-matched band.
- A cross-fit bias shift per (delay band, NM-unmatched) cell makes it
  worse: 285.53 -> 286.03 (trimmed 222.49 -> 223.13).

**Conclusion:** like LFPG (§13), no recording quirk like LIRF's echo. The
error is genuine disruption-day congestion and rare long holds. METAR was
the candidate signal for those afternoons and was already tested and not
adopted (§24-§27). Nothing here to pre-register.

## 54. Where the 46-point board gap probably is: a few huge labels, not the base model (2026-10-02)

`tests/board_gap_anatomy.py`, log `logs/board_gap_anatomy.log`. No training.

- **The gap in rows.** v21 270.2 vs the leader ~224 over 344,841 ranking
  rows is 7.9e9 squared seconds. That equals one row off by 89,000 s, or 10
  rows off by 28,000 s, or 100 off by 8,900 s. Fixing one row off by
  40,000 s is worth ~9 points. Closing the gap uniformly would need -31% MSE
  on every row.
- **The holdout is that concentrated.** v21 cross-fit full RMSE 327.5.
  Without the top 5 rows it's 265.1 (they carry 35% of SSE); without the
  top 100, 238.8. The top 5: the two LFPG monsters (84,240 s, 58,206 s)
  and three LIRF ~87,000 s "+24 h"-type rows. All five are NM-unmatched.
- **Our base model isn't behind.** arnavhm13 reported holdout Jan 357 /
  Jul 333, and Jan ~218 with labels > 1 h removed. v21 on the same cuts:
  Jan 347 / Jul 311, Jan 204.7 (Jul 249.5).
- **So the gap is most likely in how a few dozen giant 2026 labels are
  hedged**, or plain luck on unpredictable ones (LFPG-type). It isn't in
  base-model quality. GREKI's own remark: "two rows were ~30% of our
  squared error".
- **Blind spot.** Every decision since §47 uses trimmed RMSE, and the OOF
  corrector skips base predictions > 7,200 s. Neither can see these rows.
- **LIRF delayed departures** are the one population where the hedge can
  be checked. 1-3 h NM-unmatched: echo rate 0.28, offset median 6,358 s,
  prediction median 3,278 s (a squared-loss hedge would sit near ~2,600 s).
  3-6 h NM-unmatched: echo rate 0.50, prediction median 8,841 s vs offset
  13,380 s. This motivates `reports/lirf_hedge_preregistration.md`, which is
  decided on full RMSE with a board A/B as the final arbiter.

## 55. LIRF long-delay hedge recalibration -- holdout gate PASS; v22 built, board pending (2026-10-02)

Pre-registered in `reports/lirf_hedge_preregistration.md` (684eb91, before
results); full results there.

- **Change:** for LIRF departures 1-6 h late, a shrunk mean-residual shift
  per (delay band {1-3 h, 3-6 h} x NM-unmatched), applied on top of v21.
- **Gate (full RMSE decides):** pooled -0.77 (P(worse) 0.002); Jul -1.03,
  Jan -0.47; trimmed guard -0.99. ADOPT.
- **Mechanism:** the gain is in the NM-unmatched cells. v21 over-hedges
  toward the schedule echo there, and the shift is stable at about -630 s
  (1-3 h) in both directions. The NM-matched cells flip sign between
  months and are slightly worse. Kept per the rule; noted as the first
  suspect if the board disagrees.
- **v22** = `data/submissions/smart-jigsaw_v22.parquet`: v21 plus the
  shifts (fitted on Jan+Jul), 4,142 ranking rows changed, RMS 22.7 s from
  v21. Keep iff the board < 270.2.
- The variant that adds a 6 h+ band scored better on the holdout (-1.50)
  but rests on 5-16 fit rows per cell and is the v20 failure zone. Not used.

**Board (2026-10-02): v22 = 270.86, +0.66 vs v21. REJECTED; v21 (270.2)
stays.** `src/post/lirf_hedge.py` is kept for the record but isn't part of
the submission path.

**Lesson (third time):** a LIRF delay-band rule learned on 2025 passed a
cross-fit holdout gate and lost on the board. The others were v20 (+24 h)
and §42 (echo mixture). GREKI's "for NM-unmatched rows nothing beat a
calibrated hedge; rules learned on 2025 rarely carried over" holds for
us. Jan<->Jul cross-fit doesn't protect against 2025 -> 2026 drift in LIRF's
label recording. **Don't spend more submissions on LIRF hedge rules.**

| version | what | score |
|---|---|---|
| **v21** | OOF-corrected CatBoost stack + ADS-B | **270.2** (best) |
| v22 | v21 + LIRF long-delay hedge shift | 270.86 (rejected) |

## 56. More ADS-B pushbacks: inferred stands + fallback match (v3 detector) -- scoped, pre-registered, gate pending (2026-10-02)

**Brainstorm screens, both null** (non-holdout days 2025-09-15/11-15, honest OOF base):
- Neighbour ADS-B residual (mean ADS-B-minus-base of observed departures at
  the same airport within ±15/30/60 min, applied to the others): ~±1 s,
  sign differs by day.
- Arrival taxi-in excess near T (arrival taxi-in was never a feature):
  corr with the residual 0.00 at every window.

**Where ADS-B is lost (2026 ranking).** Every day has coverage at EDDF,
EGLL, LEMD, LSZH, yet 42% / 22% / 32% / 15% of their departures match no
track (LFPG 90%, but its tracks are too sparse to reach stands).
- The match needs the takeoff roll in the track. On 2026-01-15 and
  07-15, among unmatched departures with stand coords, an aircraft sat
  at the own stand in [T-90 m, T-2 m] and was later seen moving for:
  EGLL 30/41%, LSZH 23-26%, LEMD ~10%, EDDF 3-9%, LFPG 0%.
- 9,455 ranking rows have a matched track but no stand coordinates.
- Holdout: observed-pushback rows score RMSE 90-140, against 240-320 for
  the rest at the same airports.

**Built (split-blind):**
- `src/ingest/stand_infer.py` -> `data/external/stands_inferred.csv`
  (ODbL row added to DATA_SOURCES.md): 50 stands. Validated on 848 Gateway
  stands: median 7 m off, 97% within 100 m (LFPG fails, excluded).
- `src/link/adsb_pushback.py --v3` -> `cache/adsb_pushback_v3/`: the
  inferred stands plus a fallback match for unmatched departures (tiers
  `fb_dwell` / `fb_appear`). Production output is unchanged. Existing
  detections change on 34 of 159,138 rows. Adds 4,773 appear/dwell via
  inferred stands and 29,430 fallback rows (2025 + 2026).

**Scoping (`tests/adsb_v3_scope.py`):**
- Inferred-stand rows: ADS-B RMSE 153 vs base 157, LS w 0.52 (-> 133).
- Fallback: median error ~0 but RMSE ~1,200 (wrong events). Wrong events
  have implied taxi > 30 min or far from the base prediction.
- Gated on |ADS-B - lag - base| <= 300 s: 65-69% within ±120 s, base
  142 -> 123 at w ~0.5. All rows -0.25 s. On 4 sample 2026 days: fallback
  3.7% of rows, gated 1.5%, inferred-stand 0.7%.
- Expected: ~0.5-1 board point.

Gate pre-registered in `reports/adsb_v3_preregistration.md` (trimmed
RMSE decides); script `tests/adsb_v3_test.py`.

**Gate (2026-10-02): PASS.** Pooled trimmed -0.62 (P 0.000), Jan -0.88,
Jul -0.48, full -0.50. LSZH -6.4, EGLL -1.3, EDDF -1.2. Each part works
alone (inferred stands -0.30, gated fallback -0.35). Ungated, the fallback
gets w ~0.01: the 300 s gate carries it.

**v23 built:** `data/submissions/smart-jigsaw_v23.parquet` =
`src/post/stack_submit.py --cat-corrected --adsb-v3`. 89,591 rows differ
from v21 (RMS 19.4 s). Board pending; keep iff < 270.2. (v22 was the
rejected LIRF hedge, so `cat_ctr2`'s pre-registered "v23" build becomes the
next free number.)

**Board (2026-10-02): v23 = 269.50, -0.70 vs v21. ADOPTED, new best.** The
board gain (-0.70) is about 1.1x the holdout gain (-0.62); earlier ADS-B
stages ran 2-4x.

| version | what | score |
|---|---|---|
| v21 | OOF-corrected CatBoost stack + ADS-B | 270.2 |
| v22 | v21 + LIRF long-delay hedge | 270.86 (rejected) |
| **v23** | v21 on v3 detections (inferred stands + gated fallback) | **269.50** (best) |

## 57. Pseudo-label corrector killed; v4 matched-row fallback; learned ADS-B combiner pre-registered (2026-10-02)

**Pseudo-label corrector (2026 ADS-B as labels): dead, no holdout gate
needed.**
- Label-free check: the pseudo-residual (ADS-B taxi - lag - stack) on
  observed rows has stable stand structure across days in both years
  (odd/even-day split R2 0.21).
- On 2025, stand/operator means of the *pseudo*-residual make the *true*
  residual worse, on both observed (183 -> 199) and unobserved
  (290 -> 291) rows. The structure is ADS-B timing artefact, not model
  error.
- Even *true*-residual group means barely help (290 -> 290): the stack is
  already calibrated by stand and operator.
- Per-airport pseudo-residuals also shift between years (EGLL -177 ->
  -100), but v18 showed that EGLL lag shift doesn't transfer.

**v4 detector** (`src/link/adsb_pushback.py --v4` ->
`cache/adsb_pushback_v4/`):
- The fallback also serves matched rows with no appear/dwell tier, taking
  only stand events before the matched run's first sample (the stand
  visit sits in an earlier piece of the track).
- Non-holdout days: 656 new rows; 177 pass the 300 s gate (58% within
  ±120 s, none off by > 600 s); gated LS w 0.34 (155 -> 144). Alone, about
  -0.15 s.
- Not gated separately: it feeds the combiner below.

**Learned ADS-B combiner** pre-registered in
`reports/adsb_combiner_preregistration.md`, script
`tests/adsb_combiner_test.py`.
- One LightGBM on (taxi - stack), from the stack prediction and every
  ADS-B detector field (v4).
- It replaces the four linear ADS-B stages, cross-fit Jan<->Jul.
- Bar: trimmed rule vs v23's pipeline; board must beat 269.50.

## 58. Learned ADS-B combiner -- gate PASS (-2.63 trimmed); v24 built, board pending (2026-10-02)

Full results in `reports/adsb_combiner_preregistration.md`.
- **Gate:** pooled trimmed -2.63 (Jan -3.82, Jul -1.94, all P 0.000),
  full -2.08. About 4x v23's gain.
- **Gains:** EHAM -16.6, LSZH -10.3, LEBL -9.3, EDDM -9.2, EDDF -6.7.
- **The flaw:** rows with no ADS-B get worse (274.4 -> 276.4; LTFM +8.0,
  LFPG +0.9). The combiner learns month-specific airport/hour biases from
  `s`, hour and airport. The candidate follow-up is to apply it only to
  rows with ADS-B information. That's post-hoc, so it needs its own
  pre-registration.
- **v24** = `data/submissions/smart-jigsaw_v24.parquet`
  (`src/post/adsb_combiner.py`): RMS 75.6 s from v23. Keep iff the board
  < 269.50.

**Board (2026-10-02): v24 = 265.66, -3.84 vs v23. ADOPTED, new best.** The
board gain is ~1.5x the holdout gain (-2.63).

| version | what | score |
|---|---|---|
| v23 | v21 on v3 detections | 269.50 |
| **v24** | learned ADS-B combiner (v4 detections) | **265.66** (best) |

**v25 (restricted combiner) built, board-only A/B**
(`reports/adsb_combiner_restricted_preregistration.md`).
- v24's combiner applied only to rows with ADS-B information (matched or
  a fallback tier). The other 143,239 ranking rows keep v23's value.
- LTFM/LFPG changes vs v23 drop from 36/49 s RMS to 2/20 s. v25 vs v24:
  113,703 rows differ, RMS 21.8 s.
- The holdout can't decide this (the "no ADS-B" group was already seen),
  so the board is the only test: keep iff < 265.66.
- `src/post/adsb_combiner.py --restricted` ->
  `data/submissions/smart-jigsaw_v25.parquet`. The rebuilt v24 (no flag)
  is deterministic: same 75.6 s RMS vs v23.

**Board (2026-10-02): v25 = 264.39, -1.27 vs v24. ADOPTED, new best.** The
holdout's "no ADS-B rows get worse" finding carried to the board.

| version | what | score |
|---|---|---|
| v24 | learned ADS-B combiner | 265.66 |
| **v25** | v24 restricted to rows with ADS-B information | **264.39** (best) |

## 59. Combiner v2 (ADS-B rows only, regularised) -- gate narrow PASS; v26 built, board pending (2026-10-02)

`reports/adsb_combiner_v2_preregistration.md`.
- **Gate vs v24-restricted:** pooled trimmed -0.12 (P 0.013); Jul -0.19,
  Jan -0.02; full -0.10. Per airport it's mixed (EDDM +0.9, LSZH +0.4).
- **Fits:** best_iter 4,999 / 3,014; the production fit hits the
  5,000-round cap.
- **v26** = `src/post/adsb_combiner.py --v2`: RMS 21.8 s from v25. Keep iff
  the board < 264.39. The expected effect is a fraction of a point either
  way.

**v26 (combiner v2) NOT submitted**: only one submission was left today
and its expected gain was a fraction of a point. It was set aside for v5
below. The `smart-jigsaw_v26.parquet` name is reused for whichever build
is submitted next.

## 60. Brainstorm round 2: flight-number encoding dead; v5 runway-ending matches pre-registered (2026-10-02)

**Flight-number / callsign residual encoding: dead.** On the 10 OOF
training months (non-LIRF, labels <= 5 h, residual RMS 205), even/odd-month
shrunk group means of the OOF residual barely move it:
- (airport, FLIGHT_mvt): 194.8 -> 195.3 and 211.5 -> 211.8;
- (airport, callsign) and (airport, flight, hour): the same;
- stand and operator controls: the same.

The base model is calibrated at every identity level; what's left is
day-to-day, row-level error.

**Weather** was already adopted in §24 (it's in production), so it's not
a new lever.

**v5 runway-ending matches** (`src/link/adsb_pushback.py --v5` ->
`cache/adsb_pushback_v5/`):
- At EDDF/EGLL/LSZH/LEBL/LEMD, 60-95% of unmatched departures have a
  surface run ending within 200 m of a runway in [T-300 s, T+30 s] whose
  climb-out wasn't received. LFPG has only 9-13%.
- A second match pass takes them (flag `adsb_rwy_end`). Non-holdout days:
  +568 / +308 matches; 95 get an at-stand pushback (69% within ±120 s);
  the rest add only first-sighting info (corr 0.03).
- Gate pre-registered in `reports/adsb_v5_preregistration.md`: v25's
  restricted combiner on v5 with the flag vs v25's method, trimmed rule.
  Build: `src/post/adsb_combiner.py --v5`.

## 61. Inbound-arrival ADS-B identity (hex) for pushback -- killed at scoping (2026-10-02)

`tests/adsb_inbound_hex_scope.py`; non-holdout days only.
- **Method:** link each departure to its inbound arrival (Stage 1), match
  the arrival's landing in ADS-B (gs drops through 40 kt within ±180 s of
  the landing time) to get the hex, take the parking spot where it comes
  to rest, and read the pushback as the last moment that hex is at the
  spot before T.
- **Identity alone is unreliable.** The hex is found for 54-61% of
  departures, but parallel-runway landings make the landing-time match
  ambiguous: 38-50% of readings are off by more than 600 s.
- **Confirmed identity** (the same hex takes off within ±180 s of T;
  ~1,000-1,300 rows a day) is accurate (median -90 s, 12-18% off by more
  than 600 s). But 86-88% of those rows already have an appear/dwell tier
  in v4. The new ones (1.1-1.2% of departures) are poor: 8-25% within
  ±120 s.
- **Conclusion:** the departures we can't observe aren't an identity
  problem. The receivers don't see the aircraft at its stand. Not
  pursued.

**v5 gate (2026-10-02): REJECT.** Pooled trimmed +0.04 (P(worse) 0.68).
The 19,085 newly covered rows get worse (272.1 -> 273.1). The extra
runway-ending matches add mostly noise: their first-sighting information
is weak (corr 0.03 at scoping) and their pushbacks are few. v25 stays.

## 62. Plan: ADS-B for the other 2025 months, to retrain the combiner (2026-10-02)

- **Why:** the combiner (§58) is trained only on the Jan/Jul 2025 holdout
  (~318k rows). Its fits are unstable (best_iter 4,614 vs 1,008). Its
  coverage mix also differs from 2026: EGLL detected pushbacks are ~3% in
  Jan/Jul 2025 vs 22-75% in 2026.
- **The other 10 months already have honest out-of-fold base
  predictions** (`cache/oof/{lgb,cat}_mixed/`), so their ADS-B would give a
  combiner several times more training rows, from months whose coverage
  looks more like 2026.
- **Fetch:** the Colab cell "ADS-B fetch, the other 2025 months" in
  `notebooks/colab_train.py` pulls days 1-10 of each month (Sep-Dec
  first) into its own Drive folder `adsb_restofyear/`. Copy it back to
  `external-data/adsb-restofyear/`; `src/ingest/normalise_adsb.py` now
  reads that folder too.
- **Then:** a pre-registered combiner retrain, still cross-fit by month
  and scored on the Jan/Jul holdout.

**Inbound-identity and v5 runway-end matching are both closed (§61, v5
gate). The remaining ADS-B lever is training data, not detection.**

## 63. Rome: arrival echo as a live echo signal -- real but low reach (2026-10-02)

`tests/lirf_arrival_echo_scope.py`; the 10 non-holdout training months, no
holdout.
- **Context:** LIRF is ~36% of the estimated 2026 squared error. Most of
  it is the echo hedge on delayed flights (label = normal taxi or
  T - SOBT).
- **Correction to my earlier claim:** the echo rate among delayed LIRF
  departures is 8-14% in *every* month. The "0.8% outside Jan/Jul" figure
  came from a buggy exact-equality echo definition.
- **Same hour (±30/60/180 min arrival echo rate): weak.** Delayed-flight
  echo goes 0.09 -> 0.13-0.22 only in the top bins.
- **Same operator, same day: strong, and it adds to the operator's
  long-run (leave-month-out) echo rate.** For operators with a long-run
  rate of 0.15-0.30, same-day arrival echo > 0.6 lifts departure echo
  0.16 -> 0.94. For long-run > 0.5, it lifts 0.46 -> 0.93. Echo looks
  like a per-operator, per-day recording failure that the arrivals reveal
  (arrival block times are visible in 2026).
- **But reach is low where it matters:** only 55 of 436 delayed-echo rows
  (13%) fall on days with same-day operator arrival echo > 0.3. Most rows
  it flags are punctual, where echo ~ physical taxi anyway.
- **Verdict:** a valid, split-blind feature candidate (same-day operator
  arrival echo rate at LIRF) for a future corrector or base retrain.
  Expected < 1 board point. Not pursued now.

## 64. Combiner retrained with Sep-Dec 2025 ADS-B -- gate PASS (-1.22 trimmed); v26 built (2026-10-02)

Pre-registered in `reports/adsb_combiner_12m_preregistration.md`, with
four dated amendments, all before the data reached the repo: Sep-Dec only,
days spread over each month, 09-15/11-15 re-downloaded, thin-day rule.
- **Data:** 50 days pulled on Colab (`adsb_restofyear`). The thin-day rule
  dropped 6, leaving 44 days and 253k labelled rows with month-wise OOF
  base predictions.
  - `normalise_adsb.py` now skips empty extracts (2025-10-04 was a single
    empty trace).
  - The 7 later-month days that arrived early are parked in
    `external-data/adsb-later/`, unused.
- **Gate:** pooled trimmed -1.22 (P 0.000), Jan -1.31, Jul -1.19, full
  -0.98. Every airport improves (EHAM -5.7, EDDM -3.6, LSZH -2.6).
  Best_iter 1,996 / 1,386, against v24's 4,614 / 1,008.
- **Trained on Sep-Dec alone** (no holdout month): -1.08. Later-2025
  coverage transfers.
- **v26** = `src/post/adsb_combiner.py --m12`: RMS 23.1 s from v25. Keep
  iff the board < 264.39. (The earlier combiner-v2 build that held this
  name was never submitted.)
- **Next if adopted:** more months (Feb-Aug), which need a new
  pre-registration. The later-month days already in `adsb-later/` count
  toward it.

**Board (2026-10-03): v26 = 262.70, -1.69 vs v25. ADOPTED, new best** (~1.4x
the holdout gain).

| version | what | score |
|---|---|---|
| v23 | v3 detections (inferred stands + gated fallback) | 269.50 |
| v24 | learned ADS-B combiner | 265.66 |
| v25 | combiner restricted to ADS-B rows | 264.39 |
| **v26** | combiner retrained with Sep-Dec 2025 ADS-B | **262.70** (best) |

## 65. Combiner retrained with all of 2025 (Feb-Jun, Aug added) -- gate PASS (-0.48); v27 built (2026-10-03)

`reports/adsb_combiner_fullyear_preregistration.md`.
- **Data:** all 122 days pulled; the thin-day rule kept 115 (660k OOF-month
  rows).
- **Gate vs v26's method:** pooled trimmed -0.48 (P 0.000), Jan -0.63, Jul
  -0.40, full -0.38. All airports better except LFPG (+0.05).
- **Diminishing returns** vs Sep-Dec's -1.22.
- **v27** = `src/post/adsb_combiner.py --full`: RMS 15.8 s from v26. Keep
  iff the board < 262.70.

**Board (2026-10-03): v27 = 262.13, -0.57 vs v26. ADOPTED, new best** (~1.2x
the holdout gain).
