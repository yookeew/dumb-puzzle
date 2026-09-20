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

Items 3-5 (queue, `ADES_mvt`, and a second weather seed, all under `mixed`)
are running; results follow in the next entry once they land.
