# Why LIRF (Rome–Fiumicino) is the worst airport

Holdout RMSE by airport (Stage 3a global model): LIRF **601 s** vs 197–307 s for
the other nine. This is not a modelling bug — LIRF's target is genuinely harder,
for two compounding reasons.

## 1. LIRF's taxi-out distribution has a huge right tail

| airport | mean taxi | **std** | p99 |
|---|---|---|---|
| **LIRF** | 1195 | **1332** | **4019** |
| EGLL | 1364 | 422 | 2701 |
| LFPG | 1020 | 453 | 2409 |
| LEMD | 1015 | 313 | 1932 |
| … all others | — | 300–450 | 1700–2700 |

LIRF's taxi sd is **3–4× every other airport**. 3.2 % of LIRF departures have
recorded taxi > 40 min (max 131 167 s ≈ 36 h). These long values cluster in
**July** (worst month, and a ranking month), hours 08–14, runway 25 (single-runway
ops → queueing).

## 2. ~18 % of LIRF rows: the block-time feed echoes the schedule

`d = BLOCK_TIME_UTC_mvt − SCHED_TIME_UTC_mvt` (off-block minus scheduled):

| airport | fraction with \|d\| < 30 s ("echo") |
|---|---|
| **LIRF** | **18.1 %** |
| LEBL | 5.6 % |
| every other airport | 2.9–5.1 % |

For LIRF "echo" rows, `BLOCK_TIME_UTC_mvt` is a near-exact copy of
`SCHED_TIME_UTC_mvt` (off by seconds). The recorded taxi then equals
**takeoff − scheduled**, i.e. it absorbs the entire departure delay:

- LIRF echo rows: `RMSE(taxi, takeoff − SCHED)` = **3 s** (taxi ≡ the schedule→takeoff gap)
- LIRF echo rows: `RMSE(taxi, takeoff − AOBT_3)` = 922 s (AOBT_3 reflects the *real* pushback, which is *not* what the label records)
- LIRF non-echo rows: `RMSE(taxi, takeoff − AOBT_3)` = 440 s (normal)

So for ~1 in 5 LIRF departures the "taxi-out" label is really "schedule delay +
taxi", and AOBT_3 (the actual off-block) disagrees with it by design.

### The echo rows are partially predictable

| signal | echo rate |
|---|---|
| IOBT delay > 60 min | **0 %** |
| IOBT delay ≤ 0 | 22 % |
| AOBT_3 / EOBT_1 / FLIGHT_ID missing | 48 % |
| night hours (00–01, 22–23) | 26–48 % |
| midday (12–16) | 14–16 % |
| operator (anonymised) | ranges **1.7 % → 54 %** across operators |
| stand prefix "2" vs "4" | 32 % vs 11 % |

`AIRCRAFT_OPERATOR_flt`, `STAND_mvt`, `IOBT`/`EOBT` deltas and hour are already
model features, so the structure is learnable — but a single pooled model spends
its capacity on the 9 well-behaved airports. LIRF's `d` distribution (sd ≈ 1780 s
vs 300–500 s elsewhere) is simply a different problem.

## What this implies for the model

- **The ranking set has the same feed behaviour** (Jan/Jul 2026 LIRF), so RMSE
  *rewards* predicting the inflated values. We must model them, not discard them.
- **Raise the training label ceiling** from 5400 s so the tail is learned
  (superseded — current `fit.py` trains on `taxi` in `[LABEL_LO, LABEL_HI] =
  [30, 7200]` s, dropping rows outside that range entirely rather than
  clipping; see §4 for what this means for LIRF's extreme tail).
- **Per-airport residual head** — give LIRF its own correction on top of the
  global `d` model, trained on out-of-fold global predictions. This is the
  targeted fix; results in `reports/stage3a_resid.md`.
- Longer term: an explicit "echo probability" feature (per airport×operator×
  stand historical echo rate, out-of-fold) and possibly a two-part model
  (P(echo) × schedule-delay-taxi + (1−P) × normal-taxi).

## 3. Update (2026-09-17): the echo classifier shipped, but LIRF still sits at
   857-908s — the remaining gap is NOT the echo mechanism

The P(echo) classifier (`fit_echo_classifier`, `fit.py`) and the
`_reconstruct_taxi` blend described above were built and validated (see
PROGRESS.md §8) — a large win overall (-18%) but LIRF only dropped from
1320s to 857-908s, still 3-4x every other airport. Ran a targeted
diagnostic (`lirf_diag.py`/`lirf_analyze.py`, holdout-only, `target="direct"`,
row-level `ev` saved to `cache/lirf_diag_ev.parquet`) to find out why.

**Ruled out: classifier miscalibration.** Hypothesis was that P(echo) is
systematically underconfident (low recall = 9.6% at the P>0.5 cutoff could
mean probabilities are shifted low). Checked directly: binned `echo_prob`
into deciles and compared mean predicted probability to actual echo rate,
both globally and for LIRF alone. They track almost exactly at every decile
(e.g. LIRF top decile: mean_pred=0.679 vs actual_rate=0.675). The
classifier is well-calibrated — recall is low because most rows genuinely
have low individual-row echo probability, not because of a systematic bias
in the score. Calibration (Platt/isotonic) is **not** worth pursuing.

**Confirmed: even the classifier's own echo_pred=False LIRF rows are 2.4x
worse than the same lane everywhere else** (714.1s RMSE, n=24,136 vs 295.0s
RMSE, n=317,797 for the rest of the fleet). So the dominant remaining error
is not an echo-classification gap at all — matches this doc's original
"genuine congestion tail" hypothesis (§1), not the echo hypothesis (§2).
(For reference, the classifier's *confident* echo lane, `echo_pred=True`,
still resolves at 1999.6s for LIRF — but that's statistically identical to
the global `echo_pred=True` lane's 1994s, i.e. LIRF's confident-echo rows
aren't uniquely hard, everyone's are.)

**New finding: LIRF's error is a regression-to-mean compression across the
whole distribution, worst in the tail.** Per-decile bias on true taxi:

| decile | taxi range (s) | rmse | bias | echo_rate | n |
|---|---|---|---|---|---|
| d0 | 56–710 | 507.8 | **+272.4** | 0.15 | 2,653 |
| d1 | 710–786 | 456.0 | +183.1 | 0.12 | 2,653 |
| d2 | 786–892 | 385.8 | +142.3 | 0.12 | 2,653 |
| d3 | 892–962 | 455.8 | +126.6 | 0.16 | 2,653 |
| d4 | 962–1069 | 329.4 | +106.7 | 0.16 | 2,652 |
| d5 | 1069–1146 | 485.7 | +90.8 | 0.15 | 2,653 |
| d6 | 1146–1267 | 446.2 | +54.6 | 0.15 | 2,653 |
| d7 | 1267–1448 | 465.0 | +14.5 | 0.16 | 2,653 |
| d8 | 1448–1807 | 466.0 | -71.4 | 0.16 | 2,653 |
| d9 | 1807–88,132 | **2539.2** | **-820.2** | 0.35 | 2,652 |

Bias walks smoothly from over-predicting the shortest taxis (+272s) to
badly under-predicting the longest (-820s) — the model isn't spreading its
predictions as wide as the true distribution requires. d9 alone (top 10%)
has 35% echo rate (vs ~15% elsewhere), but 65% of d9 rows are *not*
classifier-flagged echoes and the tail is still wildly under-predicted, so
this isn't just "more echoes in the tail" either.

**Ruled out: Huber's alpha=800 gradient cap.** Ran an identical
`target="direct"` holdout pass with `loss="l2"` to isolate the loss
function (`lirf_diag_l2.py`). Result: LIRF d9 is nearly unchanged — rmse
2539.2 (huber) vs 2517.6 (l2), bias -820.2 vs -774.9 — and every other
decile is actually slightly *worse* under L2. Predicted-value ceilings are
also nearly identical (max pred ≈ 75.8k huber vs 75.9k l2, both well under
the true max of 88k-131k). The compression is not a loss-function
artifact; it persists almost identically under L2. Not pursuing
alpha-tuning or a per-airport alpha for this.

**New finding: the compression is a GBM extrapolation limit on
`sched_takeoff_offset`, not a missing-feature problem.** Two checks:

1. Pulled the existing congestion features (`sat_run`, `prev_gap`,
   `n_deprwy_30m_prev`, `n_dep_30m_prev`, `n_active_dep_rwy`,
   `mins_since_cfg_change`) by decile for LIRF. **All are flat** — no trend
   into d9 (e.g. `sat_run` 0.89-1.18 with no monotonic pattern,
   `n_active_dep_rwy` 1.05-1.16, `mins_since_cfg_change` actually *lower*
   in d9 than elsewhere). The existing congestion feature family carries
   **zero discriminative signal** for which LIRF departures land in the
   extreme tail — ruling out "just add more congestion features in the
   current style" as a fix.
2. Pulled `sched_takeoff_offset` (already the #1 feature by gain
   project-wide) and its correlation with true taxi, by decile:

   | decile | mean offset (s) | corr(offset, taxi) |
   |---|---|---|
   | d0-d8 | 1686 → 2808 | **0.02-0.04** (flat, near zero) |
   | d9 | **4252** | **0.901** |
   | LIRF overall | — | 0.616 |

   Within the extreme-taxi decile specifically, `sched_takeoff_offset`
   (takeoff − scheduled) correlates with true taxi at **r=0.90** — nearly a
   straight line — while it carries almost no signal (r≈0.03) everywhere
   else in LIRF's distribution. This generalizes the §2 echo mechanism
   (`|d|<30s` exact schedule-copy) to a **continuum**: LIRF departures with
   an unusually large schedule-to-takeoff gap have taxi tracking that gap
   almost linearly even when `d` isn't small enough to trip the echo
   classifier — i.e. a large fraction of recorded "taxi" time for badly
   delayed LIRF departures is schedule-related ground holding, not
   physical taxi distance, and this effect is not confined to the strict
   echo population.

   **Working theory for why the model doesn't already exploit this:**
   gradient-boosted trees cannot extrapolate a linear relationship past the
   split points seen in training — once `offset` exceeds the range most
   rows (LIRF or otherwise) have, the relevant leaf's predicted value is
   just the mean of the few training rows that landed there, which
   plateaus instead of continuing to track the true near-1:1 slope. That
   would produce exactly the observed pattern: under-prediction that grows
   with decile and is worst at the most extreme offsets. Not yet directly
   verified (would need e.g. checking leaf assignment / prediction vs.
   offset scatter for the most extreme rows) — flagging as the leading
   explanation, not a confirmed mechanism.

   **Candidate fix (not yet built):** extend `_reconstruct_taxi`'s
   echo blend from a binary P(echo) trigger to a **continuous
   large-offset blend** — when `sched_takeoff_offset` is unusually large
   for that airport (not just when `|d|<30s`), pull the prediction toward
   an offset-anchored reconstruction, weighted by how extreme the offset
   is. This is the same "targeted, not blanket" shape that worked for the
   echo blend (§8/PROGRESS.md) and deliberately avoids repeating the
   blanket-CEIL-raise regression (PROGRESS.md §6) — it would only touch
   rows with genuinely extreme offsets, most of which are already the ones
   driving d9's error. Needs holdout validation before submitting, same
   discipline as every other change in this project.

## 4. Correction (2026-09-17): §3's "compression" and r=0.90 were mostly
   binning artifacts — the real story is a handful of very extreme rows

A colleague's review of §3 flagged two statistical problems before any fix
was built, both confirmed by direct re-analysis (`lirf_critique_check.py`,
same holdout `ev` used throughout §3):

**Problem 1 — the decile-by-true-taxi bias table was regression-to-the-mean,
not model compression.** Any imperfect predictor shows over-prediction in
low outcome deciles and under-prediction in high ones purely from binning
on the *outcome*, even if E[taxi | features] is estimated correctly. The
real test is binning by the **prediction** instead. Redone:

| decile (by pred) | mean pred | mean taxi | bias | rmse |
|---|---|---|---|---|
| p0 | 675 | 701 | -25.4 | 202 |
| p4 | 1073 | 1051 | +22.4 | 282 |
| p8 | 1612 | 1591 | +20.9 | 580 |
| p9 | 2929 | 2947 | **-17.8** | 2688 |

Bias is small and roughly flat at every prediction level, including the
top decile (-17.8s, not -820s). **There is no real compression to fix.**
The model's conditional-mean estimate is close to unbiased throughout;
the RMSE blowup in the top bins is variance, not bias.

**Problem 2 — the r=0.90 correlation was inflated by selecting on the
outcome.** `sched_takeoff_offset ≡ taxi + d` exactly (verified: max
deviation = 0 across all LIRF holdout rows — it's an algebraic identity,
not an empirical fact). Selecting the top true-taxi decile mechanically
shrinks d's variance relative to taxi's variance within that slice, which
pushes corr(offset, taxi) toward 1 almost by construction, independent of
any real signal. Redone correctly — binned by **offset** instead (what's
actually known pre-hoc):

| decile (by offset) | mean d | std d | std taxi | corr(offset,taxi) | bias | rmse |
|---|---|---|---|---|---|---|
| o0-o8 | -451 → 2204 | 254-994 | 227-919 | **0.04-0.13** | -24 to +57 | 174-681 |
| o9 (top 10%, offset 4561-93535) | 5411 | 3938 | 5468 | **0.740** | -47.3 | 2663 |

The real correlation in the extreme-offset bin is 0.74, not 0.90 — still a
genuine jump from the ~0.06 baseline elsewhere (std_d and std_taxi are
comparable in o9, so this isn't pure mechanical inflation), but weaker than
originally reported, and bias stays small (-47s) even there. The "large
offset predicts large taxi almost linearly" claim was overstated.

**What actually explains LIRF's 908s RMSE — an error budget, not a bias
story:**

- Bias² is only **10.2%** of total LIRF MSE. Removing every bit of bias
  (impossible, since §4's Problem 1 shows there's barely any left to
  remove) would only take LIRF from 908s to an unreachable floor of
  **860s** — a 5% ceiling, not the large win §3 implied.
- **d9 alone is 78.2% of LIRF's total squared error**, and even within d9,
  bias is only 10.4% of its MSE — the rest is variance in which specific
  rows go extreme, not a directional miscalibration.
- **Tail concentration is extreme**: the single worst holdout row is 14.8%
  of LIRF's *entire* squared error; the top 10 rows are 58.4%; the top 100
  are 74.8% (of 26,528 total LIRF rows). This is a handful of transactions
  driving almost everything.
- The top-10 worst rows are true `taxi` values of 45,000-88,000s (12.5-24.5h)
  with wildly inconsistent `d` (from -28,623s to +6,365s) — these look like
  genuine operational anomalies (severe multi-hour disruptions, possibly
  schedule-record staleness) rather than a learnable congestion pattern.
  All of them have true taxi > 7200s, so **none of them were in the
  training set at all** — `LABEL_HI=7200` (not 14,400 as this doc's §"What
  this implies" section says; that text is now stale relative to
  `fit.py`) excludes 113/26,528 (0.4%) of LIRF training rows on this basis
  alone. The model has never seen a single training example at this
  magnitude, for any airport — asking it to predict these precisely is
  closer to extrapolation into an unobserved regime than pattern
  recognition, and no amount of feature engineering on the existing
  congestion family changes that.
- Predictions up to 75,767s on rows whose raw regressor never trained past
  taxi=7200s are **not** unconstrained tree extrapolation — they come from
  `_reconstruct_taxi`'s echo-probability blend, which explicitly targets
  `offset` (uncapped to `ECHO_WIDE_CEIL=140000`) by design. E.g. the single
  worst row (taxi=88,132, offset=84,236, echo_prob=0.780, pred=66,514) is
  mostly the offset-anchored blend component, not the base model
  extrapolating on its own initiative.
- **Confirmed this isn't a holdout-only fluke**: the actual `ranking.parquet`
  (Jan+Jul 2026) has the same profile for LIRF — offset (not blanked, so
  directly observable) reaches max=111,654s, with 27 rows >30,000s and 5
  rows >60,000s out of 26,899 LIRF ranking rows. Whatever produces this
  tail is present in what we're actually scored on, not a holdout
  artifact.

**Revised conclusion:** the candidate "continuous large-offset blend" fix
from §3 is **not worth building as first proposed** — its addressable
ceiling is ~5% (908→860s, and that's an unreachable upper bound), not the
large win the compression framing implied. The dominant remaining error is
concentrated in a tiny number of extreme rows that the model has never
trained on and that may not be predictable from any feature in the current
dataset (no ATFM/regulation/ground-stop data available). Two cheap,
lower-expectation experiments are still worth trying opportunistically
(not yet run): (a) LightGBM `linear_tree=True` with `offset` as a leaf
regressor, to test whether tree-extrapolation is even part of the residual
5-10% gap; (b) reparametrizing the target as `taxi/offset` (a bounded
share, reconstructed as `ŝ · offset`) so any genuine linear relationship
doesn't need the trees to extrapolate at all. Neither should be expected
to move LIRF's RMSE by more than a few percent given the error budget
above — the honest takeaway is that most of LIRF's remaining gap looks
like irreducible variance from a handful of extreme-magnitude departures,
and effort may be better spent on LFPG or cross-airport features unless a
new data source (e.g. ATFM regulation records) becomes available.

## 5. The "irreducible variance" call in §4 was wrong — a real, targeted
   fix was found and shipped (2026-09-17)

A second review pushed back on §4's "irreducible" framing: `taxi ≡ offset
- d` exactly, so the error of any offset-anchored reconstruction is just
the error of `d̂`, and the question was never "is this predictable" but
"does the *existing* model already predict it and get thrown away
downstream." It did. Full mechanism, validation (byte-identical on the
other 9 airports, paired-bootstrap P(worse)=0.0000, both months improve),
and the shipped fix are documented in **PROGRESS.md §12**, not duplicated
here — short version: `CEIL=10800` was clipping the `flip` target's raw
`offset - d̂` reconstruction before blending, which only matters for
`flip` (not `direct`, whose raw output never gets near 10800 — trained
labels are capped at 7200 so it never learns to). Relaxing that cap
**only** for `ADEP_mvt=="LIRF"` (every other airport genuinely needs the
tight cap — checked directly, their rare over-cap rows are 8-84x worse
raw vs. capped) took LIRF from 857s to **660.6s** on the same holdout.
Shipped as `smart-jigsaw_v12` (337.6s overall, -8.8% vs v11).

**§4's "irreducible" and "~5% ceiling from bias" framing should be read as
correct only for the *then-current* model** — bias was genuinely small at
the time it was measured, but that was measuring bias in a model that was
still silently discarding correct large predictions. Once that discard
was removed, there was far more real signal available than the bias
decomposition suggested. Lesson for next time: an error-budget analysis
answers "how much can recalibrating *this* model help," not "how much
could a *different* correct model help" — don't conflate the two again.

## 6. Two follow-up ideas tested and rejected (2026-09-17, same session as §5)

Both are pure post-processing changes on already-cached model outputs
(`lirf_next_tests.py`) — no retraining needed, results in seconds.

**AOBT_3-availability blend — rejected, the §5 fix already solved it via a
better mechanism.** Before the CEIL fix, `RMSE(aobt3_taxi, taxi)=1372` beat
the model's own `1827` on the `offset>7200 & aobt3_taxi present` subset
(n=744, non-circular), suggesting a ~5% further lever. But after the CEIL
fix, the model's own RMSE on that *exact* subset dropped to **770.4** —
better than `aobt3_taxi`'s static 1372. The CEIL fix's `d̂`-based
reconstruction already captures this population's signal better than the
cruder AOBT_3 proxy did. Tested anyway to confirm: blending toward
`aobt3_taxi` (narrow gate matching the original evidence, or a broader
"any LIRF row with aobt3 present" gate) makes LIRF **worse**, not better —
661→678 (narrow) or 661→754 (broad). Not adopted. General lesson: an
unbuilt lever's estimated ceiling can evaporate once a different fix lands
first — re-check before building, not just before shipping.

**Bound `d̂` directly instead of bounding reconstructed taxi (the
airport-agnostic generalization flagged in §3-4) — rejected, reproduces
the §6/PROGRESS.md blanket-CEIL-raise regression.** Clipped `d̂` to
`[-3600, 10800]` (reusing `encode.py`'s existing `D_CLIP`) before
reconstructing, then applied one **uniform** wide ceiling (140000,
no airport name) to everyone. Catastrophic: overall holdout RMSE
376.0→**1155.4**, every non-LIRF airport blew up (EDDF 231→**2630**, an
11x regression), and LIRF itself barely moved (855, not the fix's 660.6).
Mechanism: bounding `d̂` doesn't stop `offset` itself from pushing the
*reconstructed* value up for ordinary rows at every airport — the tight
final `CEIL=10800` was doing real work as a general noise backstop for
routine prediction error at every airport, not just catching the rare
wild-`d̂` cases. Loosening the *output* ceiling broadly, even with `d̂`
pre-clipped, lets that routine noise back in. Confirms: there is no
airport-agnostic free lunch here: the airport-gated design in §5 is
necessary, not just simpler-to-implement. Not adopted.

**Quantile (median) objective for `d`, `flip` target — rejected, no
meaningful effect where it should matter most.** Hypothesis: LIRF's tail
is so heavy (taxi sd ~1332s vs 300-450s elsewhere) that a conditional-
median estimator (LightGBM `objective="quantile"`, `alpha=0.5`) should be
structurally more robust to it than even Huber, which is still
fundamentally a mean-estimator. Tested: one retrain, `target="flip"`,
holdout-only, same train/valid split as every other experiment
(`lirf_quantile_test.py`; `LOSS_OBJECTIVE`/`alpha` plumbing added to
`fit.py` for the test, then reverted since not adopted). Compared against
a fair same-target baseline (`flip` + Huber + the already-shipped CEIL
fix, all 10 airports — not the production `mixed` number, since `mixed`
uses `direct` for 9 of the 10 airports and target has to match to isolate
the loss function's effect):

| | flip+Huber+CEILfix | flip+quantile-median+CEILfix |
|---|---|---|
| overall | 344.0 | 345.6 (slightly worse) |
| **LIRF** | **661** | **657.8** (-0.5%, noise-level) |
| other 9 airports | — | worse on 7, ~unchanged on 2, none dramatically |

LIRF — the airport this was specifically supposed to help most — barely
moved, and every other airport got marginally worse. Matches two earlier
findings rather than contradicting them: (a) this doc's own L2-vs-Huber
test (§4) found LIRF's tail compression nearly identical under either
loss, and (b) PROGRESS.md §5's note that other contestants found
loss/hyperparameter tuning was not significant once tested properly. Not
adopted; `fit.py` reverted to the pre-test state.

Simplified summary: LIRF (Rome) was by far our worst airport — 857 seconds of error, 3–4x every other airport. We found that for a tiny sliver of LIRF flights (22 out of 26,528, the ones with genuinely huge multi-hour delays), the model's raw prediction was being silently chopped down by a safety cap that exists to stop other airports' models from going haywire. For every other airport, that cap is doing its job correctly and must stay — but for LIRF specifically, those big raw predictions are usually right, not noise. So we relaxed the cap for LIRF only, leaving all 9 other airports mathematically untouched (verified: zero difference, not just similar). Result: LIRF error dropped 857s → 661s, and the overall model improved 370s → 338s — the single biggest jump of this whole investigation, from a 2-line, narrowly-targeted, thoroughly-validated change.