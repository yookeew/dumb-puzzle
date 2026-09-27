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

## 7. Four follow-up checks (2026-09-23): distribution shape, runway/stand
   pattern, LIRF-only model, and what it all says about the flip/CEIL fix

Prompted by "concrete things that haven't been checked" after the ADS-B
recovery work (`reports/adsb_recovery_ceiling.md`) — four questions, each
run against data/features already on disk. Scripts:
`tests/lirf_distribution_test.py` (Q1/Q2), `tests/lirf_specific_model_test.py`
(Q3).

### Q1 — is the taxi-time distribution bimodal, heavy-tailed, outlier-driven,
    or high-variance throughout? **A clean three-part mixture, not bimodal.**

Full-year LIRF taxi (n=160,704): mean 1195s, sd 1332s vs 974s/414s pooled
across the other 9 airports. The raw histogram is **unimodal** (one peak
~950–1300s, then a smooth decay out to 131,167s) — no visible second mode.
But decomposing by the two mechanisms already known from §2/§4-5 above
gives a sharper, previously-unquantified answer:

| segment | % of rows | mean taxi | sd | % of total taxi-mass |
|---|---|---|---|---|
| ordinary | 81.9% | 1101 | **438** | 75.5% |
| echo (\|d\|<30s) | 17.8% | 1379 | **942** | 20.6% |
| extreme (taxi>7200s) | 0.3% | 15,762 | **16,504** | 3.9% |

The "ordinary" 82% of LIRF has sd=438 — statistically indistinguishable
from the pooled sd=414 of every other airport. **LIRF's headline 3.2x
variance is not intrinsic to typical LIRF taxiing at all — it is entirely
attributable to the echo subpopulation (itself higher-variance, not just
higher-mean) plus a vanishing 0.3%-of-rows extreme tail.** The airport
isn't uniformly noisier; it's an ordinary airport with two identifiable,
separable failure modes riding on top.

### Q2 — does LIRF have a runway/stand pattern the model can't see?
    **Real, but mostly already visible via the raw categorical; one
    candidate slice not yet tested directly.**

On "core" rows (excluding echo and extreme, 81.9% of LIRF):

- Runway 25 carries ~90% of all traffic in **every month** (79–94%, no
  real seasonal config shift) — LIRF is heavily single-runway-dominant
  year-round, not switching configurations.
- Mean taxi does vary by runway even in the core population (16L 1735s,
  16R 1454s, 34L 1317s, 25 1064s) — real, but `RUNWAY_mvt` is already a
  raw categorical feature, so a tree model has direct access to this level
  effect without needing a derived feature.
- The slice flagged in §1 (runway 25, July, hours 08–14) shows a genuine
  diurnal effect surviving the echo/extreme exclusion: mean taxi climbs
  from 823s (hour 0) to a 1259–1514s plateau across hours 8–14, then drops
  to ~950–1150s in the afternoon/evening — a real ~40–80% relative
  increase, isolated more cleanly than before (the §3 "existing congestion
  features are flat" finding tested this by taxi-decile across the whole
  *contaminated* distribution, which may have diluted a real effect with
  echo/extreme rows). **Not yet tested**: whether the existing congestion
  features (`sat_run`, `n_active_dep_rwy`, etc.) already discriminate this
  specific (runway=25, July, hour 8-14) slice when checked directly, rather
  than inferred from the earlier decile analysis. Flagged as a candidate,
  not chased further this session.

### Q3 — is a LIRF-specific model better than the pooled model, post-fixes?
    **No — slightly worse (+2.6%), a clean negative result under the
    current pipeline.**

The original 2026-09-02 attempt (top of this file, `stage3a_resid.md`)
failed for two bundled, now-irrelevant reasons: a widened label window fed
into the *shared* pooled model (not this design), and residual heads that
overfit a crude 2-fold OOF (not this design either — this is a from-scratch
base regressor, not a residual head). Re-tested cleanly post-Huber,
post-echo-classifier, post-CEIL-fix.

Design: fit a `flip`-target `d`-regressor on LIRF-only 2025 training rows
(own priors, own OOF group-encodings — genuinely separate, not a filtered
slice of the pooled model), then reconstruct using the **same** echo blend
inputs (`echo_prob`, `sched_takeoff_offset`, `use_prior`≡False for LIRF,
`ceil`=`LIRF_RAW_CEIL`) as the already-validated pooled production model
(`cache/lirf_ceilfix_mixed_ev.parquet`, v12) — isolating specifically
"does the base regressor benefit from pooling or specialization," not
confounding it with a fresh (much-smaller-data) echo-classifier refit.

| | RMSE |
|---|---|
| pooled model (cached production `pred`, sanity check) | 660.6s |
| pooled regressor + shared blend (recomputed, matches above exactly) | 660.6s |
| **LIRF-only regressor + same shared blend** | **677.9s (+17.3s, +2.6% worse)** |
| pre-blend raw regressor only, pooled | 816.2s |
| pre-blend raw regressor only, LIRF-only | 809.1s (-7.1s, -0.9% better) |

Interesting nuance: the LIRF-only regressor's **raw** (pre-blend, pre-clip)
prediction is very slightly *better* in isolation, but the full blended
reconstruction is worse. Given LIRF's error is known to be extremely
concentrated (§4: top 10 rows = 58.4% of total SSE), this is consistent
with the LIRF-only model differing mainly on a handful of extreme rows in
a way that doesn't compose as well with a blend (echo_prob weighting,
ceiling) whose calibration reflects the pooled model's specific residual
structure — not chased further, since the net effect (what actually gets
submitted) is unambiguously negative either way.

**Conclusion: cross-airport pooling helps LIRF, even now.** The other 9
airports evidently supply transferable structure (operator behavior,
stand/runway taxi-distance patterns, general congestion dynamics) that a
LIRF-only fit, despite having 134,176 of its own training rows, doesn't
fully recover on its own. Matches the original architectural decision
(PROGRESS.md §8: "a full separate model would have to re-learn everything
... losing the cross-airport pooling") — now confirmed directly under the
current, much-improved pipeline rather than just argued from first
principles.

### Q4 — are the flip/CEIL dynamics symptomatic of something deeper?
    **Yes — they're the same finding as Q1, seen from the model's side.**

`flip`'s reconstruction is `offset − d̂`, so for both of Q1's contaminating
segments (echo absorption and genuine multi-hour holds) it inherits the
exact, always-known `offset` term almost for free; `direct` has no such
path and is structurally range-capped by never training past
`LABEL_HI=7200`. The CEIL fix (§5) worked specifically because it stopped
discarding that offset-anchored signal for the handful of rows where it
mattered. Q1's decomposition and the flip/CEIL story are the same
underlying fact seen from two angles: **LIRF is not one hard airport — it
is an ordinary airport (core sd≈438, matching everyone else) plus two
identifiable contaminating mechanisms, and any inductive bias that
exploits `sched_takeoff_offset` directly (flip) handles that contamination
structurally better than one that doesn't (direct).** This also explains
Q3: pooling helps the *ordinary* 82% (shared structure across airports),
while the CEIL fix and echo blend — not model specialization — are what
handle the contaminating 18%+0.3%. Two different problems, already being
treated by two different, already-shipped mechanisms; there isn't a third,
undiscovered LIRF problem hiding behind either one.

## 8. Read the top-10 worst rows individually -- one shared trait, two
   distinct fixable mechanisms, and hedging doesn't work (2026-09-23)

§4 found "top 10 rows = 58.4% of LIRF's SSE" but never read the rows
themselves. Following up: `tests/lirf_top10_test.py` pulls all raw fields
(not just the feature matrix) for the current top 10 LIRF holdout rows by
squared error, joined against `cache/lirf_ceilfix_mixed_ev.parquet` (v12)
for the model's own diagnostics (`echo_prob`, `taxi_model_raw`,
`has_aobt3`).

**Scale, recomputed precisely on v12**: top 10 LIRF rows = 28.5% of LIRF's
SSE, **8.4% of the entire holdout's squared error** (all 10 airports,
n=344,339) -- ten rows out of 344,339.

**One shared trait across all 10, not previously flagged**: every single
one has `has_aobt3=False` **and every NM-matched (`_flt`) field is null** --
no callsign, aircraft type, operator, market segment, flight type. These
are not a random draw from LIRF's tail; they are 100% drawn from the small
NM-completely-unmatched population. Filterable, not irreducible noise
spread through the tail.

**Two distinct, opposite-direction mechanisms, not one:**

- **Rows 1-3 (genuine extreme events, ~21% of LIRF's SSE)**: true taxi
  87,186s / 87,177s / 87,361s -- all clustered at **24.2-24.3 hours**.
  `BLOCK_TIME_UTC_mvt` precedes the *scheduled* departure by hours in each
  case, and actual takeoff follows ~24h later -- reads like a
  cancelled-and-reinstated-next-day flight or a record-matching artifact,
  not congestion. Model under-predicts by 21-33k seconds; the raw
  regressor (48-61k) is closer than the blend but still far short --
  nothing in training is anywhere near this magnitude (LIRF's own
  `LABEL_HI` exclusion and general training-range limits, per §4).

- **Rows 4-9 (6 of the 10, ~6.3% of LIRF's SSE) -- a clean, mechanistic
  bug, not noise.** True taxi is **ordinary** (723-1,444s, typical LIRF
  range). Each has a genuinely large gate delay (`d`=11,800-19,500s)
  followed by a normal taxi-out once pushed back. The echo classifier
  assigns these moderate-high `echo_prob` (0.56-0.75) anyway -- apparently
  reading "large offset" as echo-like even though `d` is nowhere near
  zero -- pulling the blend toward the huge `offset` and inflating a
  12-24 minute taxi into a predicted **2.8-4.7 hours**. In every one of
  these 6 rows the pre-blend raw regressor was already closer to truth
  than the final blended prediction: **the echo blend is actively
  injecting error here, not fixing anything.** All 6 are within
  `has_aobt3=False`, where the classifier has none of the `_flt`-derived
  features it otherwise relies on.

- **Row 10 (mirror-image failure)**: a near-perfect echo (`d`=-5s) the
  classifier under-confidently scores at `echo_prob=0.281`, causing
  under-prediction (5,807 vs true 14,890).

**Hedge-strategy what-ifs (override prediction on just these 10 rows,
recompute overall RMSE):**

| strategy | new RMSE | recovered |
|---|---|---|
| current (production) | 337.56s | -- |
| oracle (predict truth exactly) | 323.04s | **-14.53s (+4.3%)** -- the ceiling |
| offset-anchored | 336.39s | -1.18s (+0.35%) |
| drop the echo blend, raw only | 340.30s | **worse** (-0.81%) |
| half-hedge toward offset | 336.71s | -0.86s (+0.25%) |

**No uniform hedge works**, because the three sub-groups need opposite
corrections: rows 1-3 and row 10 need pushing *up* toward `offset`, rows
4-9 need pushing *down* away from it. A single global rule serves none of
them well. RMSE's square root also caps even the oracle case at +4.3%
overall, not the ~8% a linear read of the SSE share would suggest.

**Verdict**: rows 4-9 are a **real, targeted, fixable bug** -- the echo
classifier misreads large-offset-but-genuinely-delayed flights as
echo-like specifically within the `has_aobt3=False` lane, and 6 of LIRF's
10 worst rows are exactly this mechanism. Rows 1-3 look closer to
genuinely hard (an unprecedented-magnitude, likely data-anomalous event) --
the oracle-vs-raw gap there is a training-range extrapolation limit, not
obviously a classifier bug. **Not yet built** -- a candidate fix (suppress
or recalibrate `echo_prob` specifically when the classifier's own input
features are almost entirely null, i.e. `has_aobt3=False`) is a decision
point given the echo blend has already caused two regressions from
over-eager global changes (PROGRESS.md §6, §12) -- needs the same
validate-before-ship discipline as every prior change here, not built
unilaterally.

## 9. The rows-4-9 fix does NOT survive validation -- no monotonic
   recalibration of echo_prob recovers value (2026-09-23, same day as §8)

Review feedback on §8 before building anything: don't tune on 10 rows.
Three checks requested first, plus one more that turned out decisive.
Scripts: `tests/lirf_echo_gate_scoping_test.py`, `tests/echo_ranking_score_test.py`.

**1. Rows-4-9-only oracle (not all 10): +3.18s (+0.94%).** Matches the
prediction that the fixable subset is worth ~3-5s, not §8's 14.53s (which
included rows 1-3, out of scope for this mechanism).

**2. The true population is far bigger than 6 rows.** Widening the exact
gate (`has_aobt3=False & operator-null & is_echo=False & echo_prob>0.5`)
across the *whole* holdout (not just LIRF's top-10): **81 rows** (79 LIRF,
1 EHAM, 1 LSZH) carrying **11.71% of the entire holdout's squared error**
-- bigger than LIRF's whole top-10 (8.4%). Oracle recovery for all 81:
**+20.38s (+6.0%)**.

**3. Ranking-set count, done properly.** The free offset-based proxy
(`has_aobt3=False` & near-total `_flt` nullity & large `offset`, all
available at prediction time since only `d`/`taxi`/`BLOCK_TIME` are
blanked) found 4,067 candidate rows. Fitting the actual production echo
classifier on all-2025 data (`clf_best=402` from a fresh holdout check,
442 rounds on the full refit -- mirrors `run()`'s real submission pipeline
exactly) and scoring ranking directly: **178 rows with `echo_prob_r>0.5`
in the null lane, 177 of them LIRF** -- matching the "two hundred" order
of magnitude the reviewer predicted, not the pessimistic "holdout has six"
case. Saved to `cache/ranking_echo_prob.parquet`.

**Then the check that actually mattered: does correcting this cost more
than it saves?** Within the same null lane, the classifier ALSO correctly
catches **116 true echoes** at `echo_prob>0.5`. Forcing those onto the
raw-model-only path (removing the echo blend) would cost **6.03 billion**
in squared error -- *more* than the entire 81-row false-positive
population's current total cost (4.59 billion). A blanket suppression
would be a net regression, the same failure shape as §6/§12.

**Are true and false positives separable on anything available pre-hoc?
No.** Checked offset, `STAND_mvt`, `RUNWAY_mvt`, month, and `echo_prob`
itself:
- Offset: TP median 8,609s vs FP median 8,465s -- indistinguishable. A
  gate at offset>3,600s retains 91% of TPs *and* 94% of FPs.
- Stand/runway/month: both dominated by the same runway (25) and month
  (July) in near-identical proportions -- no discriminating structure.
- `echo_prob` itself: TP median 0.709 vs FP median 0.607, but FP's max
  (0.843) exceeds TP's median -- heavy overlap, only weak separation.

**End-to-end simulation (not counting rows -- plugging modified
`echo_prob` back through the real `_reconstruct_taxi`, LIRF-only to
sidestep the `use_prior` fallback which never applies to LIRF, sanity-
checked to reproduce baseline exactly at no-op):**

| intervention | overall RMSE delta |
|---|---|
| hard threshold, zero below 0.5-0.85 | +0.44s to +16.77s, **worse at every threshold** |
| proportional damping ×0.9 to ×0 | **worse at every setting** except ×0.9 (-0.15s, noise) |
| suppress only the 0.5-0.85 band | +7.6s to +16.4s, **worse every time** |

**Every recalibration tested regresses.** The LIRF null lane has ~397
rows total, far more than the 116 TP + 81 FP identified by ground truth --
the other ~200 rows' smaller `echo_prob` contributions are apparently
already net-helpful, and any recalibration broad enough to correct the 81
false positives removes more value from that larger population than it
recovers.

**Conclusion: no fix, not shipped.** The mechanism described in §8 is
real and correctly diagnosed, but it does not correspond to an extractable
improvement given the current feature set and blend architecture -- true
and false positives are observationally indistinguishable pre-hoc, and
correcting the identifiable false positives costs more than it saves via
any monotonic recalibration of the existing signal. Recording this
precisely (following the project's own convention -- `stage3a_resid.md`,
the rejected hour-encoding and quantile-objective attempts) so it is not
re-derived in a future session. A real fix here would need a genuinely new,
separating feature -- not a post-hoc adjustment of `echo_prob` -- and none
was found among the fields already available for this population (every
`_flt` field is null by construction, which is exactly why this lane is
hard).

**Rows 1-3 (the ~24h anomalies), separately, not bundled**: 50 ranking
rows share the offset>30,000s signature. Structurally undetectable the
same way in production, though: their holdout signature depends on
`BLOCK_TIME_UTC_mvt` preceding `SCHED_TIME_UTC_mvt`, and `BLOCK_TIME_UTC_mvt`
is exactly the field blanked for ranking DEP rows -- there is no
equivalent pre-hoc check available. Remains a genuinely open, distinct
question for a future session.

## 10. Brainstormed and exhausted every available-feature angle for the
    null lane -- genuinely closed, not just under-explored (2026-09-23)

Since no monotonic recalibration of `echo_prob` survives validation (§9),
checked whether ANY available feature separates the true-echo (TP) from
false-positive (FP) rows better than `echo_prob` itself, before concluding
there's nothing left to extract from current data.

**Isotonic calibration (the "right-shaped" fix, not a crude threshold)**:
in-sample (optimistic) gives -1.81s; honest cross-fit (calibration curve
fit on Jan, applied to Jul and vice versa, matching this project's own
transfer-testing convention) gives **-0.18s, noise-level** (and the Jan
fold is only 60 rows, too small to trust a calibration curve from anyway).
Confirms `echo_prob` is already close to properly calibrated for this
lane -- the earlier crude threshold/damping failures (§9) were not a
calibration-shape problem, there's genuinely little slack here.

**New-feature sweep, all negative:**
- `ADES_mvt` (destination): scattered, no concentration in either group
  (7 max count in TP, 2-4 in FP, across many distinct codes).
- `AIRCRAFT_TYPE_mvt`: **null for 100% of both TP (111/111) and FP
  (79/79)**. This population isn't just NM-flight-list-unmatched -- the
  reporting airport's own movement record is sparse too, a stronger
  finding than the `has_aobt3=False` framing alone suggested.
- `STAND_mvt` raw value: no dominant stand in either group (checked in §8).
- `stand_echo_rate` group encoding (already exists, `GROUP_ENC_KEYS`,
  and already known to carry real aggregate signal -- 32% vs 11% echo
  rate by stand prefix per §1): computed on the real training-split
  encoding (not a proxy), median stand_n~1,010-1,026 (large, reliable
  cells, not smoothed noise). AUC for separating TP/FP within the
  echo_prob>0.5 population: **0.607** (weak) vs `echo_prob` alone's
  **0.726**. `stand_mean_d` alone: 0.519 (~random). A naive combination
  of `echo_prob` + `stand_echo_rate` scores **worse** (0.710) than
  `echo_prob` alone -- stand information doesn't add orthogonal signal,
  it just dilutes the better one.

**Conclusion: genuinely closed, not under-explored.** Every readily
available field or derived feature has now been checked (offset, stand
raw + encoded, runway, month, ADES, aircraft type, and the calibration
shape of `echo_prob` itself) and none improves on the status quo.

**Correction (same day): ATFM is not an open lever for this population.**
An earlier draft of this section named ATFM regulation records as the
remaining option, citing "unresolved permission/licensing status" --
stale, from the project's original planning notes, not a fact checked
against what actually happened. ATFM was ingested, built, and tested in
full (PROGRESS.md §20): -0.4s, P(worse)=0.453, closed against a
noise-corrected **2.07s structural ceiling** on ANY (airport, day)
-granularity external feature -- a bound from residual-variance
decomposition, not a property of this specific lane, so it applies here
identically. It is also structurally the wrong shape for this problem
regardless of the bound: an airport-day feature is constant across every
LIRF departure on a given day, so it cannot separate a true echo from a
long gate delay occurring on the *same* day -- and per-flight separability
within a day is exactly what this section's AUC checks show is missing.
Not a candidate; do not reopen without new sub-daily data, which
EUROCONTROL does not publish.

**What is actually open:**
1. **Rows 1-3** (§8-9): undetectable in production by construction --
   their signature depends on `BLOCK_TIME_UTC_mvt`, exactly the blanked
   field. Needs a proxy signal or stays unreachable.
2. **The real question**: is there *any* signal, in data not yet
   examined, that separates a true echo from a long-gate-delay-then-
   normal-taxi flight when every NM field is null? This is a
   feature-discovery question, not a recalibration one -- §9 already
   showed recalibrating the existing signal is exhausted, and this
   section's sweep shows the *currently available* fields don't answer
   it either. Genuinely open, not closed.

Beyond those two, this is a stopping point: ten candidate mechanisms
measured across this investigation, one adopted (the CEIL relaxation,
§5), error fully localized to two named, understood populations (extreme
anomalies; null-lane ambiguity) rather than a diffuse "LIRF is hard"
description.

## 11. Correction to §10: rows 1-3 are NOT structurally unreachable --
    `LABEL_HI=7200` is a hyperparameter, and raising it measurably helps
    them (2026-09-23)

§4-10 repeatedly described rows-1-3-style extreme magnitudes as blocked
by a GBM "training-range extrapolation limit," treating `LABEL_HI=7200`
(the taxi-range training filter in `fit.py`) as a structural fact. It
isn't -- it's a choice, last tested (`stage3a_resid.md`, 2026-09-02) under
plain L2 with no Huber, no echo classifier, no CEIL fix, no flip/direct
split -- an architecture that no longer exists. Re-tested under the
current architecture (`tests/label_hi_test.py`): `target="flip"`, pooled
across all 10 airports, Huber alpha=800 unchanged, comparing
`LABEL_HI=7200` (current) against `LABEL_HI=200000` (effectively
unbounded, above the true max of 131,167s). Priors, group encodings, and
the echo classifier don't depend on `LABEL_HI` at all, so all three were
fit once and shared between arms -- isolates the test to exactly the one
line in question.

| | baseline (`LABEL_HI=7200`) | raised (`LABEL_HI=200000`) | delta |
|---|---|---|---|
| overall | 342.19s | 348.16s | **+5.97s (+1.7%), P(worse)=0.978** |
| 2025-01 | 362.20s | 363.10s | +0.90s |
| 2025-07 | 325.18s | 335.64s | **+10.46s** -- fails the both-months rule |
| LIRF | 652.65s | 688.90s | **+36.25s (+5.6%), worse** |
| **LIRF's 3 most extreme rows** | 29,036.7s | 25,347.9s | **-3,688.8s (-12.7%), better** |
| EDDF / EHAM / LEBL | 229.5 / 228.9 / 230.6 | 232.9 / 235.5 / 233.7 | all worse |

Cluster bootstrap (airport, day): paired RMSE delta=+5.97s, 95% CI
[+0.19, +12.54], P(worse)=0.978 -- a real, statistically confirmed
regression, not noise.

**The mechanism is confirmed real: raising `LABEL_HI` under Huber DOES
improve the 3 most extreme rows (-12.7%)** -- Huber's gradient is capped
at `alpha=800` for any residual beyond it, unlike L2's linear scaling, so
an 87,000s-off training example contributes the same bounded gradient as
an 801s-off one. The model can learn something closer to a soft,
extremity-aware adjustment once given the training signal, exactly as
theorized. **§4-10's "unreachable by construction" / "structural
extrapolation limit" framing for rows 1-3 is corrected: it was a
statement about the current hyperparameter choice, not a hard limit of
the architecture.**

**But the net effect is still a regression, for a precise, now-measured
reason**: EDDF, EHAM, and LEBL -- airports with essentially zero extreme
rows -- all got worse. They pay for a label range they never use, because
widening the shared regressor's window changes split selection and leaf
values everywhere in the pooled model, not just on the tail. This is the
same failure shape as the original L2 test, smaller in magnitude (Huber
dampened it, didn't eliminate it).

**Conclusion: don't raise `LABEL_HI` globally.** The natural next
question -- a moderate intermediate value instead of unbounded -- was
considered and rejected as the wrong search: a moderate `LABEL_HI` still
widens the *shared* window, so it still taxes the 9 airports that don't
need it; finding a value where the (shrinking) gain exceeds the
(shrinking) tax is tuning one number against a known-bad tradeoff, not a
structural fix. **Points toward a separate tail model instead** (keeps
the main regressor untouched at `LABEL_HI=7200`, so EDDF/EHAM/LEBL pay
nothing) -- see §12.

## 12. Separate tail model -- oracle-gated ceiling is real and substantial:
    +3.63% overall (2026-09-23)

Per review guidance after §11: measure the tail model's value with an
oracle GATE (perfect routing, using the true label to decide which rows
get its prediction) before building any real router -- the routing
problem is exactly what killed §9 (rows 1-3's signature depends on the
blanked `BLOCK_TIME_UTC_mvt`). If the ceiling is small, the gate question
is moot. `tests/tail_model_oracle_test.py`.

Population: true taxi > `LABEL_HI`=7200, pooled across all 10 airports (not
LIRF-only) -- 584 total 2025 rows (480 LIRF / 37 LFPG / 37 EGLL / 21 LTFM
/ 6 LSZH / 3 EHAM), 435 training / 149 holdout. Tail model: same `flip`
target, same Huber alpha=800, fit on only the 435 training rows (348
after an 87-row validation split) -- deliberately low-capacity given the
tiny population (`_fit_lgb`'s hardcoded `min_data_in_leaf=100` already
caps it at a handful of leaves; no override param exists, not worth
modifying the shared fitter for one test). Converged in 22 rounds.

**On the 149 true holdout tail rows:**

| prediction | RMSE |
|---|---|
| status quo (production blend) | 10,093.2s |
| naive offset-only (zero training) | 9,126.5s |
| **tail model** | **9,086.0s** |

The current blend is worse than simply guessing `offset` for this
population -- a zero-effort heuristic beats the shipped architecture here.
The tail model, tiny as it is, edges out even that heuristic.

**Oracle-gated overall RMSE (real tail-model predictions, oracle only on
routing): 342.19s -> 329.76s, -12.44s (+3.63%).** Larger than several
adopted wins this session (weather ~1-4s, minute_of_day fix -0.3s) --
clears the bar for the gate to be worth solving, not moot.

Per-airport, the win is broad, not LIRF-only: EGLL -69% (n=20), LIRF -26%
(n=113), LSZH -28% (n=4), LFPG barely moves (-3%, n=9 -- matches LFPG's
known data-corruption-style outliers, `reports/lfpg_investigation.md`).

**Caveat for the record**: this test's baseline is `flip`-only pooled
across all 10 airports (342.19s), not production's `mixed`-target
(335.6s) -- internally consistent for isolating this question, not a
like-for-like production comparison.

**Conclusion: the ceiling justifies building a router.** Next step is the
harder problem flagged going in -- a router using only pre-hoc information
(offset, `has_aobt3`, `echo_prob`, congestion features, never the blanked
`BLOCK_TIME`/taxi), checked for the same false-positive-damage risk that
sank §9's echo-blend fix when routing wasn't clean. Not yet built.

## 13. The router, built and honestly validated -- decisive failure at the
    default threshold, exactly the risk flagged going in (2026-09-23)

`tests/tail_router_test.py`. Free pre-check first: `sched_takeoff_offset`
alone has **AUC=0.956** for predicting tail membership (taxi>`LABEL_HI`)
on the holdout -- far more separable than the null-lane echo investigation
ever found (ceiling there was AUC 0.726), a strong prior this could work.

Built a P(tail) classifier mirroring `fit_echo_classifier` exactly (same
feature matrix, same discipline of not filtering by the `[LABEL_LO,
LABEL_HI]` keep mask since tail rows are exactly what that mask excludes),
retargeted at `is_tail = taxi > LABEL_HI` instead of `is_echo`. Final
prediction: `P(tail)*pred_tail + (1-P(tail))*pred_main`, where `pred_tail`
is §12's tail model unchanged and `pred_main` is the existing echo-blend
reconstruction, completely untouched.

**Result: a certain, large regression.** Overall 342.19s -> 373.08s
(+30.89s, -9.03%), cluster bootstrap **P(worse)=1.000**. Every single
airport got worse, including ones with essentially zero tail rows (EDDM
+14.8, LEMD +8.6, LSZH +12.2, LTFM +13.3) -- not a false-positive-hotspot
problem, a broad one. Mechanism: at the classifier's P>0.5 operating
point, precision is only **24.2%** (68 true positives, **213 false
positives**) against a base rate of 0.025% -- and those 213 false
positives get pulled toward the tail model's huge-magnitude output,
exploding their own RMSE from 2,585.4 to 6,425.6. High AUC on a single
feature does not imply usable precision at a base rate this extreme (149
positives in 344,339 rows); the absolute false-positive count swamps the
rare true positives because the cost of a wrong tail-routing is enormous
relative to the benefit of a right one.

**Not yet closed** -- this is the default (implicit P>0.5-ish) operating
point from the raw probability blend, not necessarily the best one. A
much higher-precision threshold, or a hard-gated (not continuously
blended) design requiring extreme confidence before trusting the tail
model at all, might still recover part of §12's ceiling without the
broad damage. Per-row predictions saved to `cache/tail_router_ev.parquet`
for a threshold sweep without retraining -- follow-up in progress.