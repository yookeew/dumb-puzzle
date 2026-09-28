# Handoff — PRC 2026 taxi-out challenge (as of 2026-09-28)

You are picking up a competition entry: predict departure taxi-out time
(`TAXITIME_SEC_mvt`) at 10 European airports, scored by RMSE on Jan + Jul 2026.
Read `CLAUDE.md` (problem, rules, architecture) and `PROGRESS.md` §37–§47 (this
week's work) before changing anything. **The freeze is 2026-10-04.**

## Where things stand

- **Branch `adsbround2`**, in sync with origin at `074deb1`.
- **Best leaderboard score: v21 = 270.2** (OOF-corrected CatBoost, PROGRESS.md §49; was v19 = 274.25). The top team is at ~224.
- **v19 is built by `src/post/stack_submit.py`** (output
  `data/submissions/stack_lgb_cat_adsbq_partial.parquet`; a rebuild is
  byte-identical to the uploaded v19). Its stages:
  1. **LightGBM + CatBoost**, each `run(engine=..., target="mixed")` in
     `src/models/fit.py`, combined by a cross-fit NNLS stack (weights ~0.54 /
     0.47).
  2. **ADS-B pushback blend** (`src/post/adsb_blend.py`). adsb.lol tracks
     plus X-Plane Gateway stand coordinates give an observed pushback for
     ~27% of ranking rows; it nudges the prediction per tier, with
     per-airport lags and quality factors (§39, §43).
  3. **Partial-track estimate** (`src/post/adsb_partial.py`) for tracks
     first seen < 1 km from the stand (§41).
- **Leaderboard history:** 302 → v15 296 (stack) → v16 277 (ADS-B) → v17 275
  (partial) → v19 274.25 (quality weights). Rejected: v18 276 (EGLL lag
  shift), v20 278 (LIRF "+24 h" rule, §44/§46).

## How we work (non-negotiable; each rule exists because breaking it cost us)

- **Validation:** the Jan + Jul 2025 holdout, cross-fit (fit on Jan → score
  Jul, and the reverse). Never random or day-level folds.
- **Pre-register every test** in `reports/<name>_preregistration.md` and
  commit it before computing any holdout result. Changes after results go
  under a dated "Amendments" heading.
- **Decision metric:** the adoption rule states the metric up front.
  - **Trimmed RMSE decides:** holdout labels > 5 h excluded, a fixed set of
    31 rows.
  - **Full RMSE is a guard:** it must not be significantly worse
    (P(worse) < 0.9).
  - Use the paired (airport, day) cluster bootstrap in `tests/_harness.py`.
  - Why: a handful of monster labels (e.g. two LFPG rows at 84,240 s and
    58,206 s) dominate full RMSE. v20 gained −4.0 on the holdout from 11
    rows and lost +3.75 on the leaderboard.
- **Holdout gains ≠ leaderboard gains.** Direction usually agrees, size only
  roughly. ADS-B stages scored ~2–4× their holdout gain; label rules
  learned on 2025 did not carry over.
- **Never feed a departure's off-block time into features** (the leak
  invariant in `CLAUDE.md`).
- Record every experiment in `PROGRESS.md` as a new numbered section.
  Commit with the attribution line used in the git log.

## Task 1 (top priority): out-of-fold (OOF) corrector

**Why.** Two teams in the competitors' chat (including the leader) said a
corrector trained on **month-wise OOF base predictions** did more than any
new feature (PROGRESS.md §46). We don't have one.

**Idea.** A second-stage model learns the base model's systematic errors
from `features + base prediction → residual (taxi − base)`, trained on
predictions the base model made for months it never saw.

**Scope for the time left.** Start with **CatBoost as the base**: ~6
min/fit on a Colab T4 with `max_ctr_complexity=1` (already the default in
`_fit_cat`). LightGBM OOF would be ~20 min/fit × 2 targets × 10 months
locally, too slow. If the CatBoost-only experiment clearly helps, decide
how to integrate it with the lgb+cat stack (e.g. correct the stack using
the CatBoost-based residual model, or refit the stack on corrected CatBoost
predictions). Pre-register that choice separately.

**Steps.**

1. **Refactor `run()`** (`src/models/fit.py`). It hard-codes
   `HOLDOUT_MONTHS = ("2025-01", "2025-07")` and `VALID_MONTH = "2025-06"`.
   Write a function that trains on an arbitrary set of months and predicts
   an arbitrary month, returning the same per-row reconstruction as
   `run()`: the mixed flip/direct target, the echo classifier, the
   `_reconstruct_taxi` echo blend, and the `use_prior` fallback.
2. **Avoid leakage inside each OOF fit.** For held-out month m:
   - `fit_priors` and the group encodings (`fit_group_encodings`,
     `add_group_encodings_oof` in `src/features/encode.py`) must be fit
     **without month m**.
   - The early-stopping validation month must not be m. Use June; when m
     is June, use May.
   - The holdout months (Jan, Jul 2025) must never enter any OOF training
     fit.
3. **Generate OOF predictions** for the 10 training months (Feb–Jun,
   Aug–Dec 2025): 10 fits per target, run on Colab. Save per-row OOF
   predictions and echo probabilities to parquet as each fold finishes
   (Colab sessions drop). Use the `ev_out`-style early save;
   `PRC_MEMLOG=1` prints memory checkpoints.
4. **Train the corrector** on training-month rows:
   - Target: residual = taxi − OOF base prediction.
   - Inputs: the base features + the OOF prediction + the echo probability.
   - Model: a small, strongly regularised tree model (shallow, high
     min-leaf, Huber or L2). Early-stop on one training month.
5. **Evaluate on the holdout.** Apply the corrector to CatBoost's existing
   holdout predictions (`cache/eval/cat_mixed_holdout_ev.parquet`, trained
   on the 10 training months, so it's the same kind of prediction).
   - Compare corrected vs uncorrected CatBoost.
   - Report trimmed + full RMSE, per month, per airport, with the
     bootstrap.
6. **For ranking:** the corrector is refit on all 12 months' OOF-style
   predictions (training months' OOF + the holdout predictions) and
   applied to the all-2025 CatBoost submission's predictions. Then rerun
   the ADS-B stages on top via `src/post/stack_submit.py`.

**Pitfalls.**
- Row-order mismatches when joining OOF predictions back: always join on
  `MVT_ID_mvt`.
- An OOF fold that silently used the held-out month in priors or encodings
  will look great and fail on the leaderboard.
- Check the corrector on **trimmed** RMSE. A gain made of a few monster
  rows is noise.

## Task 2: carrier echo rate for NM-unmatched rows (cheap)

The chat reported that "echo of the planned time in the off-block is mostly
a carrier thing when there's no NM match, and stable month to month". Our
echo classifier's recall is ~9% (`fit.py` logs `echo clf holdout
precision/recall`). The existing group encodings (`op_*` in
`src/features/encode.py`) aren't conditioned on NM-unmatched rows
(`aobt3_taxi` null).

Add a per-(airport, operator) echo rate **among NM-unmatched rows**:
- OOF by month for training rows, and fit on training months for the
  holdout, exactly like `add_group_encodings_oof`.
- Echo = `|d| < 30 s` (`_ECHO_ABS_D`).
- Test it as a feature (pre-registered, trimmed metric), or as a corrector
  input in Task 1.

## Task 3: freeze deliverables (start by 2026-10-02)

- **`REPRODUCE.md`**, covering the end-to-end path:
  - challenge data fetch
  - `src/ingest/fetch_stands.py`
  - the ADS-B fetch on Colab (`src/ingest/fetch_adsb.py`, see §39/§40 for
    the cells)
  - `src/ingest/normalise_adsb.py` → `src/link/adsb_pushback.py`
  - `src/features/export_model_inputs.py`
  - the lgb run (local) and the cat run (Colab, `eta=0.05`)
  - `src/post/stack_submit.py`
- **`DATA_SOURCES.md` check:** every external source used by the final
  submission has a row. Currently: Gateway stands/runways (GPLv2),
  adsb.lol (ODbL), METAR, EUROCONTROL.
- **Make the repo public under GPLv3**, as the challenge rules require.
- **JOAS paper draft.** Framing in `CLAUDE.md` (post-ops reconstruction),
  plus the ADS-B stand-appearance detector and the label-recording
  findings (schedule echoes, LIRF "+24 h" bug, LFPG monster labels) as
  contributions.

## Already tried and rejected (don't repeat without a new reason)

| idea | where | result |
|---|---|---|
| OSM / straight-line stand→runway geometry | §38 | worse (+0.59) |
| LIRF echo-aware ADS-B mixture | §42 | nil |
| EGLL lag re-anchoring from AOBT_3 | v18 | 276 (worse) |
| LIRF "+24 h" label rule | §44/§46, v20 | 278 (worse) |
| CatBoost 20k rounds | §47 | not adopted: trimmed −0.37 but full noise |
| XGBoost | — | out of memory on free Colab |
| Per-airport residual heads, queue refinement, ATFM | earlier §§ | see PROGRESS.md |

## Practical notes

- **Colab:** use a T4 GPU runtime.
  - The setup cell pulls from GitHub; then check out `adsbround2`.
  - Feature parquets (`cache/features/*.parquet`) and
    `data/ranking/submitting.parquet` are gitignored and must be uploaded
    to Drive.
  - `run()` writes submissions to `data/submissions/` and holdout
    predictions to the `ev_out` path.
- **Leaderboard:** about one submission per day. Upload as
  `smart-jigsaw_v<n>.parquet` (`src/post/upload.py`).