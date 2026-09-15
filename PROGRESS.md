# Progress log — 2026-09-15

Context: local holdout RMSE was consistently ~200s lower than the actual
ranking RMSE returned on submission (309s local vs 508s actual on
`smart-jigsaw_v5`). This session chased that gap, fixed one real bug along
the way (which turned out not to be the cause), ruled out two more
hypotheses, and landed a real (if modest) win from a robust loss.

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

## 3. Unresolved: most likely explanation for the remaining gap

Not locally verifiable without real 2026 ground truth, but the evidence
points at genuine year-over-year variance concentrated in the tail:

- **LIRF is ~7.7% of rows but ~27% of total squared error** (576s RMSE vs
  ~260s average elsewhere) — a real-world shift in Rome's operational
  regime (echo mix, disruption pattern) between 2025 and 2026 would hit the
  leaderboard RMSE disproportionately hard through the squared-error term.
- **d9 decile (taxi > 1505s) sits at ~680s RMSE** vs 200-260s everywhere
  else — however many genuine extreme-delay events actually happened in
  real Jan/Jul 2026 vs the 2025 holdout months directly drives this.
- **Adaptive overfitting to the fixed Jan+Jul 2025 holdout** from repeated
  model comparisons (v1 vs v2, per-airport residual heads, echo features)
  is also plausible — no code bug, just optimism baked in from reusing the
  same validation window for every decision this project has made so far.

Neither is fixable from here; both point toward the same next lever
(below).

## 4. Robust loss experiment: Huber on `d`

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

This 2% local gain is real but small next to the ~200s local-vs-actual
gap — it's not expected to close that gap on its own. See §3.

## Code changes

- `src/features/build_features.py` — `CFG_GAP_RESET_MIN` (24h) gap-aware
  reset in `_runway_config`'s `mins_since_cfg_change`.
- `src/features/export_model_inputs.py` — emits
  `cache/features/holdout_gap2025.parquet` (isolated Jan+Jul 2025 frame).
- `src/models/train_lgbm.py` — holdout built from the isolated frame
  instead of sliced from the continuous one. (Simpler/faster local
  sanity-check path; `fit.py` is the actual production pipeline.)
- `src/models/fit.py`:
  - Same isolated-holdout-frame fix as `train_lgbm.py`.
  - `loss` / `huber_alpha` params threaded through `run()`, `_fit_lgb`,
    `_fit_xgb` — training objective switches between L2 and Huber per
    engine; `metric` stays `rmse` always, so early stopping and reporting
    are unaffected by the objective choice.
  - New defaults: `loss="huber"`, `huber_alpha=800.0`, `ROUNDS=8000`
    (raised back from a stray `2500` that would have undertrained Huber
    badly — see huber v1 above).
  - `_write_report` records the loss/alpha choice in the eval markdown.
- `notebooks/colab_train.py` — noted the new Huber default and that it's
  worth raising `ROUNDS` further on Colab/GPU.

## Submissions this session

- `smart-jigsaw_v5` — gap-fix only (L2). 309.2s local / **508s actual**.
- `smart-jigsaw_v6` — Huber alpha=800 (`lgb_colab_huber_a800.parquet`
  renamed for upload). 303.1s local / actual score not yet confirmed as of
  writing.

## Next steps

1. Confirm `smart-jigsaw_v6`'s actual ranking RMSE — the real test of
   whether the Huber win transfers.
2. **Weather ingest (NOAA ISD, public domain, covers 2026)** — highest
   priority next feature family. It's the one family that could let the
   model see genuine year-to-year operational differences (de-icing,
   storms) that currently just show up as unexplained tail error.
3. Consider a rotating-holdout experiment (different month-pairs) to bound
   how much of any residual gap is "which two months got picked" variance
   vs something structural.
4. Push Huber `alpha=800` to more rounds on Colab/GPU — it hadn't
   plateaued at the local 8000-round ceiling.
5. OSM routed taxi distance (Future Step B) — deprioritized; it refines the
   physical baseline, not the tail, and is unlikely to move a gap this
   size. Revisit only if time allows after the above.
