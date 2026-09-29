# Reproducing this project

This rebuilds our best submission, **v21** (leaderboard RMSE 270.2), from
the raw challenge data. The design is in [CLAUDE.md](CLAUDE.md), and the
experiment log, including every rejected idea, is in [PROGRESS.md](PROGRESS.md).

The pipeline has four parts:

1. Features, built locally.
2. Two base models: LightGBM (local) and CatBoost (Colab GPU).
3. A second-stage corrector on CatBoost, trained on out-of-fold predictions.
4. A stack of the two models, then ADS-B post-processing.

Everything under `cache/` is scratch space, gitignored, and rebuilt by the
steps below.

**Rough cost:**
- local CPU (tested on 22 cores / 32 GB RAM): about 3 h;
- Colab T4 GPU: about 4 h;
- the ADS-B download (Colab, network-bound): several hours for 124 days.

Unless a step says otherwise, run commands from the repo root with the uv
environment's Python: `.venv/Scripts/python.exe` on Windows,
`.venv/bin/python` elsewhere.

## 0. Environment and challenge data

1. `uv sync`: installs the pinned Python 3.11 environment from
   `pyproject.toml` / `uv.lock`.
2. Configure the S3 credentials for the challenge bucket (see
   [README.md](README.md), "Get your own OpenSky/challenge credentials").
   They're only used to download the challenge files; no OpenSky flight
   data is used.
3. `python -m src.ingest.fetch_challenge_data`: downloads the monthly
   training files to `data/raw/`, and `ranking.parquet` and
   `submitting.parquet` to `data/ranking/`.

## 1. External data

Each source, with its licence, is listed in [DATA_SOURCES.md](DATA_SOURCES.md).

1. **Stands and runway thresholds** (X-Plane Scenery Gateway, GPLv2): run
   `src/ingest/fetch_stands.py`. It writes `data/external/stands.csv` and
   `data/external/runways.csv`. Both are committed, so you can skip this
   step.
2. **METAR weather** (Iowa Environmental Mesonet, a republication of
   NOAA/NWS public-domain data): run `src/ingest/fetch_metar_data.py`. It
   writes to `external-data/metar/`. The features use it (family 7).
3. **ADS-B surface tracks** (adsb.lol, ODbL), on Colab or any machine with
   good bandwidth:

   ```
   python src/ingest/fetch_adsb.py --out external-data/adsb-fetch/ \
       --days 2025-01-01:2025-01-31 2025-07-01:2025-07-31 \
              2026-01-01:2026-01-31 2026-07-01:2026-07-31
   ```

   - Setting `GITHUB_TOKEN` avoids the GitHub API's rate limit.
   - Finished days are skipped, so an interrupted run can simply be
     restarted.
   - Copy `external-data/adsb-fetch/` back to the local repo.
4. `src/ingest/normalise_adsb.py`: writes the tracks to
   `data/external/adsb/day=*/`.
5. `src/link/adsb_pushback.py`: the stand-appearance pushback detector. It
   writes one file per day to `cache/adsb_pushback/day=*.parquet`. It reads
   only takeoff time and stand, never off-block time.

## 2. Features (local, a few minutes)

`src/features/export_model_inputs.py` writes `train2025.parquet`,
`holdout_gap2025.parquet`, `ranking.parquet` and `labels2025.parquet` to
`cache/features/`.

- The builder is split-blind. Departure off-block and taxi times are
  removed before any feature is computed, in training and ranking data
  alike.
- Labels are stored separately, keyed on `MVT_ID_mvt`.

## 3. Base models

Both base models use `target="mixed"`: the model predicts pushback delay at
LIRF and taxi time directly everywhere else. The holdout is Jan + Jul 2025,
trained on the other 10 months. After scoring, each model is refit on all of
2025 to predict the ranking set.

**LightGBM, local (about 2 h):**

```python
import sys; sys.path.insert(0, "src")
from models.fit import run
run(engine="lgb", target="mixed", seed=42, name="lgb_mixed", submit=True,
    ev_out="cache/eval/lgb_mixed_holdout_ev.parquet")
```

This writes `cache/eval/lgb_mixed_holdout_ev.parquet` and
`data/submissions/lgb_mixed.parquet`.

**CatBoost, Colab T4.** Upload `src/` and `cache/features/`, then follow the
setup in [notebooks/colab_train.py](notebooks/colab_train.py). Run these
three cells and copy the outputs back to the same paths locally:

1. Out-of-fold predictions for the corrector: 10 folds, one per training
   month, about 3 h. Each fold is saved as it finishes, so rerunning resumes.

   ```python
   run_oof(engine="cat", target="mixed", eta=0.05)   # -> cache/oof/cat_mixed/fold=*.parquet
   ```

2. The CatBoost baseline that the corrector's gate is scored against:

   ```python
   run(engine="cat", target="mixed", seed=42, eta=0.05, name="cat_mixed", submit=True,
       ev_out="cache/eval/cat_mixed_holdout_ev.parquet")
   ```

3. The CatBoost run v21 is built from. It also saves the per-row ranking
   predictions the corrector needs:

   ```python
   run(engine="cat", target="mixed", eta=0.05, name="cat_mixed_rerun", submit=True,
       ev_out="cache/eval/cat_mixed_rerun_holdout_ev.parquet",
       rank_out="cache/eval/cat_mixed_rank.parquet")
   ```

CatBoost on the GPU isn't bit-for-bit deterministic. Runs 2 and 3 differed
by 27 s RMS for us, and v21 uses run 3 throughout
(`reports/oof_corrector_v2_preregistration.md`). A fresh reproduction will
land close to v21 but not byte-identical.

## 4. OOF corrector on CatBoost (local, about 30 min)

1. `tests/oof_corrector_v2_test.py` runs the pre-registered holdout gate and
   writes the corrector's round count to `cache/oof/corrector_v2_meta.json`.
2. `src/post/corrector_v2.py` refits the corrector on all 12 months and
   applies it. It writes `cache/eval/catcorr_mixed_holdout_ev.parquet` and
   `data/submissions/catcorr_mixed.parquet`.

## 5. Stack and ADS-B stages (local, a few minutes)

`src/post/stack_submit.py --cat-corrected` does the following:

1. An NNLS stack of LightGBM and corrected CatBoost, with weights fit on the
   holdout (about 0.46 / 0.56).
2. The ADS-B pushback blend (`src/post/adsb_blend.py`), fit on the stacked
   holdout prediction.
3. The quality-modulated blend weights.
4. The partial-track estimate (`src/post/adsb_partial.py`).

The final file is
`data/submissions/stack_lgb_catcorr_adsbq_partial.parquet`. That is v21,
uploaded as `smart-jigsaw_v21.parquet` (`src/post/upload.py`).

## Earlier stages (not needed for v21)

These stages produced the reports that shaped the design:

- `src/ingest/audit_ranking.py` writes `reports/step0_audit.md` (leak and
  assumption checks on the ranking data).
- `src/link/validate_links.py` writes `reports/stage1_link_validation.md`
  (stand-linking bounds).
- `src/models/baseline.py` is the group-mean baseline (holdout RMSE
  ~400 s).
- Every adopted change has a pre-registration in
  `reports/*_preregistration.md` and a test script in `tests/`. The
  holdout harness (paired (airport, day) cluster bootstrap) is
  `tests/_harness.py`.
