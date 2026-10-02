# Reproducing this project

This rebuilds our best submission, **v25** (leaderboard RMSE 264.39), from
the raw challenge data. The design is in [CLAUDE.md](CLAUDE.md). The
experiment log, including every rejected idea, is in
[PROGRESS.md](PROGRESS.md). Every adopted change has a pre-registration in
`reports/*_preregistration.md`.

**The pipeline:**

1. External data: stands, runways, METAR, ADS-B.
2. ADS-B pushback detection (four detector passes).
3. Features, built locally and split-blind.
4. Two base models: LightGBM (local) and CatBoost (Colab GPU).
5. A second-stage corrector on CatBoost, trained on out-of-fold predictions.
6. An NNLS stack of the two models, then the ADS-B stages → v23.
7. A learned ADS-B combiner on rows that have ADS-B information → v25.

Everything under `cache/` is scratch space, gitignored, and rebuilt by the
steps below.

**Rough cost:**

| Step | Where | Time |
|---|---|---|
| ADS-B download (126 days) | Colab or any machine | several hours, network-bound |
| ADS-B detection (4 passes) | local | ~1 h |
| LightGBM | local (tested on 22 cores / 32 GB RAM) | ~2 h |
| CatBoost (3 runs) | Colab T4 | ~4 h |
| Everything else | local | < 1 h |

Unless a step says otherwise, run commands from the repo root with the uv
environment's Python: `.venv/Scripts/python.exe` on Windows,
`.venv/bin/python` elsewhere.

## 0. Environment and challenge data

1. `uv sync` installs the pinned Python 3.11 environment from
   `pyproject.toml` / `uv.lock`.
2. Configure the S3 credentials for the challenge bucket (see
   [README.md](README.md), "Get your own OpenSky/challenge credentials").
   They're only used to download the challenge files. No OpenSky flight or
   trajectory data is used anywhere in the pipeline.
3. `python -m src.ingest.fetch_challenge_data` downloads the monthly
   training files to `data/raw/`, and `ranking.parquet` and
   `submitting.parquet` to `data/ranking/`.

## 1. External data

Each source, with its licence, is listed in [DATA_SOURCES.md](DATA_SOURCES.md).

1. **Stands and runway thresholds** (X-Plane Scenery Gateway, GPLv2):
   `src/ingest/fetch_stands.py` writes `data/external/stands.csv` and
   `data/external/runways.csv`. Both are committed, so you can skip this
   step.
2. **METAR weather** (Iowa Environmental Mesonet, republishing NOAA/NWS
   public-domain data): `src/ingest/fetch_metar_data.py` writes to
   `external-data/metar/`. Used by feature family 7.
3. **ADS-B surface tracks** (adsb.lol, ODbL), on Colab or any machine with
   good bandwidth:

   ```
   python src/ingest/fetch_adsb.py --out external-data/adsb-fetch/ \
       --days 2025-01-01:2025-01-31 2025-07-01:2025-07-31 \
              2026-01-01:2026-01-31 2026-07-01:2026-07-31 \
              2025-09-15 2025-11-15
   ```

   - Setting `GITHUB_TOKEN` avoids the GitHub API's rate limit.
   - Finished days are skipped, so an interrupted run can be restarted.
   - The Colab cells we used are in
     [notebooks/colab_train.py](notebooks/colab_train.py).
   - The two single days (2025-09-15, 2025-11-15) were used to explore
     designs outside the holdout. The stand inference in step 2 also reads
     them.
4. `src/ingest/normalise_adsb.py` writes the tracks to
   `data/external/adsb/day=*/`. It reads `external-data/adsb/`,
   `external-data/adsb-fetch/` and `external-data/adsb-restofyear/`.

## 2. ADS-B pushback detection (local, ~1 h)

`src/link/adsb_pushback.py` is split-blind. It reads only takeoff time,
stand, ADS-B points and runway geometry, never an off-block time. Run these
in order, because later passes depend on earlier outputs:

1. `src/link/adsb_pushback.py` → `cache/adsb_pushback/`, the original
   stand-appearance detector (§37). The holdout loaders read it.
2. `src/ingest/stand_infer.py` → `data/external/stands_inferred.csv`
   (committed). Positions for stands missing from the Gateway table, from
   stationary first ADS-B samples (§56; ODbL, see `data/external/LICENSE`).
3. `src/link/adsb_pushback.py --v3` → `cache/adsb_pushback_v3/`: inferred
   stands plus the fallback match for unmatched departures (§56).
4. `src/link/adsb_pushback.py --v4` → `cache/adsb_pushback_v4/`: v3 plus
   the fallback for matched rows with no at-stand pushback (§57).

The `--v5` pass (runway-ending matches, §60) was tested and rejected, so v25
doesn't need it.

## 3. Features (local, a few minutes)

`src/features/export_model_inputs.py` writes `train2025.parquet`,
`holdout_gap2025.parquet`, `ranking.parquet` and `labels2025.parquet` to
`cache/features/`.

- The builder is split-blind. Departure off-block and taxi times are
  removed before any feature is computed, in training and ranking data
  alike.
- Labels are stored separately, keyed on `MVT_ID_mvt`.

## 4. Base models

Both base models use `target="mixed"`: the model predicts pushback delay at
LIRF and taxi time directly everywhere else. The holdout is Jan + Jul 2025,
trained on the other 10 months. After scoring, each model is refit on all of
2025 to predict the ranking set.

**LightGBM, local (~2 h):**

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
   month, ~3 h. Each fold is saved as it finishes, so a rerun resumes.

   ```python
   run_oof(engine="cat", target="mixed", eta=0.05)   # -> cache/oof/cat_mixed/fold=*.parquet
   ```

2. The CatBoost baseline that the corrector's gate is scored against:

   ```python
   run(engine="cat", target="mixed", seed=42, eta=0.05, name="cat_mixed", submit=True,
       ev_out="cache/eval/cat_mixed_holdout_ev.parquet")
   ```

3. The CatBoost run used from here on. It also saves the per-row ranking
   predictions the corrector needs:

   ```python
   run(engine="cat", target="mixed", eta=0.05, name="cat_mixed_rerun", submit=True,
       ev_out="cache/eval/cat_mixed_rerun_holdout_ev.parquet",
       rank_out="cache/eval/cat_mixed_rank.parquet")
   ```

CatBoost on the GPU isn't bit-for-bit deterministic. Runs 2 and 3 differed
by 27 s RMS for us, and our submissions use run 3 throughout
(`reports/oof_corrector_v2_preregistration.md`). A fresh reproduction will
land close to our scores but won't be byte-identical. Everything
downstream of the CatBoost runs is deterministic.

The LightGBM out-of-fold folds (`cache/oof/lgb_mixed/`) are only needed
for the rejected LightGBM corrector (§51) and the combiner retrain test
(`tests/adsb_combiner_12m_test.py`):
`run_oof(engine="lgb", target="mixed", seed=42)`, ~30-50 min per fold
locally.

## 5. OOF corrector on CatBoost (local, ~30 min)

1. `tests/oof_corrector_v2_test.py` runs the pre-registered holdout gate and
   writes the corrector's round count to `cache/oof/corrector_v2_meta.json`.
2. `src/post/corrector_v2.py` refits the corrector on all 12 months and
   applies it. It writes `cache/eval/catcorr_mixed_holdout_ev.parquet` and
   `data/submissions/catcorr_mixed.parquet`.

## 6. Stack and ADS-B stages → v23 (local, a few minutes)

```
python src/post/stack_submit.py --cat-corrected --adsb-v3
```

This runs, in order:

1. An NNLS stack of LightGBM and corrected CatBoost, with weights fit on the
   holdout (~0.46 / 0.56).
2. The ADS-B pushback blend (`src/post/adsb_blend.py`) with
   quality-modulated weights.
3. The partial-track estimate (`src/post/adsb_partial.py`).
4. The gated fallback blend (`src/post/adsb_fallback.py`).

It reads `cache/adsb_pushback_v3/` and writes
`data/submissions/stack_lgb_catcorr_v3_adsbq_partial_fb.parquet`. Copy it
to `data/submissions/smart-jigsaw_v23.parquet`, because step 7 reads that
name.

Without `--adsb-v3`, the same script rebuilds v21 byte for byte
(`stack_lgb_catcorr_adsbq_partial.parquet`).

## 7. Learned ADS-B combiner → v25 (local, ~15 min)

```
python src/post/adsb_combiner.py --restricted
```

- One LightGBM is fit on `taxi − stack` over the Jan+Jul 2025 holdout. Its
  inputs are the stack prediction and every v4 detector field
  (`reports/adsb_combiner_preregistration.md`).
- It's applied to 2026 ranking rows that have ADS-B information (a matched
  track or a fallback match). Every other row, and all of LIRF, keeps v23's
  value (`reports/adsb_combiner_restricted_preregistration.md`).
- Output: `data/submissions/smart-jigsaw_v25.parquet`.
- Without `--restricted` it builds v24 (265.66), which applies the
  combiner to every non-LIRF row.

The build imports helper code from `tests/` (`ltfm_egll_anatomy.py`,
`adsb_v3_test.py`, `adsb_combiner_test.py`). That's deliberate: the
production build uses exactly the same functions as the validated gate.

Upload with `python -m src.post.upload <file>` (credentials in `.env`, see
`.env.example`).

## Leaderboard history

| version | change | RMSE |
|---|---|---|
| v12 | LightGBM, mixed target | 302 |
| v15 | + CatBoost stack | 296 |
| v16 | + ADS-B pushback blend | 277 |
| v17 | + partial-track estimate | 275 |
| v19 | + quality-modulated ADS-B weights | 274.25 |
| v21 | CatBoost replaced by the OOF-corrected CatBoost | 270.2 |
| v23 | + inferred stands and gated fallback match (v3 detections) | 269.50 |
| v24 | + learned ADS-B combiner (v4 detections) | 265.66 |
| **v25** | combiner restricted to rows with ADS-B information | **264.39** |

Rejected submissions (v18, v20, v22) are explained in PROGRESS.md §46 and
§55.

## Earlier stages (not needed for v25)

These stages produced the reports that shaped the design:

- `src/ingest/audit_ranking.py` writes `reports/step0_audit.md` (leak and
  assumption checks on the ranking data).
- `src/link/validate_links.py` writes `reports/stage1_link_validation.md`
  (stand-linking bounds).
- `src/models/baseline.py` is the group-mean baseline (holdout RMSE
  ~400 s).
- The holdout harness (paired (airport, day) cluster bootstrap) is
  `tests/_harness.py`.
