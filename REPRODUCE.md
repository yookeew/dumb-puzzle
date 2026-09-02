# Reproducing this project

1. `uv sync` — installs the pinned Python 3.11 environment from
   `pyproject.toml` / `uv.lock`.
2. Configure pyopensky credentials (see [README.md](README.md#opensky--s3-credentials)).
3. `python -m src.ingest.fetch_challenge_data` — downloads the training,
   ranking, and submitting parquet files into `data/raw/` and
   `data/ranking/` (gitignored; rebuild from source, don't commit).
4. **Step 0 audit** — `python src/ingest/audit_ranking.py` → `reports/step0_audit.md`
   (leak / assumption checks on ranking.parquet).
5. **Stage 1 stand linking** — `python src/link/validate_links.py` →
   `reports/stage1_link_validation.md`. Linker: `src/link/stand_link.py`.
6. **Trivial baseline** — `python src/models/baseline.py` → group-mean submission
   `data/submissions/baseline_v1.parquet` (holdout RMSE ~400 s).
7. **Stage 2 features + Stage 3a model (local)** —
   `python src/models/train_lgbm.py`. Builds features (cached to
   `cache/features/`), fits a global LightGBM on d = AOBT − SOBT, scores the
   Jan+Jul 2025 holdout, writes a ranking submission. ~25 min on CPU.
8. **Model training on Colab** (faster iteration) —
   `python src/features/export_model_inputs.py` writes the three portable inputs
   to `cache/features/`; upload them with `src/` and run
   `notebooks/colab_train.py` (`from models.fit import run`).

Caches under `cache/` are local-only scratch space and must always be
rebuildable by rerunning the relevant stage script against `data/`.
