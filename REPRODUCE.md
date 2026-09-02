# Reproducing this project

1. `uv sync` — installs the pinned Python 3.11 environment from
   `pyproject.toml` / `uv.lock`.
2. Configure pyopensky credentials (see [README.md](README.md#opensky--s3-credentials)).
3. `python -m src.ingest.fetch_challenge_data` — downloads the training,
   ranking, and submitting parquet files into `data/raw/` and
   `data/ranking/` (gitignored; rebuild from source, don't commit).
4. *(stages 1+ to be filled in as they're built — stand linking, feature
   build, model training, constraint post-processing, submission.)*

Caches under `cache/` are local-only scratch space and must always be
rebuildable by rerunning the relevant stage script against `data/`.
