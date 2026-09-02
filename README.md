# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predicting `TAXITIME_SEC_mvt` for departures at 11 major European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTAI, LTFM, LSZH) for the
2026 PRC / OpenSky Network Data Challenge. Full design and rationale live in
[CLAUDE.md](CLAUDE.md).

## Setup (Windows)

### 1. Install uv

```powershell
winget install astral-sh.uv
```

### 2. Clone the repo and sync the environment

```powershell
git clone <this repo>
cd prc2025
uv sync
.\.venv\Scripts\Activate.ps1
```

`uv sync` reads `pyproject.toml` (which pins `requires-python = "==3.11.*"`)
and `uv.lock`. **You do not need to separately install Python 3.11** — if uv
doesn't find a matching interpreter on your system, it downloads one
automatically and creates `.venv` from it. This also installs every
dependency (pyopensky, traffic, polars, pyarrow, lightgbm, scikit-learn,
networkx, jupyter) at the exact locked versions.

### 3. Get your own OpenSky/challenge credentials

You need your own OSN account login and, separately, the MinIO/S3
`access_key` + `secret_key` for the `competition-data` bucket (from the
challenge organizers — this is the same thing as "S3 credentials", MinIO is
just the S3-compatible server OpenSky runs). These are per-person, not
shared — everyone on the team needs their own.

### 4. Create/edit the pyopensky config file

This file lives **outside the repo**, per-machine, and must never be
committed. Find (and auto-create) it by importing the package once:

```powershell
python -c "from pyopensky.config import opensky_config_dir; print(opensky_config_dir)"
```

On Windows this is `%LOCALAPPDATA%\pyopensky\pyopensky\settings.conf`. Open
it — the first import already generated a template — and fill it in. Note
`access_key`/`secret_key` go under a separate `[s3]` section, which starts
commented out; **uncomment it**:

```ini
[default]
username = your_osn_user
password = ...
client_id =
client_secret =

[s3]
access_key = ACCESS_KEY
secret_key = SECRET_KEY
```

(`client_id`/`client_secret` are for the REST API, unused here — fine to
leave blank.)

### 5. Download the challenge data

```powershell
python -m src.ingest.fetch_challenge_data
```

Downloads every object in the `competition-data` bucket, routed into
`data/raw/` (monthly training files) and `data/ranking/` (`ranking.parquet`,
`submitting.parquet`). Both are gitignored — this step is per-machine, not
committed. Safe to re-run: it skips files already present.