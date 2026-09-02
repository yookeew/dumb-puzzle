# Data Sources

Every external dataset used in this project, its licence, and how it was
fetched. Per the challenge rules, all external data must be openly
accessible/usable and documented here.

| Name | URL | Licence | Date accessed | Derived artefact | Fetch script |
|------|-----|---------|----------------|-------------------|---------------|
| PRC/OSN challenge data (movements + flight) | `competition-data` S3 bucket via pyopensky | Challenge terms | — | `data/raw/*.parquet` (gitignored) | `src/ingest/fetch_challenge_data.py` |

Planned additions (not yet fetched):

- **NOAA ISD** (weather, public domain) — preferred over Meteostat, whose
  terms carry non-commercial clauses incompatible with the GPLv3 openness
  requirement.
- **OpenStreetMap** (airport layout / routed taxi distances, ODbL) — derived
  tables land in `data/external/` with their own `LICENSE` file and
  `© OpenStreetMap contributors` attribution, kept separate from this
  project's GPLv3 code per ODbL's share-alike scope.
