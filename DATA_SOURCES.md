# Data Sources

Every external dataset used in this project, its licence, and how it was
fetched. Per the challenge rules, all external data must be openly
accessible/usable and documented here.

| Name | Publisher | URL | Licence | Date accessed | Derived artefact | Fetch script |
|------|-----------|-----|---------|----------------|-------------------|---------------|
| PRC/OSN challenge data (movements + flight) | OpenSky Network / PRC | `competition-data` S3 bucket via pyopensky | Challenge terms | — | `data/raw/*.parquet` (gitignored) | `src/ingest/fetch_challenge_data.py` |
| ATFM Slot Adherence | EUROCONTROL, Aviation Intelligence Unit | https://ansperformance.eu/csv/atfm_slot_adherence_2025.csv (+ `_2026` vintage) | Free to copy with attribution, non-commercial use only. Admin-approved for this challenge via Discord ruling, 2026-09-17. | 2026-09-19 | `external-data/atfm_slot_adherence_{2025,2026}.csv` | `src/ingest/fetch_atfm_data.py` |
| Arrival ATFM Delay by Cause (apt_dly) | EUROCONTROL, Aviation Intelligence Unit | https://ansperformance.eu/csv/apt_dly_2025.csv.bz2 (+ `_2026` vintage) | Free to copy with attribution, non-commercial use only. Admin-approved for this challenge via Discord ruling, 2026-09-17. | 2026-09-19 | `external-data/apt_dly_{2025,2026}.csv.bz2` | `src/ingest/fetch_atfm_data.py` |
| Airport Traffic | EUROCONTROL, Aviation Intelligence Unit | https://ansperformance.eu/csv/airport_traffic_2025.csv (+ `_2026` vintage) | Free to copy with attribution, non-commercial use only. Admin-approved for this challenge via Discord ruling, 2026-09-17. | 2026-09-19 | `external-data/airport_traffic_{2025,2026}.csv` | `src/ingest/fetch_atfm_data.py` |
| METAR (hourly surface obs, 10 challenge departure airports) | Underlying data: NOAA/NWS (US federal government work, public domain, 17 U.S.C. §105). Archive/redistribution: Iowa Environmental Mesonet (IEM), Iowa State University. | https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py (`request/download.phtml` is the human-facing form for the same endpoint) | Underlying obs are public domain. IEM states no additional license/attribution terms on the dataset page (`info/datasets/metar.html`) beyond a general site copyright notice covering the IEM website itself, not the data; no account/API key required. Data is "as-is", not the source of record (IEM recommends NCEI ISD for that) — used here as a free, easy-to-query mirror. | 2026-09-20 | `external-data/metar/{STATION}_{window}.csv` | `src/ingest/fetch_metar_data.py` |

Full licence text for the three EUROCONTROL datasets above lives in
`external-data/LICENSE` (kept separate from this project's own code
licence — see that file for why). `taxi_in_additional_time_2026.csv` is
also present in `external-data/` but is not yet accounted for here: its
provenance/licence hasn't been confirmed, so it isn't used by any code yet.
- **OpenStreetMap** (airport layout / routed taxi distances, ODbL) — derived
  tables land in `data/external/` with their own `LICENSE` file and
  `© OpenStreetMap contributors` attribution, kept separate from this
  project's GPLv3 code per ODbL's share-alike scope.
