# Data Sources

Every external dataset used in this project, its licence, and how it was
fetched. Per the challenge rules, all external data must be openly
accessible/usable and documented here.

## Used by the final submission (v25)

| Source | Licence | Role in v25 |
|---|---|---|
| PRC/OSN challenge data | Challenge terms | training labels, all features |
| METAR (NOAA/NWS via Iowa Environmental Mesonet) | Public domain | weather features (family 7) |
| ADS-B surface tracks (adsb.lol) | ODbL 1.0 | pushback detection, partial-track estimate, combiner inputs |
| Stand and runway positions (X-Plane Scenery Gateway) | GPLv2 | own-stand coordinates for the ADS-B detector |
| Inferred stand positions (derived from adsb.lol) | ODbL 1.0 | coordinates for 50 stands missing from the Gateway table |

**Fetched and tested, but not used by v25:** the three EUROCONTROL ATFM
datasets. They were rejected in PROGRESS.md §20: the daily granularity
caps any gain at ~2 s. They stay documented because the code that reads
them is in the repo.

No OpenSky Network trajectory or state-vector data is used anywhere. The
only OpenSky-hosted data is the challenge dataset itself.

## All sources

| Name | Publisher | URL | Licence | Date accessed | Derived artefact | Fetch script |
|------|-----------|-----|---------|----------------|-------------------|---------------|
| PRC/OSN challenge data (movements + flight) | OpenSky Network / PRC | `competition-data` S3 bucket via pyopensky | Challenge terms | — | `data/raw/*.parquet` (gitignored) | `src/ingest/fetch_challenge_data.py` |
| ATFM Slot Adherence | EUROCONTROL, Aviation Intelligence Unit | https://ansperformance.eu/csv/atfm_slot_adherence_2025.csv (+ `_2026` vintage) | Free to copy with attribution, non-commercial use only. Admin-approved for this challenge via Discord ruling, 2026-09-17. | 2026-09-19 | `external-data/atfm_slot_adherence_{2025,2026}.csv` | `src/ingest/fetch_atfm_data.py` |
| Arrival ATFM Delay by Cause (apt_dly) | EUROCONTROL, Aviation Intelligence Unit | https://ansperformance.eu/csv/apt_dly_2025.csv.bz2 (+ `_2026` vintage) | Free to copy with attribution, non-commercial use only. Admin-approved for this challenge via Discord ruling, 2026-09-17. | 2026-09-19 | `external-data/apt_dly_{2025,2026}.csv.bz2` | `src/ingest/fetch_atfm_data.py` |
| Airport Traffic | EUROCONTROL, Aviation Intelligence Unit | https://ansperformance.eu/csv/airport_traffic_2025.csv (+ `_2026` vintage) | Free to copy with attribution, non-commercial use only. Admin-approved for this challenge via Discord ruling, 2026-09-17. | 2026-09-19 | `external-data/airport_traffic_{2025,2026}.csv` | `src/ingest/fetch_atfm_data.py` |
| METAR (hourly surface obs, 10 challenge departure airports) | Underlying data: NOAA/NWS (US federal government work, public domain, 17 U.S.C. §105). Archive/redistribution: Iowa Environmental Mesonet (IEM), Iowa State University. | https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py (`request/download.phtml` is the human-facing form for the same endpoint) | Underlying obs are public domain. IEM states no additional license/attribution terms on the dataset page (`info/datasets/metar.html`) beyond a general site copyright notice covering the IEM website itself, not the data; no account/API key required. Data is "as-is", not the source of record (IEM recommends NCEI ISD for that) — used here as a free, easy-to-query mirror. | 2026-09-20 | `external-data/metar/{STATION}_{window}.csv` | `src/ingest/fetch_metar_data.py` |
| ADS-B/MLAT surface tracks (readsb `trace_full` per aircraft per day) | adsb.lol (community feeder network) | https://github.com/adsblol/globe_history_2025 and `globe_history_2026` — GitHub release per day, split tar; replica chosen via `PREFERRED_RELEASES.txt` in each repo root, fallbacks `prod-0`/`staging-0`/`prod-0tmp`/`staging-0tmp`. Tag used per day recorded in `manifest.csv` next to the extracts. | Database: ODbL 1.0 (`LICENSE-ODbL.txt` shipped inside each release); feeder contributions CC0. Derived tables must carry ODbL + attribution "adsb.lol contributors". | 2026-09-26 (first days pulled 2026-09-23) | Raw extracts `external-data/adsb/adsb_YYYYMMDD.parquet` (points within ±0.10° lat / ±0.15° lon of each aerodrome reference point; gitignored) → normalised `data/external/adsb/day=YYYY-MM-DD/` (gitignored) → per-departure pushback `cache/adsb_pushback/` | `src/ingest/fetch_adsb.py` → `src/ingest/normalise_adsb.py` → `src/link/adsb_pushback.py` |
| Stand/gate positions (apt.dat row codes 1300/1301) and runway ends (row code 100) | X-Plane Scenery Gateway (Laminar Research; community-contributed airport sceneries) | https://gateway.x-plane.com — public API via the `xplane_airports` package (MIT), recommended scenery pack per airport, no account | GNU GPL v2 — each downloaded pack ships a `COPYING` file with the GPLv2 text; copy kept at `data/external/stands_LICENSE_GPLv2.txt`. Scenery IDs and approval dates recorded per row in the CSV. | 2026-09-26 | `data/external/stands.csv`, `data/external/runways.csv` (committed); raw apt.dat per airport in `cache/stands/` (gitignored) | `src/ingest/fetch_stands.py` |
| Inferred stand positions (stands missing from the Gateway table) | Derived from adsb.lol (row above) + challenge STAND_mvt | — (computed locally) | ODbL 1.0, derived database of adsb.lol; attribution "adsb.lol contributors" | 2026-10-02 | `data/external/stands_inferred.csv` (50 stands; median of matched runs' stationary first samples, n >= 3, spread <= 30 m; LFPG excluded) | `src/ingest/stand_infer.py` |

## Licence files

- `LICENSE` (repo root): GNU GPL v3, covering this project's source code.
- `external-data/LICENSE`: the EUROCONTROL data terms (non-commercial,
  attribution). It covers only those data files, not the code.
- `data/external/LICENSE`: the committed derived tables.
  `stands.csv` and `runways.csv` are under GPLv2 (Scenery Gateway), with
  the full text in `data/external/stands_LICENSE_GPLv2.txt`.
  `stands_inferred.csv` is under ODbL 1.0, a derived database of adsb.lol,
  attribution "adsb.lol contributors".
- Raw adsb.lol extracts (`external-data/adsb*/`) and the normalised store
  (`data/external/adsb/`) are not committed. They are rebuilt by the fetch
  scripts and stay under ODbL 1.0.

## Not used

- `external-data/taxi_in_additional_time_2026.csv` is present locally but
  isn't read by any code. Its provenance and licence weren't confirmed, so
  it isn't part of any submission.
- **OpenStreetMap:** planned for routed taxi distances (CLAUDE.md "Future
  step B") but never built. Stand/runway geometry from the Gateway table
  added nothing as a feature (PROGRESS.md §38), so no OSM-derived data
  exists in this repo.
