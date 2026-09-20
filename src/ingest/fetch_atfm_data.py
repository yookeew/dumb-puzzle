"""Download the EUROCONTROL ATFM/traffic CSVs used by features/atfm.py.

Source: EUROCONTROL Aviation Intelligence Unit, https://ansperformance.eu.
Licence: free to copy with attribution, non-commercial use only -- see
external-data/LICENSE. Admin-approved for this challenge, Discord ruling
2026-09-17 (see DATA_SOURCES.md).

Idempotent: skips files already present, same convention as
fetch_challenge_data.py. The files currently in external-data/ were fetched
manually before this script existed; re-running it is safe and just backfills
anything missing.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://ansperformance.eu/csv"
EXTERNAL_DIR = Path(__file__).resolve().parents[2] / "external-data"

FILES = [
    "atfm_slot_adherence_2025.csv",
    "atfm_slot_adherence_2026.csv",
    "apt_dly_2025.csv.bz2",
    "apt_dly_2026.csv.bz2",
    "airport_traffic_2025.csv",
    "airport_traffic_2026.csv",
]


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = EXTERNAL_DIR / name
        if dest.exists():
            print(f"skip  {dest} (already present)")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"fetch {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print("Done.")


if __name__ == "__main__":
    main()
