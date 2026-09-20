"""Download hourly METAR observations for the 10 challenge departure airports
(LTAI/Antalya excluded -- zero departures in train or ranking, see CLAUDE.md)
from the Iowa Environmental Mesonet (IEM) ASOS/METAR archive.

Source: Iowa State University, Iowa Environmental Mesonet.
  https://mesonet.agron.iastate.edu/request/download.phtml
IEM re-publishes NOAA/NWS METAR data (public domain); IEM's own archive is
free to use, no account or API key required -- see external-data/LICENSE for
the note on this specific endpoint.

Fetches report_type=3 (routine/hourly METAR only, not SPECI) for full-year
2025 plus January and July 2026, matching the training and ranking windows.
Columns: temperature/dewpoint/humidity (de-icing signal), wind, visibility,
up to 3 sky-cover/ceiling layers, present-weather codes (snow/thunderstorm),
and 1h precipitation.

Idempotent: skips files already present, same convention as
fetch_atfm_data.py / fetch_challenge_data.py.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
EXTERNAL_DIR = Path(__file__).resolve().parents[2] / "external-data" / "metar"

# 10 challenge airports with departures (excludes LTAI -- see CLAUDE.md note).
STATIONS = [
    "EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
    "LEMD", "LFPG", "LIRF", "LTFM", "LSZH",
]

DATA_FIELDS = [
    "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby",
    "skyc1", "skyl1", "skyc2", "skyl2", "skyc3", "skyl3",
    "wxcodes", "p01i",
]

# (label, year1, month1, day1, year2, month2, day2) -- half-open [start, end)
WINDOWS = [
    ("2025_2026-01", 2025, 1, 1, 2026, 2, 1),   # full 2025 + Jan 2026
    ("2026-07", 2026, 7, 1, 2026, 8, 1),        # Jul 2026
]


def _url(station: str, y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> str:
    params = {
        "station": station,
        "data": ",".join(DATA_FIELDS),
        "year1": y1, "month1": m1, "day1": d1,
        "year2": y2, "month2": m2, "day2": d2,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",
    }
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for station in STATIONS:
        for label, y1, m1, d1, y2, m2, d2 in WINDOWS:
            dest = EXTERNAL_DIR / f"{station}_{label}.csv"
            if dest.exists():
                print(f"skip  {dest} (already present)")
                continue
            url = _url(station, y1, m1, d1, y2, m2, d2)
            print(f"fetch {station} {label} -> {dest}")
            urllib.request.urlretrieve(url, dest)
    print("Done.")


if __name__ == "__main__":
    main()
