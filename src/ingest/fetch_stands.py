"""Fetch stand/gate positions from the X-Plane Scenery Gateway.

For each challenge airport, download the Gateway's recommended scenery pack
(public API via the `xplane_airports` package, no account) and parse apt.dat
row codes:

  1300  lat lon heading type aircraft_classes name...
        type: gate | tie_down | hangar | misc
        aircraft_classes: pipe-separated subset of heavy|jets|turboprops|props|helos|fighters|all
  1301  width_code operation_type [airline codes...]   (ramp detail, follows its 1300)
        width_code: A..F (ICAO aerodrome reference code letter)
        operation_type: none | general_aviation | airline | cargo | military

  100   land runway: ... then per end [name lat lon displaced_m overrun_m ...]
        (end lat/lon = that designator's threshold, where its takeoff roll starts)

Writes data/external/stands.csv (one row per 1300 line) and
data/external/runways.csv (one row per runway end), and caches the raw
apt.dat text per airport under cache/stands/ for reproducibility.
`--from-cache` rebuilds runways.csv from cache/stands/ without the network.

Gateway scenery packs are distributed under the GNU GPL v2 (each pack ships a
COPYING file); see DATA_SOURCES.md.

Run:  .venv/Scripts/python.exe src/ingest/fetch_stands.py [--from-cache]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from xplane_airports import gateway

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "stands.csv"
OUT_RWY = ROOT / "data" / "external" / "runways.csv"
RAW = ROOT / "cache" / "stands"
LICENSE = ROOT / "data" / "external" / "stands_LICENSE_GPLv2.txt"  # COPYING shipped in each pack

# LTAI has no departures in train or ranking (CLAUDE.md), so it is skipped.
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
            "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]

FIELDS = ["airport", "stand_name", "lat", "lon", "heading", "stand_type",
          "aircraft_classes", "width_code", "operation_type", "airlines",
          "gateway_scenery_id", "gateway_date_approved"]


def parse_stands(lines: list[str]) -> list[dict]:
    rows: list[dict] = []
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "1300" and len(tok) >= 6:
            rows.append({
                "lat": float(tok[1]), "lon": float(tok[2]),
                "heading": float(tok[3]), "stand_type": tok[4],
                "aircraft_classes": tok[5],
                "stand_name": " ".join(tok[6:]),
                "width_code": "", "operation_type": "", "airlines": "",
            })
        elif tok[0] == "1301" and rows and len(tok) >= 3:
            rows[-1]["width_code"] = tok[1]
            rows[-1]["operation_type"] = tok[2]
            rows[-1]["airlines"] = " ".join(tok[3:])
    return rows


RWY_FIELDS = ["airport", "runway", "lat", "lon", "displaced_m", "width_m", "opposite"]


def parse_runways(lines: list[str]) -> list[dict]:
    rows: list[dict] = []
    for line in lines:
        tok = line.split()
        if len(tok) >= 20 and tok[0] == "100":
            ends = [(tok[8], tok[9], tok[10], tok[11]), (tok[17], tok[18], tok[19], tok[20])]
            for k, (name, lat, lon, disp) in enumerate(ends):
                rows.append({"runway": name, "lat": float(lat), "lon": float(lon),
                             "displaced_m": float(disp), "width_m": float(tok[1]),
                             "opposite": ends[1 - k][0]})
    return rows


def write_runways(texts: dict[str, str]) -> int:
    rows = []
    for icao, text in texts.items():
        for r in parse_runways(text.splitlines()):
            r["airport"] = icao
            rows.append(r)
    with OUT_RWY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RWY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    texts: dict[str, str] = {}
    if "--from-cache" in sys.argv:
        texts = {icao: (RAW / f"{icao}.dat").read_text(encoding="utf-8") for icao in AIRPORTS}
        print(f"wrote {write_runways(texts)} runway ends -> {OUT_RWY.relative_to(ROOT)} (from cache)")
        return
    for icao in AIRPORTS:
        pack = gateway.scenery_pack(icao)
        text = "\n".join(pack.apt.raw_lines) if pack.apt.raw_lines else str(pack.apt.text)
        (RAW / f"{icao}.dat").write_text(text, encoding="utf-8")
        texts[icao] = text
        if pack.copying:
            LICENSE.write_text(pack.copying, encoding="utf-8")
        meta = pack.pack_metadata
        stands = parse_stands(text.splitlines())
        for s in stands:
            s.update(airport=icao, gateway_scenery_id=meta.get("sceneryId"),
                     gateway_date_approved=meta.get("dateApproved"))
        out_rows.extend(stands)
        print(f"{icao}: scenery {meta.get('sceneryId')} ({meta.get('type')}, "
              f"approved {meta.get('dateApproved')}) -> {len(stands)} stands")
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {OUT.relative_to(ROOT)}")
    print(f"wrote {write_runways(texts)} runway ends -> {OUT_RWY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
