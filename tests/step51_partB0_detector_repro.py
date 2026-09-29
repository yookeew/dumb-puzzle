"""§51 Part B0: does the unchanged detector on the local normalised points
reproduce cache/adsb_pushback/ (the output the shipped ADS-B stages were fitted
and applied on)?

Re-runs src/link/adsb_pushback.detect_day in memory for every day in
data/external/adsb/ and compares with the cached day file, row by row on
MVT_ID_mvt. Writes nothing to cache/. Label-free.

Expected: identical for every day fetched by src/ingest/fetch_adsb.py and
normalised the same way. 2025-09-15 / 2025-11-15 were cached from the older
date-filtered Colab extracts and are now re-fetched, so they may differ; the
size of that difference is reported.

Run:  .venv/Scripts/python.exe tests/step51_partB0_detector_repro.py > logs/step51_partB0_detector_repro.log
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingest.normalise_adsb import available_days  # noqa: E402
from src.ingest.stands import load_stands, stand_positions  # noqa: E402
from src.link.adsb_pushback import detect_day  # noqa: E402

DET = ROOT / "cache" / "adsb_pushback"
FLOATS = ["adsb_pushback_ts", "adsb_takeoff_gap_s", "adsb_first_ts", "adsb_first_gs", "adsb_first_own_m",
          "adsb_min_own_m", "adsb_pb_gap_s", "adsb_pb_dist_m", "adsb_pb_gs"]


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    pos = stand_positions(load_stands())
    rows = []
    for day in available_days():
        new = detect_day(day, pos).sort("MVT_ID_mvt")
        old = pl.read_parquet(DET / f"day={day.isoformat()}.parquet").sort("MVT_ID_mvt")
        same_ids = new["MVT_ID_mvt"].equals(old["MVT_ID_mvt"])
        j = old.join(new, on="MVT_ID_mvt", how="full", suffix="_new", coalesce=True)
        tier_diff = int((j["adsb_tier"].fill_null("-") != j["adsb_tier_new"].fill_null("-")).sum())
        match_diff = int((j["adsb_matched"].fill_null(False) != j["adsb_matched_new"].fill_null(False)).sum())
        num_diff = 0
        for c in FLOATS:
            a, b = j[c], j[f"{c}_new"]
            num_diff += int(((a - b).abs() > 1e-6).fill_null(a.is_null() != b.is_null()).sum())
        rows.append(dict(day=day.isoformat(), n=old.height, same_ids=same_ids, tier_diff=tier_diff,
                         matched_diff=match_diff, float_cells_diff=num_diff,
                         identical=same_ids and tier_diff == 0 and match_diff == 0 and num_diff == 0))
        print(f"  {day}: rows {old.height:,}  tier diff {tier_diff}  matched diff {match_diff}  "
              f"float-cell diff {num_diff}", flush=True)
    r = pl.DataFrame(rows)
    with pl.Config(tbl_rows=200, tbl_formatting="ASCII_MARKDOWN"):
        print(r.filter(~pl.col("identical")))
    print(f"identical days: {r['identical'].sum()} / {r.height}")
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
