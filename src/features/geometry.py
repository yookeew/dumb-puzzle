"""Family 8: stand/runway geometry from X-Plane Gateway apt.dat.

Per DEP row, from STAND_mvt and RUNWAY_mvt (both unblanked at ranking time):

  stand_e_m, stand_n_m   stand position, metres east/north of the airport's
                         mean stand position (lets trees localise piers and
                         remote aprons across stands the categorical splits
                         rarely see)
  dist_stand_rwy_m       straight-line distance stand -> threshold of the
                         departure runway end (where the takeoff roll starts)
  rwy_e_m, rwy_n_m       that threshold, same frame

Straight-line distance is the cheap stand-in for the routed OSM taxi distance
(CLAUDE.md "Future step B"). Nulls where the stand name does not resolve
(see src/ingest/stands.py) or the runway is unknown (NA, helipad "H").
Inputs: data/external/stands.csv, data/external/runways.csv
(src/ingest/fetch_stands.py; GPLv2, DATA_SOURCES.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ingest.stands import attach_stand_coords, load_stands, stand_positions

ROOT = Path(__file__).resolve().parents[2]
RUNWAYS = ROOT / "data" / "external" / "runways.csv"
GEOM_COLS = ["stand_e_m", "stand_n_m", "rwy_e_m", "rwy_n_m", "dist_stand_rwy_m"]


def geometry_context(feats: pl.DataFrame) -> pl.DataFrame:
    """MVT_ID_mvt + GEOM_COLS for the DEP rows of `feats` (needs ADEP_mvt, STAND_mvt, RUNWAY_mvt)."""
    stands = load_stands()
    origin = stands.group_by("airport").agg(lat0=pl.col("lat").mean(), lon0=pl.col("lon").mean())
    rwy = pl.read_csv(RUNWAYS).select(
        "airport", pl.col("runway").cast(pl.Utf8), rwy_lat="lat", rwy_lon="lon")

    d = attach_stand_coords(feats.select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"),
                            stand_positions(stands))
    d = (d.join(origin, left_on="ADEP_mvt", right_on="airport", how="left")
         .join(rwy, left_on=["ADEP_mvt", "RUNWAY_mvt"], right_on=["airport", "runway"], how="left"))
    kx = 111320.0 * (pl.col("lat0") * np.pi / 180).cos()
    ky = 110540.0
    d = d.with_columns(
        stand_e_m=(pl.col("stand_lon") - pl.col("lon0")) * kx,
        stand_n_m=(pl.col("stand_lat") - pl.col("lat0")) * ky,
        rwy_e_m=(pl.col("rwy_lon") - pl.col("lon0")) * kx,
        rwy_n_m=(pl.col("rwy_lat") - pl.col("lat0")) * ky,
    ).with_columns(
        dist_stand_rwy_m=((pl.col("stand_e_m") - pl.col("rwy_e_m")).pow(2)
                          + (pl.col("stand_n_m") - pl.col("rwy_n_m")).pow(2)).sqrt(),
    )
    return d.select("MVT_ID_mvt", *[pl.col(c).cast(pl.Float32) for c in GEOM_COLS])
