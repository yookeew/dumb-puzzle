"""Produce the portable model-stage inputs under cache/features/.

Feature building is Polars/CPU and stays local; model training moves to Colab
(GPU). This writes the three files the model stage needs — upload them to Colab:

  train2025.parquet   base features for all 2025 departures (label-free)
  ranking.parquet     base features for the ranking departures
  labels2025.parquet  MVT_ID_mvt, ADEP/RUNWAY/STAND, taxi, d, ym  (train only)

sched_takeoff_offset (the T − SOBT term used to recover taxi) already lives in
the two base-feature files.

Run:  .venv/Scripts/python.exe src/features/export_model_inputs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import build_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAIN_GLOB = str(ROOT / "data" / "raw" / "training_2025-*.parquet")
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
OUT = ROOT / "cache" / "features"

HOLDOUT_MONTHS = ("2025-01", "2025-07")

FRAME_COLS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_TYPE_flt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_mvt",
    "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "AOBT_3_flt",
]


def blind(df: pl.DataFrame) -> pl.DataFrame:
    dep = pl.col("PHASE_mvt") == "DEP"
    return df.with_columns(
        pl.when(dep).then(None).otherwise(pl.col("BLOCK_TIME_UTC_mvt")).alias("BLOCK_TIME_UTC_mvt"),
        pl.when(dep).then(None).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"),
    )


def main() -> None:
    # weather=True: family 7 (METAR) adopted under target="mixed", PROGRESS.md
    # S24 -- -1.21s, 95% CI [-2.48, -0.13], P(worse)=0.013, both months, no
    # airport regression beyond noise. Requires external-data/metar/ (fetch
    # via src/ingest/fetch_metar_data.py). atfm/queue stay off (S20/S23).
    OUT.mkdir(parents=True, exist_ok=True)

    frame = pl.read_parquet(TRAIN_GLOB, columns=FRAME_COLS)
    build_features(blind(frame), priors=None, weather=True).write_parquet(OUT / "train2025.parquet")

    labels = frame.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
        taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Int64),
        d=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
        ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    )
    labels.write_parquet(OUT / "labels2025.parquet")

    # Isolated two-month frame (just Jan+Jul 2025, no other months present) so
    # holdout features see the same Jan->Jul discontinuity ranking.parquet
    # actually has, instead of the continuous 12-month timeline train2025.parquet
    # gives them. fit.py reads this for f_ho instead of slicing train2025.
    frame_ho = frame.filter(
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").is_in(HOLDOUT_MONTHS)
    )
    build_features(blind(frame_ho), priors=None, weather=True).write_parquet(OUT / "holdout_gap2025.parquet")

    rank = pl.read_parquet(RANKING, columns=FRAME_COLS)
    build_features(blind(rank), priors=None, weather=True).write_parquet(OUT / "ranking.parquet")

    for f in ("train2025.parquet", "labels2025.parquet", "holdout_gap2025.parquet", "ranking.parquet"):
        p = OUT / f
        print(f"  {f:24} {p.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
