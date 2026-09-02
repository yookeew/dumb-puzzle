"""Minimal end-to-end baseline — the submission safety net (CLAUDE.md NOTE).

Predict TAXITIME_SEC_mvt directly with a hierarchical group-mean and backoff:
    (airport, stand, takeoff-hour)
 -> (airport, stand)
 -> (airport, runway, takeoff-hour)
 -> (airport, runway)
 -> (airport)
 -> global

Means are computed on labels trimmed to [TAXI_FLOOR_SEC, TAXI_CEIL_SEC] to keep
the ~36 h poison rows out. Evaluated on the Jan+Jul 2025 holdout, then refit on
all of 2025 and written as a valid submission.

Run:  .venv/Scripts/python.exe src/models/baseline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.holdout import (  # noqa: E402
    TAXI_CEIL_SEC,
    TAXI_FLOOR_SEC,
    TRAIN_GLOB,
    load_dep,
    report,
)
from post.submit import write_submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"

FEAT_COLS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]

LEVELS = [
    (["ADEP_mvt", "STAND_mvt", "hh"], 50),
    (["ADEP_mvt", "STAND_mvt"], 30),
    (["ADEP_mvt", "RUNWAY_mvt", "hh"], 50),
    (["ADEP_mvt", "RUNWAY_mvt"], 30),
    (["ADEP_mvt"], 1),
]


def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hh"))


def fit(train: pl.DataFrame) -> tuple[list[tuple[list[str], pl.DataFrame]], float]:
    t = add_keys(train).filter(
        pl.col("taxi").is_between(TAXI_FLOOR_SEC, TAXI_CEIL_SEC)
    )
    tables = []
    for keys, min_n in LEVELS:
        tbl = (
            t.group_by(keys)
            .agg(pl.col("taxi").mean().alias("m"), pl.len().alias("n"))
            .filter(pl.col("n") >= min_n)
            .select(*keys, "m")
        )
        tables.append((keys, tbl))
    global_mean = float(t["taxi"].mean())
    return tables, global_mean


def predict(
    df: pl.DataFrame,
    tables: list[tuple[list[str], pl.DataFrame]],
    global_mean: float,
) -> pl.Series:
    d = add_keys(df).with_row_index("__i")
    pred = pl.Series("pred", [None] * d.height, dtype=pl.Float64)
    remaining = d
    for keys, tbl in tables:
        if remaining.height == 0:
            break
        hit = remaining.join(tbl, on=keys, how="inner").select("__i", "m")
        idx = hit["__i"].to_numpy()
        pred[idx] = hit["m"]
        remaining = remaining.join(tbl, on=keys, how="anti")
    if remaining.height:
        pred[remaining["__i"].to_numpy()] = global_mean
    return pred.clip(TAXI_FLOOR_SEC, TAXI_CEIL_SEC)


def main() -> None:
    print("=== fit on 10-month train, evaluate on Jan+Jul 2025 holdout ===")
    train = load_dep("train", columns=FEAT_COLS)
    holdout = load_dep("holdout", columns=FEAT_COLS)
    tables, gm = fit(train)
    ho = holdout.with_columns(predict(holdout, tables, gm).alias("pred"))
    report(ho.filter(pl.col("taxi").is_between(0, 4 * 3600)))

    # naive references
    from eval.holdout import rmse

    print(
        f"\nreference RMSE (global mean {gm:.0f}): "
        f"{rmse(pl.Series([gm] * holdout.height), holdout['taxi']):.1f}"
    )

    print("\n=== refit on all 2025, write submission ===")
    full = load_dep("all", columns=FEAT_COLS)
    tables, gm = fit(full)

    rank = pl.read_parquet(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", *FEAT_COLS])
    rank_dep = rank.filter(pl.col("PHASE_mvt") == "DEP")
    preds = rank_dep.select(
        "MVT_ID_mvt",
        predict(rank_dep, tables, gm).alias("TAXITIME_SEC_mvt"),
    )
    write_submission(preds, "baseline_v1")


if __name__ == "__main__":
    main()
