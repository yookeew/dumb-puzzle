"""Evaluation harness (CLAUDE.md Stage 5).

Holdout = Jan 2025 + Jul 2025, train on the other ten months. Never random
k-fold — neighbouring flights share congestion state.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TRAIN_GLOB = str(ROOT / "data" / "raw" / "training_2025-*.parquet")

HOLDOUT_MONTHS = ("2025-01", "2025-07")

# Physical sanity clip: applied to every prediction and used to drop poisoned
# training labels (max observed label is 131167 s ~= 36 h).
TAXI_FLOOR_SEC = 60
TAXI_CEIL_SEC = 5400


def load_dep(split: str = "all", columns: list[str] | None = None) -> pl.DataFrame:
    """Training DEP movements with a `ym` month key and Int64 `taxi` label.

    split: "all" | "train" (10 months) | "holdout" (Jan+Jul 2025).
    """
    need = {
        "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "ADEP_mvt",
    }
    if columns:
        need |= set(columns)
    lf = (
        pl.scan_parquet(TRAIN_GLOB)
        .select(sorted(need))
        .filter(pl.col("PHASE_mvt") == "DEP")
        .with_columns(
            pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("taxi"),
        )
    )
    if split == "train":
        lf = lf.filter(~pl.col("ym").is_in(HOLDOUT_MONTHS))
    elif split == "holdout":
        lf = lf.filter(pl.col("ym").is_in(HOLDOUT_MONTHS))
    elif split != "all":
        raise ValueError(split)
    return lf.collect()


def rmse(pred, true) -> float:
    e = (pl.Series(pred).cast(pl.Float64) - pl.Series(true).cast(pl.Float64))
    return float((e.pow(2).mean()) ** 0.5)


def report(df: pl.DataFrame, pred_col: str = "pred", true_col: str = "taxi") -> float:
    e = (pl.col(pred_col).cast(pl.Float64) - pl.col(true_col).cast(pl.Float64))
    overall = float(df.select(e.pow(2).mean()).item() ** 0.5)
    print(f"\nOVERALL RMSE: {overall:.2f} s   (n={df.height:,})")

    for key, label in [("ADEP_mvt", "airport"), ("ym", "month")]:
        if key not in df.columns:
            continue
        g = (
            df.group_by(key)
            .agg(e.pow(2).mean().sqrt().alias("rmse"), pl.len().alias("n"))
            .sort(key)
        )
        print(f"\nper {label}:")
        for r in g.iter_rows(named=True):
            print(f"  {str(r[key]):<10} rmse={r['rmse']:8.1f}  n={r['n']:,}")

    dec = df.with_columns(
        ((pl.col(true_col).rank("ordinal") - 1) * 10 // pl.len()).alias("dq")
    )
    g = (
        dec.group_by("dq")
        .agg(
            e.pow(2).mean().sqrt().alias("rmse"),
            pl.col(true_col).min().alias("lo"),
            pl.col(true_col).max().alias("hi"),
            pl.len().alias("n"),
        )
        .sort("dq")
    )
    print("\nper true-taxi decile:")
    for r in g.iter_rows(named=True):
        print(f"  d{r['dq']} taxi[{r['lo']:>5},{r['hi']:>5}] rmse={r['rmse']:8.1f} n={r['n']:,}")
    return overall
