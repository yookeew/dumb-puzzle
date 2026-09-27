"""LIRF "+24 h" label adjustment (reports/lirf_dayplus_preregistration.md, PROGRESS.md §44).

The LIRF block-time feed sometimes stamps the real pushback clock time onto
the scheduled date, so for flights taking off many hours late the label becomes
real taxi + 86,400 s. On segment rows (LIRF, NM-unmatched, T - SOBT >= 12 h):
    new = (1 - p) * pred + p * (86,400 + t0)
p = +24 h share among segment rows and t0 = median normal LIRF NM-unmatched
taxi, both from the 10 training months (2025 minus Jan and Jul).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HOLDOUT = ("2025-01", "2025-07")
MIN_OFFSET_S = 12 * 3600
DAYPLUS = (86400 - 600, 86400 + 5400)


def fit() -> dict:
    r = (pl.scan_parquet(str(RAW / "training_*.parquet"))
         .filter(pl.col("PHASE_mvt") == "DEP", pl.col("ADEP_mvt") == "LIRF",
                 pl.col("AOBT_3_flt").is_null())
         .select("SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt").collect()
         .with_columns(ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
                       off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
         .filter(~pl.col("ym").is_in(HOLDOUT)))
    seg = r.filter(pl.col("off") >= MIN_OFFSET_S)
    p = float(seg["TAXITIME_SEC_mvt"].is_between(*DAYPLUS).mean())
    t0 = float(r.filter(pl.col("TAXITIME_SEC_mvt") <= 7200)["TAXITIME_SEC_mvt"].median())
    return {"p": p, "t0": t0, "n_seg": seg.height}


def segment() -> pl.Expr:
    """Needs ADEP_mvt, nm_unmatched (bool), offset_s (T - SOBT, seconds)."""
    return ((pl.col("ADEP_mvt") == "LIRF") & pl.col("nm_unmatched")
            & (pl.col("offset_s") >= MIN_OFFSET_S)).fill_null(False)


def apply(df: pl.DataFrame, col: str, prm: dict) -> pl.Expr:
    return (pl.when(segment()).then((1 - prm["p"]) * pl.col(col) + prm["p"] * (86400 + prm["t0"]))
            .otherwise(pl.col(col)))
