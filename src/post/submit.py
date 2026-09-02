"""Write and validate a submission against submitting.parquet.

Submission rule (CLAUDE.md): fill TAXITIME_SEC_mvt in submitting.parquet, exact
MVT_ID_mvt match, no missing or extra rows. Upload as <team-name>_v<n>.parquet.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
OUT_DIR = ROOT / "data" / "submissions"


def write_submission(preds: pl.DataFrame, name: str) -> Path:
    """preds: columns MVT_ID_mvt, TAXITIME_SEC_mvt. Returns the written path."""
    template = pl.read_parquet(TEMPLATE)
    tmpl_ids = template.select("MVT_ID_mvt")

    preds = preds.select(
        pl.col("MVT_ID_mvt"),
        pl.col("TAXITIME_SEC_mvt").round().cast(pl.Int32),
    )

    merged = tmpl_ids.join(preds, on="MVT_ID_mvt", how="left")

    missing = merged.filter(pl.col("TAXITIME_SEC_mvt").is_null()).height
    if missing:
        raise ValueError(f"{missing} template rows have no prediction")
    extra = preds.join(tmpl_ids, on="MVT_ID_mvt", how="anti").height
    if extra:
        raise ValueError(f"{extra} predictions not in the template")
    if merged.height != template.height:
        raise ValueError(f"row count {merged.height} != template {template.height}")
    if (merged["TAXITIME_SEC_mvt"] < 0).any():
        raise ValueError("negative predictions")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.parquet"
    merged.write_parquet(path)
    print(
        f"wrote {path.relative_to(ROOT)}  rows={merged.height:,}  "
        f"pred: min={merged['TAXITIME_SEC_mvt'].min()} "
        f"median={merged['TAXITIME_SEC_mvt'].median()} "
        f"max={merged['TAXITIME_SEC_mvt'].max()}"
    )
    return path
