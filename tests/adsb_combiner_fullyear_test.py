"""ADS-B combiner retrained with all of 2025, per reports/adsb_combiner_fullyear_preregistration.md.

Base: v26's method cross-fit (restricted combiner, other holdout month + Sep-Dec days).
Treatment: the same with Feb-Jun, Aug days added. Same day sample and thin-day rule
as tests/adsb_combiner_12m_test.py. Trimmed RMSE decides.

Run:  .venv/Scripts/python.exe tests/adsb_combiner_fullyear_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_12m_test as T  # noqa: E402
import adsb_combiner_test as C  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

SEP_DEC = ("2025-09", "2025-10", "2025-11", "2025-12")
FULL = ("2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08") + SEP_DEC


def extra(months: tuple[str, ...]) -> pl.DataFrame:
    T.MONTHS = months
    return T.oof_frame()


def main() -> None:
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)

    ex_sd, ex_full = extra(SEP_DEC), extra(FULL)
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12):
        print("days kept per month:", ex_full.group_by("ym").agg(pl.col("day").n_unique()).sort("ym"))
    base, info_b = T.combiner_12m(d4, ex_sd, v23, use_holdout=True)        # v26's method
    new, info_n = T.combiner_12m(d4, ex_full, v23, use_holdout=True)
    print("base (v26 method):", info_b)
    print("full-year:", info_n)
    V.evaluate(d4, base, new, "PRIMARY: full-year combiner vs v26 method", decisive=True)


if __name__ == "__main__":
    main()
