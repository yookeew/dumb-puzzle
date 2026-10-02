"""ADS-B v5 (runway-ending matches) through the restricted combiner, per
reports/adsb_v5_preregistration.md.

Base: v25's method cross-fit = v24 combiner on v4 detections, applied only to rows
with ADS-B information, v23 pipeline value elsewhere. Treatment: the same on v5
detections with the `rwy_end` flag as an extra input. Trimmed RMSE decides.

Run:  .venv/Scripts/python.exe tests/adsb_v5_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

JAN, JUL = A.JAN, A.JUL
V5 = ROOT / "cache" / "adsb_pushback_v5"


def read_det_v5() -> pl.DataFrame:
    det = pl.concat([pl.read_parquet(p) for p in sorted(V5.glob("day=2025-0[17]*.parquet"))])
    return det.select("MVT_ID_mvt", *V.DET_COLS, "adsb_fallback", "adsb_rwy_end")


def features_v5(df: pl.DataFrame, s: np.ndarray, airports: list[str]) -> np.ndarray:
    """v24 inputs plus `rwy_end`, appended last (categorical indices unchanged)."""
    X = C.features(df, s, airports).to_numpy().astype(np.float64)
    flag = df["adsb_rwy_end"].fill_null(False).cast(pl.Float64).to_numpy()
    return np.column_stack([X, flag])


def combiner_v5(df: pl.DataFrame, base: np.ndarray) -> tuple[np.ndarray, dict]:
    """C.combiner with the extra input, applied only to rows with ADS-B information."""
    airports = sorted(df["ADEP_mvt"].unique().to_list())
    out, info = base.copy(), {}
    non_lirf = (df["ADEP_mvt"] != "LIRF").to_numpy()
    pop = df.select(C2.has_adsb()).to_series().to_numpy()
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = C.stack(df, fm)
        X = features_v5(df, s, airports)
        fit_rows = ((df["ym"] == fm) & (df["taxi"] <= C.TRIM)).to_numpy() & non_lirf
        y = df["taxi"].to_numpy() - s
        inner = (df["day"].dt.day() % 5 == 0).to_numpy()[fit_rows]
        m, best = C.fit_combiner(X[fit_rows], y[fit_rows], inner)
        app = (df["ym"] == am).to_numpy() & non_lirf & pop
        out[app] = s[app] + m.predict(X[app])
        info[f"fit {fm}"] = {"best_iter": best}
    return out, info


def main() -> None:
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    d5 = V.with_det(df, read_det_v5())
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    v24, info24 = C.combiner(d4, v23)
    pop4 = d4.select(C2.has_adsb()).to_series().to_numpy()
    base = np.where(pop4, v24, v23)                      # v25's method
    pop5 = d5.select(C2.has_adsb()).to_series().to_numpy()
    new_rows = pop5 & ~pop4
    print("v24 combiner:", info24)
    print(f"rows with ADS-B information: v4 {pop4.sum():,}, v5 {pop5.sum():,} (+{new_rows.sum():,}); "
          f"runway-ending matches {d5['adsb_rwy_end'].fill_null(False).sum():,}")

    new, info = combiner_v5(d5, v23)
    print("v5 combiner:", info)
    V.evaluate(d5, base, new, "PRIMARY: restricted combiner on v5 vs v25 method", decisive=True)
    tr = (df["taxi"] <= C.TRIM).to_numpy()
    for nm, m in (("newly covered rows", new_rows & tr), ("rows covered in both", pop4 & pop5 & tr)):
        y = df["taxi"].to_numpy()[m]
        print(f"  {nm:<22} n={m.sum():>6}: base {np.sqrt(np.mean((base[m] - y) ** 2)):.1f} -> "
              f"new {np.sqrt(np.mean((new[m] - y) ** 2)):.1f}")


if __name__ == "__main__":
    main()
