"""Stage A screen for reports/history_encodings_preregistration.md.

On the 10 OOF training months only (no holdout): does the OOF stack residual have
structure by stand x runway, stand x runway x 3h, operator x stand, runway x hour that
shrunk group means fitted on other months can remove? Pass iff the four groups together
cut residual RMS by >= 0.5% in both directions (even -> odd months and odd -> even).

Run:  .venv/Scripts/python.exe tests/history_enc_screen.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
FEAT, OOF = ROOT / "cache" / "features", ROOT / "cache" / "oof"
TRIM, SMOOTH, PASS_CUT = 18000, 30.0, 0.005
GROUPS = {
    "sr": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "srh": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "h3"],
    "os": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt"],
    "rh": ["ADEP_mvt", "RUNWAY_mvt", "h1"],
}


def frame() -> pl.DataFrame:
    oof = lambda e: pl.concat([pl.read_parquet(p) for p in sorted((OOF / f"{e}_mixed").glob("fold=*.parquet"))])  # noqa: E731
    lgb = oof("lgb").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ym", lgb="pred")
    cat = oof("cat").select(pl.col("MVT_ID_mvt").cast(pl.Int64), cat="pred")
    lab = pl.read_parquet(FEAT / "labels2025.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "taxi")
    f = pl.read_parquet(FEAT / "train2025.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
                                                             "AIRCRAFT_OPERATOR_flt", "T"])
    f = f.select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt",
                 h1=pl.col("T").dt.hour(), h3=pl.col("T").dt.hour() // 3)
    d = lgb.join(cat, on="MVT_ID_mvt").join(lab, on="MVT_ID_mvt").join(f, on="MVT_ID_mvt")
    d = d.filter(pl.col("taxi") <= TRIM).with_columns(pl.col("taxi").cast(pl.Float64))
    w, _ = nnls(d.select("lgb", "cat").to_numpy(), d["taxi"].to_numpy())
    print(f"OOF rows (label <= 5 h): {d.height:,}; months {sorted(d['ym'].unique().to_list())}; NNLS w {np.round(w, 3)}")
    return d.with_columns(r=pl.col("taxi") - (w[0] * pl.col("lgb") + w[1] * pl.col("cat")),
                          even=pl.col("ym").str.slice(5, 2).cast(pl.Int32) % 2 == 0)


def shrunk(fit: pl.DataFrame, keys: list[str], col: str) -> pl.DataFrame:
    return fit.group_by(keys).agg(_s=pl.col(col).sum(), _n=pl.len()).select(
        keys, adj=pl.col("_s") / (pl.col("_n") + SMOOTH))


def apply(fit: pl.DataFrame, ev: pl.DataFrame, groups: list[str]) -> np.ndarray:
    """Sequential additive shrunk means: each group fit on the residual left by the previous ones."""
    fit, ev = fit.with_columns(rr=pl.col("r")), ev.with_columns(rr=pl.col("r"))
    for g in groups:
        t = shrunk(fit, GROUPS[g], "rr")
        fit = fit.join(t, on=GROUPS[g], how="left").with_columns(rr=pl.col("rr") - pl.col("adj").fill_null(0)).drop("adj")
        ev = ev.join(t, on=GROUPS[g], how="left").with_columns(rr=pl.col("rr") - pl.col("adj").fill_null(0)).drop("adj")
    return ev


def main() -> None:
    d = frame()
    ok = True
    for name, fit_m, ev_m in (("even -> odd", True, False), ("odd -> even", False, True)):
        fit, ev = d.filter(pl.col("even") == fit_m), d.filter(pl.col("even") == ev_m)
        base = float(np.sqrt((ev["r"] ** 2).mean()))
        print(f"\n== {name}: eval rows {ev.height:,}, residual RMS {base:.2f}")
        for gs in [[g] for g in GROUPS] + [list(GROUPS)]:
            out = apply(fit, ev, gs)
            new = float(np.sqrt((out["rr"] ** 2).mean()))
            print(f"  {'+'.join(gs):<14} {base:.2f} -> {new:.2f} ({(new / base - 1) * 100:+.2f}%)")
            if len(gs) == len(GROUPS):
                ok &= new <= base * (1 - PASS_CUT)
                for apt in ("LIRF", "LTFM", "LFPG"):
                    a = out.filter(pl.col("ADEP_mvt") == apt)
                    print(f"    {apt}: {np.sqrt((a['r'] ** 2).mean()):.2f} -> {np.sqrt((a['rr'] ** 2).mean()):.2f}")
    print(f"\nSTAGE A DECISION: {'PASS -> stage B' if ok else 'FAIL -> stop'}")


if __name__ == "__main__":
    main()
