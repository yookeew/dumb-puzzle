"""ADS-B partial-track estimate (reports/adsb_partial_preregistration.md, PROGRESS.md §41).

For departures matched to an ADS-B track that never shows pushback, first seen
< 1 km from the own stand (LIRF excluded):
  est   = (MVT_TIME - adsb_first_ts) + a_airport + b * first_dist_km
  final = base + w_band * (est - base),  bands: first sighting < 500 m / 500-1000 m
Same arithmetic as tests/adsb_partial_test.py (secondary variant, adopted).
"""

from __future__ import annotations

import numpy as np
import polars as pl

D_MAX, D_SPLIT = 1000.0, 500.0
MIN_AP_ROWS = 100


def with_inputs(df: pl.DataFrame) -> pl.DataFrame:
    """Add L and dkm; df needs mvt_ts, adsb_first_ts, adsb_first_own_m."""
    return df.with_columns(L=pl.col("mvt_ts") - pl.col("adsb_first_ts"),
                           dkm=pl.col("adsb_first_own_m") / 1000.0)


def mask() -> pl.Expr:
    return (pl.col("adsb_matched").fill_null(False)
            & ~pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False)
            & (pl.col("adsb_first_own_m") < D_MAX).fill_null(False)
            & (pl.col("ADEP_mvt") != "LIRF"))


def _est(df: pl.DataFrame, p: dict) -> np.ndarray:
    a = df["ADEP_mvt"].replace_strict(p["a"], default=p["a_pooled"], return_dtype=pl.Float64).to_numpy()
    return df["L"].to_numpy() + a + p["b"] * df["dkm"].to_numpy()


def fit(df: pl.DataFrame) -> dict:
    """df needs taxi, base, L, dkm, adsb_first_own_m, ADEP_mvt + mask() columns."""
    f = df.filter(mask())
    unseen = (f["taxi"] - f["L"]).to_numpy()
    lo, hi = np.percentile(unseen, [1, 99])
    y = np.clip(unseen, lo, hi)
    counts = f.group_by("ADEP_mvt").len()
    big = sorted(counts.filter(pl.col("len") >= MIN_AP_ROWS)["ADEP_mvt"].to_list())
    ap = f["ADEP_mvt"].to_numpy()
    X = np.column_stack([(ap == a).astype(float) for a in big]
                        + [(~np.isin(ap, big)).astype(float), f["dkm"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = {"a": dict(zip(big, coef[:len(big)])), "a_pooled": float(coef[len(big)]), "b": float(coef[-1])}
    est, base = _est(f, p), f["base"].to_numpy()
    r, dl = f["taxi"].to_numpy() - base, est - base
    near = (f["adsb_first_own_m"] < D_SPLIT).to_numpy()

    def w(m):
        return float(np.clip((r[m] * dl[m]).sum() / max((dl[m] ** 2).sum(), 1e-9), 0, 1))

    p["w"] = {"near": w(near), "far": w(~near)}
    return p


def apply(df: pl.DataFrame, p: dict) -> np.ndarray:
    """New prediction for every row of df (needs base + the fit inputs)."""
    out = df["base"].to_numpy().astype(float).copy()
    m = df.select(mask()).to_series().to_numpy()
    sub = df.filter(mask())
    w = np.where((sub["adsb_first_own_m"] < D_SPLIT).to_numpy(), p["w"]["near"], p["w"]["far"])
    out[m] = out[m] + w * (_est(sub, p) - out[m])
    return out
