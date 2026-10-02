"""Gated ADS-B fallback blend (reports/adsb_v3_preregistration.md, PROGRESS.md §56).

On rows with a fallback pushback (tier fb_dwell / fb_appear from
src/link/adsb_pushback.py --v3), LIRF excluded, where the lag-corrected ADS-B
taxi c = adsb_taxi - lag_airport agrees with the incoming prediction to within
GATE_S:  final = base + w_tier * (c - base),  w_tier by least squares, in [0, 1].
lag_airport comes from the main ADS-B blend's fit (adsb_blend.fit()["lags"]).
Same arithmetic as tests/adsb_v3_test.py.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TIERS = ("fb_dwell", "fb_appear")
GATE_S = 300.0
EXCLUDED = ("LIRF",)


def _c(p_blend: dict) -> pl.Expr:
    lag = pl.col("ADEP_mvt").replace_strict(p_blend["lags"], default=p_blend["pooled"], return_dtype=pl.Float64)
    return pl.col("adsb_taxi") - lag


def mask(p_blend: dict) -> pl.Expr:
    return (pl.col("adsb_tier").is_in(TIERS).fill_null(False) & ~pl.col("ADEP_mvt").is_in(EXCLUDED)
            & ((_c(p_blend) - pl.col("base")).abs() <= GATE_S).fill_null(False))


def fit(df: pl.DataFrame, p_blend: dict) -> dict:
    """df: rows with taxi, base, adsb_taxi, adsb_tier, ADEP_mvt."""
    f = df.filter(mask(p_blend)).with_columns(c=_c(p_blend))
    w = {}
    for t in TIERS:
        g = f.filter(pl.col("adsb_tier") == t)
        r, dl = (g["taxi"] - g["base"]).to_numpy(), (g["c"] - g["base"]).to_numpy()
        w[t] = float(np.clip((r * dl).sum() / max((dl * dl).sum(), 1e-9), 0, 1)) if g.height else 0.0
    return {"w": w, "n_fit": f.height}


def apply(df: pl.DataFrame, p_blend: dict, p: dict) -> np.ndarray:
    m = df.select(mask(p_blend)).to_series().to_numpy()
    base = df["base"].to_numpy().astype(float)
    c = df.select(_c(p_blend)).to_series().to_numpy()
    w = df["adsb_tier"].replace_strict(p["w"], default=0.0, return_dtype=pl.Float64).to_numpy()
    return np.where(m, base + w * (c - base), base)
