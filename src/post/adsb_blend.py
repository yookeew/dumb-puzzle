"""ADS-B pushback blend (PROGRESS.md §39, reports/adsb_blend_preregistration.md).

final = pred + w * (adsb_taxi - lag_airport - pred) on eligible rows
(adsb_tier in {appear, dwell}, LIRF excluded), where
  adsb_taxi   = MVT_TIME - adsb_pushback_ts        (src/link/adsb_pushback.py)
  lag_airport = median(adsb_taxi - taxi) on the fit rows at that airport
                (pooled median for airports with < 100 fit rows)
  w           = least-squares weight per tier, clipped to [0, 1]; per
                airport-tier cell where the cell has >= 200 fit rows (the
                pre-registered secondary variant, adopted in §39).

Same arithmetic as tests/adsb_blend_test.py, which is the validated reference.

Quality modulation (§43, reports/adsb_quality_preregistration.md): each
eligible row's weight is multiplied by q[tier, quality] and clipped to [0, 1],
where quality is "good" iff the pushback sample is <= 70 m from the stand and
(appear) its speed <= 5 kt or (dwell) the gap after the last stationary
sample <= 40 s. fit_quality / apply(..., q=) mirror tests/adsb_quality_test.py.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TIERS = ("appear", "dwell")
MIN_LAG_ROWS = 100
MIN_CELL_ROWS = 200
EXCLUDED = ("LIRF",)


def eligible_mask() -> pl.Expr:
    return (pl.col("adsb_tier").is_in(TIERS).fill_null(False)
            & ~pl.col("ADEP_mvt").is_in(EXCLUDED))


def _correct(df: pl.DataFrame, p: dict) -> pl.DataFrame:
    lag = pl.col("ADEP_mvt").replace_strict(p["lags"], default=p["pooled"], return_dtype=pl.Float64)
    return df.with_columns(adsb_c=pl.col("adsb_taxi") - lag)


def fit(df: pl.DataFrame, per_airport: bool = True) -> dict:
    """df: rows with pred, taxi, adsb_taxi, adsb_tier, ADEP_mvt; only eligible rows are used."""
    f = df.filter(eligible_mask())
    pooled = f.select((pl.col("adsb_taxi") - pl.col("taxi")).median()).item()
    lags = {a: (g.select((pl.col("adsb_taxi") - pl.col("taxi")).median()).item()
                if g.height >= MIN_LAG_ROWS else pooled)
            for (a,), g in f.group_by("ADEP_mvt")}
    f = _correct(f, {"lags": lags, "pooled": pooled})

    def w(g: pl.DataFrame) -> float:
        dlt = (g["adsb_c"] - g["pred"]).to_numpy()
        r = (g["taxi"] - g["pred"]).to_numpy()
        return float(np.clip((r * dlt).sum() / max((dlt * dlt).sum(), 1e-9), 0, 1))

    weights = {t: w(f.filter(pl.col("adsb_tier") == t)) for t in TIERS
               if f.filter(pl.col("adsb_tier") == t).height}
    cells = {}
    if per_airport:
        for (a, t), g in f.group_by("ADEP_mvt", "adsb_tier"):
            if g.height >= MIN_CELL_ROWS:
                cells[(a, t)] = w(g)
    return {"lags": lags, "pooled": pooled, "w": weights, "cells": cells}


D_OK, GS_OK, GAP_OK = 70.0, 5.0, 40.0


def quality() -> pl.Expr:
    """'good'/'poor' per row (needs adsb_tier, adsb_pb_dist_m, adsb_pb_gs, adsb_pb_gap_s)."""
    good = (pl.col("adsb_pb_dist_m") <= D_OK) & pl.when(pl.col("adsb_tier") == "appear").then(
        pl.col("adsb_pb_gs") <= GS_OK).otherwise(pl.col("adsb_pb_gap_s") <= GAP_OK)
    return pl.when(good.fill_null(False)).then(pl.lit("good")).otherwise(pl.lit("poor"))


def _weight(p: dict) -> pl.Expr:
    wt = pl.col("adsb_tier").replace_strict(p["w"], default=0.0, return_dtype=pl.Float64)
    if p["cells"]:
        key = pl.concat_str("ADEP_mvt", pl.lit("|"), "adsb_tier")
        cell = {f"{a}|{t}": v for (a, t), v in p["cells"].items()}
        wt = key.replace_strict(cell, default=None, return_dtype=pl.Float64).fill_null(wt)
    return wt


def fit_quality(df: pl.DataFrame, p: dict) -> dict:
    """q[(tier, quality)] >= 0 by least squares given the production weights, on eligible rows."""
    d = _correct(df, p).filter(eligible_mask()).with_columns(_w=_weight(p), _q=quality())
    q = {}
    for (tier, ql), g in d.group_by("adsb_tier", "_q"):
        u = (g["_w"] * (g["adsb_c"] - g["pred"])).to_numpy()
        r = (g["taxi"] - g["pred"]).to_numpy()
        q[(tier, ql)] = float(max((r * u).sum() / max((u * u).sum(), 1e-9), 0.0))
    return q


def apply(df: pl.DataFrame, p: dict, q: dict | None = None) -> pl.Series:
    """Blended prediction for every row of df (needs pred, adsb_taxi, adsb_tier, ADEP_mvt;
    plus the quality fields when q is given)."""
    d = _correct(df, p)
    wt = _weight(p)
    if q:
        key = pl.concat_str("adsb_tier", pl.lit("|"), quality())
        mult = key.replace_strict({f"{t}|{ql}": v for (t, ql), v in q.items()}, default=1.0,
                                  return_dtype=pl.Float64)
        wt = (wt * mult).clip(0, 1)
    return d.select(pl.when(eligible_mask())
                    .then(pl.col("pred") + wt * (pl.col("adsb_c") - pl.col("pred")))
                    .otherwise(pl.col("pred")).alias("blend"))["blend"]
