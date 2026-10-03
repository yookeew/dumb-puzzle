"""Neighbour-state stage gate, per reports/neighbour_state_preregistration.md.

Errors are shared in time between departures at the same airport/runway (PROGRESS.md §70).
Label-free proxies of that shared state -- neighbours' NM pseudo-residual, neighbours'
ADS-B pseudo-residual, neighbours' delay, arrivals' taxi-in excess -- are added to
  (a) the full-year ADS-B combiner (rows with ADS-B information), and
  (b) a neighbour corrector on s for every other row.
Base = v27's method cross-fit. Trimmed RMSE on the Jan/Jul 2025 holdout decides.

Run:  .venv/Scripts/python.exe tests/neighbour_state_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_12m_test as T  # noqa: E402
import adsb_combiner_fullyear_test as TF  # noqa: E402
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

JAN, JUL = A.JAN, A.JUL
RAW = ROOT / "data" / "raw"
CEIL = 140000
WINDOWS = (900, 1800, 3600)
OBS = ("appear", "dwell")
NB_FEATS = ([f"{k}_{sc}_{w}" for sc in ("apt", "rwy") for w in WINDOWS for k in ("nbp", "nbq", "nbq_n", "nboff")]
            + [f"arrx_{w}" for w in WINDOWS] + ["arrx_prev"])
B_FEATS = ["airport", "hour", "s", "p_own", "offset"]


# ---------------------------------------------------------------- neighbour builder
def _win_mean(t: np.ndarray, v: np.ndarray, tq: np.ndarray, lo_off: float, hi_off: float,
              excl_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean of non-NaN v over samples with t in [tq + lo_off, tq + hi_off]; t sorted."""
    ok = ~np.isnan(v)
    vv = np.where(ok, v, 0.0)
    cs, cn = np.concatenate([[0.0], np.cumsum(vv)]), np.concatenate([[0], np.cumsum(ok)])
    lo, hi = np.searchsorted(t, tq + lo_off, "left"), np.searchsorted(t, tq + hi_off, "right")
    s, n = cs[hi] - cs[lo], cn[hi] - cn[lo]
    if excl_self:
        s, n = s - vv, n - ok
    return np.where(n >= 2, s / np.maximum(n, 1), np.nan), n.astype(np.float64)


def arrivals(paths: list[Path], base_paths: list[Path]) -> pl.DataFrame:
    """Arrival landing time + taxi-in excess over the 2025 (airport, stand, runway) median."""
    cols = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
    prep = lambda ps: pl.concat([pl.read_parquet(p, columns=cols) for p in ps]).filter(  # noqa: E731
        (pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).with_columns(
        tin=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0).filter(pl.col("tin").is_between(0, 3600))
    med = prep(base_paths).group_by("ADES_mvt", "STAND_mvt", "RUNWAY_mvt").agg(med=pl.col("tin").median())
    a = prep(paths).join(med, on=["ADES_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
    return a.select("ADES_mvt", "t", xin=pl.col("tin") - pl.col("med")).drop_nulls().sort("ADES_mvt", "t")


def nb_frame(df: pl.DataFrame, s: np.ndarray, lag: dict[str, float], arr: pl.DataFrame) -> pl.DataFrame:
    """NB_FEATS for every row of df (row order kept). df needs MVT_ID_mvt, ADEP_mvt, RUNWAY_mvt,
    mvt_ts, aobt3_taxi, offset, adsb_tier, adsb_taxi. Uses no labels."""
    d = df.select("ADEP_mvt", "RUNWAY_mvt", "mvt_ts", "aobt3_taxi", "offset", "adsb_tier", "adsb_taxi").with_columns(
        s=pl.Series(s), _i=pl.int_range(pl.len()))
    d = d.with_columns(
        p=pl.when(pl.col("aobt3_taxi").is_between(1, 5399)).then(pl.col("aobt3_taxi") - pl.col("s")),
        q=pl.when(pl.col("adsb_tier").is_in(OBS)).then(
            (pl.col("adsb_taxi") + pl.col("ADEP_mvt").replace_strict(lag, default=0.0) - pl.col("s")).clip(-1800, 1800)),
        off=pl.col("offset").cast(pl.Float64).clip(-3600, 14400))
    out = {c: np.full(d.height, np.nan) for c in NB_FEATS}
    for scope, keys in (("apt", ["ADEP_mvt"]), ("rwy", ["ADEP_mvt", "RUNWAY_mvt"])):
        for _, g in d.group_by(keys):
            g = g.sort("mvt_ts")
            t, idx = g["mvt_ts"].to_numpy(), g["_i"].to_numpy()
            for w in WINDOWS:
                for k, col in (("nbp", "p"), ("nbq", "q"), ("nboff", "off")):
                    m, n = _win_mean(t, g[col].fill_null(np.nan).to_numpy().astype(np.float64), t, -w, w, True)
                    out[f"{k}_{scope}_{w}"][idx] = m
                    if k == "nbq":
                        out[f"nbq_n_{scope}_{w}"][idx] = n
    for (apt,), g in d.group_by(["ADEP_mvt"]):
        a = arr.filter(pl.col("ADES_mvt") == apt)
        g = g.sort("mvt_ts")
        t, idx = g["mvt_ts"].to_numpy(), g["_i"].to_numpy()
        ta, xa = a["t"].to_numpy(), a["xin"].to_numpy()
        for w in WINDOWS:
            out[f"arrx_{w}"][idx] = _win_mean(ta, xa, t, -w, w, False)[0]
        out["arrx_prev"][idx] = _win_mean(ta, xa, t, -1800, -900, False)[0]
    return pl.DataFrame(out)


# ---------------------------------------------------------------- models
def x_a(df: pl.DataFrame, s: np.ndarray, nb: pl.DataFrame, airports: list[str]) -> np.ndarray:
    return np.column_stack([C.features(df, s, airports).to_numpy().astype(np.float64), nb.to_numpy()])


def x_b(df: pl.DataFrame, s: np.ndarray, nb: pl.DataFrame, airports: list[str]) -> np.ndarray:
    base = df.with_columns(s=pl.Series(s)).select(
        airport=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(airports)}, default=-1,
                                                  return_dtype=pl.Int32),
        hour=pl.col("hour").cast(pl.Float64), s=pl.col("s"),
        p_own=pl.col("aobt3_taxi").cast(pl.Float64) - pl.col("s"), offset=pl.col("offset").cast(pl.Float64))
    return np.column_stack([base.to_numpy().astype(np.float64), nb.to_numpy()])


def fit(X: np.ndarray, y: np.ndarray, inner: np.ndarray, cat_idx: list[int]) -> tuple[lgb.Booster, int]:
    tr = lgb.Dataset(X[~inner], y[~inner], categorical_feature=cat_idx, free_raw_data=False)
    va = lgb.Dataset(X[inner], y[inner], categorical_feature=cat_idx, reference=tr, free_raw_data=False)
    m = lgb.train(C.PARAMS, tr, C.MAX_ROUNDS, valid_sets=[va], callbacks=[lgb.early_stopping(C.ES_PATIENCE, verbose=False)])
    best = max(m.best_iteration, 1)
    return lgb.train(C.PARAMS, lgb.Dataset(X, y, categorical_feature=cat_idx), best), best


CAT_A = [C.FEATS.index(c) for c in C.CATS]
CAT_B = [0]


def train_extra() -> tuple[pl.DataFrame, dict[str, float]]:
    """OOF-month ADS-B day rows (v27's training set) with the raw columns the builder needs."""
    T.MONTHS = TF.FULL
    ex = T.oof_frame()
    raw = (pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "RUNWAY_mvt",
                   aobt3_taxi=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
                   offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).collect())
    ex = ex.join(raw, on="MVT_ID_mvt", how="left")
    lag = dict(ex.filter(pl.col("adsb_tier").is_in(OBS) & (pl.col("taxi") <= C.TRIM)).group_by("ADEP_mvt").agg(
        (pl.col("taxi") - pl.col("adsb_taxi")).median()).iter_rows())
    return ex, lag


def main() -> None:
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    ex, lag = train_extra()
    base, _ = T.combiner_12m(d4, ex, v23, use_holdout=True)          # v27's method
    print(f"lag by airport: { {k: round(v) for k, v in sorted(lag.items())} }")

    paths = sorted(RAW.glob("training_*.parquet"))
    arr = arrivals(paths, paths)
    airports = sorted(d4["ADEP_mvt"].unique().to_list())
    nb_ex = nb_frame(ex, ex["s"].to_numpy(), lag, arr)
    print(f"neighbour features on {ex.height:,} training rows; null share "
          f"{ {c: round(nb_ex[c].is_null().mean() + nb_ex[c].is_nan().mean(), 2) for c in NB_FEATS[:4] + ['arrx_900']} }")

    ex_ok = (ex["taxi"] <= C.TRIM).to_numpy()
    ex_a = ex_ok & (ex["ADEP_mvt"] != "LIRF").to_numpy()
    yex = (ex["taxi"] - ex["s"]).to_numpy()
    iex = (ex["day"].dt.day() % 5 == 0).to_numpy()
    Xa_ex, Xb_ex = x_a(ex, ex["s"].to_numpy(), nb_ex, airports), x_b(ex, ex["s"].to_numpy(), nb_ex, airports)

    pop = d4.select(C2.has_adsb()).to_series().to_numpy()
    non_lirf = (d4["ADEP_mvt"] != "LIRF").to_numpy()
    new_a, new_b = base.copy(), base.copy()
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = C.stack(d4, fm)
        nb = nb_frame(d4, s, lag, arr)
        Xa, Xb = x_a(d4, s, nb, airports), x_b(d4, s, nb, airports)
        y = d4["taxi"].to_numpy() - s
        ih = (d4["day"].dt.day() % 5 == 0).to_numpy()
        fr = ((d4["ym"] == fm) & (d4["taxi"] <= C.TRIM)).to_numpy()
        ma, ba = fit(np.vstack([Xa_ex[ex_a], Xa[fr & non_lirf]]), np.concatenate([yex[ex_a], y[fr & non_lirf]]),
                     np.concatenate([iex[ex_a], ih[fr & non_lirf]]), CAT_A)
        mb, bb = fit(np.vstack([Xb_ex[ex_ok], Xb[fr]]), np.concatenate([yex[ex_ok], y[fr]]),
                     np.concatenate([iex[ex_ok], ih[fr]]), CAT_B)
        app = (d4["ym"] == am).to_numpy()
        app_a, app_b = app & non_lirf & pop, app & ~pop
        new_a[app_a] = np.clip(s[app_a] + ma.predict(Xa[app_a]), 0, CEIL)
        new_b[app_b] = np.clip(s[app_b] + mb.predict(Xb[app_b]), 0, CEIL)
        for nm, m, names in (("a", ma, C.FEATS + NB_FEATS), ("b", mb, B_FEATS + NB_FEATS)):
            g = m.feature_importance("gain")
            share = g[-len(NB_FEATS):].sum() / g.sum()
            top = sorted(zip(names, g), key=lambda t: -t[1])[:6]
            print(f"fit {fm} part ({nm}): best_iter {ba if nm == 'a' else bb}, neighbour gain share {share:.2f}, "
                  f"top {[f'{k}:{v / g.sum():.2f}' for k, v in top]}")
        print(f"  applied to {am}: (a) {app_a.sum():,} rows, (b) {app_b.sum():,} rows", flush=True)

    new = base.copy()
    new[new_a != base] = new_a[new_a != base]
    new[new_b != base] = new_b[new_b != base]
    V.evaluate(d4, base, new, "PRIMARY: neighbour state (a)+(b) vs v27 method", decisive=True)
    V.evaluate(d4, base, new_a, "REPORTED: part (a) alone")
    V.evaluate(d4, base, new_b, "REPORTED: part (b) alone")


if __name__ == "__main__":
    main()
