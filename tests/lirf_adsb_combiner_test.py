"""Rome (LIRF) ADS-B combiner gate, per reports/lirf_adsb_combiner_preregistration.md.

Trained on LIRF rows with ADS-B information from the 2025 OOF-month days (OOF stack),
scored on Jul 2025 holdout LIRF rows with ADS-B information. Base = v27's method
cross-fit. Trimmed RMSE on the Jul holdout decides.

Run:  .venv/Scripts/python.exe tests/lirf_adsb_combiner_test.py
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
LIRF_CEIL = 140000
PARAMS = {**C.PARAMS, "num_leaves": 15, "min_data_in_leaf": 200, "lambda_l2": 10.0, "feature_fraction": 0.9}
EXTRA_FEATS = ["offset", "nm_unmatched", "inbound_echo_day"]


def arrival_echo_day(mv: pl.DataFrame) -> pl.DataFrame:
    """Per (operator, UTC date): share of LIRF arrivals with |block - sched| < 30 s (arrivals only)."""
    a = mv.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")
                  & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    return a.group_by(op="AIRCRAFT_OPERATOR_flt", date=pl.col("MVT_TIME_UTC_mvt").dt.date()).agg(
        inbound_echo_day=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() < 30)
        .mean())


def dep_extras(mv: pl.DataFrame) -> pl.DataFrame:
    """MVT_ID -> offset, nm_unmatched, inbound_echo_day for LIRF departures in mv."""
    d = mv.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), op="AIRCRAFT_OPERATOR_flt", date=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        nm_unmatched=pl.col("AOBT_3_flt").is_null().cast(pl.Float64))
    return d.join(arrival_echo_day(mv), on=["op", "date"], how="left").select(
        "MVT_ID_mvt", "offset", "nm_unmatched", "inbound_echo_day")


def features(df: pl.DataFrame, s: np.ndarray, airports: list[str]) -> np.ndarray:
    X = C.features(df, s, airports).to_numpy().astype(np.float64)
    return np.column_stack([X, df.select(EXTRA_FEATS).to_numpy().astype(np.float64)])


def fit(X: np.ndarray, y: np.ndarray, inner: np.ndarray) -> tuple[lgb.Booster, int]:
    cat_idx = [C.FEATS.index(c) for c in C.CATS]
    tr = lgb.Dataset(X[~inner], y[~inner], categorical_feature=cat_idx, free_raw_data=False)
    va = lgb.Dataset(X[inner], y[inner], categorical_feature=cat_idx, reference=tr, free_raw_data=False)
    m = lgb.train(PARAMS, tr, C.MAX_ROUNDS, valid_sets=[va], callbacks=[lgb.early_stopping(C.ES_PATIENCE, verbose=False)])
    best = max(m.best_iteration, 1)
    return lgb.train(PARAMS, lgb.Dataset(X, y, categorical_feature=cat_idx), best), best


def train_frame() -> pl.DataFrame:
    T.MONTHS = TF.FULL
    ex = T.oof_frame().filter(pl.col("ADEP_mvt") == "LIRF")
    mv = pl.concat([pl.read_parquet(p) for p in sorted((ROOT / "data" / "raw").glob("training_*.parquet"))],
                   how="diagonal_relaxed")
    ex = ex.join(dep_extras(mv), on="MVT_ID_mvt", how="left")
    return ex.filter(ex.select(C2.has_adsb()).to_series() & (pl.col("taxi") <= C.TRIM))


def main() -> None:
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    T.MONTHS = TF.FULL
    base, _ = T.combiner_12m(d4, T.oof_frame(), v23, use_holdout=True)     # v27's method

    mv = pl.concat([pl.read_parquet(p) for p in sorted((ROOT / "data" / "raw").glob("training_*.parquet"))],
                   how="diagonal_relaxed")
    d4 = d4.join(dep_extras(mv).drop("offset"), on="MVT_ID_mvt", how="left")
    tr = train_frame()
    airports = sorted(df["ADEP_mvt"].unique().to_list())
    Xt = features(tr, tr["s"].to_numpy(), airports)
    yt = (tr["taxi"] - tr["s"]).to_numpy()
    inner = (tr["day"].dt.day() % 5 == 0).to_numpy()
    m, best = fit(Xt, yt, inner)
    imp = sorted(zip(C.FEATS + EXTRA_FEATS, m.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(v for _, v in imp)
    print(f"LIRF training rows {len(yt):,} on {tr['day'].n_unique()} days ({sorted(tr['ym'].unique().to_list())}); "
          f"best_iter {best}; top gain {[f'{k}:{v / tot:.2f}' for k, v in imp[:8]]}")

    s = C.stack(d4, JAN)
    app = ((d4["ym"] == JUL) & (d4["ADEP_mvt"] == "LIRF")).to_numpy() & d4.select(C2.has_adsb()).to_series().to_numpy()
    new = base.copy()
    new[app] = np.clip(s[app] + m.predict(features(d4, s, airports)[app]), 0, LIRF_CEIL)
    print(f"Jul 2025 LIRF rows with ADS-B information: {app.sum():,}")

    jul = (d4["ym"] == JUL).to_numpy()
    trim = (d4["taxi"] <= C.TRIM).to_numpy()
    sub = lambda msk: d4.filter(pl.Series(msk))  # noqa: E731
    print("\n=== PRIMARY: LIRF ADS-B combiner, Jul 2025 holdout ===")
    ptT, pwT = V.score(sub(jul & trim), base[jul & trim], new[jul & trim], "Jul trimmed (decides)")
    _, pwF = V.score(sub(jul), base[jul], new[jul], "Jul full (guard)")
    ok = ptT < 0 and pwT < 0.05 and pwF < 0.9
    print(f"PRIMARY DECISION: {'ADOPT -> build v28, board A/B' if ok else 'REJECT -> stop'}")

    y = d4["taxi"].to_numpy()
    for nm, msk in (("LIRF-with-ADS-B trimmed", app & trim), ("LIRF-with-ADS-B full", app)):
        print(f"  {nm:<26} n={msk.sum():>6}: base {np.sqrt(np.mean((base[msk] - y[msk]) ** 2)):.1f} -> "
              f"new {np.sqrt(np.mean((new[msk] - y[msk]) ** 2)):.1f}")
    off = d4["offset"].to_numpy()
    for lo, hi in ((-1e9, 1800), (1800, 3600), (3600, 1e9)):
        msk = app & trim & (off > lo) & (off <= hi)
        if msk.sum():
            print(f"  delay band ({lo:>6.0f},{hi:>6.0f}] n={msk.sum():>5}: base "
                  f"{np.sqrt(np.mean((base[msk] - y[msk]) ** 2)):.1f} -> new {np.sqrt(np.mean((new[msk] - y[msk]) ** 2)):.1f}")


if __name__ == "__main__":
    main()
