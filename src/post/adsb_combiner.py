"""Learned ADS-B combiner -> v24 (reports/adsb_combiner_preregistration.md, PROGRESS.md §58).

s = NNLS(lgb, catcorr) stack (production weights, fit on the Jan+Jul 2025
holdout); final = s + LightGBM(taxi - s) on the stack prediction and the v4
ADS-B detector fields, fit on the whole holdout with rounds from the
day-of-month inner split. Same arithmetic as tests/adsb_combiner_test.py.
LIRF rows keep v23's value. Clipped as in production.

Run:  .venv/Scripts/python.exe src/post/adsb_combiner.py [--restricted]
  -> data/submissions/smart-jigsaw_v24.parquet
  --restricted (reports/adsb_combiner_restricted_preregistration.md): the combiner only on
  rows with ADS-B information (matched, or a fallback tier); every other row keeps v23's
  value -> data/submissions/smart-jigsaw_v25.parquet
  --v2 (reports/adsb_combiner_v2_preregistration.md): combiner v2, fit and applied only on
  rows with ADS-B information, stronger regularisation (implies --restricted)
  -> data/submissions/smart-jigsaw_v26.parquet
  --v5 (reports/adsb_v5_preregistration.md): v25's restricted combiner on v5 detections
  (runway-ending matches) with the rwy_end input (implies --restricted)
  -> data/submissions/smart-jigsaw_v26.parquet
  --m12 (reports/adsb_combiner_12m_preregistration.md): v25's restricted combiner, trained on
  the holdout plus the other 2025 months' ADS-B days (OOF stack) (implies --restricted)
  -> data/submissions/smart-jigsaw_m12.parquet (rename to the next free version on upload)
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import adsb_combiner_test as C  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

SUB = ROOT / "data" / "submissions"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
V2 = "--v2" in sys.argv
V5 = "--v5" in sys.argv
M12 = "--m12" in sys.argv
RESTRICTED = "--restricted" in sys.argv or V2 or V5 or M12
V23 = SUB / "smart-jigsaw_v23.parquet"
V24 = SUB / ("smart-jigsaw_m12.parquet" if M12 else "smart-jigsaw_v26.parquet" if (V2 or V5) else "smart-jigsaw_v25.parquet" if RESTRICTED
             else "smart-jigsaw_v24.parquet")
ENG = ["lgb", "catcorr"]
DET_DIR = ROOT / "cache" / ("adsb_pushback_v5" if V5 else "adsb_pushback_v4")
CEIL, LIRF_CEIL = 10800, 140000


def ranking_frame() -> pl.DataFrame:
    rk = None
    for e in ENG:
        s = pl.read_parquet(SUB / f"{e}_mixed.parquet").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(e))
        rk = s if rk is None else rk.join(s, on="MVT_ID_mvt", how="inner")
    mv = (pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt",
                  hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
                  mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0))
    det = pl.concat([pl.read_parquet(p) for p in sorted(DET_DIR.glob("day=2026-*.parquet"))])
    det = det.select("MVT_ID_mvt", *V.DET_COLS, "adsb_fallback", *(["adsb_rwy_end"] if V5 else []))
    return (rk.join(mv, on="MVT_ID_mvt", how="left").join(det, on="MVT_ID_mvt", how="left")
            .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))


def main() -> None:
    if V5:
        import adsb_v5_test as T5
        ho = V.with_det(A.load(), T5.read_det_v5())
    else:
        ho = V.with_det(A.load(), V.read_det(C.V4))
    feats = (lambda d, s, a: T5.features_v5(d, s, a)) if V5 else         (lambda d, s, a: C.features(d, s, a).to_numpy().astype(np.float64))
    w, _ = nnls(ho.select(ENG).to_numpy(), ho["taxi"].to_numpy())
    print("stack weights:", dict(zip(ENG, np.round(w, 4))))
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    s_ho = ho.select(ENG).to_numpy() @ w
    X = feats(ho, s_ho, airports)
    fit = ((ho["taxi"] <= C.TRIM) & (ho["ADEP_mvt"] != "LIRF")).to_numpy()
    if V2:
        import adsb_combiner_v2_test as C2
        fit = fit & ho.select(C2.has_adsb()).to_series().to_numpy()
    y = ho["taxi"].to_numpy() - s_ho
    inner = (ho["day"].dt.day() % 5 == 0).to_numpy()[fit]
    if M12:
        import adsb_combiner_12m_test as T12
        ex = T12.oof_frame()
        ex_rows = ((ex["taxi"] <= C.TRIM) & (ex["ADEP_mvt"] != "LIRF")).to_numpy()
        X = np.vstack([X[fit], feats(ex, ex["s"].to_numpy(), airports)[ex_rows]])
        y = np.concatenate([y[fit], (ex["taxi"] - ex["s"]).to_numpy()[ex_rows]])
        inner = np.concatenate([inner, (ex["day"].dt.day() % 5 == 0).to_numpy()[ex_rows]])
        fit = np.ones(len(y), bool)
    m, best = (C2.fit_v2 if V2 else C.fit_combiner)(X[fit], y[fit], inner)
    print(f"combiner fit on {fit.sum():,} holdout rows; best_iter {best}")

    rk = ranking_frame()
    assert rk["ADEP_mvt"].null_count() == 0
    s_rk = rk.select(ENG).to_numpy() @ w
    Xr = feats(rk, s_rk, airports)
    pred = s_rk + m.predict(Xr)
    v23 = pl.read_parquet(V23).select(pl.col("MVT_ID_mvt").cast(pl.Int64), v23=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    rk = rk.with_columns(comb=pl.Series(pred)).join(v23, on="MVT_ID_mvt", how="left")
    hi = pl.when(pl.col("ADEP_mvt") == "LIRF").then(LIRF_CEIL).otherwise(CEIL)
    has_adsb = (pl.col("adsb_matched").fill_null(False)
                | pl.col("adsb_tier").is_in(["fb_dwell", "fb_appear"]).fill_null(False))
    keep_v23 = (pl.col("ADEP_mvt") == "LIRF") | (~has_adsb if RESTRICTED else pl.lit(False))
    print("rows keeping v23:", rk.select(keep_v23.sum()).item())
    rk = rk.with_columns(final=pl.when(keep_v23).then(pl.col("v23"))
                         .otherwise(pl.col("comb").clip(lower_bound=0).clip(upper_bound=hi)))

    tpl = pl.read_parquet(TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(
        rk.select(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype),
                  pl.col("final").round().cast(pl.Int32).alias("TAXITIME_SEC_mvt")), on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert out["MVT_ID_mvt"].n_unique() == out.height and (out["TAXITIME_SEC_mvt"] >= 0).all()
    out.write_parquet(V24)
    dd = rk.with_columns(diff=pl.col("final").round() - pl.col("v23"))
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12, float_precision=1):
        print(dd.group_by("ADEP_mvt").agg(pl.len(), pl.col("diff").mean().alias("mean_diff"),
              pl.col("diff").pow(2).mean().sqrt().alias("rms_diff")).sort("ADEP_mvt"))
    print(f"wrote {V24.name}: RMS diff vs v23 {np.sqrt((dd['diff'] ** 2).mean()):.1f} s; "
          f"median {out['TAXITIME_SEC_mvt'].median():.0f}, mean {out['TAXITIME_SEC_mvt'].mean():.1f}")


if __name__ == "__main__":
    main()
