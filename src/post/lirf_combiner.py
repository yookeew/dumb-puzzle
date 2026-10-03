"""Rome (LIRF) ADS-B combiner -> v28 (reports/lirf_adsb_combiner_preregistration.md, PROGRESS.md §66).

v27 with its LIRF ranking rows that have ADS-B information (matched or a fallback tier)
replaced by s + LightGBM(taxi - s). The model is fit on the 2025 OOF-month LIRF rows (OOF
stack) plus the Jul 2025 holdout LIRF rows (production stack weights), with the gate's
inputs and hyperparameters (tests/lirf_adsb_combiner_test.py). Rounds come from the same
day-of-month inner split.

Run:  .venv/Scripts/python.exe src/post/lirf_combiner.py
  -> data/submissions/smart-jigsaw_v28.parquet
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
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import lirf_adsb_combiner_test as L  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
from post import adsb_combiner as PC  # noqa: E402

SUB = ROOT / "data" / "submissions"
V27, V28 = SUB / "smart-jigsaw_v27.parquet", SUB / "smart-jigsaw_v28.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
ENG = ["lgb", "catcorr"]


def main() -> None:
    ho = V.with_det(A.load(), V.read_det(C.V4))
    w, _ = nnls(ho.select(ENG).to_numpy(), ho["taxi"].to_numpy())
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    mv = pl.concat([pl.read_parquet(p) for p in sorted((ROOT / "data" / "raw").glob("training_*.parquet"))],
                   how="diagonal_relaxed")
    ho = ho.join(L.dep_extras(mv).drop("offset"), on="MVT_ID_mvt", how="left")
    jul = ho.filter((pl.col("ym") == A.JUL) & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("taxi") <= C.TRIM)
                    & C2.has_adsb())
    s_j = jul.select(ENG).to_numpy() @ w
    tr = L.train_frame()
    X = np.vstack([L.features(tr, tr["s"].to_numpy(), airports), L.features(jul, s_j, airports)])
    y = np.concatenate([(tr["taxi"] - tr["s"]).to_numpy(), jul["taxi"].to_numpy() - s_j])
    inner = np.concatenate([(tr["day"].dt.day() % 5 == 0).to_numpy(), (jul["day"].dt.day() % 5 == 0).to_numpy()])
    m, best = L.fit(X, y, inner)
    print(f"LIRF combiner fit on {len(y):,} rows (OOF {tr.height:,} + Jul 2025 {jul.height:,}); best_iter {best}")

    rk = PC.ranking_frame()
    rmv = pl.read_parquet(ROOT / "data" / "ranking" / "ranking.parquet")
    rk = rk.join(L.dep_extras(rmv), on="MVT_ID_mvt", how="left")
    app = ((rk["ADEP_mvt"] == "LIRF").to_numpy() & rk.select(C2.has_adsb()).to_series().to_numpy())
    s_rk = rk.select(ENG).to_numpy() @ w
    pred = np.clip(s_rk + m.predict(L.features(rk, s_rk, airports)), 0, L.LIRF_CEIL)
    v27 = pl.read_parquet(V27)
    rk = rk.join(v27.select(pl.col("MVT_ID_mvt").cast(pl.Int64), v27=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)),
                 on="MVT_ID_mvt", how="left")
    final = np.where(app, np.round(pred), rk["v27"].to_numpy())
    rk = rk.with_columns(final=pl.Series(final))
    print(f"LIRF ranking rows replaced: {app.sum():,} (by month: "
          f"{dict(rk.filter(pl.Series(app)).group_by(pl.from_epoch(pl.col('mvt_ts').cast(pl.Int64), 's').dt.month()).len().iter_rows())})")
    ch = rk.filter(pl.Series(app))
    print(f"  mean shift vs v27 {(ch['final'] - ch['v27']).mean():+.1f}s, RMS {np.sqrt(((ch['final'] - ch['v27']) ** 2).mean()):.1f}s")

    tpl = pl.read_parquet(TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(
        rk.select(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype),
                  pl.col("final").cast(v27["TAXITIME_SEC_mvt"].dtype).alias("TAXITIME_SEC_mvt")),
        on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert out["MVT_ID_mvt"].n_unique() == out.height and (out["TAXITIME_SEC_mvt"] >= 0).all()
    out.write_parquet(V28)
    d = out.join(v27, on="MVT_ID_mvt").select((pl.col("TAXITIME_SEC_mvt") - pl.col("TAXITIME_SEC_mvt_right"))
                                              .cast(pl.Float64)).to_series().to_numpy()
    print(f"wrote {V28.name}: {(d != 0).sum():,} rows differ from v27, RMS over all rows {np.sqrt((d ** 2).mean()):.1f}s")


if __name__ == "__main__":
    main()
