"""Neighbour-state stage -> v29 (reports/neighbour_state_preregistration.md, PROGRESS.md §70-71).

v27 with
  (a) rows with ADS-B information (non-LIRF): the full-year combiner retrained with the
      neighbour features added, and
  (b) every row without ADS-B information (all airports): s + a neighbour corrector,
both fit on the Jan+Jul 2025 holdout (production NNLS stack) plus the 115 OOF-month ADS-B
days (OOF stack), with the gate's inputs and settings (tests/neighbour_state_test.py).
LIRF rows with ADS-B information keep v27's value. Clipped to [0, 140,000].

Run:  .venv/Scripts/python.exe src/post/neighbour_state.py [--v2]
  -> data/submissions/smart-jigsaw_v29.parquet
  --v2 (reports/neighbour_state_v2_preregistration.md): part (b) only, with its correction
  demeaned per (airport, UTC day) over the rows it is applied to; rows with ADS-B information
  keep v27 -> data/submissions/smart-jigsaw_v30.parquet
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
import ltfm_egll_anatomy as A  # noqa: E402
import neighbour_state_test as N  # noqa: E402
from post import adsb_combiner as PC  # noqa: E402

SUB = ROOT / "data" / "submissions"
V2 = "--v2" in sys.argv
V27 = SUB / "smart-jigsaw_v27.parquet"
V29 = SUB / ("smart-jigsaw_v30.parquet" if V2 else "smart-jigsaw_v29.parquet")
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
ENG = ["lgb", "catcorr"]


def main() -> None:
    ho = V.with_det(A.load(), V.read_det(C.V4))
    w, _ = nnls(ho.select(ENG).to_numpy(), ho["taxi"].to_numpy())
    print("stack weights:", dict(zip(ENG, np.round(w, 4))))
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    ex, lag = N.train_extra()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr25 = N.arrivals(paths, paths)

    s_ho, s_ex = ho.select(ENG).to_numpy() @ w, ex["s"].to_numpy()
    nb_ho, nb_ex = N.nb_frame(ho, s_ho, lag, arr25), N.nb_frame(ex, s_ex, lag, arr25)
    ok_ho, ok_ex = (ho["taxi"] <= C.TRIM).to_numpy(), (ex["taxi"] <= C.TRIM).to_numpy()
    nl_ho, nl_ex = (ho["ADEP_mvt"] != "LIRF").to_numpy(), (ex["ADEP_mvt"] != "LIRF").to_numpy()
    y = np.concatenate([ho["taxi"].to_numpy() - s_ho, (ex["taxi"] - ex["s"]).to_numpy()])
    inner = np.concatenate([(ho["day"].dt.day() % 5 == 0).to_numpy(), (ex["day"].dt.day() % 5 == 0).to_numpy()])
    Xa = np.vstack([N.x_a(ho, s_ho, nb_ho, airports), N.x_a(ex, s_ex, nb_ex, airports)])
    Xb = np.vstack([N.x_b(ho, s_ho, nb_ho, airports), N.x_b(ex, s_ex, nb_ex, airports)])
    fa, fb = np.concatenate([ok_ho & nl_ho, ok_ex & nl_ex]), np.concatenate([ok_ho, ok_ex])
    ma, ba = (None, 0) if V2 else N.fit(Xa[fa], y[fa], inner[fa], N.CAT_A)
    mb, bb = N.fit(Xb[fb], y[fb], inner[fb], N.CAT_B)
    print(f"part (a) fit on {fa.sum():,} rows, best_iter {ba}; part (b) fit on {fb.sum():,} rows, best_iter {bb}")

    rk = PC.ranking_frame()
    raw = pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "RUNWAY_mvt",
        aobt3_taxi=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
        offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
    rk = rk.join(raw, on="MVT_ID_mvt", how="left")
    arr26 = N.arrivals([RANKING], paths)
    s_rk = rk.select(ENG).to_numpy() @ w
    nb_rk = N.nb_frame(rk, s_rk, lag, arr26)
    pop = rk.select(C2.has_adsb()).to_series().to_numpy()
    lirf = (rk["ADEP_mvt"] == "LIRF").to_numpy()
    c = mb.predict(N.x_b(rk, s_rk, nb_rk, airports))
    if V2:
        import neighbour_state_v2_test as N2
        day = pl.from_epoch(rk["mvt_ts"].cast(pl.Int64), "s").dt.date().to_numpy()
        c = N2.day_demean(c, rk["ADEP_mvt"].to_numpy(), day, ~pop)
        lirf = np.ones(rk.height, bool)          # every row with ADS-B information keeps v27
    pred_b = np.clip(s_rk + c, 0, N.CEIL)
    pred_a = pred_b if V2 else np.clip(s_rk + ma.predict(N.x_a(rk, s_rk, nb_rk, airports)), 0, N.CEIL)
    v27 = pl.read_parquet(V27)
    rk = rk.join(v27.select(pl.col("MVT_ID_mvt").cast(pl.Int64), v27=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)),
                 on="MVT_ID_mvt", how="left")
    final = np.where(pop & ~lirf, pred_a, np.where(~pop, pred_b, rk["v27"].to_numpy()))
    rk = rk.with_columns(final=pl.Series(np.round(final)), part=pl.Series(np.where(pop & ~lirf, "a", np.where(~pop, "b", "v27"))))
    print("rows by part:", dict(rk.group_by("part").len().iter_rows()))

    tpl = pl.read_parquet(TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(
        rk.select(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype),
                  pl.col("final").cast(v27["TAXITIME_SEC_mvt"].dtype).alias("TAXITIME_SEC_mvt")),
        on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert out["MVT_ID_mvt"].n_unique() == out.height and (out["TAXITIME_SEC_mvt"] >= 0).all()
    out.write_parquet(V29)
    dd = rk.with_columns(diff=pl.col("final") - pl.col("v27"))
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=24, float_precision=1):
        print(dd.group_by("ADEP_mvt", "part").agg(pl.len(), pl.col("diff").mean().alias("mean_diff"),
              pl.col("diff").pow(2).mean().sqrt().alias("rms_diff")).sort("ADEP_mvt", "part"))
    print(f"wrote {V29.name}: RMS diff vs v27 {np.sqrt((dd['diff'] ** 2).mean()):.1f} s")


if __name__ == "__main__":
    main()
