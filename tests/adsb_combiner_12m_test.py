"""ADS-B combiner retrained with the other 2025 months, per
reports/adsb_combiner_12m_preregistration.md.

Training rows = OOF-month ADS-B days (stack from month-wise OOF lgb + cat) plus the
other holdout month; scored on the holdout month, restricted to ADS-B rows as in v25.
Base = v25's method cross-fit. Trimmed RMSE decides.

Run:  .venv/Scripts/python.exe tests/adsb_combiner_12m_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

JAN, JUL = A.JAN, A.JUL
OOF = ROOT / "cache" / "oof"
RAW = ROOT / "data" / "raw"
MONTHS = ("2025-09", "2025-10", "2025-11", "2025-12")   # amendment 2026-10-02: Sep-Dec only
DAYS = {1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28}      # second amendment: spread over the month
EXTRA_DAYS = {"2025-09-15", "2025-11-15"}                # third amendment: re-downloaded fresh
MANIFEST = ROOT / "external-data" / "adsb-restofyear" / "manifest.csv"
MIN_FRAC = 0.5                                           # fourth amendment: thin-day rule


def good_days() -> set[str]:
    """Fourth amendment: keep a listed day iff its last manifest row has status ok/tolerant and
    >= MIN_FRAC x the median points of the listed days in its month. Label-free."""
    m = pl.read_csv(MANIFEST).with_row_index("_i").sort("_i").group_by("day").last()
    m = m.filter(pl.col("day").str.slice(0, 7).is_in(MONTHS)
                 & (pl.col("day").str.slice(8, 2).cast(pl.Int32).is_in(list(DAYS))
                    | pl.col("day").is_in(list(EXTRA_DAYS))))
    m = m.with_columns(med=pl.col("points").filter(pl.col("status").is_in(["ok", "tolerant"]))
                       .median().over(pl.col("day").str.slice(0, 7)))
    keep = m.filter(pl.col("status").is_in(["ok", "tolerant"]) & (pl.col("points") >= MIN_FRAC * pl.col("med")))
    drop = m.join(keep.select("day"), on="day", how="anti")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=60):
        print(f"thin-day rule: {keep.height} days kept, {drop.height} dropped")
        if drop.height:
            print(drop.select("day", "status", "bad_members", "points", "med").sort("day"))
    return set(keep["day"].to_list())


def oof_frame() -> pl.DataFrame:
    """Labelled DEP rows on non-holdout 2025 days that have v4 detections, with the OOF stack."""
    keep = good_days()
    paths = [p for p in sorted(C.V4.glob("day=2025-*.parquet")) if p.name[4:14] in keep]
    det = pl.concat([pl.read_parquet(p) for p in paths]).select("MVT_ID_mvt", *V.DET_COLS, "adsb_fallback")
    pr = None
    for e in ("lgb", "cat"):
        s = pl.read_parquet(OOF / f"{e}_mixed" / "*.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                                                                       pl.col("pred").alias(e))
        pr = s if pr is None else pr.join(s, on="MVT_ID_mvt")
    mv = (pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                  ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
                  hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
                  mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0).collect())
    d = det.join(mv, on="MVT_ID_mvt", how="inner").join(pr, on="MVT_ID_mvt", how="inner")
    d = d.with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts"))
    w, _ = nnls(d.select("lgb", "cat").to_numpy(), d["taxi"].to_numpy())
    print(f"OOF-month ADS-B rows: {d.height:,} on {d['day'].n_unique()} days "
          f"({sorted(d['ym'].unique().to_list())}); OOF stack weights {np.round(w, 3)}")
    return d.with_columns(s=pl.Series(d.select("lgb", "cat").to_numpy() @ w))


def combiner_12m(ho: pl.DataFrame, extra: pl.DataFrame, base: np.ndarray,
                 use_holdout: bool = True) -> tuple[np.ndarray, dict]:
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    out, info = base.copy(), {}
    ex_rows = ((extra["taxi"] <= C.TRIM) & (extra["ADEP_mvt"] != "LIRF")).to_numpy()
    Xe = C.features(extra, extra["s"].to_numpy(), airports).to_numpy().astype(np.float64)[ex_rows]
    ye = (extra["taxi"] - extra["s"]).to_numpy()[ex_rows]
    ie = (extra["day"].dt.day() % 5 == 0).to_numpy()[ex_rows]
    non_lirf = (ho["ADEP_mvt"] != "LIRF").to_numpy()
    pop = ho.select(C2.has_adsb()).to_series().to_numpy()
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = C.stack(ho, fm)
        X = C.features(ho, s, airports).to_numpy().astype(np.float64)
        y = ho["taxi"].to_numpy() - s
        Xs, ys, ins = [Xe], [ye], [ie]
        if use_holdout:
            fr = ((ho["ym"] == fm) & (ho["taxi"] <= C.TRIM)).to_numpy() & non_lirf
            Xs.append(X[fr]); ys.append(y[fr]); ins.append((ho["day"].dt.day() % 5 == 0).to_numpy()[fr])
        m, best = C.fit_combiner(np.vstack(Xs), np.concatenate(ys), np.concatenate(ins))
        app = (ho["ym"] == am).to_numpy() & non_lirf & pop
        out[app] = s[app] + m.predict(X[app])
        info[f"fit {fm}{'+OOF' if use_holdout else ' (OOF only)'}"] = {
            "best_iter": best, "n_train": int(sum(len(v) for v in ys))}
    return out, info


def main() -> None:
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    v24, _ = C.combiner(d4, v23)
    pop = d4.select(C2.has_adsb()).to_series().to_numpy()
    base = np.where(pop, v24, v23)                       # v25's method
    extra = oof_frame()

    new, info = combiner_12m(d4, extra, v23, use_holdout=True)
    print("12-month combiner:", info)
    V.evaluate(d4, base, new, "PRIMARY: combiner + OOF months vs v25 method", decisive=True)

    new2, info2 = combiner_12m(d4, extra, v23, use_holdout=False)
    print("\nOOF-months-only combiner:", info2)
    V.evaluate(d4, base, new2, "reported: combiner trained on OOF months only")


if __name__ == "__main__":
    main()
