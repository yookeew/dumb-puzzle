"""Build the stacked submissions (reports/stack_preregistration.md, PROGRESS.md §39).

Writes two files so the leaderboard shows each gain separately:
  data/submissions/stack_lgb_cat.parquet        NNLS stack of LightGBM + CatBoost
  data/submissions/stack_lgb_cat_adsb.parquet   the same + ADS-B pushback blend

Parameters come from the Jan+Jul 2025 holdout only, fit on both months together
(the cross-fit evaluation that justified them is tests/stack_test.py):
  - stack weights: NNLS, no intercept, on the engines' holdout predictions
  - ADS-B blend: src/post/adsb_blend.py fit on the *stacked* holdout prediction
They are applied to each engine's all-2025 refit submission
(run(..., submit=True) -> data/submissions/<engine>_mixed.parquet) and, for the
ADS-B part, to 2026 ranking rows with a recovered pushback
(cache/adsb_pushback/day=2026-*.parquet).

Run:  .venv/Scripts/python.exe src/post/stack_submit.py
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
from post import adsb_blend  # noqa: E402

EVAL = ROOT / "cache" / "eval"
SUB = ROOT / "data" / "submissions"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
RAW = ROOT / "data" / "raw"
DET = ROOT / "cache" / "adsb_pushback"
ENGINES = ["lgb", "cat"]
CEIL, LIRF_CEIL = 10800, 140000   # models.fit CEIL / LIRF_RAW_CEIL


def _ev(engine: str) -> pl.DataFrame:
    p = EVAL / f"{engine}_mixed_holdout_ev.parquet"
    if engine == "lgb" and not p.exists():
        p = EVAL / "prod_mixed_holdout_ev.parquet"
    return pl.read_parquet(p).select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "taxi",
                                     pl.col("pred").alias(engine))


def _det(prefix: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(DET.glob(f"day={prefix}*.parquet"))]).select(
        "MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts")


def _mvt_ts(src) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt",
                    mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0).collect())


def _clip(df: pl.DataFrame, col: str) -> pl.Expr:
    hi = pl.when(pl.col("ADEP_mvt") == "LIRF").then(LIRF_CEIL).otherwise(CEIL)
    return pl.col(col).clip(lower_bound=0).clip(upper_bound=hi)


def _write(ids_preds: pl.DataFrame, name: str) -> None:
    tpl = pl.read_parquet(TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(
        ids_preds.select(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype),
                         pl.col("final").round().cast(pl.Int32).alias("TAXITIME_SEC_mvt")),
        on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height, "row count differs from template"
    assert out["TAXITIME_SEC_mvt"].null_count() == 0, "missing predictions"
    assert out["MVT_ID_mvt"].n_unique() == out.height, "duplicate ids"
    assert (out["TAXITIME_SEC_mvt"] >= 0).all()
    SUB.mkdir(exist_ok=True)
    out.write_parquet(SUB / f"{name}.parquet")
    s = out["TAXITIME_SEC_mvt"]
    print(f"wrote {name}.parquet  n={out.height:,}  median={s.median():.0f}  mean={s.mean():.1f}  "
          f"max={s.max()}")


def main() -> None:
    # ---- stack weights from the full holdout
    ho = _ev(ENGINES[0])
    for e in ENGINES[1:]:
        ho = ho.join(_ev(e).select("MVT_ID_mvt", e), on="MVT_ID_mvt", how="inner")
    w, _ = nnls(ho.select(ENGINES).to_numpy(), ho["taxi"].to_numpy())
    print("stack weights (full holdout):", dict(zip(ENGINES, np.round(w, 4))))
    ho = ho.with_columns(pred=pl.Series(ho.select(ENGINES).to_numpy() @ w))
    print(f"holdout in-sample RMSE: lgb {np.sqrt(((ho['lgb'] - ho['taxi']) ** 2).mean()):.2f}  "
          f"stack {np.sqrt(((ho['pred'] - ho['taxi']) ** 2).mean()):.2f}")

    # ---- ADS-B blend parameters from the stacked holdout prediction
    ho = (ho.join(_mvt_ts(str(RAW / "training_*.parquet")).select("MVT_ID_mvt", "mvt_ts"),
                  on="MVT_ID_mvt", how="left")
          .join(_det("2025"), on="MVT_ID_mvt", how="left")
          .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))
    p = adsb_blend.fit(ho, per_airport=True)
    print("ADS-B blend: tier w", {k: round(v, 3) for k, v in p["w"].items()},
          "| lags", {a: round(v) for a, v in sorted(p["lags"].items())},
          "| cells", {f"{a}/{t}": round(v, 2) for (a, t), v in sorted(p["cells"].items())})

    # ---- ranking: stack the engines' submissions
    rk = None
    for e in ENGINES:
        s = pl.read_parquet(SUB / f"{e}_mixed.parquet").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(e))
        rk = s if rk is None else rk.join(s, on="MVT_ID_mvt", how="inner")
    rk = rk.join(_mvt_ts(RANKING), on="MVT_ID_mvt", how="left")
    assert rk["ADEP_mvt"].null_count() == 0, "submission ids missing from ranking.parquet"
    rk = rk.with_columns(pred=pl.Series(rk.select(ENGINES).to_numpy() @ w))
    rk = rk.with_columns(final=_clip(rk, "pred"))
    _write(rk, "stack_lgb_cat")

    # ---- ranking: + ADS-B blend
    rk = (rk.join(_det("2026"), on="MVT_ID_mvt", how="left")
          .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))
    rk = rk.with_columns(blend=adsb_blend.apply(rk, p)).with_columns(final=_clip(rk, "blend"))
    changed = rk.filter(adsb_blend.eligible_mask())
    print(f"ADS-B blended ranking rows: {changed.height:,} of {rk.height:,} "
          f"({changed.height / rk.height * 100:.1f}%); mean shift "
          f"{(changed['final'] - changed['pred']).mean():+.1f}s")
    with pl.Config(tbl_rows=20, tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        print(changed.group_by("ADEP_mvt").agg(
            pl.len().alias("n"), (pl.col("final") - pl.col("pred")).mean().alias("mean_shift_s"),
            (pl.col("final") - pl.col("pred")).abs().mean().alias("mean_abs_shift_s")).sort("ADEP_mvt"))
    _write(rk, "stack_lgb_cat_adsb")


if __name__ == "__main__":
    main()
