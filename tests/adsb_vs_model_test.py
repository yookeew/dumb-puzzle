"""ADS-B stand-detector pushback vs the production model, on the same rows.

2025-01-15 lies in the Jan+Jul 2025 holdout, so the production model's
holdout prediction exists for every row tests/adsb_stand_detector_test.py
recovered. Compares taxi error on those rows for: the model, ADS-B taxi
(MVT_TIME - adsb_pushback), AOBT_3 taxi, and naive combinations. Answers
whether a recovered ADS-B pushback carries information the model lacks,
before any bulk download.

Run:  .venv/Scripts/python.exe tests/adsb_stand_detector_test.py   (first)
      .venv/Scripts/python.exe tests/adsb_vs_model_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.fit import run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET  # noqa: E402

DET = ROOT / "logs" / "adsb_stand_detector.parquet"
EV_CACHE = ROOT / "cache" / "eval" / f"prod_{DEFAULT_TARGET}_holdout_ev.parquet"


def rmse(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def main() -> None:
    if EV_CACHE.exists():
        ev = pl.read_parquet(EV_CACHE)
    else:
        _, ev = run(engine="lgb", target=DEFAULT_TARGET, submit=False, seed=42)
        EV_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ev.write_parquet(EV_CACHE)
    print("ev columns:", ev.columns[:12], "...")
    base = rmse(ev["pred"] - ev["taxi"])
    print(f"production holdout RMSE (all rows): {base:.1f}s, n={ev.height:,}")

    det = pl.read_parquet(DET).with_columns(pl.col("mvt_id").cast(ev["MVT_ID_mvt"].dtype))
    j = det.join(ev.select("MVT_ID_mvt", "ADEP_mvt", "pred", "taxi"),
                 left_on="mvt_id", right_on="MVT_ID_mvt", how="inner")
    print(f"recovered rows joined to holdout: {j.height}/{det.height}")
    # taxi error of a pushback estimate = (MVT - pb) - (MVT - block) = block - pb = -diff
    j = j.with_columns(e_model=pl.col("pred") - pl.col("taxi"),
                       e_adsb=-pl.col("diff_adsb"),
                       e_aobt3=-pl.col("diff_aobt3"))
    j = j.with_columns(e_mean=(pl.col("e_model") + pl.col("e_adsb")) / 2)

    def row(df, label):
        return dict(set=label, n=df.height, model=rmse(df["e_model"]), adsb=rmse(df["e_adsb"]),
                    aobt3=rmse(df["e_aobt3"].drop_nulls()),
                    mean_model_adsb=rmse(df["e_mean"]),
                    adsb_closer_pct=float((df["e_adsb"].abs() < df["e_model"].abs()).mean() * 100),
                    model_le60=float((df["e_model"].abs() <= 60).mean() * 100),
                    adsb_le60=float((df["e_adsb"].abs() <= 60).mean() * 100))

    out = [row(j, "ALL")]
    for t in ("dwell", "appear", "pass"):
        out.append(row(j.filter(pl.col("tier") == t), t))
    for ap in sorted(j["airport"].unique()):
        out.append(row(j.filter(pl.col("airport") == ap), ap))
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, tbl_formatting="ASCII_MARKDOWN",
                   float_precision=1):
        print("\nTaxi-time RMSE (s) on ADS-B-recovered rows, 2025-01-15")
        print(pl.DataFrame(out))

    # What would a whole-holdout RMSE look like if every Jan+Jul day at these
    # airports recovered this share of rows with this error? Scale the day's
    # squared-error reduction per recovered row to the holdout.
    good = j.filter(pl.col("tier") != "pass")
    day = ev.filter(pl.col("ADEP_mvt").is_in(good["airport"].unique().to_list()))
    for label, col in (("replace with ADS-B", "e_adsb"), ("mean(model, ADS-B)", "e_mean")):
        dse_per_row = float((good[col] ** 2 - good["e_model"] ** 2).mean())
        # recovered share per covered-airport departure on the day
        share = good.height / 2629
        new_mse = (ev["pred"] - ev["taxi"]).pow(2).sum() + dse_per_row * share * day.height
        print(f"extrapolated holdout RMSE, {label} on dwell+appear rows "
              f"(share {share * 100:.1f}% of covered-airport DEP): "
              f"{base:.1f}s -> {np.sqrt(new_mse / ev.height):.1f}s")


if __name__ == "__main__":
    main()
