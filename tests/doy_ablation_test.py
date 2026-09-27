"""Follow-up to the S26 weather ablation: is the benign-weather arm's gain
(temp/dewpoint/wind, -1.04s overall, -3.3s LIRF) a real weather effect, or a
smoother proxy for the seasonal signal our existing month/hour categoricals
encode coarsely?

Two cyclical calendar features, built from the movement timestamp alone (no
external data):
  doy_sin  = sin(2*pi*doy/365.25)
  doy_cos  = cos(2*pi*doy/365.25)   (sin/cos so 31 Dec and 1 Jan are adjacent)
  hour_sin/hour_cos -- same idea for minute-of-day, the optional diurnal
  counterpart (existing `hour` is already a 24-level integer/categorical, so
  this is a much smaller effect than doy_sin/cos, but cheap to include).

NOTE: `minute_of_day` (build_features.py, family "calendar") is silently
wrong -- `pl.col("T").dt.hour() * 60 + pl.col("T").dt.minute()` computes in
Int8 arithmetic and wraps for every hour >= 3 (e.g. hour=20 -> -80, not
1200). Confirmed by direct repro, not used here -- hour_sin/cos below are
computed fresh from `T` with an explicit Int32 cast. Flagging in
PROGRESS.md/report separately; not fixed in this script since it's a
production feature file and out of scope for this ablation.

Two-step test, both under target="mixed" (S25: mixed is the correct
instrument for feature testing, not direct -- see tests/_harness.py):

  step 1: baseline (no weather, no doy) vs baseline + doy
          -- does calendar resolution help on its own?
  step 2: baseline + doy vs baseline + doy + benign weather
          -- does the benign arm's S26 gain survive once smooth seasonality
             is already in the model?

Reading: if the benign gain mostly disappears in step 2, it was a seasonal
proxy. If it holds, temp/dewpoint/wind carry something calendar can't.

Caveat: training is 2025-only, holdout is Jan+Jul 2025, so day-of-year is
fully covered there -- but ranking is Jan+Jul 2026, the same days of year,
so doy_sin/cos face no extrapolation risk at inference time either.

Same discipline as tests/metar_ablation_test.py: paired per-row squared
errors on identical holdout rows, (airport, day) cluster bootstrap, both
months reported separately, LIRF reported specifically.

Run:  .venv/Scripts/python.exe tests/doy_ablation_test.py
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET, cluster_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
BASE_DIR = ROOT / "cache" / "features_doy_base_tmp"
DOY_DIR = ROOT / "cache" / "features_doy_doy_tmp"
DOYWX_DIR = ROOT / "cache" / "features_doy_doywx_tmp"

# Same partition as tests/metar_ablation_test.py (S26).
ADVERSE_COLS = [
    "has_ceiling", "vis_mi", "ceiling_ft", "below_freezing", "deicing_risk",
    "wx_snow", "wx_freezing", "wx_tstorm", "wx_obscuration", "flight_category",
    "precip_1h_in",
]
BENIGN_COLS = ["temp_c", "dewpoint_c", "temp_dewpoint_spread_c", "wind_dir", "wind_kt", "gust_kt"]
ALL_WX_COLS = ADVERSE_COLS + BENIGN_COLS

MODEL_SEED = 42
N_RESAMPLES = 3000
BOOT_SEED = 0

TWO_PI = 2.0 * math.pi
DOY_YEAR = 365.25


def _add_doy(df: pl.DataFrame) -> pl.DataFrame:
    minute_of_day_safe = pl.col("T").dt.hour().cast(pl.Int32) * 60 + pl.col("T").dt.minute()
    return df.with_columns(
        doy_sin=(pl.col("doy").cast(pl.Float32) * (TWO_PI / DOY_YEAR)).sin().cast(pl.Float32),
        doy_cos=(pl.col("doy").cast(pl.Float32) * (TWO_PI / DOY_YEAR)).cos().cast(pl.Float32),
        hour_sin=(minute_of_day_safe.cast(pl.Float32) * (TWO_PI / 1440.0)).sin().cast(pl.Float32),
        hour_cos=(minute_of_day_safe.cast(pl.Float32) * (TWO_PI / 1440.0)).cos().cast(pl.Float32),
    )


def _prep(dest: Path, keep_wx: list[str], add_doy: bool) -> None:
    """dest gets FEAT_DIR with (ALL_WX_COLS - keep_wx) dropped, and doy/hour
    cyclical features added on top if add_doy."""
    drop = [c for c in ALL_WX_COLS if c not in keep_wx]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in ("train2025.parquet", "holdout_gap2025.parquet"):
        df = pl.read_parquet(FEAT_DIR / f)
        df = df.drop([c for c in drop if c in df.columns])
        if add_doy:
            df = _add_doy(df)
        df.write_parquet(dest / f)
    shutil.copy(FEAT_DIR / "labels2025.parquet", dest / "labels2025.parquet")


def _fit(dest: Path, name: str, target: str, seed: int):
    return run(engine="lgb", feat_dir=dest, name=name, target=target,
               submit=False, seed=seed)


def _report(label: str, ev_a: pl.DataFrame, ev_b: pl.DataFrame, tag: str) -> pl.DataFrame:
    """ev_a = baseline (control), ev_b = treatment. Returns the paired ev
    frame with se_base/se_treat for further use."""
    ev = (
        ev_a.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt", pred_base="pred")
        .join(ev_b.select("MVT_ID_mvt", pred_treat="pred"), on="MVT_ID_mvt", how="inner")
    )
    assert ev.height == ev_a.height == ev_b.height, f"row mismatch ({tag})"
    day = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = ev.join(day, on="MVT_ID_mvt", how="left").with_columns(
        se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
        se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2),
    )

    rb, rt = ev["se_base"].mean() ** 0.5, ev["se_treat"].mean() ** 0.5
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"overall RMSE   baseline={rb:.1f}  treatment={rt:.1f}  delta={rt - rb:+.1f}")
    for ym in HOLDOUT_MONTHS:
        s = ev.filter(pl.col("ym") == ym)
        a, b = s["se_base"].mean() ** 0.5, s["se_treat"].mean() ** 0.5
        print(f"  {ym}: baseline={a:.1f}  treatment={b:.1f}  delta={b - a:+.1f}")

    lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF")
    la, lb = lirf["se_base"].mean() ** 0.5, lirf["se_treat"].mean() ** 0.5
    print(f"  LIRF: baseline={la:.1f}  treatment={lb:.1f}  delta={lb - la:+.1f}  n={lirf.height:,}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("sb"), pl.col("se_treat").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, pw = cluster_bootstrap(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                                         cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED)
    print(f"cluster bootstrap: paired RMSE delta={point:+.2f}s  95% CI=[{lo:+.2f}, {hi:+.2f}]  "
          f"P(worse)={pw:.3f}")
    return ev


def main(target: str = DEFAULT_TARGET, seed: int = MODEL_SEED) -> None:
    print(f"=== building 3 arms (target={target}, seed={seed}) ===")
    _prep(BASE_DIR, keep_wx=[], add_doy=False)
    _prep(DOY_DIR, keep_wx=[], add_doy=True)
    _prep(DOYWX_DIR, keep_wx=BENIGN_COLS, add_doy=True)

    print("\n=== fitting baseline (no weather, no doy) ===")
    _, ev_base = _fit(BASE_DIR, "doy_base", target, seed)

    print("\n=== fitting baseline + doy/hour cyclical ===")
    _, ev_doy = _fit(DOY_DIR, "doy_doy", target, seed)

    print("\n=== fitting baseline + doy + benign weather ===")
    _, ev_doywx = _fit(DOYWX_DIR, "doy_doywx", target, seed)

    ev1 = _report("STEP 1: baseline+doy vs baseline (does calendar resolution help alone?)",
                   ev_base, ev_doy, "step1")
    ev2 = _report("STEP 2: baseline+doy+benign-weather vs baseline+doy "
                   "(does benign weather's S26 gain survive?)",
                   ev_doy, ev_doywx, "step2")

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    d1 = ev1["se_treat"].mean() ** 0.5 - ev1["se_base"].mean() ** 0.5
    d2 = ev2["se_treat"].mean() ** 0.5 - ev2["se_base"].mean() ** 0.5
    print(f"step 1 (doy alone):              {d1:+.2f}s")
    print(f"step 2 (benign wx given doy):     {d2:+.2f}s")
    print("reference (S26, no doy in either arm): benign-only vs no-weather = -1.04s "
          "overall, -3.3s LIRF")
    if abs(d2) < 0.3 * 1.04:
        print("-> benign gain mostly disappears once doy is present: consistent with "
              "a seasonal proxy, not a genuine weather effect.")
    elif abs(d2) > 0.7 * 1.04:
        print("-> benign gain survives doy: temp/dewpoint/wind carry something "
              "calendar resolution can't.")
    else:
        print("-> partial: some of the benign gain is seasonal proxy, some survives.")


if __name__ == "__main__":
    _t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    _s = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_SEED
    main(target=_t, seed=_s)
