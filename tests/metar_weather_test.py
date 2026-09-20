"""Pre-registered test of family 7 (METAR weather/de-icing context,
build_features.py / features/weather.py) against a fresh baseline. Same
discipline as tests/atfm_v1_test.py (PROGRESS.md S14/S16/S19/S20):

  PRIMARY METRIC:  paired overall RMSE delta (treatment - baseline) on the
                    Jan+Jul 2025 holdout. target defaults to "direct" (fast,
                    day-to-day signal); pass "mixed" on the command line to
                    re-check under the actual production target, where LIRF
                    uses flip (908s->660.6s post-S12, see PROGRESS.md) instead
                    of direct's 908s -- the airport that absorbed all of
                    direct's weather-collateral damage (S24) is far more
                    robust under mixed, so the "not adopted" verdict from a
                    direct-only test may not hold at the production config.
  UNCERTAINTY:      (airport, day) cluster bootstrap, 3000 resamples.
  BOTH-MONTHS RULE: adopt only if both 2025-01 and 2025-07 improve, not just
                    the pooled number (contestants' lesson, PROGRESS.md S5).
  MECHANISM TEST:   the pre-registered condition specific to this feature --
                    the gain must CONCENTRATE on low-visibility/freezing
                    conditions (flight_category != VFR, or below_freezing).
                    A uniform gain across conditions means the feature isn't
                    measuring weather, the same logic that rejected the
                    queue feature (S23, no gain gradient with queue depth).

Baseline vs treatment is built by adding the 16 weather columns onto the
already-built (weather-less) cache/features/*.parquet directly -- the
weather join is a pure left-join on (ADEP_mvt, hour bucket of T) that adds
columns without touching anything else, same efficiency trick atfm_v1_test.py
used in the opposite direction (there: drop columns from an included cache;
here: add columns to an excluded cache).

Run:  .venv/Scripts/python.exe tests/metar_weather_test.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.weather import load_metar_hourly  # noqa: E402
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET, cluster_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
WX_DIR = ROOT / "cache" / "features_metar_tmp"

WX_COLS = [
    "has_ceiling", "vis_mi", "ceiling_ft", "temp_c", "dewpoint_c",
    "temp_dewpoint_spread_c", "below_freezing", "deicing_risk",
    "wx_snow", "wx_freezing", "wx_tstorm", "wx_obscuration",
    "flight_category", "wind_dir", "wind_kt", "gust_kt", "precip_1h_in",
]

N_RESAMPLES = 3000
MODEL_SEED = 42
BOOT_SEED = 0


def _add_weather(df: pl.DataFrame, wx: pl.DataFrame) -> pl.DataFrame:
    d = df.with_columns(hour=pl.col("T").dt.truncate("1h")).join(
        wx, left_on=["ADEP_mvt", "hour"], right_on=["station", "hour"], how="left",
    ).with_columns(
        has_ceiling=pl.col("ceiling_ft").is_not_null().cast(pl.Int8),
        ceiling_ft=pl.col("ceiling_ft").fill_null(99999.0),
    )
    # explicit null-rate check right after the join (PROGRESS.md S5 -- a
    # tz-aware-vs-naive mismatch silently nulled a weather feature for 19
    # submissions on another team; fail loud instead of fitting on nulls).
    nr = d.select(pl.col("vis_mi").is_null().mean()).item()
    print(f"    vis_mi null rate after join: {nr:.4%}  (n={d.height:,})")
    assert nr < 0.02, f"weather join null rate {nr:.2%} looks like a broken join, not missing data"
    return d.drop("hour")


def _make_wx_dir() -> None:
    if WX_DIR.exists():
        shutil.rmtree(WX_DIR)
    WX_DIR.mkdir(parents=True)
    wx = load_metar_hourly()
    for f in ("train2025.parquet", "holdout_gap2025.parquet", "ranking.parquet"):
        print(f"  {f}:")
        df = pl.read_parquet(FEAT_DIR / f)
        _add_weather(df, wx).write_parquet(WX_DIR / f)
    shutil.copy(FEAT_DIR / "labels2025.parquet", WX_DIR / "labels2025.parquet")


def _cluster_key(ev: pl.DataFrame) -> pl.DataFrame:
    t = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date()
    )
    return ev.join(t, on="MVT_ID_mvt", how="left")


def _weather_regime(ev: pl.DataFrame) -> pl.DataFrame:
    """Pull the per-row weather regime flags used by the mechanism test,
    straight from the treatment feature frame (not re-derived) so this
    matches exactly what the model saw."""
    wx = pl.read_parquet(WX_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", "flight_category", "below_freezing", "deicing_risk",
    )
    return ev.join(wx, on="MVT_ID_mvt", how="left")


def main(target: str = DEFAULT_TARGET, seed: int = MODEL_SEED) -> None:
    suffix = f"_{target}" if target != "direct" else ""
    suffix += f"_seed{seed}" if seed != MODEL_SEED else ""
    print(f"=== building treatment feature dir (weather columns added), "
          f"target={target}, seed={seed} ===")
    _make_wx_dir()

    print("\n=== baseline (no weather) ===")
    _, ev_base = run(engine="lgb", feat_dir=FEAT_DIR, name=f"metar_baseline{suffix}",
                      target=target, submit=False, seed=seed)

    print("\n=== treatment (weather family 7 included) ===")
    _, ev_treat = run(engine="lgb", feat_dir=WX_DIR, name=f"metar_treatment{suffix}",
                       target=target, submit=False, seed=seed)

    base = ev_base.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt", pred_base="pred")
    treat = ev_treat.select("MVT_ID_mvt", pred_treat="pred")
    ev = base.join(treat, on="MVT_ID_mvt", how="inner")
    assert ev.height == ev_base.height == ev_treat.height, "row mismatch between runs"

    ev = ev.with_columns(
        se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
        se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2),
    )
    ev = _cluster_key(ev)

    rmse_base = ev["se_base"].mean() ** 0.5
    rmse_treat = ev["se_treat"].mean() ** 0.5
    print(f"\noverall RMSE   baseline={rmse_base:.1f}  treatment={rmse_treat:.1f}  "
          f"delta={rmse_treat - rmse_base:+.1f}")

    for ym in HOLDOUT_MONTHS:
        sub = ev.filter(pl.col("ym") == ym)
        rb, rt = sub["se_base"].mean() ** 0.5, sub["se_treat"].mean() ** 0.5
        print(f"  {ym}: baseline={rb:.1f}  treatment={rt:.1f}  delta={rt - rb:+.1f}")

    print("\nper airport (treatment - baseline RMSE):")
    for r in (
        ev.group_by("ADEP_mvt")
        .agg(pl.col("se_base").mean().sqrt().alias("rb"),
             pl.col("se_treat").mean().sqrt().alias("rt"),
             pl.len().alias("n"))
        .sort("ADEP_mvt")
        .iter_rows(named=True)
    ):
        print(f"  {r['ADEP_mvt']:<6} base={r['rb']:7.1f}  treat={r['rt']:7.1f}  "
              f"delta={r['rt'] - r['rb']:+7.1f}  n={r['n']:,}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("se_base_sum"),
        pl.col("se_treat").sum().alias("se_treat_sum"),
        pl.len().alias("n"),
    )
    point, lo, hi, p_worse = cluster_bootstrap(
        cl["se_base_sum"].to_numpy(), cl["se_treat_sum"].to_numpy(),
        cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED,
    )
    print(f"\ncluster bootstrap (airport, day), {N_RESAMPLES} resamples, "
          f"{cl.height} clusters:")
    print(f"  paired RMSE delta (treatment - baseline): {point:+.2f}s")
    print(f"  95% CI: [{lo:+.2f}, {hi:+.2f}]")
    print(f"  P(worse) = {p_worse:.3f}")

    print(f"\n{'=' * 78}\nMECHANISM TEST -- pre-registered: gain must concentrate on "
          f"low-vis / freezing conditions\n{'=' * 78}")
    print("Each row also reports its SHARE OF THE TOTAL REALIZED GAIN (not an\n"
          "isolated hypothetical) -- for the three complementary pairs below\n"
          "(low_vis/VFR, freezing, deicing) each pair sums to ~100% by\n"
          "construction, since they partition all rows. A pre-registered\n"
          "mechanism claim needs the ADVERSE side to dominate this share, not\n"
          "just have the larger per-row delta -- a huge per-row effect on a\n"
          "tiny population can still lose the share to a tiny per-row effect\n"
          "on the 80-99% majority. Report both; don't call it a clean pass\n"
          "on the per-row delta alone (PROGRESS.md S24/S25).")
    evr = _weather_regime(ev)
    evr = evr.with_columns(
        low_vis=(pl.col("flight_category") != "VFR").fill_null(False),
    )
    total_mse_delta = rmse_treat**2 - rmse_base**2
    for label, mask_expr in [
        ("low_vis (IFR/MVFR/LIFR)", pl.col("low_vis")),
        ("VFR (ordinary)", ~pl.col("low_vis")),
        ("below_freezing", pl.col("below_freezing").fill_null(False)),
        ("above freezing", ~pl.col("below_freezing").fill_null(False)),
        ("deicing_risk", pl.col("deicing_risk").fill_null(False)),
        ("no deicing_risk", ~pl.col("deicing_risk").fill_null(False)),
    ]:
        sub = evr.filter(mask_expr)
        if sub.height < 200:
            print(f"  {label:<26} n={sub.height:<8} (too small, skipped)")
            continue
        rb, rt = sub["se_base"].mean() ** 0.5, sub["se_treat"].mean() ** 0.5
        contrib = sub.height * (rt**2 - rb**2) / ev.height
        share = contrib / total_mse_delta if total_mse_delta != 0 else float("nan")
        print(f"  {label:<26} n={sub.height:<8} rows={sub.height / ev.height:6.1%}  "
              f"base={rb:7.1f}  treat={rt:7.1f}  delta={rt - rb:+7.1f}  "
              f"share_of_total_gain={share:6.1%}")

    print("\nA uniform-looking per-row delta (both sides negative/positive by a "
          "similar\namount) means the gain isn't concentrated and shouldn't be "
          "read as weather\nsignal, same logic that rejected the queue feature "
          "(S23). But even a correctly\nconcentrated per-row delta can still "
          "have most of its AGGREGATE share come from\nthe majority population "
          "-- that's not a mechanism failure, just an honest\naccounting of "
          "where the realized RMSE gain actually comes from.")


if __name__ == "__main__":
    _t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    _s = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_SEED
    main(target=_t, seed=_s)
