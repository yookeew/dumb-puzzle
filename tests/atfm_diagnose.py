"""Why did the ATFM family (build_features.py family 5) come back null?

Runs the cheap, decisive checks BEFORE acting on any story about why. Uses
saved row-level eval frames from models trained WITHOUT atfm features
(cache/lirf_ceilfix_mixed_ev.parquet = production mixed model / smart-jigsaw_v12;
cache/echoclf_baseline_direct_ev.parquet = direct-target baseline), so the
residuals here are genuine baseline residuals and no weak-proxy model is
involved -- the exact failure mode that misled PROGRESS.md S19.

  check 2  residual gradient vs ATFM deciles (ADEP and ADES) -- DECISIVE.
           No gradient => no information to extract, close it out.
           Gradient the model missed => encoding problem, not information.
  check 3  Jan vs Jul data quality, 2025 (the holdout months, which produced
           the observed split) and 2026 (the ranking months, deployment risk).
  check 4  destination join match rate + whether ADES features actually vary
           flight-to-flight WITHIN an (airport, day) -- the premise of the
           whole dual-join design.
  check 5  variance decomposition of taxi: between (airport, day) vs within.
           A day-level feature can only ever address the between component.
  reframe  RMSE restricted to high-regulation slices -- a feature that only
           matters for a minority of flights can be invisible in the aggregate.

Run:  .venv/Scripts/python.exe tests/atfm_diagnose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.atfm import load_atfm_daily  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"
BASE_EV = ROOT / "cache" / "echoclf_baseline_direct_ev.parquet"

ATFM_CTX = ["reg_share", "in_slot_share", "late_share", "dly_min_per_flight",
            "dly_weather_share", "dly_staffing_share", "traffic_dep",
            "traffic_arr", "traffic_tot"]

RULE = "=" * 78


def _holdout_atfm() -> pl.DataFrame:
    """Holdout rows with their atfm columns + day, keyed MVT_ID_mvt."""
    cols = ["MVT_ID_mvt", "ades_in_panel", "T", "ADEP_mvt"] + \
        [f"dep_atfm_{c}" for c in ATFM_CTX] + [f"des_atfm_{c}" for c in ATFM_CTX]
    f = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet")
    have = [c for c in cols if c in f.columns]
    return f.select(have).with_columns(day=pl.col("T").dt.date()).drop("T")


def _decile_gradient(ev: pl.DataFrame, col: str, label: str) -> None:
    """Mean/abs residual and RMSE by decile of `col`. A gradient in mean
    residual is systematic bias the model failed to capture."""
    d = ev.filter(pl.col(col).is_not_null())
    if d.height < 1000:
        print(f"  {label}: only {d.height} non-null rows, skipped")
        return
    d = d.with_columns(
        bucket=((pl.col(col).rank("ordinal") - 1) * 10 // pl.len()).alias("bucket")
    )
    g = (
        d.group_by("bucket")
        .agg(
            pl.col(col).min().alias("lo"),
            pl.col(col).max().alias("hi"),
            pl.col("resid").mean().alias("mean_resid"),
            pl.col("resid").abs().mean().alias("mae"),
            pl.col("resid").pow(2).mean().sqrt().alias("rmse"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )
    print(f"\n  {label}  (n={d.height:,})")
    print(f"    {'dec':<4}{'range':<22}{'mean_resid':>12}{'MAE':>10}{'RMSE':>10}{'n':>10}")
    for r in g.iter_rows(named=True):
        rng = f"[{r['lo']:.4g}, {r['hi']:.4g}]"
        print(f"    d{r['bucket']:<3}{rng:<22}{r['mean_resid']:>12.1f}"
              f"{r['mae']:>10.1f}{r['rmse']:>10.1f}{r['n']:>10,}")
    spread = g["mean_resid"].max() - g["mean_resid"].min()
    print(f"    -> mean-resid spread across deciles: {spread:.1f}s")


def check2(ev_prod: pl.DataFrame, ev_base: pl.DataFrame) -> None:
    print(f"\n{RULE}\nCHECK 2 (DECISIVE) -- baseline residual gradient vs ATFM deciles")
    print("resid = taxi - pred (positive = model underpredicted)")
    print(RULE)
    for name, ev in [("PRODUCTION model (mixed, smart-jigsaw_v12)", ev_prod),
                     ("BASELINE model (direct target)", ev_base)]:
        print(f"\n--- {name} ---")
        for col, lab in [
            ("dep_atfm_reg_share", "ADEP regulated-departure share"),
            ("des_atfm_reg_share", "ADES regulated-departure share"),
            ("dep_atfm_dly_min_per_flight", "ADEP arrival ATFM delay min/flight"),
            ("des_atfm_dly_min_per_flight", "ADES arrival ATFM delay min/flight"),
            ("dep_atfm_dly_weather_share", "ADEP weather share of delay"),
            ("dep_atfm_dly_staffing_share", "ADEP staffing share of delay"),
        ]:
            _decile_gradient(ev, col, lab)


def check3() -> None:
    print(f"\n{RULE}\nCHECK 3 -- Jan vs Jul data quality (2025 = holdout, 2026 = ranking)")
    print(RULE)
    atfm = load_atfm_daily()
    a = atfm.with_columns(
        year=pl.col("date").dt.year(), month=pl.col("date").dt.month()
    ).filter(pl.col("month").is_in([1, 7]))
    for year in (2025, 2026):
        sub = a.filter(pl.col("year") == year)
        if sub.height == 0:
            print(f"\n  {year}: NO ROWS")
            continue
        print(f"\n  --- {year} ---")
        g = sub.group_by("month").agg(
            pl.len().alias("rows"),
            pl.col("APT_ICAO").n_unique().alias("airports"),
            *[pl.col(c).is_not_null().mean().alias(f"nn_{c}")
              for c in ["reg_share", "dly_min_per_flight", "traffic_tot"]],
            pl.col("reg_share").mean().alias("reg_share_mean"),
            pl.col("dly_min_per_flight").mean().alias("dly_mean"),
            pl.col("traffic_tot").mean().alias("traffic_mean"),
        ).sort("month")
        for r in g.iter_rows(named=True):
            mon = "Jan" if r["month"] == 1 else "Jul"
            print(f"    {mon}: rows={r['rows']:,}  airports={r['airports']}  "
                  f"non-null: reg={r['nn_reg_share']:.3f} "
                  f"dly={r['nn_dly_min_per_flight']:.3f} "
                  f"traffic={r['nn_traffic_tot']:.3f}")
            print(f"         means: reg_share={r['reg_share_mean']:.4f}  "
                  f"dly/flt={r['dly_mean']:.3f}  traffic={r['traffic_mean']:.1f}")


def check4(ho: pl.DataFrame) -> None:
    print(f"\n{RULE}\nCHECK 4 -- destination join rate + within-(airport,day) ADES variation")
    print(RULE)
    rank = pl.read_parquet(FEAT_DIR / "ranking.parquet")
    for nm, df in [("holdout (Jan+Jul 2025)", ho), ("ranking (Jan+Jul 2026)", rank)]:
        if "des_atfm_reg_share" not in df.columns:
            continue
        rate = df.select(pl.col("des_atfm_reg_share").is_not_null().mean()).item()
        panel = df.select(pl.col("ades_in_panel").fill_null(0).mean()).item()
        print(f"\n  {nm}: ADES join rate={rate:.3f}   ades_in_panel={panel:.3f}  "
              f"n={df.height:,}")

    # the premise of the dual join: does ADES context vary flight-to-flight
    # within one airport-day, or is it effectively constant like the ADEP side?
    print("\n  within-(ADEP, day) variation of ADES-side features (holdout):")
    for col in ["des_atfm_reg_share", "des_atfm_dly_min_per_flight"]:
        g = ho.filter(pl.col(col).is_not_null()).group_by("ADEP_mvt", "day").agg(
            pl.col(col).std().alias("sd"), pl.col(col).mean().alias("mu"),
            pl.len().alias("n")
        ).filter(pl.col("n") >= 20)
        overall_sd = ho.select(pl.col(col).std()).item()
        print(f"    {col}: median within-day sd={g['sd'].median():.4f}  "
              f"overall sd={overall_sd:.4f}  "
              f"ratio={g['sd'].median() / overall_sd:.2f}  ({g.height} airport-days)")


def check5(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\nCHECK 5 (STRUCTURAL) -- where does taxi variance actually live?")
    print(RULE)
    g = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("taxi").mean().alias("cell_mean"), pl.len().alias("n")
    )
    grand = ev.select(pl.col("taxi").mean()).item()
    total_var = ev.select(pl.col("taxi").var()).item()
    between = g.select(
        ((pl.col("n") * (pl.col("cell_mean") - grand).pow(2)).sum() / pl.col("n").sum())
    ).item()
    print(f"\n  total taxi variance:                {total_var:,.0f}")
    print(f"  between (airport, day) cells:       {between:,.0f}  "
          f"({between / total_var * 100:.1f}%)")
    print(f"  within  (airport, day) cells:       {total_var - between:,.0f}  "
          f"({(total_var - between) / total_var * 100:.1f}%)")
    print("\n  A day-level feature can only address the BETWEEN component.")

    # same decomposition on the production model's residuals -- the part the
    # model has NOT already explained is what a new feature could still win.
    gr = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("resid").mean().alias("cell_mean"), pl.len().alias("n")
    )
    grand_r = ev.select(pl.col("resid").mean()).item()
    var_r = ev.select(pl.col("resid").var()).item()
    between_r = gr.select(
        ((pl.col("n") * (pl.col("cell_mean") - grand_r).pow(2)).sum() / pl.col("n").sum())
    ).item()
    print(f"\n  production RESIDUAL variance:       {var_r:,.0f}")
    print(f"  between (airport, day) cells:       {between_r:,.0f}  "
          f"({between_r / var_r * 100:.1f}%)   <- the actual addressable ceiling")
    print(f"  within  (airport, day) cells:       {var_r - between_r:,.0f}  "
          f"({(var_r - between_r) / var_r * 100:.1f}%)")
    ceiling = var_r ** 0.5 - (var_r - between_r) ** 0.5
    print(f"\n  Perfectly explaining the between-day residual component would cut")
    print(f"  RMSE by at most ~{ceiling:.1f}s (from {var_r ** 0.5:.1f} to "
          f"{(var_r - between_r) ** 0.5:.1f}).")


def reframe(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\nREFRAME -- is the feature working on the minority it should affect?")
    print(RULE)
    print("\n  production-model RMSE on high- vs low-regulation slices:")
    for col, lab in [("dep_atfm_reg_share", "ADEP regulated share"),
                     ("des_atfm_reg_share", "ADES regulated share"),
                     ("dep_atfm_dly_min_per_flight", "ADEP delay min/flight")]:
        d = ev.filter(pl.col(col).is_not_null())
        if d.height < 1000:
            continue
        q90 = d.select(pl.col(col).quantile(0.9)).item()
        hi = d.filter(pl.col(col) >= q90)
        lo = d.filter(pl.col(col) < q90)
        print(f"    {lab:<28} top decile (>={q90:.4g}): "
              f"rmse={hi.select(pl.col('resid').pow(2).mean().sqrt()).item():7.1f} "
              f"n={hi.height:,}   |   rest: "
              f"rmse={lo.select(pl.col('resid').pow(2).mean().sqrt()).item():7.1f} "
              f"n={lo.height:,}")


def main() -> None:
    ho = _holdout_atfm()
    # ev frames already carry ADEP_mvt -- drop the duplicate to avoid a suffix
    ho_j = ho.drop("ADEP_mvt")
    prod = pl.read_parquet(PROD_EV).join(ho_j, on="MVT_ID_mvt", how="left")
    base = pl.read_parquet(BASE_EV).join(ho_j, on="MVT_ID_mvt", how="left")
    for df in (prod, base):
        assert "dep_atfm_reg_share" in df.columns, "atfm columns missing from holdout frame"
    prod = prod.with_columns(resid=pl.col("taxi") - pl.col("pred"))
    base = base.with_columns(resid=pl.col("taxi") - pl.col("pred"))
    print(f"joined: production n={prod.height:,}  baseline n={base.height:,}")

    check2(prod, base)
    check3()
    check4(ho)
    check5(prod)
    reframe(prod)


if __name__ == "__main__":
    main()
