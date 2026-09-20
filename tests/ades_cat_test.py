"""Test ADES_mvt (destination identity) as a plain categorical feature.

Surfaced by the S20 ceiling work: ADES_mvt was never a model feature at all
(CAT_COLS had ADEP_mvt only), yet much of the (airport, day, ADES) oracle
bound looked like destination identity rather than ATFM values. Destination
sets the SID/departure route, hence runway exit and holding point, so there
is a physical story -- but unlike family 5 this one can't be bounded away in
advance, so it gets a real test.

The ATFM family (5) is dropped from BOTH arms, since S20 did not adopt it.
This isolates ADES_mvt exactly:

  baseline  = production feature set (no atfm, no ADES_mvt)
  treatment = production feature set + ADES_mvt

Same discipline as tests/atfm_v1_test.py: paired per-row squared errors on
identical holdout rows, (airport, day) cluster bootstrap, both months
reported separately, primary metric pre-registered as the paired overall
RMSE delta.

Run:  .venv/Scripts/python.exe tests/ades_cat_test.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
BASE_DIR = ROOT / "cache" / "features_ades_base_tmp"
TREAT_DIR = ROOT / "cache" / "features_ades_treat_tmp"

ATFM_CTX = ["reg_share", "in_slot_share", "late_share", "dly_min_per_flight",
            "dly_weather_share", "dly_staffing_share", "traffic_dep",
            "traffic_arr", "traffic_tot"]
ATFM_COLS = ["ades_in_panel"] + [f"dep_atfm_{c}" for c in ATFM_CTX] + \
    [f"des_atfm_{c}" for c in ATFM_CTX]

N_RESAMPLES = 3000
MODEL_SEED = 42
BOOT_SEED = 0


def _prep(dest: Path, drop: list[str]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in ("train2025.parquet", "holdout_gap2025.parquet"):
        df = pl.read_parquet(FEAT_DIR / f)
        df.drop([c for c in drop if c in df.columns]).write_parquet(dest / f)
    shutil.copy(FEAT_DIR / "labels2025.parquet", dest / "labels2025.parquet")


def _cluster_bootstrap(se_base_sum, se_treat_sum, n_by_cluster, n_resamples, seed):
    rng = np.random.default_rng(seed)
    k = len(se_base_sum)
    n_tot = n_by_cluster.sum()
    point = (se_treat_sum.sum() / n_tot) ** 0.5 - (se_base_sum.sum() / n_tot) ** 0.5
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, k, k)
        n_b = n_by_cluster[idx].sum()
        boots[b] = (se_treat_sum[idx].sum() / n_b) ** 0.5 - \
            (se_base_sum[idx].sum() / n_b) ** 0.5
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, float((boots > 0).mean())


def main() -> None:
    print("=== baseline (no atfm, no ADES_mvt) ===")
    _prep(BASE_DIR, ATFM_COLS + ["ADES_mvt"])
    _, ev_base = run(engine="lgb", feat_dir=BASE_DIR, name="ades_cat_baseline",
                     target="direct", submit=False, seed=MODEL_SEED)

    print("\n=== treatment (+ ADES_mvt categorical) ===")
    _prep(TREAT_DIR, ATFM_COLS)
    _, ev_treat = run(engine="lgb", feat_dir=TREAT_DIR, name="ades_cat_treatment",
                      target="direct", submit=False, seed=MODEL_SEED)

    ev = (
        ev_base.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt", pred_base="pred")
        .join(ev_treat.select("MVT_ID_mvt", pred_treat="pred"),
              on="MVT_ID_mvt", how="inner")
    )
    assert ev.height == ev_base.height == ev_treat.height, "row mismatch"

    day = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = ev.join(day, on="MVT_ID_mvt", how="left").with_columns(
        se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
        se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2),
    )

    rb = ev["se_base"].mean() ** 0.5
    rt = ev["se_treat"].mean() ** 0.5
    print(f"\noverall RMSE   baseline={rb:.1f}  treatment={rt:.1f}  delta={rt - rb:+.1f}")
    for ym in HOLDOUT_MONTHS:
        s = ev.filter(pl.col("ym") == ym)
        a, b = s["se_base"].mean() ** 0.5, s["se_treat"].mean() ** 0.5
        print(f"  {ym}: baseline={a:.1f}  treatment={b:.1f}  delta={b - a:+.1f}")

    print("\nper airport:")
    for r in (ev.group_by("ADEP_mvt").agg(
            pl.col("se_base").mean().sqrt().alias("rb"),
            pl.col("se_treat").mean().sqrt().alias("rt"),
            pl.len().alias("n")).sort("ADEP_mvt").iter_rows(named=True)):
        print(f"  {r['ADEP_mvt']:<6} base={r['rb']:7.1f}  treat={r['rt']:7.1f}  "
              f"delta={r['rt'] - r['rb']:+7.1f}  n={r['n']:,}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("sb"),
        pl.col("se_treat").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, p_worse = _cluster_bootstrap(
        cl["sb"].to_numpy(), cl["st"].to_numpy(), cl["n"].to_numpy(),
        N_RESAMPLES, BOOT_SEED)
    print(f"\ncluster bootstrap (airport, day), {N_RESAMPLES} resamples, "
          f"{cl.height} clusters:")
    print(f"  paired RMSE delta (treatment - baseline): {point:+.2f}s")
    print(f"  95% CI: [{lo:+.2f}, {hi:+.2f}]    P(worse) = {p_worse:.3f}")


if __name__ == "__main__":
    main()
