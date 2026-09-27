"""Test ADES_mvt (destination identity) as a plain categorical feature.

Surfaced by the S20 ceiling work: ADES_mvt was never a model feature at all
(CAT_COLS had ADEP_mvt only), yet much of the (airport, day, ADES) oracle
bound looked like destination identity rather than ATFM values. Destination
sets the SID/departure route, hence runway exit and holding point, so there
is a physical story -- but unlike family 5 this one can't be bounded away in
advance, so it gets a real test.

  baseline  = production feature set (FEAT_DIR as-is: weather on, atfm off)
  treatment = production feature set + ADES_mvt, added by joining the raw
              column onto a freshly rebuilt feature frame and temporarily
              registering it in encode.CAT_COLS for this run only (restored
              after, production encode.py is never touched on disk).

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

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features.encode as encode  # noqa: E402
from features.build_features import build_features  # noqa: E402
from features.export_model_inputs import FRAME_COLS, HOLDOUT_MONTHS as _HO, TRAIN_GLOB, blind  # noqa: E402
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET, cluster_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
TREAT_DIR = ROOT / "cache" / "features_ades_treat_tmp"

N_RESAMPLES = 3000
MODEL_SEED = 42
BOOT_SEED = 0


def _make_treat_dir() -> None:
    """Rebuild train2025/holdout_gap2025 with the current production flags
    (weather=True as of S24/S25, atfm=False) plus a raw ADES_mvt passthrough
    column joined on afterward -- build_features() itself never selects
    ADES_mvt (S21 removed it), so it has to be added here, not toggled."""
    if TREAT_DIR.exists():
        shutil.rmtree(TREAT_DIR)
    TREAT_DIR.mkdir(parents=True)

    frame = pl.read_parquet(TRAIN_GLOB, columns=FRAME_COLS)
    ades = frame.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "ADES_mvt")

    build_features(blind(frame), priors=None, weather=True) \
        .join(ades, on="MVT_ID_mvt", how="left") \
        .write_parquet(TREAT_DIR / "train2025.parquet")

    frame_ho = frame.filter(
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").is_in(_HO)
    )
    ades_ho = frame_ho.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "ADES_mvt")
    build_features(blind(frame_ho), priors=None, weather=True) \
        .join(ades_ho, on="MVT_ID_mvt", how="left") \
        .write_parquet(TREAT_DIR / "holdout_gap2025.parquet")

    shutil.copy(FEAT_DIR / "labels2025.parquet", TREAT_DIR / "labels2025.parquet")


def main(target: str = DEFAULT_TARGET) -> None:
    suffix = f"_{target}" if target != "direct" else ""
    print(f"=== baseline (production cache, no ADES_mvt), target={target} ===")
    _, ev_base = run(engine="lgb", feat_dir=FEAT_DIR, name=f"ades_cat_baseline{suffix}",
                     target=target, submit=False, seed=MODEL_SEED)

    print("\n=== treatment (+ ADES_mvt categorical) ===")
    _make_treat_dir()
    original_cat_cols = list(encode.CAT_COLS)
    encode.CAT_COLS.append("ADES_mvt")
    try:
        _, ev_treat = run(engine="lgb", feat_dir=TREAT_DIR, name=f"ades_cat_treatment{suffix}",
                          target=target, submit=False, seed=MODEL_SEED)
    finally:
        encode.CAT_COLS[:] = original_cat_cols

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
    point, lo, hi, p_worse = cluster_bootstrap(
        cl["sb"].to_numpy(), cl["st"].to_numpy(), cl["n"].to_numpy(),
        N_RESAMPLES, BOOT_SEED)
    print(f"\ncluster bootstrap (airport, day), {N_RESAMPLES} resamples, "
          f"{cl.height} clusters:")
    print(f"  paired RMSE delta (treatment - baseline): {point:+.2f}s")
    print(f"  95% CI: [{lo:+.2f}, {hi:+.2f}]    P(worse) = {p_worse:.3f}")


if __name__ == "__main__":
    main(target=sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
