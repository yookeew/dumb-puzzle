"""Before/after significance check for the minute_of_day Int8-overflow fix
(PROGRESS.md S27 finding, fixed in build_features.py).

`minute_of_day` was computed as `pl.col("T").dt.hour() * 60 +
pl.col("T").dt.minute()` in Int8 arithmetic, silently wrapping for every
hour >= 3 (e.g. hour=20 -> -80, not 1200) -- confirmed by direct repro. It
is a live numeric feature (not excluded by NON_FEATURES), so the model has
been training on wrapped garbage for most of the day this whole time. Fix:
cast dt.hour() to Int16 before the multiply.

This is a correctness fix, not a hypothesis test -- but per this project's
own discipline (PROGRESS.md S5/S6), no feature-affecting change ships
without a before/after holdout check, since a "fix" can still move
generalization in either direction (LightGBM may have been getting
incidental split value out of the wrapped pattern).

BEFORE = cache/features_before_minfix_tmp/ (backup of the pre-fix
          production cache, copied off before export_model_inputs.py was
          rerun)
AFTER  = cache/features/ (current production cache, rebuilt with the fix)

Same discipline as tests/metar_ablation_test.py / doy_ablation_test.py:
paired per-row squared error, (airport, day) cluster bootstrap,
target="mixed" (S25: the correct instrument for feature testing), both
months + LIRF reported separately.

Run:  .venv/Scripts/python.exe tests/minute_of_day_fix_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET, cluster_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEFORE_DIR = ROOT / "cache" / "features_before_minfix_tmp"
AFTER_DIR = ROOT / "cache" / "features"

MODEL_SEED = 42
N_RESAMPLES = 3000
BOOT_SEED = 0


def _fit(dest: Path, name: str, target: str, seed: int):
    return run(engine="lgb", feat_dir=dest, name=name, target=target,
               submit=False, seed=seed)


def main(target: str = DEFAULT_TARGET, seed: int = MODEL_SEED) -> None:
    print(f"=== fitting BEFORE (buggy minute_of_day, target={target}, seed={seed}) ===")
    _, ev_before = _fit(BEFORE_DIR, "minfix_before", target, seed)

    print("\n=== fitting AFTER (fixed minute_of_day) ===")
    _, ev_after = _fit(AFTER_DIR, "minfix_after", target, seed)

    ev = (
        ev_before.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt", pred_base="pred")
        .join(ev_after.select("MVT_ID_mvt", pred_treat="pred"), on="MVT_ID_mvt", how="inner")
    )
    assert ev.height == ev_before.height == ev_after.height, "row mismatch"
    day = pl.read_parquet(AFTER_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = ev.join(day, on="MVT_ID_mvt", how="left").with_columns(
        se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
        se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2),
    )

    rb, rt = ev["se_base"].mean() ** 0.5, ev["se_treat"].mean() ** 0.5
    print(f"\n{'=' * 78}\nBEFORE (buggy) vs AFTER (fixed) minute_of_day\n{'=' * 78}")
    print(f"overall RMSE   before={rb:.1f}  after={rt:.1f}  delta={rt - rb:+.1f}")
    for ym in HOLDOUT_MONTHS:
        s = ev.filter(pl.col("ym") == ym)
        a, b = s["se_base"].mean() ** 0.5, s["se_treat"].mean() ** 0.5
        print(f"  {ym}: before={a:.1f}  after={b:.1f}  delta={b - a:+.1f}")

    print("  per-airport:")
    for ap in sorted(ev["ADEP_mvt"].unique().to_list()):
        s = ev.filter(pl.col("ADEP_mvt") == ap)
        a, b = s["se_base"].mean() ** 0.5, s["se_treat"].mean() ** 0.5
        print(f"    {ap}: before={a:.1f}  after={b:.1f}  delta={b - a:+.1f}  n={s.height:,}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("sb"), pl.col("se_treat").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, pw = cluster_bootstrap(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                                         cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED)
    print(f"\ncluster bootstrap: paired RMSE delta={point:+.2f}s  95% CI=[{lo:+.2f}, {hi:+.2f}]  "
          f"P(worse)={pw:.3f}")


if __name__ == "__main__":
    _t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    _s = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_SEED
    main(target=_t, seed=_s)
