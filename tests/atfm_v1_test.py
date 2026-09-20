"""Validate the new ATFM context family (features/atfm.py, build_features.py
family 5) against a fresh baseline, per the project's established discipline
(PROGRESS.md S14/S16/S19): paired per-row squared-error deltas on the SAME
holdout rows, cluster-bootstrapped at the (airport, day) level (congestion/
disruption errors are correlated within an airport-day window), both months
checked separately, primary metric pre-registered as the paired overall RMSE
delta.

"Baseline" is simulated by dropping the 19 atfm/ades_in_panel columns from
the already-built (atfm-included) cache/features/*.parquet rather than
rebuilding from scratch -- the atfm join is a pure left-join that adds
columns without touching anything else, so this is behaviourally identical
to rerunning export_model_inputs.py with family 5 disabled, at a fraction of
the cost.

Run:  .venv/Scripts/python.exe tests/atfm_v1_test.py
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
NOATFM_DIR = ROOT / "cache" / "features_noatfm_tmp"

ATFM_CTX_COLS = [
    "reg_share", "in_slot_share", "late_share", "dly_min_per_flight",
    "dly_weather_share", "dly_staffing_share", "traffic_dep", "traffic_arr",
    "traffic_tot",
]
ATFM_COLS = ["ades_in_panel"] + [f"dep_atfm_{c}" for c in ATFM_CTX_COLS] + \
    [f"des_atfm_{c}" for c in ATFM_CTX_COLS]

N_RESAMPLES = 3000
MODEL_SEED = 42
BOOT_SEED = 0


def _make_noatfm_dir() -> None:
    if NOATFM_DIR.exists():
        shutil.rmtree(NOATFM_DIR)
    NOATFM_DIR.mkdir(parents=True)
    for f in ("train2025.parquet", "holdout_gap2025.parquet"):
        df = pl.read_parquet(FEAT_DIR / f)
        present = [c for c in ATFM_COLS if c in df.columns]
        df.drop(present).write_parquet(NOATFM_DIR / f)
    # labels2025.parquet is untouched by family 5 -- just needs to exist alongside
    shutil.copy(FEAT_DIR / "labels2025.parquet", NOATFM_DIR / "labels2025.parquet")


def _cluster_key(ev: pl.DataFrame) -> pl.DataFrame:
    """(ADEP_mvt, day) per MVT_ID_mvt, pulled from the real holdout_gap2025.parquet
    (has the raw "T" takeoff timestamp that ev itself doesn't carry)."""
    t = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date()
    )
    return ev.join(t, on="MVT_ID_mvt", how="left")


def _cluster_bootstrap(se_base_sum: np.ndarray, se_treat_sum: np.ndarray,
                        n_by_cluster: np.ndarray, n_resamples: int,
                        seed: int) -> tuple[float, float, float, float]:
    """Paired cluster bootstrap on the RMSE delta itself (nonlinear in the
    per-cluster squared-error sums, so resampled per draw rather than
    approximated by a linear MSE delta -- matches the pre-registered primary
    metric used elsewhere in this project, "paired delta in overall RMSE".
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(se_base_sum)
    point = (se_treat_sum.sum() / n_by_cluster.sum()) ** 0.5 - \
        (se_base_sum.sum() / n_by_cluster.sum()) ** 0.5
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n_clusters, n_clusters)
        n_b = n_by_cluster[idx].sum()
        rmse_b_base = (se_base_sum[idx].sum() / n_b) ** 0.5
        rmse_b_treat = (se_treat_sum[idx].sum() / n_b) ** 0.5
        boots[b] = rmse_b_treat - rmse_b_base
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_worse = float((boots > 0).mean())
    return point, lo, hi, p_worse


def main() -> None:
    print("=== baseline (atfm columns dropped) ===")
    _make_noatfm_dir()
    _, ev_base = run(engine="lgb", feat_dir=NOATFM_DIR, name="atfm_v1_baseline",
                      target="direct", submit=False, seed=MODEL_SEED)

    print("\n=== treatment (atfm family 5 included) ===")
    _, ev_treat = run(engine="lgb", feat_dir=FEAT_DIR, name="atfm_v1_treatment",
                       target="direct", submit=False, seed=MODEL_SEED)

    base = ev_base.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt",
                           pred_base="pred")
    treat = ev_treat.select("MVT_ID_mvt", pred_treat="pred")
    ev = base.join(treat, on="MVT_ID_mvt", how="inner")
    assert ev.height == ev_base.height == ev_treat.height, "row mismatch between runs"

    ev = ev.with_columns(
        se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
        se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2),
    )
    ev = _cluster_key(ev)

    rmse_base = (ev["se_base"].mean()) ** 0.5
    rmse_treat = (ev["se_treat"].mean()) ** 0.5
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
    point, lo, hi, p_worse = _cluster_bootstrap(
        cl["se_base_sum"].to_numpy(), cl["se_treat_sum"].to_numpy(),
        cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED,
    )
    print(f"\ncluster bootstrap (airport, day), {N_RESAMPLES} resamples, "
          f"{cl.height} clusters:")
    print(f"  paired RMSE delta (treatment - baseline): {point:+.2f}s")
    print(f"  95% CI: [{lo:+.2f}, {hi:+.2f}]")
    print(f"  P(worse) = {p_worse:.3f}")


if __name__ == "__main__":
    main()
