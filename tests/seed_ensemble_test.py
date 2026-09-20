"""Equal-weight seed-averaged ensemble -- the variance-reduction lever.

Motivated by a direct measurement (tests/train_gap.py, PROGRESS.md S22):
train 166.4s / inner-valid 230.7s / holdout 246.4s, i.e. a +48.1%
train->holdout gap with num_leaves=255 and ~3,310 trees against 1.56M rows.
That is a high-variance model, and variance is reducible with NO new
information.

CLAUDE.md's ensemble stage was planned and never built. S5's warning was
specifically that *fitted* blend weights backfired 4/4 times for other
contestants while EQUAL weights won every time -- so equal-weight seed
averaging is both the untried form and the safe one.

The echo classifier is seed-independent in run() (seed is threaded to the
main regressor only, per S9), so this isolates regressor variance cleanly.

Reports, on identical holdout rows:
  * each individual seed (this also measures genuine run-to-run spread,
    which S14 estimated at +-7.8s for retrain-vs-retrain comparisons)
  * the equal-weight average of all seeds
  * ensemble vs best/median single seed, (airport, day) cluster bootstrap

Run:  .venv/Scripts/python.exe tests/seed_ensemble_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
SEEDS = [42, 7, 13, 99]
N_RESAMPLES = 3000
BOOT_SEED = 0


def _boot(sb, st, n, n_resamples, seed):
    rng = np.random.default_rng(seed)
    k = len(sb)
    tot = n.sum()
    point = (st.sum() / tot) ** 0.5 - (sb.sum() / tot) ** 0.5
    out = np.empty(n_resamples)
    for b in range(n_resamples):
        i = rng.integers(0, k, k)
        nb = n[i].sum()
        out[b] = (st[i].sum() / nb) ** 0.5 - (sb[i].sum() / nb) ** 0.5
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, lo, hi, float((out > 0).mean())


def main() -> None:
    frames = {}
    for s in SEEDS:
        print(f"\n=== seed {s} ===")
        _, ev = run(engine="lgb", name=f"seed_ens_{s}", target="direct",
                    submit=False, seed=s)
        frames[s] = ev.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt",
                              **{f"p{s}": pl.col("pred")})

    ev = frames[SEEDS[0]]
    for s in SEEDS[1:]:
        ev = ev.join(frames[s].select("MVT_ID_mvt", f"p{s}"),
                     on="MVT_ID_mvt", how="inner")
    assert ev.height == frames[SEEDS[0]].height, "row mismatch across seeds"

    pcols = [f"p{s}" for s in SEEDS]
    ev = ev.with_columns(ens=pl.mean_horizontal(pcols))
    day = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = ev.join(day, on="MVT_ID_mvt", how="left")

    def rmse(col, d=ev):
        return d.select((pl.col(col) - pl.col("taxi")).pow(2).mean()).item() ** 0.5

    print("\n" + "=" * 70)
    print("INDIVIDUAL SEEDS vs EQUAL-WEIGHT ENSEMBLE")
    print("=" * 70)
    singles = {s: rmse(f"p{s}") for s in SEEDS}
    for s, v in singles.items():
        print(f"  seed {s:<4} RMSE = {v:7.2f}s")
    vals = np.array(list(singles.values()))
    print(f"\n  single-seed spread: min={vals.min():.2f} max={vals.max():.2f} "
          f"sd={vals.std(ddof=1):.2f}  (run-to-run noise)")
    ens = rmse("ens")
    mean_single = vals.mean()
    print(f"\n  mean single seed    = {mean_single:7.2f}s")
    print(f"  EQUAL-WEIGHT ENSEMBLE = {ens:7.2f}s")
    print(f"  gain vs mean single = {ens - mean_single:+.2f}s")
    print(f"  gain vs best single = {ens - vals.min():+.2f}s")

    print("\n  by month:")
    for ym in HOLDOUT_MONTHS:
        d = ev.filter(pl.col("ym") == ym)
        ms = np.mean([rmse(f"p{s}", d) for s in SEEDS])
        print(f"    {ym}: mean single={ms:7.2f}  ensemble={rmse('ens', d):7.2f}  "
              f"delta={rmse('ens', d) - ms:+.2f}")

    print("\n  by airport (ensemble vs mean single):")
    for ap in sorted(ev["ADEP_mvt"].unique().to_list()):
        d = ev.filter(pl.col("ADEP_mvt") == ap)
        ms = np.mean([rmse(f"p{s}", d) for s in SEEDS])
        e = rmse("ens", d)
        print(f"    {ap:<6} single={ms:8.1f}  ens={e:8.1f}  delta={e - ms:+7.1f}  "
              f"n={d.height:,}")

    # paired bootstrap: ensemble vs the FIRST seed (the incumbent config)
    base_col = f"p{SEEDS[0]}"
    cl = ev.with_columns(
        sb=(pl.col(base_col) - pl.col("taxi")).pow(2),
        st=(pl.col("ens") - pl.col("taxi")).pow(2),
    ).group_by("ADEP_mvt", "day").agg(
        pl.col("sb").sum().alias("sb"), pl.col("st").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, pw = _boot(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                              cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED)
    print(f"\ncluster bootstrap (airport, day), {N_RESAMPLES} resamples, "
          f"{cl.height} clusters")
    print(f"  ensemble vs seed {SEEDS[0]} (the incumbent):")
    print(f"  paired RMSE delta: {point:+.2f}s   95% CI [{lo:+.2f}, {hi:+.2f}]   "
          f"P(ensemble worse) = {pw:.3f}")


if __name__ == "__main__":
    main()
