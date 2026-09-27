"""ADS-B pushback-recovery ceiling check -- NO new data used.

Before committing to a 400 GB ADS-B download to recover ground-truth pushback
time, quantify the best case for what it could possibly be worth, using only
data already on disk. This runs entirely against the existing production
holdout (Jan+Jul 2025) and `data/ranking/ranking.parquet`.

Context (one-day adsb.lol probe, 2025-01-15, 2.1 GB): ground-level ADS-B
coverage exists at only 6 of our 10 departure airports -- EHAM, EGLL, EDDM,
EDDF, LSZH, LEBL. LFPG, LEMD, LIRF, LTFM had zero aircraft with a ground
position that day, not thin coverage. LIRF (our single largest error
contributor) is one of the zero-coverage airports.

Per reports/step0_audit.md SS4: AOBT_3_flt (already a feature, `aobt3_taxi`)
is a noisy read on the same off-block event -- 21% within +-1 min, 74.6%
within +-5 min of true BLOCK_TIME_UTC_mvt, and naive taxi from it alone gives
RMSE 384.9s vs true-taxi sd 417.5s. So ADS-B's only possible value here is
DENOISING that existing feature down to the true pushback time -- not adding
a new signal from nothing. The ceiling below reflects that: it asks "if we
had the true pushback time (equivalently, the true taxi) for every covered-
airport row, and changed nothing else," which is the best any amount of
ADS-B denoising could ever achieve, since real recovery will (a) not catch
every aircraft and (b) not be perfectly exact even when it does.

Computes, against the current production model (`target="mixed"`, the
project's adopted target per tests/_harness.py):
  1. Share of ranking DEP rows at the 6 covered airports.
  2. Their share of total holdout squared error (not just row share --
     these airports may be easier or harder than average).
  3. Oracle ceiling: zero out the holdout squared error on covered-airport
     rows (perfect recovery), recompute overall RMSE. Absolute best case.
  4. Sensitivity: same computation at 50% and 25% of covered-airport rows
     getting perfect recovery (uniform random subsample), since a one-day
     probe saw ~400 aircraft/day at EDDM -- well under a full day's
     departures there.

Run:  .venv/Scripts/python.exe tests/adsb_ceiling_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.fit import run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"

# 6 airports with nonzero ground-level ADS-B coverage in the one-day probe.
COVERED = ["EHAM", "EGLL", "EDDM", "EDDF", "LSZH", "LEBL"]
MODEL_SEED = 42
RULE = "=" * 78


def main(target: str = DEFAULT_TARGET, seed: int = MODEL_SEED) -> None:
    print(f"=== fitting production model (target={target}, seed={seed}) ===")
    _, ev = run(engine="lgb", target=target, submit=False, seed=seed)

    ev = ev.with_columns(
        se=(pl.col("pred") - pl.col("taxi")).pow(2),
        covered=pl.col("ADEP_mvt").is_in(COVERED),
    )

    n_total = ev.height
    sum_se_total = ev["se"].sum()
    base_mse = sum_se_total / n_total
    base_rmse = base_mse ** 0.5
    covered_ev = ev.filter(pl.col("covered"))
    n_covered = covered_ev.height
    sum_se_covered = covered_ev["se"].sum()

    print(f"\n{RULE}\nproduction holdout: n={n_total:,}  RMSE={base_rmse:.2f}s\n{RULE}")

    # --- 1. ranking DEP row share at covered airports ----------------------
    rk = pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP")
    n_rk = rk.height
    n_rk_covered = rk.filter(pl.col("ADEP_mvt").is_in(COVERED)).height
    print(f"\n1. ranking DEP rows: n={n_rk:,}")
    print(f"   at covered airports: n={n_rk_covered:,} "
          f"({n_rk_covered / n_rk * 100:.1f}%)")
    print("   per airport:")
    g = rk.group_by("ADEP_mvt").agg(pl.len().alias("n")).sort("n", descending=True)
    for r in g.iter_rows(named=True):
        flag = "  <-- covered" if r["ADEP_mvt"] in COVERED else ""
        print(f"     {r['ADEP_mvt']:<6} n={r['n']:>7,}  "
              f"({r['n'] / n_rk * 100:5.1f}%){flag}")

    # --- 2. holdout squared-error share by airport --------------------------
    print(f"\n2. covered-airport holdout rows: n={n_covered:,} "
          f"({n_covered / n_total * 100:.1f}% of holdout rows)")
    print(f"   covered-airport share of TOTAL squared error: "
          f"{sum_se_covered / sum_se_total * 100:.1f}%")
    print("   per-airport holdout RMSE / % of total squared error:")
    g2 = (ev.group_by("ADEP_mvt")
          .agg(pl.col("se").mean().sqrt().alias("rmse"), pl.len().alias("n"),
               (pl.col("se").sum() / sum_se_total * 100).alias("pct_sqerr"))
          .sort("pct_sqerr", descending=True))
    for r in g2.iter_rows(named=True):
        flag = "  <-- covered" if r["ADEP_mvt"] in COVERED else ""
        print(f"     {r['ADEP_mvt']:<6} rmse={r['rmse']:7.1f}  n={r['n']:>7,}  "
              f"pct_sqerr={r['pct_sqerr']:5.1f}%{flag}")

    # --- 3 & 4. oracle ceiling, 100/50/25% coverage --------------------------
    print(f"\n{RULE}\n3+4. ADS-B pushback-recovery ceiling (perfect denoising of "
          f"aobt3_taxi)\n{RULE}")
    print(f"\n{'coverage':>10}  {'new RMSE':>10}  {'gain (s)':>10}  {'gain (%)':>10}")
    for frac in (1.0, 0.5, 0.25):
        new_sum_se = sum_se_total - frac * sum_se_covered
        new_rmse = (new_sum_se / n_total) ** 0.5
        gain = base_rmse - new_rmse
        print(f"{frac * 100:>9.0f}%  {new_rmse:>10.2f}  {gain:>10.2f}  "
              f"{gain / base_rmse * 100:>9.2f}%")

    lirf_n = rk.filter(pl.col("ADEP_mvt") == "LIRF").height
    lirf_se_pct = (ev.filter(pl.col("ADEP_mvt") == "LIRF")["se"].sum()
                   / sum_se_total * 100)
    print(f"\nLIRF (zero ADS-B ground coverage in the probe): "
          f"{lirf_n:,} ranking DEP rows ({lirf_n / n_rk * 100:.1f}% of ranking "
          f"DEP), {lirf_se_pct:.1f}% of holdout squared error -- "
          "structurally out of reach for this approach.")


if __name__ == "__main__":
    _t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    _s = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_SEED
    main(target=_t, seed=_s)
