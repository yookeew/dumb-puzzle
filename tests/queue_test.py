"""Test the realised departure-queue family (build_features._realised_queue).

Family 2's congestion counts are TAKEOFF-anchored. This family is anchored
on the flight's own pushback (AOBT_3_flt) and counts only aircraft that
pushed back STRICTLY BEFORE it -- i.e. who was actually ahead of it on the
ground. Causality verified independently in tests/queue_causality_check.py
(brute-force agreement on 400 rows x 4 features; shifting a flight's own
takeoff by +3h leaves its own features unchanged 25/25).

Motivated by PROGRESS.md S22: 57% of MSE sits in ordinary flights where the
model explains only 53-58% of variance, and every cell-granularity candidate
screened so far is bounded small. A flight-level queue is not bounded by
those screens.

Same harness as the other feature tests: paired per-row squared errors on
identical holdout rows, (airport, day) cluster bootstrap, both months
reported separately, primary metric the paired overall RMSE delta.

Run:  .venv/Scripts/python.exe tests/queue_test.py
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
BASE_DIR = ROOT / "cache" / "features_q_base_tmp"

QUEUE_COLS = ["q_ahead", "q_ahead_rwy", "q_push_15m", "q_ahead_mean_wait"]

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
    print("=== baseline (queue features dropped) ===")
    _prep(BASE_DIR, QUEUE_COLS)
    _, ev_base = run(engine="lgb", feat_dir=BASE_DIR, name="queue_baseline",
                     target="direct", submit=False, seed=MODEL_SEED)

    print("\n=== treatment (+ realised queue family) ===")
    _, ev_treat = run(engine="lgb", feat_dir=FEAT_DIR, name="queue_treatment",
                      target="direct", submit=False, seed=MODEL_SEED)

    ev = (ev_base.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt", "echo_pred",
                         pred_base="pred")
          .join(ev_treat.select("MVT_ID_mvt", pred_treat="pred"),
                on="MVT_ID_mvt", how="inner"))
    assert ev.height == ev_base.height == ev_treat.height, "row mismatch"

    extra = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", "sched_takeoff_offset", "q_ahead",
        day=pl.col("T").dt.date())
    ev = ev.join(extra, on="MVT_ID_mvt", how="left").with_columns(
        se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
        se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2),
    )

    rb = ev["se_base"].mean() ** 0.5
    rt = ev["se_treat"].mean() ** 0.5
    print(f"\noverall RMSE   baseline={rb:.1f}  treatment={rt:.1f}  "
          f"delta={rt - rb:+.1f}")
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

    print("\nby offset band (S22 says 57% of MSE is in the first two):")
    for lo, hi in [(0, 1800), (1800, 3600), (3600, 7200), (7200, 14400),
                   (14400, 10**9)]:
        b = ev.filter((pl.col("sched_takeoff_offset") >= lo)
                      & (pl.col("sched_takeoff_offset") < hi))
        if b.height < 50:
            continue
        lab = f"[{lo:,}, {hi:,})" if hi < 10**9 else f">= {lo:,}"
        x, y = b["se_base"].mean() ** 0.5, b["se_treat"].mean() ** 0.5
        print(f"  {lab:<20} base={x:8.1f} treat={y:8.1f} delta={y - x:+8.1f}"
              f"  n={b.height:,}")

    print("\nby queue depth (does it help most where the queue is long?):")
    qd = ev.filter(pl.col("q_ahead").is_not_null())
    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 100)]:
        b = qd.filter((pl.col("q_ahead") >= lo) & (pl.col("q_ahead") < hi))
        if b.height < 50:
            continue
        x, y = b["se_base"].mean() ** 0.5, b["se_treat"].mean() ** 0.5
        print(f"  q_ahead [{lo:>2},{hi:>3}) base={x:8.1f} treat={y:8.1f} "
              f"delta={y - x:+8.1f}  n={b.height:,}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("sb"), pl.col("se_treat").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, pw = _boot(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                              cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED)
    print(f"\ncluster bootstrap (airport, day), {N_RESAMPLES} resamples, "
          f"{cl.height} clusters:")
    print(f"  paired RMSE delta (treatment - baseline): {point:+.2f}s")
    print(f"  95% CI: [{lo:+.2f}, {hi:+.2f}]    P(worse) = {pw:.3f}")


if __name__ == "__main__":
    main()
