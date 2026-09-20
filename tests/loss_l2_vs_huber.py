"""Re-test L2 vs Huber under the HONEST eval.

Huber (alpha=800) was adopted in PROGRESS.md S3 on 303.1s vs L2's 309.2s.
S6 later established that BOTH numbers came from the buggy eval, which
excluded `taxi > 4h` rows -- exactly the heavy-tail population where a
robust loss does its damage. Grepping the history, the comparison has never
been redone since the eval was fixed; S17 even flags one comparison as
confounded by "l2 vs huber" without isolating it.

Why it should matter:
  * the competition scores RMSE, which is minimised by the conditional MEAN
  * L2 estimates the conditional mean; Huber down-weights large residuals
    and estimates something between the mean and the median
  * these conditionals are strongly right-skewed mixtures (~98% of
    large-offset flights taxi normally, ~2% sat at the stand for hours), so
    median << mean
  => Huber should systematically UNDER-predict the heavy tail.

And tests/extreme_rows.py measured exactly that without knowing the cause:
the model under-predicts 75% of its 1,000 worst rows.

Same harness as tests/atfm_v1_test.py: paired rows, (airport, day) cluster
bootstrap, both months reported, primary metric the paired overall RMSE
delta. Treatment is L2, baseline is the incumbent Huber.

Run:  .venv/Scripts/python.exe tests/loss_l2_vs_huber.py
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
N_RESAMPLES = 3000
MODEL_SEED = 42
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
    print("=== baseline: huber alpha=800 (incumbent) ===")
    _, ev_h = run(engine="lgb", name="loss_huber_baseline", target="direct",
                  submit=False, seed=MODEL_SEED, loss="huber")

    print("\n=== treatment: plain L2 ===")
    _, ev_l = run(engine="lgb", name="loss_l2_treatment", target="direct",
                  submit=False, seed=MODEL_SEED, loss="l2")

    ev = (ev_h.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt",
                      "has_aobt3", "echo_pred", pred_h="pred")
          .join(ev_l.select("MVT_ID_mvt", pred_l="pred"),
                on="MVT_ID_mvt", how="inner"))
    assert ev.height == ev_h.height == ev_l.height, "row mismatch"

    day = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", "sched_takeoff_offset", day=pl.col("T").dt.date())
    ev = ev.join(day, on="MVT_ID_mvt", how="left").with_columns(
        se_h=(pl.col("pred_h") - pl.col("taxi")).pow(2),
        se_l=(pl.col("pred_l") - pl.col("taxi")).pow(2),
    )

    rh = ev["se_h"].mean() ** 0.5
    rl = ev["se_l"].mean() ** 0.5
    print(f"\noverall RMSE   huber={rh:.1f}  l2={rl:.1f}  delta={rl - rh:+.1f}")
    for ym in HOLDOUT_MONTHS:
        s = ev.filter(pl.col("ym") == ym)
        a, b = s["se_h"].mean() ** 0.5, s["se_l"].mean() ** 0.5
        print(f"  {ym}: huber={a:.1f}  l2={b:.1f}  delta={b - a:+.1f}")

    print("\nper airport:")
    for r in (ev.group_by("ADEP_mvt").agg(
            pl.col("se_h").mean().sqrt().alias("h"),
            pl.col("se_l").mean().sqrt().alias("l"),
            pl.len().alias("n")).sort("ADEP_mvt").iter_rows(named=True)):
        print(f"  {r['ADEP_mvt']:<6} huber={r['h']:7.1f}  l2={r['l']:7.1f}  "
              f"delta={r['l'] - r['h']:+7.1f}  n={r['n']:,}")

    print("\nby offset band (where the theory predicts the difference):")
    for lo, hi in [(0, 1800), (1800, 3600), (3600, 7200), (7200, 14400),
                   (14400, 10**9)]:
        b = ev.filter((pl.col("sched_takeoff_offset") >= lo)
                      & (pl.col("sched_takeoff_offset") < hi))
        if b.height < 50:
            continue
        lab = f"[{lo:,}, {hi:,})" if hi < 10**9 else f">= {lo:,}"
        bh, bl = b["se_h"].mean() ** 0.5, b["se_l"].mean() ** 0.5
        ub_h = b.select((pl.col("pred_h") < pl.col("taxi")).mean()).item()
        ub_l = b.select((pl.col("pred_l") < pl.col("taxi")).mean()).item()
        print(f"  {lab:<20} huber={bh:8.1f} l2={bl:8.1f} delta={bl - bh:+8.1f}"
              f"   under-pred: huber {ub_h * 100:.0f}% l2 {ub_l * 100:.0f}%  n={b.height:,}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_h").sum().alias("sb"), pl.col("se_l").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, pw = _boot(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                              cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED)
    print(f"\ncluster bootstrap (airport, day), {N_RESAMPLES} resamples, "
          f"{cl.height} clusters:")
    print(f"  paired RMSE delta (l2 - huber): {point:+.2f}s")
    print(f"  95% CI: [{lo:+.2f}, {hi:+.2f}]    P(l2 worse) = {pw:.3f}")


if __name__ == "__main__":
    main()
