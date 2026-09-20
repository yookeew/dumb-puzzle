"""Verify the realised-queue features are causal BEFORE trusting any RMSE.

The specific hazard: a queue must contain only aircraft that pushed back
STRICTLY BEFORE this flight. Include later pushbacks and it stops being a
queue-ahead, becomes correlated with taxi duration in-sample, and looks like
a win right up until the leaderboard disagrees.

So this does not take the vectorised searchsorted implementation on trust.
It recomputes the features by brute force on a random sample, directly from
the definition, and requires exact agreement.

Also checks:
  * self-exclusion (a flight never counts itself)
  * no dependence on the flight's own takeoff time T (anchoring on T is the
    exact mistake this feature is designed to avoid)
  * 2025-vs-2026 distribution parity, per S5's rule that a new feature's
    train/ranking distributions be compared before training on it

Run:  .venv/Scripts/python.exe tests/queue_causality_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.build_features import _realised_queue, build_features  # noqa: E402
from features.export_model_inputs import FRAME_COLS, blind  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULE = "=" * 78
N_SAMPLE = 400
RNG = np.random.default_rng(0)


def _brute(dep: pl.DataFrame, row: dict) -> tuple:
    """Recompute the four features for one row straight from the definition."""
    a_i = row["a"]
    # only valid taxi intervals are countable (takeoff strictly after pushback)
    same_ap = dep.filter((pl.col("airport") == row["airport"])
                         & (pl.col("t") > pl.col("a")))
    # ONLY aircraft that pushed back strictly before this one
    ahead = same_ap.filter(pl.col("a") < a_i)
    active = ahead.filter(pl.col("t") > a_i)
    q_ahead = active.height
    q_rwy = active.filter(pl.col("RUNWAY_mvt") == row["RUNWAY_mvt"]).height
    q15 = ahead.filter(pl.col("a") >= a_i - 900).height
    wait = (a_i - active["a"].mean()) if active.height else float("nan")
    return q_ahead, q_rwy, q15, wait


def main() -> None:
    frame = pl.read_parquet(
        ROOT / "data" / "raw" / "training_2025-03-01_2025-04-01.parquet",
        columns=FRAME_COLS)
    frame = blind(frame)
    q = _realised_queue(frame)

    dep = (
        frame.filter(pl.col("PHASE_mvt") == "DEP")
        .select("MVT_ID_mvt", "RUNWAY_mvt", airport=pl.col("ADEP_mvt"),
                a=pl.col("AOBT_3_flt").dt.epoch("s"),
                t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"))
        .filter(pl.col("a").is_not_null() & pl.col("t").is_not_null())
    )
    j = dep.join(q, on="MVT_ID_mvt", how="inner")
    print(f"departures with AOBT_3: {dep.height:,}   joined: {j.height:,}")

    print(f"\n{RULE}\n1. BRUTE-FORCE AGREEMENT (the causality test)\n{RULE}")
    idx = RNG.choice(j.height, size=min(N_SAMPLE, j.height), replace=False)
    sample = j[idx]
    bad = 0
    for row in sample.iter_rows(named=True):
        qa, qr, q15, w = _brute(dep, row)
        checks = [
            (row["q_ahead"], qa, "q_ahead"),
            (row["q_ahead_rwy"], qr, "q_ahead_rwy"),
            (row["q_push_15m"], q15, "q_push_15m"),
        ]
        for got, exp, nm in checks:
            if got != exp:
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH {nm}: vectorised={got} brute={exp} "
                          f"mvt={row['MVT_ID_mvt']}")
        gw, ew = row["q_ahead_mean_wait"], w
        g_missing = gw is None or (isinstance(gw, float) and np.isnan(gw))
        e_missing = ew is None or (isinstance(ew, float) and np.isnan(ew))
        if g_missing != e_missing or (
                not g_missing and abs(gw - ew) > 1e-6):
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH q_ahead_mean_wait: {gw} vs {ew}")
    print(f"\n  checked {sample.height} rows x 4 features: "
          f"{'ALL MATCH' if bad == 0 else f'{bad} MISMATCHES'}")

    print(f"\n{RULE}\n2. SELF-EXCLUSION AND ANCHOR SANITY\n{RULE}")
    # a flight taxiing alone must see an empty queue
    solo = j.filter(pl.col("q_ahead") == 0)
    print(f"  rows with q_ahead == 0: {solo.height:,} "
          f"({solo.height / j.height * 100:.1f}%)  -- nonzero share expected")
    # mean wait must be positive: those ahead pushed back before me
    w = j.filter(pl.col("q_ahead_mean_wait").is_not_null())
    neg = w.filter(pl.col("q_ahead_mean_wait") < 0).height
    print(f"  negative q_ahead_mean_wait: {neg} "
          f"(MUST be 0 -- negative means someone 'ahead' pushed back later)")
    print(f"  q_ahead_rwy <= q_ahead violations: "
          f"{j.filter(pl.col('q_ahead_rwy') > pl.col('q_ahead')).height} (must be 0)")

    print(f"\n{RULE}\n3. NO DEPENDENCE ON OWN TAKEOFF TIME T\n{RULE}")
    print("  Recomputing with every T shifted +1h: the queue-ahead count is")
    print("  defined at pushback, so counts for OTHER aircraft may change but")
    print("  a flight's own T must not enter its own anchor.")
    shifted = frame.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("MVT_TIME_UTC_mvt") + pl.duration(hours=1))
        .otherwise(pl.col("MVT_TIME_UTC_mvt")).alias("MVT_TIME_UTC_mvt"))
    q2 = _realised_queue(shifted).rename({"q_push_15m": "q15_b"})
    cmp = q.join(q2.select("MVT_ID_mvt", "q15_b"), on="MVT_ID_mvt", how="inner")
    same = cmp.filter(pl.col("q_push_15m") == pl.col("q15_b")).height
    print(f"  q_push_15m unchanged under a uniform T shift: "
          f"{same:,}/{cmp.height:,} ({same / cmp.height * 100:.2f}%)")
    print("  (this one depends only on pushback times, so it must be 100%)")

    print(f"\n{RULE}\n4. 2025 vs 2026 DISTRIBUTION PARITY\n{RULE}")
    rank = blind(pl.read_parquet(ROOT / "data" / "ranking" / "ranking.parquet",
                                 columns=FRAME_COLS))
    qr = _realised_queue(rank)
    print(f"\n  {'feature':<22}{'2025-03 mean':>15}{'2026 mean':>13}"
          f"{'2025 sd':>11}{'2026 sd':>11}{'null%2026':>12}")
    for c in ["q_ahead", "q_ahead_rwy", "q_push_15m", "q_ahead_mean_wait"]:
        a, b = q[c], qr[c]
        nullpct = b.is_null().mean() * 100
        aa, bb = a.drop_nulls(), b.drop_nulls()
        print(f"  {c:<22}{aa.mean():>15.2f}{bb.mean():>13.2f}"
              f"{aa.std():>11.2f}{bb.std():>11.2f}{nullpct:>11.1f}%")


if __name__ == "__main__":
    main()
