"""Quantify what the check-2 gradients and the check-5 structure are actually
worth in RMSE, rather than eyeballing them.

Two things the first diagnostic pass left open:

  A. The decile gradients in check 2 look like 5-34s of mean-residual spread,
     but a bias confined to one decile contributes only (n_d/N)*bias^2 to MSE.
     Compute the exact MSE reduction from perfectly removing each feature's
     decile-level bias -- in-sample (optimistic) AND Jan->Jul transfer (honest,
     matching the S18 methodology: derive on one month, test on the other).

  B. Check 5's "1.4% of residual variance is between (airport, day)" bound
     strictly applies only to features constant within an airport-day. The
     ADES-side features are NOT (check 4: within-day sd ratio 0.60-0.70), so
     the honest ceiling for the dual-join design is the (ADEP, day, ADES)
     decomposition, with a noise correction -- tiny cells make a naive oracle
     look good for free.

Run:  .venv/Scripts/python.exe tests/atfm_ceiling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"

ATFM_CTX = ["reg_share", "in_slot_share", "late_share", "dly_min_per_flight",
            "dly_weather_share", "dly_staffing_share", "traffic_dep",
            "traffic_arr", "traffic_tot"]
FEATURES = [f"dep_atfm_{c}" for c in ATFM_CTX] + [f"des_atfm_{c}" for c in ATFM_CTX]

RULE = "=" * 78
N_BUCKETS = 10


def _load() -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "T"] + FEATURES
    f = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet")
    f = f.select([c for c in cols if c in f.columns]).with_columns(
        day=pl.col("T").dt.date()
    ).drop("T")
    ev = pl.read_parquet(PROD_EV).join(f, on="MVT_ID_mvt", how="left")
    return ev.with_columns(resid=pl.col("taxi") - pl.col("pred"))


def part_a(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\nA. What each feature's decile-level bias is WORTH in RMSE")
    print(RULE)
    base_mse = ev.select(pl.col("resid").pow(2).mean()).item()
    base_rmse = base_mse ** 0.5
    print(f"\nproduction holdout RMSE = {base_rmse:.2f}s   (n={ev.height:,})")
    print("\nin-sample = fit decile means on ALL holdout rows (optimistic, an")
    print("upper bound). transfer = fit on Jan, apply to Jul and vice versa")
    print("(honest -- this is what a real feature would have to achieve).\n")
    print(f"  {'feature':<32}{'in-sample':>12}{'transfer':>12}   (RMSE gain, s)")

    rows = []
    for col in FEATURES:
        d = ev.filter(pl.col(col).is_not_null())
        if d.height < 5000:
            continue
        d = d.with_columns(
            b=((pl.col(col).rank("ordinal") - 1) * N_BUCKETS // pl.len())
        )
        # in-sample: MSE reduction = E[bias^2] over deciles
        g = d.group_by("b").agg(pl.col("resid").mean().alias("mu"),
                                pl.len().alias("n"))
        red = g.select(
            (pl.col("n") * pl.col("mu").pow(2)).sum() / pl.col("n").sum()
        ).item()
        mse_here = d.select(pl.col("resid").pow(2).mean()).item()
        share = d.height / ev.height
        # gain on the FULL holdout: only the covered rows improve
        new_mse = base_mse - share * red
        gain_in = base_rmse - new_mse ** 0.5

        # transfer: decile means from the other month
        gain_tr = 0.0
        for src, dst in [("2025-01", "2025-07"), ("2025-07", "2025-01")]:
            tr = d.filter(pl.col("ym") == src)
            te = d.filter(pl.col("ym") == dst)
            if tr.height < 1000 or te.height < 1000:
                continue
            # bucket edges from the source month, applied to the target
            edges = tr.select(
                pl.col(col).quantile(q / N_BUCKETS).alias(f"q{q}")
                for q in range(1, N_BUCKETS)
            ).row(0)
            bexpr = pl.lit(0, dtype=pl.Int32)
            for e in edges:
                bexpr = bexpr + (pl.col(col) > e).cast(pl.Int32)
            mu = tr.with_columns(b=bexpr).group_by("b").agg(
                pl.col("resid").mean().alias("mu_src")
            )
            t = te.with_columns(b=bexpr).join(mu, on="b", how="left").with_columns(
                pl.col("mu_src").fill_null(0.0)
            )
            before = t.select(pl.col("resid").pow(2).mean()).item()
            after = t.select((pl.col("resid") - pl.col("mu_src")).pow(2).mean()).item()
            gain_tr += (before - after) * t.height
        gain_tr = gain_tr / ev.height
        # convert an MSE reduction on the full holdout into an RMSE gain
        gain_tr_rmse = base_rmse - max(base_mse - gain_tr, 1e-9) ** 0.5
        rows.append((col, gain_in, gain_tr_rmse, d.height))
        print(f"  {col:<32}{gain_in:>12.3f}{gain_tr_rmse:>12.3f}   n={d.height:,}")

    best_in = max(r[1] for r in rows)
    best_tr = max(r[2] for r in rows)
    print(f"\n  best single feature: in-sample {best_in:.3f}s, transfer {best_tr:.3f}s")
    print("  (measured -0.4s from the actual retrain sits inside this range)")


def _decomp(ev: pl.DataFrame, keys: list[str], label: str, base_var: float) -> None:
    """Between/within decomposition of residual variance over `keys` cells,
    with a noise correction: under pure noise, k cells of a random split
    already capture ~(k-1)/N of the variance."""
    g = ev.group_by(keys).agg(pl.col("resid").mean().alias("mu"),
                              pl.len().alias("n"))
    grand = ev.select(pl.col("resid").mean()).item()
    between = g.select(
        (pl.col("n") * (pl.col("mu") - grand).pow(2)).sum() / pl.col("n").sum()
    ).item()
    k, N = g.height, ev.height
    # expected between-variance from noise alone with k cells
    noise = base_var * (k - 1) / N
    adj = max(between - noise, 0.0)
    rmse0 = base_var ** 0.5
    gain_raw = rmse0 - max(base_var - between, 1e-9) ** 0.5
    gain_adj = rmse0 - max(base_var - adj, 1e-9) ** 0.5
    print(f"\n  {label}")
    print(f"    cells={k:,}  between={between:,.0f} ({between / base_var * 100:.1f}%)"
          f"  noise-expected={noise:,.0f}")
    print(f"    naive oracle RMSE gain={gain_raw:.2f}s   "
          f"noise-corrected={gain_adj:.2f}s")


def part_b(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\nB. Honest oracle ceiling by feature granularity")
    print(RULE)
    base_var = ev.select(pl.col("resid").pow(2).mean()).item()
    print(f"\nproduction residual MSE = {base_var:,.0f}  (RMSE {base_var ** 0.5:.1f}s)")
    print("An oracle knowing the TRUE mean residual of each cell is the best any")
    print("feature at that granularity could ever do. Noise correction matters:")
    print("small cells fit noise for free.")

    _decomp(ev, ["ADEP_mvt", "day"], "(airport, day) -- bounds ALL ADEP-side features",
            base_var)
    des = ev.filter(pl.col("des_atfm_reg_share").is_not_null())
    print(f"\n  [ADES-side rows only: n={des.height:,}]")
    bv_des = des.select(pl.col("resid").pow(2).mean()).item()
    _decomp(des, ["ADEP_mvt", "day"], "(airport, day), ADES-covered rows", bv_des)
    _decomp(des, ["ADEP_mvt", "day", "des_atfm_reg_share"],
            "(airport, day, ADES reg_share) -- bounds the FULL dual join", bv_des)
    _decomp(ev, ["ADEP_mvt", "day", "hour"] if "hour" in ev.columns
            else ["ADEP_mvt", "day"],
            "(airport, day, hour) -- what HOURLY data could reach", base_var)


def main() -> None:
    ev = _load()
    # hour is not in the ev frame; pull it for the granularity comparison
    f = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", "hour"
    )
    ev = ev.join(f, on="MVT_ID_mvt", how="left")
    part_a(ev)
    part_b(ev)


if __name__ == "__main__":
    main()
