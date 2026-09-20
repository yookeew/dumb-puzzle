"""Pre-screen OSM routed taxi distance BEFORE building any geometry.

CLAUDE.md's "Future Step B" (per-airport OSM graph, stand -> runway
threshold shortest path) has been deferred since day one. tests/atfm_ceiling.py
(S20) established the habit of bounding a candidate feature at its native
granularity first. Routed distance is a deterministic function of
(airport, stand, runway), so that cell structure bounds it exactly.

The catch, and the reason this needs care: the model ALREADY has STAND_mvt
and RUNWAY_mvt as categoricals plus per-stand/-runway unimpeded-taxi priors.
So a (airport, stand, runway) oracle largely re-measures what is already in
the model. What routed distance can add over that is:
  (a) metric structure -- "100m further" means something, a category code
      does not; and
  (b) generalisation to rare or unseen (stand, runway) combinations, where
      a categorical has little or no training support.

So this reports BOTH: the raw oracle bound, and -- more importantly -- the
bound restricted to thin-support cells, which is where geometry should
actually pay if it pays at all.

Run:  .venv/Scripts/python.exe tests/geometry_screen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"
RULE = "=" * 78


def _decomp(ev: pl.DataFrame, keys: list[str], label: str,
            base_var: float) -> tuple[float, float]:
    """Noise-corrected between-cell oracle gain (same method as S20)."""
    g = ev.group_by(keys).agg(pl.col("resid").mean().alias("mu"),
                              pl.len().alias("n"))
    grand = ev.select(pl.col("resid").mean()).item()
    between = g.select(
        (pl.col("n") * (pl.col("mu") - grand).pow(2)).sum() / pl.col("n").sum()
    ).item()
    k, N = g.height, ev.height
    noise = base_var * (k - 1) / N
    adj = max(between - noise, 0.0)
    r0 = base_var ** 0.5
    raw = r0 - max(base_var - between, 1e-9) ** 0.5
    cor = r0 - max(base_var - adj, 1e-9) ** 0.5
    print(f"  {label:<44}{k:>8,}{raw:>10.2f}{cor:>12.2f}")
    return raw, cor


def main() -> None:
    ev = pl.read_parquet(PROD_EV).with_columns(
        resid=pl.col("taxi") - pl.col("pred"))
    # training support per (airport, stand, runway) -- from the TRAIN frame,
    # not the holdout, since that is what the categorical actually learned from
    tr = pl.read_parquet(FEAT_DIR / "train2025.parquet").select(
        "MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt")
    lab = pl.read_parquet(FEAT_DIR / "labels2025.parquet").select("MVT_ID_mvt", "ym")
    tr = tr.join(lab, on="MVT_ID_mvt", how="left").filter(
        ~pl.col("ym").is_in(["2025-01", "2025-07"]))
    support = tr.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").agg(
        pl.len().alias("support"))
    ev = ev.join(support, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
                 how="left").with_columns(pl.col("support").fill_null(0))

    base_var = ev.select(pl.col("resid").pow(2).mean()).item()
    print(f"\n{RULE}\nORACLE BOUNDS AT GEOMETRY-RELEVANT GRANULARITY\n{RULE}")
    print(f"\nproduction residual MSE = {base_var:,.0f} (RMSE {base_var ** 0.5:.1f}s), "
          f"n={ev.height:,}")
    print("\nRouted distance is a deterministic function of (airport, stand,")
    print("runway), so that cell structure bounds what it could ever add.")
    print(f"\n  {'granularity':<44}{'cells':>8}{'raw':>10}{'corrected':>12}")
    _decomp(ev, ["ADEP_mvt"], "airport", base_var)
    _decomp(ev, ["ADEP_mvt", "RUNWAY_mvt"], "airport + runway", base_var)
    _decomp(ev, ["ADEP_mvt", "STAND_mvt"], "airport + stand", base_var)
    _decomp(ev, ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
            "airport + stand + runway  <- routed distance", base_var)

    print(f"\n{RULE}\nWHERE GEOMETRY SHOULD ACTUALLY PAY: THIN-SUPPORT CELLS\n{RULE}")
    print("\nA categorical memorises well-sampled (stand, runway) pairs. Routed")
    print("distance helps most where training support is thin -- there the")
    print("category code carries little and a metric generalises.")
    print(f"\n  {'training support':<20}{'rows':>10}{'% rows':>9}{'RMSE':>9}"
          f"{'% of MSE':>11}")
    tot = ev.select(pl.col("resid").pow(2).sum()).item()
    for lo, hi, lab in [(0, 1, "0 (unseen)"), (1, 10, "1-9"), (10, 50, "10-49"),
                        (50, 200, "50-199"), (200, 10**9, ">=200")]:
        d = ev.filter((pl.col("support") >= lo) & (pl.col("support") < hi))
        if d.height == 0:
            continue
        se = d.select(pl.col("resid").pow(2).sum()).item()
        print(f"  {lab:<20}{d.height:>10,}{d.height / ev.height * 100:>8.2f}%"
              f"{(se / d.height) ** 0.5:>9.1f}{se / tot * 100:>10.2f}%")

    thin = ev.filter(pl.col("support") < 50)
    if thin.height > 1000:
        bv = thin.select(pl.col("resid").pow(2).mean()).item()
        print(f"\n  thin-support subset (<50): n={thin.height:,}, "
              f"RMSE={bv ** 0.5:.1f}s, {thin.select(pl.col('resid').pow(2).sum()).item() / tot * 100:.2f}% of MSE")
        print(f"\n  {'granularity':<44}{'cells':>8}{'raw':>10}{'corrected':>12}")
        _, cor = _decomp(thin, ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
                         "stand+runway within thin support", bv)
        # translate a gain on the subset into a gain on the full holdout
        share = thin.height / ev.height
        full = base_var ** 0.5 - max(base_var - share * (bv - (bv ** 0.5 - cor) ** 2),
                                     1e-9) ** 0.5
        print(f"\n  => perfectly solving that subset is worth {full:.2f}s "
              f"on the FULL holdout")


if __name__ == "__main__":
    main()
