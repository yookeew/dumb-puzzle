"""Is there any signal left to find, or is the model at the information floor?

Motivated by the hypothesis that recent feature attempts (S19 arrival
disruption, S20 ATFM, S21 ADES, teammate's weather) all came back flat
because the model is overfitting existing data and has no room for more.

This pass is FREE -- it reads the saved production holdout frame
(cache/lirf_ceilfix_mixed_ev.parquet = smart-jigsaw_v12, 337.6s local /
302s board) and asks where the remaining error actually is:

  1. concentration -- how much of total MSE lives in how few rows
  2. composition  -- are those rows the known-pathological populations
                     (echo / NM-unmatched / LIRF tail) or ordinary flights
  3. clean-lane   -- what RMSE looks like with the pathologies removed
  4. noise floor  -- within-group spread of taxi for near-identical flights.
                     No feature set can beat the irreducible component; if
                     the model's residual is already at that spread, new
                     features cannot help regardless of what they measure.

Run:  .venv/Scripts/python.exe tests/error_floor.py
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


def _load() -> pl.DataFrame:
    f = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", "hour", "AIRCRAFT_TYPE_mvt", "stand_group",
        "n_dep_30m_prev", day=pl.col("T").dt.date()
    )
    ev = pl.read_parquet(PROD_EV).join(f, on="MVT_ID_mvt", how="left")
    return ev.with_columns(
        resid=pl.col("taxi") - pl.col("pred"),
        se=(pl.col("taxi") - pl.col("pred")).pow(2),
    )


def concentration(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n1. ERROR CONCENTRATION\n{RULE}")
    tot = ev["se"].sum()
    n = ev.height
    rmse = (tot / n) ** 0.5
    print(f"\nproduction holdout: RMSE={rmse:.1f}s  n={n:,}")
    print(f"\n  {'top-k rows':<14}{'% of total MSE':>16}{'RMSE if solved':>18}")
    s = ev.sort("se", descending=True)
    for k in (10, 50, 100, 500, 1000, 5000):
        share = s.head(k)["se"].sum() / tot
        rest = (tot - s.head(k)["se"].sum()) / n
        print(f"  {k:<14,}{share * 100:>15.1f}%{rest ** 0.5:>17.1f}s")
    for pct in (1.0, 5.0):
        k = int(n * pct / 100)
        share = s.head(k)["se"].sum() / tot
        rest = (tot - s.head(k)["se"].sum()) / n
        print(f"  top {pct}% ({k:,}){share * 100:>9.1f}%{rest ** 0.5:>17.1f}s")


def composition(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n2. WHAT ARE THE WORST ROWS?\n{RULE}")
    tot = ev["se"].sum()
    s = ev.sort("se", descending=True)
    for k in (100, 1000):
        t = s.head(k)
        print(f"\n  top {k} rows by squared error ({t['se'].sum() / tot * 100:.1f}% of MSE):")
        print(f"    is_echo (|d|<30s):        {t['is_echo'].mean() * 100:5.1f}%   "
              f"(holdout base rate {ev['is_echo'].mean() * 100:.1f}%)")
        print(f"    has_aobt3 missing:        {(~t['has_aobt3']).mean() * 100:5.1f}%   "
              f"(base rate {(~ev['has_aobt3']).mean() * 100:.1f}%)")
        print(f"    use_prior (NM-unmatched): {t['use_prior'].mean() * 100:5.1f}%   "
              f"(base rate {ev['use_prior'].mean() * 100:.1f}%)")
        print(f"    LIRF:                     {(t['ADEP_mvt'] == 'LIRF').mean() * 100:5.1f}%   "
              f"(base rate {(ev['ADEP_mvt'] == 'LIRF').mean() * 100:.1f}%)")
        print(f"    median true taxi:         {t['taxi'].median():.0f}s   "
              f"(holdout median {ev['taxi'].median():.0f}s)")


def clean_lane(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n3. THE ORDINARY POPULATION\n{RULE}")
    lanes = [
        ("everything", ev),
        ("has_aobt3 only", ev.filter(pl.col("has_aobt3"))),
        ("has_aobt3 & not echo", ev.filter(pl.col("has_aobt3") & ~pl.col("is_echo"))),
        ("...& not LIRF", ev.filter(pl.col("has_aobt3") & ~pl.col("is_echo")
                                    & (pl.col("ADEP_mvt") != "LIRF"))),
        ("...& taxi<=3600s", ev.filter(pl.col("has_aobt3") & ~pl.col("is_echo")
                                       & (pl.col("ADEP_mvt") != "LIRF")
                                       & (pl.col("taxi") <= 3600))),
    ]
    print(f"\n  {'lane':<26}{'n':>10}{'% rows':>9}{'RMSE':>9}{'% of tot MSE':>14}")
    tot = ev["se"].sum()
    for name, d in lanes:
        print(f"  {name:<26}{d.height:>10,}{d.height / ev.height * 100:>8.1f}%"
              f"{(d['se'].mean()) ** 0.5:>9.1f}{d['se'].sum() / tot * 100:>13.1f}%")


def noise_floor(ev: pl.DataFrame) -> None:
    print(f"\n{RULE}\n4. IRREDUCIBLE NOISE FLOOR (near-identical flights)\n{RULE}")
    print("\nFlights matched on progressively more of what's observable. The")
    print("within-group sd of TRUE taxi bounds what ANY model could achieve on")
    print("rows it cannot otherwise distinguish. Compare to the model's own")
    print("residual sd on the same rows.")
    clean = ev.filter(pl.col("has_aobt3") & ~pl.col("is_echo")
                      & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("taxi") <= 3600))
    print(f"\n  (computed on the ordinary lane: n={clean.height:,}, "
          f"model RMSE={clean['se'].mean() ** 0.5:.1f}s)")
    keysets = [
        (["ADEP_mvt", "day"], "airport + day"),
        (["ADEP_mvt", "day", "hour"], "+ hour"),
        (["ADEP_mvt", "day", "hour", "RUNWAY_mvt"], "+ runway"),
        (["ADEP_mvt", "day", "hour", "RUNWAY_mvt", "stand_group"], "+ stand group"),
        (["ADEP_mvt", "day", "hour", "RUNWAY_mvt", "stand_group",
          "AIRCRAFT_TYPE_mvt"], "+ aircraft type"),
    ]
    print(f"\n  {'matched on':<26}{'groups':>9}{'rows':>10}{'within sd':>12}{'model resid sd':>16}")
    for keys, label in keysets:
        g = clean.group_by(keys).agg(
            pl.col("taxi").std().alias("sd"),
            pl.col("resid").std().alias("rsd"),
            pl.len().alias("n"),
        ).filter(pl.col("n") >= 3)
        if g.height == 0:
            continue
        rows = g["n"].sum()
        # pooled within-group sd, weighted by (n-1)
        pooled = (
            g.select(((pl.col("n") - 1) * pl.col("sd").pow(2)).sum()
                     / (pl.col("n") - 1).sum()).item() ** 0.5
        )
        pooled_r = (
            g.select(((pl.col("n") - 1) * pl.col("rsd").pow(2)).sum()
                     / (pl.col("n") - 1).sum()).item() ** 0.5
        )
        print(f"  {label:<26}{g.height:>9,}{rows:>10,}{pooled:>12.1f}{pooled_r:>16.1f}")
    print("\n  If 'model resid sd' has already fallen to ~'within sd' at the")
    print("  finest matching, the model is extracting essentially everything")
    print("  these covariates contain and new features must bring genuinely")
    print("  orthogonal information to beat it.")


def main() -> None:
    ev = _load()
    concentration(ev)
    composition(ev)
    clean_lane(ev)
    noise_floor(ev)


if __name__ == "__main__":
    main()
