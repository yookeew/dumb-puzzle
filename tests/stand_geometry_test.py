"""Pre-registered test of family 8 (stand/runway geometry, features/geometry.py)
against the production feature set. Same discipline as
tests/metar_weather_test.py:

  PRIMARY METRIC:  paired overall RMSE delta (treatment - baseline) on the
                   Jan+Jul 2025 holdout, target = DEFAULT_TARGET ("mixed").
  UNCERTAINTY:     (airport, day) cluster bootstrap, 3000 resamples.
  BOTH-MONTHS RULE: adopt only if both 2025-01 and 2025-07 improve.
  MECHANISM TEST:  STAND_mvt and RUNWAY_mvt are already categorical features
                   and feed the taxi priors, so for frequent (stand, runway)
                   pairs the model has already learned the geometry. Distance
                   can only add information where those splits are data-poor:
                   the gain must CONCENTRATE on holdout rows whose
                   (airport, stand, runway) pair has < RARE_N training
                   departures. A uniform gain means the columns act as generic
                   extra capacity, not geometry.

Treatment = the production feature frames (cache/features/) with the five
geometry columns left-joined on (a pure left join, so baseline is FEAT_DIR
as-is). Refuses to run if FEAT_DIR already contains them.

Run:  .venv/Scripts/python.exe tests/stand_geometry_test.py [target] [seed]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.geometry import GEOM_COLS, geometry_context  # noqa: E402
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET, cluster_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
GEO_DIR = ROOT / "cache" / "features_geometry_tmp"
N_RESAMPLES = 3000
MODEL_SEED = 42
BOOT_SEED = 0
RARE_N = 50


def _make_geo_dir() -> None:
    cols = pl.read_parquet(FEAT_DIR / "train2025.parquet", n_rows=1).columns
    assert not set(GEOM_COLS) & set(cols), "FEAT_DIR already has geometry columns"
    if GEO_DIR.exists():
        shutil.rmtree(GEO_DIR)
    GEO_DIR.mkdir(parents=True)
    for f in ("train2025.parquet", "holdout_gap2025.parquet", "ranking.parquet"):
        df = pl.read_parquet(FEAT_DIR / f)
        out = df.join(geometry_context(df), on="MVT_ID_mvt", how="left")
        assert out.height == df.height
        nr = out.select(pl.col("dist_stand_rwy_m").is_null().mean()).item()
        print(f"  {f}: dist_stand_rwy_m null rate {nr:.2%}  (n={out.height:,})")
        assert nr < 0.15, "geometry join null rate looks like a broken join"
        out.write_parquet(GEO_DIR / f)
    shutil.copy(FEAT_DIR / "labels2025.parquet", GEO_DIR / "labels2025.parquet")


def main(target: str = DEFAULT_TARGET, seed: int = MODEL_SEED) -> None:
    suffix = f"_{target}" + (f"_seed{seed}" if seed != MODEL_SEED else "")
    print(f"=== building treatment frames ({target=}, {seed=}) ===")
    _make_geo_dir()

    print("\n=== baseline (production features) ===")
    _, ev_base = run(engine="lgb", feat_dir=FEAT_DIR, name=f"geom_baseline{suffix}",
                     target=target, submit=False, seed=seed)
    print("\n=== treatment (+ family 8 geometry) ===")
    _, ev_treat = run(engine="lgb", feat_dir=GEO_DIR, name=f"geom_treatment{suffix}",
                      target=target, submit=False, seed=seed)

    ev = (ev_base.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
                         pred_base="pred")
          .join(ev_treat.select("MVT_ID_mvt", pred_treat="pred"), on="MVT_ID_mvt", how="inner"))
    assert ev.height == ev_base.height == ev_treat.height, "row mismatch between runs"
    ev = ev.with_columns(se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
                         se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2))
    days = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = ev.join(days, on="MVT_ID_mvt", how="left")

    def line(df: pl.DataFrame, label: str) -> None:
        rb, rt = df["se_base"].mean() ** 0.5, df["se_treat"].mean() ** 0.5
        print(f"  {label:<28} base={rb:7.1f}  treat={rt:7.1f}  delta={rt - rb:+6.2f}  n={df.height:,}")

    print("\n=== results ===")
    line(ev, "OVERALL")
    for ym in HOLDOUT_MONTHS:
        line(ev.filter(pl.col("ym") == ym), str(ym))
    print("\nper airport:")
    for ap in sorted(ev["ADEP_mvt"].unique()):
        line(ev.filter(pl.col("ADEP_mvt") == ap), ap)

    cl = ev.group_by("ADEP_mvt", "day").agg(pl.col("se_base").sum(), pl.col("se_treat").sum(),
                                            pl.len().alias("n"))
    point, lo, hi, p_worse = cluster_bootstrap(cl["se_base"].to_numpy(), cl["se_treat"].to_numpy(),
                                               cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED)
    print(f"\ncluster bootstrap: delta={point:+.2f}s  95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"P(worse)={p_worse:.3f}")

    # mechanism: rare vs common (airport, stand, runway) pairs in training
    tr = pl.read_parquet(FEAT_DIR / "train2025.parquet").select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt")
    lab = pl.read_parquet(FEAT_DIR / "labels2025.parquet").select("MVT_ID_mvt", "ym")
    tr_ids = pl.read_parquet(FEAT_DIR / "train2025.parquet").select("MVT_ID_mvt")
    tr = tr.with_columns(tr_ids["MVT_ID_mvt"]).join(lab, on="MVT_ID_mvt", how="left").filter(
        ~pl.col("ym").is_in(list(HOLDOUT_MONTHS)))
    cnt = tr.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").agg(pl.len().alias("n_train"))
    ev = ev.join(cnt, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left").with_columns(
        pl.col("n_train").fill_null(0))
    print(f"\nmechanism (pre-registered): gain should concentrate on pairs with < {RARE_N} "
          "training departures")
    line(ev.filter(pl.col("n_train") < RARE_N), f"rare (<{RARE_N})")
    line(ev.filter(pl.col("n_train") >= RARE_N), f"common (>={RARE_N})")
    ev.write_parquet(ROOT / "cache" / f"geom_ev{suffix}.parquet")


if __name__ == "__main__":
    _t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    _s = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_SEED
    main(target=_t, seed=_s)
