"""Ablate family 7 (METAR weather) into adverse-condition vs benign-condition
subsets, to find out whether the S24/S25 gain is weather-as-operations or a
proxy for time-of-year/diurnal patterns the model's existing month/hour
features encode weakly.

Pre-registered split (user-specified):
  ADVERSE = visibility, ceiling, freezing, precipitation, present-weather
            (the low-vis-procedure / de-icing mechanism)
  BENIGN  = temperature, dewpoint, wind (no pressure column -- not fetched,
            see fetch_metar_data.py's DATA_FIELDS)

If BENIGN alone produces most of the combined gain, the effect isn't
weather-as-operations -- it's more likely a seasonal/diurnal proxy. If
ADVERSE alone produces most of it, that's consistent with the mechanism this
project has assumed since CLAUDE.md's original feature priority order.

Cheap sanity check already run (not repeated here): Pearson correlation of
each weather column against month/hour is low (|r|<=0.27), but
`flight_category`'s non-VFR rate swings 6.4%(Jun)->42.6%(Dec) and
~14%(afternoon)->29%(night), a real threshold-level seasonal/diurnal
concentration linear correlation doesn't capture. Not decisive -- hence
this ablation.

Three arms, same seed/target, built by DROPPING columns from FEAT_DIR
(already has all 17 weather columns baked in as of S24/S25 adoption -- cheap,
no rebuild needed):
  baseline  = all 17 weather columns dropped (pure no-weather)
  adverse   = only the 11 ADVERSE_COLS kept
  benign    = only the 6 BENIGN_COLS kept

Same discipline as the rest of this project's feature tests: paired per-row
squared errors on identical holdout rows, (airport, day) cluster bootstrap,
both months reported separately, LIRF reported specifically (S25 lesson --
LIRF absorbs collateral damage other airports don't, so its own number needs
watching, not just the pooled one).

Run:  .venv/Scripts/python.exe tests/metar_ablation_test.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.fit import HOLDOUT_MONTHS, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import DEFAULT_TARGET, cluster_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
BASE_DIR = ROOT / "cache" / "features_ablate_base_tmp"
ADVERSE_DIR = ROOT / "cache" / "features_ablate_adverse_tmp"
BENIGN_DIR = ROOT / "cache" / "features_ablate_benign_tmp"

ADVERSE_COLS = [
    "has_ceiling", "vis_mi", "ceiling_ft", "below_freezing", "deicing_risk",
    "wx_snow", "wx_freezing", "wx_tstorm", "wx_obscuration", "flight_category",
    "precip_1h_in",
]
BENIGN_COLS = ["temp_c", "dewpoint_c", "temp_dewpoint_spread_c", "wind_dir", "wind_kt", "gust_kt"]
ALL_WX_COLS = ADVERSE_COLS + BENIGN_COLS

MODEL_SEED = 42
N_RESAMPLES = 3000
BOOT_SEED = 0


def _prep(dest: Path, keep_extra: list[str]) -> None:
    """dest gets FEAT_DIR with (ALL_WX_COLS - keep_extra) dropped -- i.e.
    keeps only keep_extra of the weather columns, drops the rest."""
    drop = [c for c in ALL_WX_COLS if c not in keep_extra]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in ("train2025.parquet", "holdout_gap2025.parquet"):
        df = pl.read_parquet(FEAT_DIR / f)
        df.drop([c for c in drop if c in df.columns]).write_parquet(dest / f)
    shutil.copy(FEAT_DIR / "labels2025.parquet", dest / "labels2025.parquet")


def _fit(dest: Path, name: str, target: str, seed: int):
    return run(engine="lgb", feat_dir=dest, name=name, target=target,
               submit=False, seed=seed)


def _report(label: str, ev_a: pl.DataFrame, ev_b: pl.DataFrame, tag: str) -> pl.DataFrame:
    """ev_a = baseline, ev_b = treatment (some weather subset). Returns the
    paired ev frame with se_base/se_treat for further use."""
    ev = (
        ev_a.select("MVT_ID_mvt", "taxi", "ym", "ADEP_mvt", pred_base="pred")
        .join(ev_b.select("MVT_ID_mvt", pred_treat="pred"), on="MVT_ID_mvt", how="inner")
    )
    assert ev.height == ev_a.height == ev_b.height, f"row mismatch ({tag})"
    day = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", day=pl.col("T").dt.date())
    ev = ev.join(day, on="MVT_ID_mvt", how="left").with_columns(
        se_base=(pl.col("pred_base") - pl.col("taxi")).pow(2),
        se_treat=(pl.col("pred_treat") - pl.col("taxi")).pow(2),
    )

    rb, rt = ev["se_base"].mean() ** 0.5, ev["se_treat"].mean() ** 0.5
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"overall RMSE   baseline={rb:.1f}  treatment={rt:.1f}  delta={rt - rb:+.1f}")
    for ym in HOLDOUT_MONTHS:
        s = ev.filter(pl.col("ym") == ym)
        a, b = s["se_base"].mean() ** 0.5, s["se_treat"].mean() ** 0.5
        print(f"  {ym}: baseline={a:.1f}  treatment={b:.1f}  delta={b - a:+.1f}")

    lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF")
    la, lb = lirf["se_base"].mean() ** 0.5, lirf["se_treat"].mean() ** 0.5
    print(f"  LIRF: baseline={la:.1f}  treatment={lb:.1f}  delta={lb - la:+.1f}  n={lirf.height:,}")

    cl = ev.group_by("ADEP_mvt", "day").agg(
        pl.col("se_base").sum().alias("sb"), pl.col("se_treat").sum().alias("st"),
        pl.len().alias("n"))
    point, lo, hi, pw = cluster_bootstrap(cl["sb"].to_numpy(), cl["st"].to_numpy(),
                                         cl["n"].to_numpy(), N_RESAMPLES, BOOT_SEED)
    print(f"cluster bootstrap: paired RMSE delta={point:+.2f}s  95% CI=[{lo:+.2f}, {hi:+.2f}]  "
          f"P(worse)={pw:.3f}")
    return ev


def main(target: str = DEFAULT_TARGET, seed: int = MODEL_SEED) -> None:
    print(f"=== building 3 ablation arms (target={target}, seed={seed}) ===")
    _prep(BASE_DIR, keep_extra=[])
    _prep(ADVERSE_DIR, keep_extra=ADVERSE_COLS)
    _prep(BENIGN_DIR, keep_extra=BENIGN_COLS)

    print("\n=== fitting baseline (no weather) ===")
    _, ev_base = _fit(BASE_DIR, "ablate_base", target, seed)

    print("\n=== fitting adverse-only (vis/ceiling/freezing/precip) ===")
    _, ev_adverse = _fit(ADVERSE_DIR, "ablate_adverse", target, seed)

    print("\n=== fitting benign-only (temp/dewpoint/wind) ===")
    _, ev_benign = _fit(BENIGN_DIR, "ablate_benign", target, seed)

    ev_a = _report("ADVERSE-ONLY vs baseline", ev_base, ev_adverse, "adverse")
    ev_b = _report("BENIGN-ONLY vs baseline", ev_base, ev_benign, "benign")

    print(f"\n{'=' * 78}\nSUMMARY -- which subset explains the combined gain?\n{'=' * 78}")
    print("Combined-weather reference (S24/S25, seed42, mixed): overall -1.21s, "
          "LIRF -4.9s.")
    rb = ev_a["se_base"].mean() ** 0.5
    ra = ev_a["se_treat"].mean() ** 0.5
    rb2 = ev_b["se_base"].mean() ** 0.5
    rbenign = ev_b["se_treat"].mean() ** 0.5
    print(f"  adverse-only delta: {ra - rb:+.2f}s")
    print(f"  benign-only  delta: {rbenign - rb2:+.2f}s")
    combined_ref = -1.21
    if abs(ra - rb) > abs(rbenign - rb2):
        print("  -> ADVERSE dominates: consistent with a weather-as-operations "
              "mechanism (low-vis/de-icing).")
    else:
        print("  -> BENIGN dominates: consistent with a seasonal/diurnal proxy, "
              "not weather-as-operations specifically.")
    print(f"  (reference: combined gain was {combined_ref:+.2f}s -- if adverse+benign "
          "sum roughly to this, the two subsets are close to additive/independent; "
          "if not, there's interaction between them.)")


if __name__ == "__main__":
    _t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    _s = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_SEED
    main(target=_t, seed=_s)
