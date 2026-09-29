"""Screen 1 (PROGRESS.md §50): ADS-B tracks first seen already taxiing.

Population: departures matched to a takeoff-ending surface run (adsb_matched)
that have no appear/dwell pushback AND are not eligible for the §41
partial-track stage (src/post/adsb_partial.mask(): first sighting < 1 km from a
known own stand, LIRF excluded). What's left: LIRF, first sighting >= 1 km from
the stand, or no stand coordinates.

L = MVT_TIME - adsb_first_ts is a floor on taxi (unless the match is wrong).
Explored ONLY on the training-month ADS-B days (2025-09-15, 2025-11-15) with
the OOF CatBoost (mixed) fold prediction as the base. Jan/Jul 2025 labels are
not read; for those days only row counts are reported.

No training, no stage, no submission.

Run:  .venv/Scripts/python.exe tests/screen1_adsb_taxiing_floor.py > logs/screen1_adsb_taxiing_floor.log
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from post import adsb_partial  # noqa: E402

DET = ROOT / "cache" / "adsb_pushback"
OOF = ROOT / "cache" / "oof" / "cat_mixed"
LAB = ROOT / "cache" / "features" / "labels2025.parquet"
RAW = ROOT / "data" / "raw"
TRAIN_DAYS = ("2025-09-15", "2025-11-15")
MONSTER_S = 5 * 3600
MARGINS = (0, 60, 120)

AD = pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False)
MATCHED = pl.col("adsb_matched").fill_null(False)


def partial() -> pl.Expr:
    return adsb_partial.mask()          # the §41 stage's own eligibility, verbatim


def pop() -> pl.Expr:
    return MATCHED & ~AD & ~partial()


def reason() -> pl.Expr:
    return (pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.lit("LIRF"))
            .when(pl.col("adsb_first_own_m").is_null()).then(pl.lit("no stand coords"))
            .otherwise(pl.lit(">= 1 km from stand")))


def load_det() -> pl.DataFrame:
    files = sorted(DET.glob("day=*.parquet"))
    print(f"detector files: {len(files)}  newest mtime "
          f"{dt.datetime.fromtimestamp(max(f.stat().st_mtime for f in files)):%Y-%m-%d %H:%M}")
    d = pl.concat([pl.read_parquet(f).with_columns(day=pl.lit(f.stem.split("=", 1)[1]))
                   for f in files])
    return d.rename({"airport": "ADEP_mvt"})


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_float_precision(2)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    det = load_det().with_columns(ad=AD, part=partial(), pop=pop())
    det = det.with_columns(grp=pl.when(pl.col("day").str.starts_with("2026")).then(pl.lit("ranking 2026"))
                           .when(pl.col("day").is_in(TRAIN_DAYS)).then(pl.lit("2025 training days"))
                           .otherwise(pl.lit("2025 holdout days (Jan/Jul)")))

    # ------------------------------------------------------------------ 1
    section("1. POPULATION")
    ov_ad = det.filter(pl.col("pop") & pl.col("ad")).height
    ov_pt = det.filter(pl.col("pop") & pl.col("part")).height
    ov_adpt = det.filter(pl.col("ad") & pl.col("part")).height
    print(f"overlap pop & appear/dwell = {ov_ad}; pop & partial = {ov_pt}; "
          f"(appear/dwell & partial = {ov_adpt})")
    if ov_ad or ov_pt:
        raise SystemExit("population overlaps an existing stage")
    rank_ids = pl.read_parquet(ROOT / "data" / "ranking" / "submitting.parquet")["MVT_ID_mvt"].cast(pl.Int64)
    d26 = det.filter(pl.col("grp") == "ranking 2026")
    print(f"ranking DEP rows {rank_ids.len():,}; detector 2026 rows {d26.height:,}; "
          f"in submitting template {d26['MVT_ID_mvt'].is_in(rank_ids.implode()).sum():,}")
    summ = det.group_by("grp").agg(
        pl.len().alias("dep"), pl.col("adsb_matched").sum().alias("matched"),
        pl.col("ad").sum().alias("appear_dwell"), pl.col("part").sum().alias("partial"),
        pl.col("pop").sum().alias("POP")).with_columns(
        (100 * pl.col("POP") / pl.col("dep")).alias("POP_%dep")).sort("grp")
    print(summ)
    print("\nPOP by reason:")
    print(det.filter("pop").with_columns(reason=reason()).group_by("grp", "reason").len()
          .pivot(on="reason", index="grp", values="len").sort("grp"))
    print("\nPOP by tier (null = censored, never reached the stand):")
    print(det.filter("pop").group_by("grp", "adsb_tier").len()
          .pivot(on="adsb_tier", index="grp", values="len").sort("grp"))
    print("\nPOP share of DEP by airport, ranking 2026 vs 2025 training days:")
    print(det.filter(pl.col("grp") != "2025 holdout days (Jan/Jul)").group_by("grp", "ADEP_mvt")
          .agg(pl.len().alias("dep"), pl.col("pop").sum().alias("pop"))
          .with_columns(pct=100 * pl.col("pop") / pl.col("dep"))
          .pivot(on="grp", index="ADEP_mvt", values=["pop", "pct"]).sort("ADEP_mvt"))

    # ------------------------------------------------------------------ 2
    section("2. FLOOR L = MVT_TIME - adsb_first_ts ON TRAINING DAYS (base = OOF CatBoost mixed)")
    oof = pl.concat([pl.read_parquet(OOF / f"fold={d[:7]}.parquet") for d in TRAIN_DAYS])
    lab = pl.read_parquet(LAB).filter(pl.col("ym").is_in([d[:7] for d in TRAIN_DAYS]))
    mvt = (pl.concat([pl.scan_parquet(next(RAW.glob(f"training_{d[:7]}-01_*.parquet"))) for d in TRAIN_DAYS])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64), mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0)
           .collect())
    tr = (det.filter(pl.col("day").is_in(TRAIN_DAYS))
          .join(mvt, on="MVT_ID_mvt", how="left")
          .join(oof.select(pl.col("MVT_ID_mvt").cast(pl.Int64), "pred"), on="MVT_ID_mvt", how="left")
          .join(lab.select(pl.col("MVT_ID_mvt").cast(pl.Int64), "taxi"), on="MVT_ID_mvt", how="left")
          .with_columns(L=pl.col("mvt_ts") - pl.col("adsb_first_ts"), reason=reason()))
    print(f"training-day DEP rows {tr.height:,}; with label {tr['taxi'].is_not_null().sum():,}; "
          f"with OOF pred {tr['pred'].is_not_null().sum():,}")
    p = tr.filter(pl.col("pop") & pl.col("taxi").is_not_null() & pl.col("pred").is_not_null()
                  & (pl.col("taxi") >= 0))
    print(f"POP rows with label + OOF pred: {p.height:,}  "
          f"(monster labels > 5 h in POP: {(p['taxi'] > MONSTER_S).sum()})")
    print(p.group_by("day", "reason").len().pivot(on="reason", index="day", values="len").sort("day"))
    print("\nL distribution (s):", p.select(pl.col("L").quantile(q).alias(f"q{int(q * 100)}")
                                           for q in (0.01, 0.1, 0.5, 0.9, 0.99)).row(0))

    print("\n2a. match-error proxy: taxi < L - 60 s")
    print(p.group_by("day", "reason").agg(pl.len().alias("n"),
          (100 * (pl.col("taxi") < pl.col("L") - 60).mean()).alias("pct_taxi_lt_L-60"),
          (100 * (pl.col("taxi") < pl.col("L") - 300).mean()).alias("pct_taxi_lt_L-300"))
          .sort("day", "reason"))
    print("pooled:", p.select(n=pl.len(), pct_lt_L60=100 * (pl.col("taxi") < pl.col("L") - 60).mean(),
                               pct_lt_L300=100 * (pl.col("taxi") < pl.col("L") - 300).mean()).row(0))

    print("\n2b-d. pred -> max(pred, L - m), per day (SSE in units of 1e6 s^2)")
    rows = []
    for day in (*TRAIN_DAYS, "pooled"):
        f = p if day == "pooled" else p.filter(pl.col("day") == day)
        t, b, L = f["taxi"].to_numpy().astype(float), f["pred"].to_numpy(), f["L"].to_numpy()
        keep = t <= MONSTER_S
        se_b = (b - t) ** 2
        for m in MARGINS:
            new = np.maximum(b, L - m)
            hit = b < L - m
            se_n = (new - t) ** 2
            rows.append(dict(day=day, m=m, n=len(t), pct_hit=100 * hit.mean(),
                             hit_share_of_SSE=100 * se_b[hit].sum() / se_b.sum(),
                             SSE_full_base=se_b.sum() / 1e6, dSSE_full=(se_n - se_b).sum() / 1e6,
                             dSSE_trim=(se_n - se_b)[keep].sum() / 1e6,
                             rmse_pop_base=np.sqrt(se_b.mean()), rmse_pop_new=np.sqrt(se_n.mean()),
                             rmse_pop_trim_base=np.sqrt(se_b[keep].mean()),
                             rmse_pop_trim_new=np.sqrt(se_n[keep].mean()),
                             mean_lift_on_hit=float((new - b)[hit].mean()) if hit.any() else 0.0,
                             pct_hit_improved=100 * float((se_n < se_b)[hit].mean()) if hit.any() else 0.0))
    res = pl.DataFrame(rows)
    print(res)
    print("\nsame, by reason (pooled days, m = 60):")
    rr = []
    for (rs,), f in p.group_by("reason"):
        t, b, L = f["taxi"].to_numpy().astype(float), f["pred"].to_numpy(), f["L"].to_numpy()
        new = np.maximum(b, L - 60)
        rr.append(dict(reason=rs, n=len(t), pct_hit=100 * (b < L - 60).mean(),
                       dSSE_full=((new - t) ** 2 - (b - t) ** 2).sum() / 1e6,
                       pct_taxi_lt_L60=100 * (t < L - 60).mean()))
    print(pl.DataFrame(rr).sort("reason"))
    print("\n2e. mechanism: does the run start before this departure's own inbound in-block?")
    feats = pl.read_parquet(ROOT / "cache" / "features" / "train2025.parquet",
                            columns=["MVT_ID_mvt", "inbound_mvt_id", "link_confidence"])
    arr = (pl.concat([pl.scan_parquet(next(RAW.glob(f"training_{d[:7]}-01_*.parquet"))) for d in TRAIN_DAYS])
           .filter(pl.col("PHASE_mvt") == "ARR")
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("inbound_mvt_id"),
                   in_block_ts=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                   landing_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0)
           .collect())
    mech = (p.join(feats.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64),
                                      pl.col("inbound_mvt_id").cast(pl.Int64)), on="MVT_ID_mvt", how="left")
            .join(arr, on="inbound_mvt_id", how="left")
            .with_columns(hit=pl.col("pred") < pl.col("L") - 60,
                          starts_before_inblock=pl.col("adsb_first_ts") < pl.col("in_block_ts"),
                          starts_near_landing=(pl.col("adsb_first_ts") - pl.col("landing_ts")).abs() < 600))
    print(mech.group_by("hit").agg(
        pl.len().alias("n"), pl.col("in_block_ts").is_not_null().mean().alias("has_inbound"),
        (100 * pl.col("starts_before_inblock").mean()).alias("pct_run_starts_before_inbound_inblock"),
        (100 * pl.col("starts_near_landing").mean()).alias("pct_run_starts_within_10min_of_inbound_landing"),
        (100 * (pl.col("taxi") < pl.col("L") - 60).mean()).alias("pct_taxi_lt_L-60"),
        pl.col("L").median().alias("med_L"), pl.col("taxi").median().alias("med_taxi"),
        pl.col("adsb_n_pts").median().alias("med_pts")).sort("hit"))
    ok = mech.filter(~pl.col("starts_before_inblock").fill_null(False))
    t, b, L = ok["taxi"].to_numpy().astype(float), ok["pred"].to_numpy(), ok["L"].to_numpy()
    print(f"excluding runs that start before the inbound in-block (n={ok.height}): "
          f"taxi < L-60 in {100 * (t < L - 60).mean():.2f}%; floor m=60 dSSE_full "
          f"{(((np.maximum(b, L - 60) - t) ** 2) - (b - t) ** 2).sum() / 1e6:+.2f}e6, "
          f"hit {100 * (b < L - 60).mean():.2f}%")
    for day in TRAIN_DAYS:
        o = ok.filter(pl.col("day") == day)
        t, b, L = o["taxi"].to_numpy().astype(float), o["pred"].to_numpy(), o["L"].to_numpy()
        print(f"  {day}: n={o.height}  " + "  ".join(
            f"m={m}: dSSE {(((np.maximum(b, L - m) - t) ** 2) - (b - t) ** 2).sum() / 1e6:+.3f}e6"
            for m in MARGINS))
    both = res.filter(pl.col("day") != "pooled").group_by("m").agg(
        (pl.col("dSSE_full") < 0).all().alias("full_both"), (pl.col("dSSE_trim") < 0).all().alias("trim_both"))
    print("\nfloor reduces SSE on BOTH days:"); print(both.sort("m"))
    err = float(p.select((pl.col("taxi") < pl.col("L") - 60).mean()).item())
    passed = bool(both["full_both"].any() or both["trim_both"].any()) and err < 0.05
    print(f"match-error proxy {100 * err:.2f}% (< 5% required)")

    # ------------------------------------------------------------------ 3
    section("3. UNSEEN PART (taxi - L) vs FIRST SIGHTING")
    q = p.with_columns(unseen=pl.col("taxi") - pl.col("L"))

    def by(expr, name):
        print(f"\nunseen = taxi - L by {name}:")
        print(q.with_columns(b=expr).group_by("b").agg(
            pl.len().alias("n"), pl.col("unseen").median().alias("median"),
            pl.col("unseen").quantile(0.25).alias("q25"), pl.col("unseen").quantile(0.75).alias("q75"),
            pl.col("unseen").mean().alias("mean"),
            (pl.col("pred") - pl.col("L")).median().alias("med_pred-L")).sort("b"))

    by(pl.col("adsb_first_own_m").cut([1000, 1500, 2000, 3000, 5000]).cast(pl.Utf8).fill_null("no stand"),
       "first-sighting distance to own stand (m)")
    by(pl.col("adsb_first_gs").cut([1, 5, 10, 15, 20, 30]).cast(pl.Utf8), "first-sighting ground speed (kt)")
    by(pl.col("reason"), "reason")

    # L + predicted unseen part vs plain floor, cross-fit between the two days
    print("\n'L + predicted unseen' vs plain floor, cross-fit day A -> day B (unseen clipped to its 1-99 pct):")
    out = []
    for fit_d, app_d in ((TRAIN_DAYS[0], TRAIN_DAYS[1]), (TRAIN_DAYS[1], TRAIN_DAYS[0])):
        fa, fb = q.filter(pl.col("day") == fit_d), q.filter(pl.col("day") == app_d)
        lo, hi = np.percentile(fa["unseen"].to_numpy(), [1, 99])
        fa = fa.with_columns(u=pl.col("unseen").clip(lo, hi),
                             db=pl.col("adsb_first_own_m").cut([2000, 3000]).cast(pl.Utf8).fill_null("no stand"))
        fb = fb.with_columns(db=pl.col("adsb_first_own_m").cut([2000, 3000]).cast(pl.Utf8).fill_null("no stand"))
        med = fa.group_by("db").agg(pl.col("u").mean().alias("u_hat"))
        fb = fb.join(med, on="db", how="left").with_columns(pl.col("u_hat").fill_null(float(fa["u"].mean())))
        t, b, L = fb["taxi"].to_numpy().astype(float), fb["pred"].to_numpy(), fb["L"].to_numpy()
        est = L + fb["u_hat"].to_numpy()
        ra = fa["taxi"].to_numpy() - fa["pred"].to_numpy()
        fa_est = fa["L"].to_numpy() + fa.join(med, on="db", how="left")["u_hat"].to_numpy()
        dl = fa_est - fa["pred"].to_numpy()
        w = float(np.clip((ra * dl).sum() / (dl ** 2).sum(), 0, 1))
        keep = t <= MONSTER_S
        for name, v in (("base", b), ("floor m=60", np.maximum(b, L - 60)),
                        ("L + u_hat (replace)", est), (f"nudge w={w:.2f}", b + w * (est - b)),
                        ("floor, then nudge", np.maximum(b, L - 60) + w * (est - np.maximum(b, L - 60)))):
            out.append(dict(fit=fit_d, apply=app_d, method=name, n=len(t),
                            rmse_full=np.sqrt(((v - t) ** 2).mean()),
                            rmse_trim=np.sqrt(((v - t) ** 2)[keep].mean())))
    print(pl.DataFrame(out))

    section("VERDICT")
    print(f"pass rule: floor reduces SSE on BOTH days for some m, and match errors < 5%  ->  "
          f"{'PASS' if passed else 'FAIL'}")
    print("caveat: two training days only; small sample.")
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
