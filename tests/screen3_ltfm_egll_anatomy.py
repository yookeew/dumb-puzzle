"""Screen 3 (PROGRESS.md §50; §45 item 2): LTFM and EGLL error anatomy.

Eval frame: cache/eval/catcorr_mixed_holdout_ev.parquet, the v21 gate's
corrected-CatBoost holdout predictions (Jan + Jul 2025). v21 itself is the
lgb/catcorr stack plus ADS-B stages; LTFM has no ADS-B, EGLL's blend touches
~1% of its holdout rows, so the corrected CatBoost is the relevant frame (the
in-sample stack is printed alongside for reference).

A. error concentration: top-10/100/1000 share of squared error, NM-unmatched,
   echo (|block - sched| < 30 s), labels > 5 h, by label band and by day.
B. §18 transfer test: Jan-vs-Jul correlation of mean residual by hour,
   runway, stand group, operator, and the RMSE change of applying one
   month's (shrunk) group means to the other.
C. recording patterns over all of 2025 (the way §44 found the LIRF +24 h bug):
   label vs T - AOBT_3/EOBT/LOBT/IOBT/SOBT coincidences, minute rounding,
   repeated block stamps, day offsets, then their holdout squared-error share.

Findings only. No training, no stage, no submission.

Run:  .venv/Scripts/python.exe tests/screen3_ltfm_egll_anatomy.py > logs/screen3_ltfm_egll_anatomy.log
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "cache" / "eval"
FEAT = ROOT / "cache" / "features"
RAW = ROOT / "data" / "raw"
AIRPORTS = ("LTFM", "EGLL")
MONSTER_S = 5 * 3600
ECHO_ABS_D = 30
MIN_GROUP_N = 30
SHRINK = 30.0
STACK_W = {"lgb": 0.4589, "catcorr": 0.5551}      # logs/stack_submit_v21.log


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def load_raw() -> pl.DataFrame:
    return (pl.scan_parquet(str(RAW / "training_*.parquet"))
            .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(AIRPORTS))
            .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt",
                    "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
                    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt",
                    "EOBT_1_flt", "LOBT_flt", "IOBT_flt", "TAXITIME_SEC_mvt")
            .collect()
            .with_columns(ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
                          taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                          nm_unmatched=pl.col("FLIGHT_ID_mvt").is_null(),
                          d=secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
                          delay=secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
                          lab_minus_aobt3=secs("AOBT_3_flt", "BLOCK_TIME_UTC_mvt"),
                          lab_minus_eobt=secs("EOBT_1_flt", "BLOCK_TIME_UTC_mvt"),
                          lab_minus_lobt=secs("LOBT_flt", "BLOCK_TIME_UTC_mvt"),
                          lab_minus_iobt=secs("IOBT_flt", "BLOCK_TIME_UTC_mvt"),
                          block_sec=pl.col("BLOCK_TIME_UTC_mvt").dt.second(),
                          mvt_sec=pl.col("MVT_TIME_UTC_mvt").dt.second()))


def flags() -> dict[str, pl.Expr]:
    """Recording-pattern flags (label-side). Each is a candidate 'kind of label'."""
    return {
        "echo |block-sched|<30": pl.col("d").abs() < ECHO_ABS_D,
        "block == AOBT_3 (+-30s)": pl.col("lab_minus_aobt3").abs() < 30,
        "block == EOBT_1 (+-30s)": pl.col("lab_minus_eobt").abs() < 30,
        "block == LOBT (+-30s)": pl.col("lab_minus_lobt").abs() < 30,
        "block == IOBT (+-30s)": pl.col("lab_minus_iobt").abs() < 30,
        "taxi <= 0": pl.col("taxi") <= 0,
        "taxi < 120": pl.col("taxi") < 120,
        "taxi > 2 h": pl.col("taxi") > 7200,
        "taxi > 5 h": pl.col("taxi") > MONSTER_S,
        "taxi within 90 min of k*24h (k>=1)": ((pl.col("taxi") / 86400).round() >= 1)
        & ((pl.col("taxi") - (pl.col("taxi") / 86400).round() * 86400).abs() < 5400),
        "block stamped on :00 s": pl.col("block_sec") == 0,
        "block shared by >= 3 dep (same airport)": pl.col("block_dup") >= 3,
        "NM-unmatched": pl.col("nm_unmatched"),
    }


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    for p in (EVAL / "catcorr_mixed_holdout_ev.parquet", EVAL / "catrerun_mixed_holdout_ev.parquet",
              EVAL / "lgb_mixed_holdout_ev.parquet", FEAT / "holdout_gap2025.parquet"):
        print(f"  input {p.relative_to(ROOT)}  mtime {dt.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(2)
    pl.Config.set_fmt_str_lengths(40)

    ev = pl.read_parquet(EVAL / "catcorr_mixed_holdout_ev.parquet").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ym").cast(pl.Utf8))
    lgb = pl.read_parquet(EVAL / "lgb_mixed_holdout_ev.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias("lgb"))
    ctx = pl.read_parquet(FEAT / "holdout_gap2025.parquet",
                          columns=["MVT_ID_mvt", "stand_group", "hour", "T"]).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64))
    raw = load_raw()
    raw = raw.join(raw.group_by("ADEP_mvt", "BLOCK_TIME_UTC_mvt").agg(pl.len().alias("block_dup")),
                   on=["ADEP_mvt", "BLOCK_TIME_UTC_mvt"], how="left")
    tot_sse = float(((ev["pred"] - ev["taxi"]) ** 2).sum())
    df = (ev.join(lgb, on="MVT_ID_mvt", how="left").join(ctx, on="MVT_ID_mvt", how="left")
          .join(raw.drop("ADEP_mvt", "ym", "taxi"), on="MVT_ID_mvt", how="left")
          .filter(pl.col("ADEP_mvt").is_in(AIRPORTS))
          .with_columns(res=pl.col("taxi") - pl.col("pred"),
                        stack=STACK_W["lgb"] * pl.col("lgb") + STACK_W["catcorr"] * pl.col("pred"),
                        day=pl.col("T").dt.date())
          .with_columns(se=pl.col("res") ** 2))
    print(f"holdout rows all airports {ev.height:,}; total SSE {tot_sse:.4g}")

    # ------------------------------------------------------------------ A
    section("A. ERROR CONCENTRATION (catcorr = v21 gate eval frame)")
    for ap in AIRPORTS:
        a = df.filter(pl.col("ADEP_mvt") == ap)
        se = np.sort(a["se"].to_numpy())[::-1]
        keep = a["taxi"] <= MONSTER_S
        print(f"\n--- {ap}: n={a.height:,}  RMSE full {np.sqrt(se.mean()):.1f}  trimmed "
              f"{np.sqrt(a.filter(keep)['se'].mean()):.1f}  (in-sample stack full "
              f"{np.sqrt(((a['stack'] - a['taxi']) ** 2).mean()):.1f})  share of all-airport SSE "
              f"{100 * se.sum() / tot_sse:.1f}%")
        for k in (10, 100, 1000):
            print(f"  top-{k:<5} rows ({100 * k / a.height:.2f}% of rows): {100 * se[:k].sum() / se.sum():.1f}% of SSE")
        for ym in ("2025-01", "2025-07"):
            m = a.filter(pl.col("ym") == ym)
            print(f"  {ym}: n={m.height:,} RMSE full {np.sqrt(m['se'].mean()):.1f} trimmed "
                  f"{np.sqrt(m.filter(pl.col('taxi') <= MONSTER_S)['se'].mean()):.1f} mean res {m['res'].mean():+.1f}")
        a = a.with_columns(echo=pl.col("d").abs() < ECHO_ABS_D, monster=pl.col("taxi") > MONSTER_S)
        print(a.group_by("nm_unmatched", "echo").agg(
            pl.len().alias("n"), (100 * pl.len() / a.height).alias("pct_rows"),
            (100 * pl.col("se").sum() / a["se"].sum()).alias("pct_SSE"),
            pl.col("se").mean().sqrt().alias("rmse"), pl.col("res").mean().alias("mean_res"),
            pl.col("monster").sum().alias("labels_gt_5h")).sort("nm_unmatched", "echo"))
        print("  by label band:")
        print(a.with_columns(band=pl.col("taxi").cut([600, 1200, 1800, 3600, 7200, MONSTER_S]).cast(pl.Utf8))
              .group_by("band").agg(pl.len().alias("n"), (100 * pl.col("se").sum() / a["se"].sum()).alias("pct_SSE"),
                                    pl.col("se").mean().sqrt().alias("rmse"), pl.col("res").mean().alias("mean_res"),
                                    pl.col("taxi").min().alias("lo")).sort("lo"))
        print("  worst days:")
        print(a.group_by("day").agg(pl.len().alias("n"), (100 * pl.col("se").sum() / a["se"].sum()).alias("pct_SSE"),
                                     pl.col("se").mean().sqrt().alias("rmse"), pl.col("res").mean().alias("mean_res"),
                                     pl.col("taxi").median().alias("med_taxi"), pl.col("pred").median().alias("med_pred"))
              .sort("pct_SSE", descending=True).head(8))
        print("  top-15 rows:")
        print(a.sort("se", descending=True).head(15).select(
            "ym", "FLIGHT_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt",
            "nm_unmatched", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt",
            "taxi", "pred", pl.col("lab_minus_aobt3").alias("aobt3-blk"), pl.col("d").alias("blk-sched")))

    # ------------------------------------------------------------------ B
    section("B. §18 TRANSFER TEST: Jan-vs-Jul mean residual by group")
    rows = []
    for ap in AIRPORTS:
        a = df.filter(pl.col("ADEP_mvt") == ap)
        for key in ("hour", "RUNWAY_mvt", "stand_group", "AIRCRAFT_OPERATOR_flt"):
            for trim in (False, True):
                b = a.filter(pl.col("taxi") <= MONSTER_S) if trim else a
                g = b.group_by(key, "ym").agg(pl.col("res").mean().alias("m"), pl.len().alias("n"))
                w = (g.filter(pl.col("n") >= MIN_GROUP_N).pivot(on="ym", index=key, values=["m", "n"])
                     .drop_nulls())
                if w.height < 3:
                    rows.append(dict(ap=ap, key=key, trimmed=trim, groups=w.height))
                    continue
                r = float(np.corrcoef(w["m_2025-01"].to_numpy(), w["m_2025-07"].to_numpy())[0, 1])
                # apply one month's shrunk group means to the other (both directions)
                d_rmse = {}
                for src, dst in (("2025-01", "2025-07"), ("2025-07", "2025-01")):
                    gs = (b.filter(pl.col("ym") == src).group_by(key)
                          .agg(pl.col("res").sum().alias("s"), pl.len().alias("n"))
                          .with_columns(adj=pl.col("s") / (pl.col("n") + SHRINK)))
                    t = b.filter(pl.col("ym") == dst).join(gs.select(key, "adj"), on=key, how="left") \
                        .with_columns(pl.col("adj").fill_null(0.0))
                    d_rmse[dst] = float(np.sqrt(((t["res"] - t["adj"]) ** 2).mean()) - np.sqrt((t["res"] ** 2).mean()))
                rows.append(dict(ap=ap, key=key, trimmed=trim, groups=w.height, corr=r,
                                 d_rmse_Jul=d_rmse["2025-07"], d_rmse_Jan=d_rmse["2025-01"]))
    print(pl.DataFrame(rows).sort("ap", "key", "trimmed"))

    # ------------------------------------------------------------------ C
    section("C. RECORDING PATTERNS over all 2025 DEP (training months + holdout)")
    fl = flags()
    for ap in AIRPORTS:
        a = raw.filter((pl.col("ADEP_mvt") == ap) & pl.col("taxi").is_not_null())
        print(f"\n--- {ap}: 2025 DEP rows {a.height:,}")
        print(a.select([(100 * e.fill_null(False).mean()).alias(k) for k, e in fl.items()])
              .transpose(include_header=True, header_name="flag", column_names=["pct_rows_2025"]))
        print("  by month (% rows):")
        print(a.group_by("ym").agg([(100 * fl[k].fill_null(False).mean()).alias(k[:18]) for k in (
            "echo |block-sched|<30", "block == AOBT_3 (+-30s)", "block == EOBT_1 (+-30s)",
            "block stamped on :00 s", "block shared by >= 3 dep (same airport)", "NM-unmatched", "taxi > 2 h")]
                                  + [pl.col("taxi").median().alias("med_taxi"), pl.len().alias("n")]).sort("ym"))
        print("  most frequent label values:")
        print(a.group_by("taxi").len().sort("len", descending=True).head(8)
              .with_columns(pct=100 * pl.col("len") / a.height))
        print("  label vs other off-block fields: |field - block| quantiles (s), NM-matched rows")
        mm = a.filter(~pl.col("nm_unmatched"))
        print(pl.DataFrame([dict(field=c, n=int(mm[c].is_not_null().sum()),
                                 **{f"q{int(q * 100)}": float(mm[c].abs().quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                                 exact_pct=float(100 * (mm[c] == 0).mean()))
                            for c in ("lab_minus_aobt3", "lab_minus_eobt", "lab_minus_lobt", "lab_minus_iobt", "d")]))
        print("  large delays (T - SOBT >= 3 h): label kinds, NM-matched vs unmatched")
        big = a.filter(pl.col("delay") >= 3 * 3600).with_columns(
            kind=pl.when(pl.col("d").abs() < ECHO_ABS_D).then(pl.lit("echo"))
            .when(pl.col("taxi") > 12 * 3600).then(pl.lit(">12h"))
            .when(pl.col("taxi") > 7200).then(pl.lit("2-12h"))
            .otherwise(pl.lit("normal <=2h")))
        print(big.group_by("nm_unmatched", "kind").len().pivot(on="kind", index="nm_unmatched", values="len"))
        print("  NM-unmatched: block date vs sched date for labels > 2 h (LIRF-bug check)")
        un = a.filter(pl.col("nm_unmatched") & (pl.col("taxi") > 7200)).with_columns(
            same_date=pl.col("BLOCK_TIME_UTC_mvt").dt.date() == pl.col("SCHED_TIME_UTC_mvt").dt.date(),
            plus1d_before_T=(secs("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt") - 86400).is_between(0, 5400))
        print(un.select(pl.len().alias("n"), pl.col("same_date").mean().alias("block_date==sched_date"),
                        pl.col("plus1d_before_T").sum().alias("block+1d_within_90min_before_T")))

        h = df.filter(pl.col("ADEP_mvt") == ap)
        print(f"  holdout ({ap}): squared-error share by flag")
        print(pl.DataFrame([dict(flag=k, n=int(h.select(e.fill_null(False).sum()).item()),
                                 pct_rows=float(100 * h.select(e.fill_null(False).mean()).item()),
                                 pct_SSE=float(100 * h.filter(e.fill_null(False))["se"].sum() / h["se"].sum()),
                                 rmse=float(np.sqrt(h.filter(e.fill_null(False))["se"].mean() or 0)),
                                 mean_res=float(h.filter(e.fill_null(False))["res"].mean() or 0))
                            for k, e in fl.items()]))
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
