"""Screen 2 (PROGRESS.md §50): neighbour residual, "how is the airport running now".

Question: does the recent residual of nearby departures predict this row's OOF
residual? §20 put the (airport, day, hour) oracle ceiling at ~10.8 s.

Pre-check (done by hand, recorded in §50): no production feature measures
neighbour taxi DURATIONS. Family 6's q_ahead_mean_wait (elapsed wait of the
aircraft ahead, from AOBT_3) is the nearest thing; it is off (rejected §23) and
is not a residual.

For each training-month departure i, over same-airport departures j != i with
AOBT_3 present (training months only; the Jan/Jul 2025 holdout is never read,
not even its predictions, except in the label-free drift section):
  r_j = clip((T_j - AOBT_3_j) - oof_pred_j, -1800, 1800)   (T - AOBT_3 = aobt3_taxi)
  windows |T_j - T_i| <= 30 / 60 min (post-ops: later flights allowed)
  -> nb{30,60}_mean, nb{30,60}_median, nb{30,60}_n, and same-runway rw{30,60}_mean/_n
Secondary: arrivals' taxi-in residual against a leave-one-month-out
(airport, runway) median, clipped +-1800, landing within +-30/60 min of T_i.
Arrival taxi-in is unblanked in ranking; DEP BLOCK/TAXITIME are never used.

Base = OOF CatBoost (mixed) folds, cache/oof/cat_mixed/. Own residual
e_i = taxi_i - oof_pred_i. Trimmed = labels <= 5 h.

No training, no stage, no submission.

Run:  .venv/Scripts/python.exe tests/screen2_neighbour_residual.py > logs/screen2_neighbour_residual.log
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
from models.fit import TRAIN_MONTHS  # noqa: E402

FEAT = ROOT / "cache" / "features"
OOF = ROOT / "cache" / "oof" / "cat_mixed"
EVAL = ROOT / "cache" / "eval"
RAW = ROOT / "data" / "raw"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
MONSTER_S = 5 * 3600
CLIP = 1800.0
WINDOWS = (30, 60)
MIN_N = 1          # a window mean needs >= MIN_N neighbours, else null
DRIFT_SD = 0.25

NB_COLS = [f"nb{w}_{s}" for w in WINDOWS for s in ("mean", "median")] + [f"rw{w}_mean" for w in WINDOWS] \
    + [f"arr{w}_mean" for w in WINDOWS]
N_COLS = [f"nb{w}_n" for w in WINDOWS] + [f"rw{w}_n" for w in WINDOWS] + [f"arr{w}_n" for w in WINDOWS]


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def window_stats(tq, idq, tn, rn, idn, w_s, median: bool):
    """Mean/median/count of rn over |tn - tq| <= w_s, excluding the query's own id.
    tn must be sorted."""
    lo = np.searchsorted(tn, tq - w_s, side="left")
    hi = np.searchsorted(tn, tq + w_s, side="right")
    cs = np.concatenate([[0.0], np.cumsum(rn)])
    s, n = cs[hi] - cs[lo], (hi - lo).astype(float)
    pos = {v: k for k, v in enumerate(idn)}
    own = np.array([pos.get(v, -1) for v in idq])
    inside = (own >= lo) & (own < hi) & (own >= 0)
    s[inside] -= rn[own[inside]]
    n[inside] -= 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n >= MIN_N, s / np.maximum(n, 1), np.nan)
    med = np.full(len(tq), np.nan)
    if median:
        for k in range(len(tq)):
            if n[k] < MIN_N:
                continue
            a, b = lo[k], hi[k]
            if inside[k]:
                o = own[k]
                med[k] = np.median(np.concatenate([rn[a:o], rn[o + 1:b]]))
            else:
                med[k] = np.median(rn[a:b])
    return mean, med, n


def build(q: pl.DataFrame, nb: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """q: query DEP rows (MVT_ID_mvt, ADEP_mvt, RUNWAY_mvt, t). nb: neighbour DEP rows
    (same cols + r). arr: arrivals (ADEP_mvt, t, a). Returns q + feature columns."""
    out = []
    for (ap,), qa in q.group_by("ADEP_mvt", maintain_order=True):
        na = nb.filter(pl.col("ADEP_mvt") == ap).sort("t")
        aa = arr.filter(pl.col("ADEP_mvt") == ap).sort("t")
        tq, idq = qa["t"].to_numpy().astype(float), qa["MVT_ID_mvt"].to_numpy()
        tn, rn, idn = na["t"].to_numpy().astype(float), na["r"].to_numpy(), na["MVT_ID_mvt"].to_numpy()
        cols = {}
        for w in WINDOWS:
            m, md, n = window_stats(tq, idq, tn, rn, idn, 60.0 * w, median=True)
            cols.update({f"nb{w}_mean": m, f"nb{w}_median": md, f"nb{w}_n": n})
            m, _, n = window_stats(tq, idq,
                                   aa["t"].to_numpy().astype(float), aa["a"].to_numpy(),
                                   np.full(aa.height, -1), 60.0 * w, median=False)
            cols.update({f"arr{w}_mean": m, f"arr{w}_n": n})
            rm, rnn = np.full(len(tq), np.nan), np.zeros(len(tq))
            rq = qa["RUNWAY_mvt"].fill_null("?").to_numpy()
            rnb = na["RUNWAY_mvt"].fill_null("?").to_numpy()
            for rw in np.unique(rq):
                sel_q, sel_n = rq == rw, rnb == rw
                m, _, n = window_stats(tq[sel_q], idq[sel_q], tn[sel_n], rn[sel_n], idn[sel_n],
                                       60.0 * w, median=False)
                rm[sel_q], rnn[sel_q] = m, n
            cols.update({f"rw{w}_mean": rm, f"rw{w}_n": rnn})
        out.append(qa.with_columns(**{k: pl.Series(k, v, nan_to_null=True) for k, v in cols.items()}))
        print(f"  {ap}: {qa.height:,} query rows, {na.height:,} neighbours, {aa.height:,} arrivals", flush=True)
    return pl.concat(out)


def dep_frame(feat_file: Path, preds: pl.DataFrame, months: list[str] | None) -> pl.DataFrame:
    f = pl.read_parquet(feat_file, columns=["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "T", "aobt3_taxi"])
    f = f.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), t=pl.col("T").dt.epoch("s"),
                       ym=pl.col("T").dt.strftime("%Y-%m"))
    if months is not None:
        f = f.filter(pl.col("ym").is_in(months))
    return f.join(preds, on="MVT_ID_mvt", how="inner").with_columns(
        r=(pl.col("aobt3_taxi") - pl.col("pred")).clip(-CLIP, CLIP))


def arrivals(src, months: list[str] | None, med: pl.DataFrame | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Arrival taxi-in residual vs (airport, runway) median. With med=None: leave-one-month-out
    medians within `months`; else the given medians. Returns (arrivals, medians over all months)."""
    a = (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "ARR")
         .select("ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", taxi_in=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                 t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"))
         .collect())
    # arrivals: the reporting airport is ADES (ADEP_mvt is the origin)
    a = a.with_columns(ap=pl.col("ADES_mvt")).filter(pl.col("taxi_in").is_between(0, 7200))
    if months is not None:
        a = a.filter(pl.col("ym").is_in(months))
    all_med = a.group_by("ap", "RUNWAY_mvt").agg(pl.col("taxi_in").median().alias("m"))
    if med is None:
        parts = []
        for mth in a["ym"].unique().to_list():
            mm = (a.filter(pl.col("ym") != mth).group_by("ap", "RUNWAY_mvt")
                  .agg(pl.col("taxi_in").median().alias("m")))
            parts.append(a.filter(pl.col("ym") == mth).join(mm, on=["ap", "RUNWAY_mvt"], how="left"))
        a = pl.concat(parts)
    else:
        a = a.join(med, on=["ap", "RUNWAY_mvt"], how="left")
    a = a.filter(pl.col("m").is_not_null()).with_columns(
        a=(pl.col("taxi_in") - pl.col("m")).clip(-CLIP, CLIP))
    return a.select(pl.col("ap").alias("ADEP_mvt"), "t", "a", "ym"), all_med


def rmse(x):
    return float(np.sqrt(np.mean(x ** 2)))


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    for p in (FEAT / "train2025.parquet", FEAT / "labels2025.parquet", FEAT / "ranking.parquet",
              EVAL / "cat_mixed_rank.parquet", *sorted(OOF.glob("*.parquet"))[:1]):
        print(f"  input {p.relative_to(ROOT)}  mtime {dt.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    months = list(TRAIN_MONTHS)

    # ------------------------------------------------------------ build (2025 training months)
    section("BUILD")
    oof = pl.concat([pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), "pred")
                     for f in sorted(OOF.glob("fold=*.parquet"))])
    dep = dep_frame(FEAT / "train2025.parquet", oof, months)
    print(f"training-month DEP rows with OOF pred: {dep.height:,}; aobt3_taxi present "
          f"{dep['aobt3_taxi'].is_not_null().mean() * 100:.1f}%; aobt3_taxi median "
          f"{dep['aobt3_taxi'].median():.0f} s")
    nb = dep.filter(pl.col("r").is_not_null())
    arr, arr_med = arrivals(str(RAW / "training_*.parquet"), months, None)
    print(f"arrivals (training months, LOMO medians): {arr.height:,}")
    tr = build(dep.select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t", "ym"), nb, arr)
    lab = pl.read_parquet(FEAT / "labels2025.parquet", columns=["MVT_ID_mvt", "taxi"]).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64))
    tr = (tr.join(oof, on="MVT_ID_mvt").join(lab, on="MVT_ID_mvt")
          .filter(pl.col("taxi") >= 0).with_columns(e=pl.col("taxi") - pl.col("pred")))
    print(f"analysis rows {tr.height:,}; monster labels > 5 h: {(tr['taxi'] > MONSTER_S).sum()}")
    print("feature coverage (non-null %) and median neighbour counts:")
    print(tr.select([pl.col(c).is_not_null().mean().mul(100).alias(c) for c in NB_COLS]))
    print(tr.select([pl.col(c).median().alias(c) for c in N_COLS]))

    # ------------------------------------------------------------ per-month correlation / slope
    section("PER-MONTH CORRELATION AND SLOPE WITH OWN OOF RESIDUAL e = taxi - oof_pred")
    rows = []
    for c in NB_COLS:
        for m in months:
            f = tr.filter((pl.col("ym") == m) & pl.col(c).is_not_null())
            for lbl, g in (("full", f), ("trim", f.filter(pl.col("taxi") <= MONSTER_S))):
                x, y = g[c].to_numpy(), g["e"].to_numpy()
                r = float(np.corrcoef(x, y)[0, 1])
                slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
                rows.append(dict(feat=c, ym=m, set=lbl, n=len(x), corr=r, slope=slope))
    cs = pl.DataFrame(rows)
    summ = cs.group_by("feat", "set").agg(
        pl.col("corr").mean().alias("mean_corr"), pl.col("corr").min().alias("min_corr"),
        pl.col("corr").max().alias("max_corr"), pl.col("slope").mean().alias("mean_slope"),
        (pl.col("slope") > 0).sum().alias("months_pos"), (pl.col("slope") < 0).sum().alias("months_neg"))
    print(summ.sort("feat", "set"))
    print("\nper-month trimmed corr, main features:")
    print(cs.filter((pl.col("set") == "trim") & pl.col("feat").is_in(["nb30_mean", "nb60_mean", "nb60_median",
                                                                         "rw60_mean", "arr60_mean"]))
          .pivot(on="feat", index="ym", values="corr").sort("ym"))

    # ------------------------------------------------------------ transfer (leave-one-month-out deciles)
    section("TRANSFER: decile-means mapping fit on 9 months, applied to the 10th")

    def mapping(fit: pl.DataFrame, app: pl.DataFrame, c: str, clip_e: bool) -> np.ndarray:
        x = fit[c].to_numpy()
        ok = ~np.isnan(x)
        edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 11)[1:-1]))
        y = fit["e"].to_numpy()
        if clip_e:
            y = np.clip(y, -CLIP, CLIP)
        b_fit = np.where(ok, np.searchsorted(edges, x, side="right"), -1)
        means = {b: float(y[b_fit == b].mean()) for b in np.unique(b_fit)}
        xa = app[c].to_numpy()
        b_app = np.where(~np.isnan(xa), np.searchsorted(edges, xa, side="right"), -1)
        return np.array([means.get(b, 0.0) for b in b_app])

    res, pooled = [], {}
    for c in NB_COLS:
        for clip_e in (False, True):
            adj = np.zeros(tr.height)
            for m in months:
                sel = (tr["ym"] == m).to_numpy()
                adj[sel] = mapping(tr.filter(~pl.Series(sel)), tr.filter(pl.Series(sel)), c, clip_e)
            e, keep = tr["e"].to_numpy(), (tr["taxi"] <= MONSTER_S).to_numpy()
            ym = tr["ym"].to_numpy()
            for m in months:
                s = ym == m
                res.append(dict(feat=c, clip_e=clip_e, ym=m,
                                d_full=rmse(e[s] - adj[s]) - rmse(e[s]),
                                d_trim=rmse(e[s & keep] - adj[s & keep]) - rmse(e[s & keep])))
            pooled[(c, clip_e)] = dict(feat=c, clip_e=clip_e, rmse_full=rmse(e), d_full=rmse(e - adj) - rmse(e),
                                       rmse_trim=rmse(e[keep]), d_trim=rmse(e[keep] - adj[keep]) - rmse(e[keep]))
    rs = pl.DataFrame(res)
    print("pooled (out-of-month) RMSE change, s (negative = better):")
    print(pl.DataFrame(list(pooled.values())).join(
        rs.group_by("feat", "clip_e").agg((pl.col("d_trim") < 0).sum().alias("months_trim_better"),
                                          (pl.col("d_full") < 0).sum().alias("months_full_better")),
        on=["feat", "clip_e"]).sort("clip_e", "d_trim"))
    print("\nper-month trimmed change, plain decile means (primary):")
    print(rs.filter(~pl.col("clip_e")).pivot(on="feat", index="ym", values="d_trim").sort("ym"))
    print("per-month full change, plain decile means:")
    print(rs.filter(~pl.col("clip_e")).pivot(on="feat", index="ym", values="d_full").sort("ym"))

    best = min((v for v in pooled.values() if not v["clip_e"]), key=lambda v: v["d_trim"])
    bc = best["feat"]
    print(f"\nbest primary feature by pooled trimmed gain: {bc} ({best['d_trim']:+.2f} s trimmed, "
          f"{best['d_full']:+.2f} s full)")
    print(f"per airport, {bc}, pooled out-of-month (plain decile means):")
    adj = np.zeros(tr.height)
    for m in months:
        sel = (tr["ym"] == m).to_numpy()
        adj[sel] = mapping(tr.filter(~pl.Series(sel)), tr.filter(pl.Series(sel)), bc, False)
    t2 = tr.with_columns(adj=pl.Series(adj))
    nmu = tr.select("MVT_ID_mvt").join(dep.select("MVT_ID_mvt", nm_unmatched=pl.col("aobt3_taxi").is_null()),
                                       on="MVT_ID_mvt", how="left")["nm_unmatched"]
    t2 = t2.with_columns(nm_unmatched=nmu)
    print("by own AOBT_3 availability (NM-unmatched rows have no aobt3_taxi of their own):")
    print(t2.group_by("nm_unmatched").agg(
        pl.len().alias("n"), pl.col("e").pow(2).mean().sqrt().alias("rmse_full"),
        ((pl.col("e") - pl.col("adj")).pow(2).mean().sqrt() - pl.col("e").pow(2).mean().sqrt()).alias("d_full"),
        ((pl.col("e") - pl.col("adj")).filter(pl.col("taxi") <= MONSTER_S).pow(2).mean().sqrt()
         - pl.col("e").filter(pl.col("taxi") <= MONSTER_S).pow(2).mean().sqrt()).alias("d_trim"),
        pl.corr(bc, "e").alias("corr")).sort("nm_unmatched"))
    print(t2.group_by("ADEP_mvt").agg(
        pl.len().alias("n"), pl.col("e").pow(2).mean().sqrt().alias("rmse_full"),
        ((pl.col("e") - pl.col("adj")).pow(2).mean().sqrt() - pl.col("e").pow(2).mean().sqrt()).alias("d_full"),
        ((pl.col("e") - pl.col("adj")).filter(pl.col("taxi") <= MONSTER_S).pow(2).mean().sqrt()
         - pl.col("e").filter(pl.col("taxi") <= MONSTER_S).pow(2).mean().sqrt()).alias("d_trim"))
        .sort("ADEP_mvt"))

    # ------------------------------------------------------------ drift
    section("DRIFT: 2025 training months (OOF preds) vs ranking 2026 (full-refit CatBoost preds)")
    rk_pred = pl.read_parquet(EVAL / "cat_mixed_rank.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "pred")
    rdep = dep_frame(FEAT / "ranking.parquet", rk_pred, None)
    rarr, _ = arrivals(str(RANKING), None, arr_med)
    print(f"ranking DEP rows with pred: {rdep.height:,}; aobt3_taxi present "
          f"{rdep['aobt3_taxi'].is_not_null().mean() * 100:.1f}%; arrivals {rarr.height:,}")
    rk = build(rdep.select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t", "ym"),
               rdep.filter(pl.col("r").is_not_null()), rarr)
    # like-for-like season: Jan/Jul 2025, 10-month-fit holdout predictions (no labels read)
    ho_pred = pl.read_parquet(EVAL / "cat_mixed_rerun_holdout_ev.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "pred")
    hdep = dep_frame(FEAT / "holdout_gap2025.parquet", ho_pred, None)
    harr, _ = arrivals(str(RAW / "training_*.parquet"), ["2025-01", "2025-07"], arr_med)
    hj = build(hdep.select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t", "ym"),
               hdep.filter(pl.col("r").is_not_null()), harr)
    drift_rows = []
    for c in ("nb30_mean", "nb60_mean", "nb60_median", "rw60_mean", "arr60_mean"):
        for (ap,), g in tr.group_by("ADEP_mvt"):
            x25 = g[c].drop_nans().drop_nulls()
            for lbl, other in (("rank2026", rk), ("JanJul2025", hj)):
                x26 = other.filter(pl.col("ADEP_mvt") == ap)[c].drop_nans().drop_nulls()
                if x26.len() == 0:
                    continue
                sd = float(x25.std())
                drift_rows.append(dict(feat=c, ADEP_mvt=ap, vs=lbl, mean25=float(x25.mean()),
                                       mean_other=float(x26.mean()), sd25=sd,
                                       shift_sd=(float(x26.mean()) - float(x25.mean())) / sd,
                                       sd_ratio=float(x26.std()) / sd))
    dr = pl.DataFrame(drift_rows).with_columns(flag=pl.col("shift_sd").abs() > DRIFT_SD)
    for c in dr["feat"].unique(maintain_order=True):
        print(f"\n{c}:")
        print(dr.filter(pl.col("feat") == c).pivot(on="vs", index="ADEP_mvt",
                                                   values=["shift_sd", "sd_ratio", "flag"]).sort("ADEP_mvt"))

    # ------------------------------------------------------------ verdict
    section("VERDICT (pass: same sign in >= 9/10 months, pooled trimmed transfer gain >= 1 s, no airport "
            f"drifting > {DRIFT_SD} sd vs ranking)")
    for c in NB_COLS:
        s = summ.filter((pl.col("feat") == c) & (pl.col("set") == "trim")).row(0, named=True)
        sign_ok = max(s["months_pos"], s["months_neg"]) >= 9
        p = pooled[(c, False)]
        gain_ok = p["d_trim"] <= -1.0
        drift_bad = dr.filter((pl.col("feat") == c) & (pl.col("vs") == "rank2026") & pl.col("flag"))
        drift_ok = None if c not in dr["feat"].to_list() else drift_bad.height == 0
        print(f"  {c:<12} sign {max(s['months_pos'], s['months_neg'])}/10 {'ok' if sign_ok else 'FAIL'} | "
              f"pooled trim {p['d_trim']:+.2f} s (full {p['d_full']:+.2f}) {'ok' if gain_ok else 'FAIL'} | "
              f"drift {'n/a' if drift_ok is None else ('ok' if drift_ok else 'FAIL ' + ','.join(drift_bad['ADEP_mvt'].to_list()))}"
              f"  -> {'PASS' if sign_ok and gain_ok and drift_ok else 'fail'}")
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
