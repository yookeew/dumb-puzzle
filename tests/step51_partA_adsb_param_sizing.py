"""§51 Part A: are the shipped ADS-B stage parameters under-fitted for 2026?

Rebuilds the v21 ADS-B fit exactly as src/post/stack_submit.py --cat-corrected
does: NNLS stack (lgb + catcorr) on the full Jan+Jul 2025 holdout, then
adsb_blend.fit (lags, tier weights, airport-tier cells), adsb_blend.fit_quality
(q factors), adsb_partial.fit on the q-blended base (intercepts a, slope b,
band weights). The shipped parameters are these fits; nothing is refit for
shipping here.

For every parameter: rows it was fitted on (holdout), 2026 ranking rows it
is applied to, and a cluster-bootstrap SE (resampling (airport, day) clusters
of the holdout, stack weights held fixed). Also, label-free: the SD of the
final 2026 prediction across bootstrap draws, per airport (how much the
submission itself wobbles from parameter noise).

Flags: fitted on < 500 rows and applied to > 5,000; or SE large relative to the
estimate: weights/factors SE/|est| > 0.25; lags/intercepts SE > 20 s;
slope b SE/|b| > 0.25.

Holdout labels are used only to refit the already-shipped parameters (the same
rows the shipped fit used); no new exploration on them.

Run:  .venv/Scripts/python.exe tests/step51_partA_adsb_param_sizing.py > logs/step51_partA_adsb_param_sizing.log
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from post import adsb_blend, adsb_partial  # noqa: E402

EVAL = ROOT / "cache" / "eval"
SUB = ROOT / "data" / "submissions"
RAW = ROOT / "data" / "raw"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
DET = ROOT / "cache" / "adsb_pushback"
ENGINES = ["lgb", "catcorr"]
N_BOOT, SEED = 300, 0
TRAIN_DAYS = ("2025-09-15", "2025-11-15")


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def det(prefix: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(DET.glob(f"day={prefix}*.parquet"))]).select(
        "MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts", "adsb_matched", "adsb_first_ts",
        "adsb_first_own_m", "adsb_pb_gap_s", "adsb_pb_dist_m", "adsb_pb_gs")


def mvt(src) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt",
                    mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                    day=pl.col("MVT_TIME_UTC_mvt").dt.date()).collect())


def fit_all(ho: pl.DataFrame) -> dict:
    p = adsb_blend.fit(ho, per_airport=True)
    q = adsb_blend.fit_quality(ho, p)
    hoq = adsb_partial.with_inputs(ho.with_columns(base=adsb_blend.apply(ho, p, q)))
    pp = adsb_partial.fit(hoq)
    return {"p": p, "q": q, "pp": pp}


def flat(f: dict) -> dict[str, float]:
    p, q, pp = f["p"], f["q"], f["pp"]
    out = {f"lag|{a}": v for a, v in p["lags"].items()}
    out["lag|pooled"] = p["pooled"]
    out.update({f"w_tier|{t}": v for t, v in p["w"].items()})
    out.update({f"w_cell|{a}/{t}": v for (a, t), v in p["cells"].items()})
    out.update({f"q|{t}/{ql}": v for (t, ql), v in q.items()})
    out.update({f"partial_a|{a}": v for a, v in pp["a"].items()})
    out["partial_a|pooled"] = pp["a_pooled"]
    out["partial_b|all"] = pp["b"]
    out.update({f"partial_w|{k}": v for k, v in pp["w"].items()})
    return out


def final_pred(rk: pl.DataFrame, f: dict) -> np.ndarray:
    r = adsb_partial.with_inputs(rk.with_columns(base=adsb_blend.apply(rk, f["p"], f["q"])))
    return adsb_partial.apply(r, f["pp"])


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    for p in [EVAL / f"{e}_mixed_holdout_ev.parquet" for e in ENGINES] + [SUB / f"{e}_mixed.parquet" for e in ENGINES]:
        print(f"  input {p.relative_to(ROOT)}  mtime {dt.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)

    # ---------------------------------------------------------- holdout frame, as stack_submit
    ho = None
    for e in ENGINES:
        d = pl.read_parquet(EVAL / f"{e}_mixed_holdout_ev.parquet").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "taxi", pl.col("pred").alias(e))
        ho = d if ho is None else ho.join(d.select("MVT_ID_mvt", e), on="MVT_ID_mvt", how="inner")
    w, _ = nnls(ho.select(ENGINES).to_numpy(), ho["taxi"].to_numpy())
    print("stack weights:", dict(zip(ENGINES, np.round(w, 4))), "(v21 log: lgb 0.4589, catcorr 0.5551)")
    ho = ho.with_columns(pred=pl.Series(ho.select(ENGINES).to_numpy() @ w))
    ho = (ho.join(mvt(str(RAW / "training_*.parquet")).select("MVT_ID_mvt", "mvt_ts", "day"),
                  on="MVT_ID_mvt", how="left")
          .join(det("2025-0"), on="MVT_ID_mvt", how="left")       # Jan/Jul 2025 + Sep (excluded below)
          .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))
    ho = ho.filter(pl.col("day").dt.month().is_in([1, 7]))
    full = fit_all(ho)
    est = flat(full)
    print("shipped fit reproduced: lags", {k.split("|")[1]: round(v) for k, v in est.items() if k.startswith("lag|")})

    # ---------------------------------------------------------- ranking frame
    rk = None
    for e in ENGINES:
        s = pl.read_parquet(SUB / f"{e}_mixed.parquet").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(e))
        rk = s if rk is None else rk.join(s, on="MVT_ID_mvt", how="inner")
    rk = (rk.join(mvt(RANKING).select("MVT_ID_mvt", "ADEP_mvt", "mvt_ts"), on="MVT_ID_mvt", how="left")
          .with_columns(pred=pl.Series(rk.select(ENGINES).to_numpy() @ w))
          .join(det("2026"), on="MVT_ID_mvt", how="left")
          .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))

    # ---------------------------------------------------------- row counts per parameter
    def counts(df: pl.DataFrame) -> dict[str, int]:
        el = df.filter(adsb_blend.eligible_mask()).with_columns(_q=adsb_blend.quality())
        pm = adsb_partial.with_inputs(df).filter(adsb_partial.mask())
        c = {}
        for (a,), g in el.group_by("ADEP_mvt"):
            c[f"lag|{a}"] = g.height
        c["lag|pooled"] = el.height
        for (a, t), g in el.group_by("ADEP_mvt", "adsb_tier"):
            c[f"w_cell|{a}/{t}"] = g.height
        cell_keys = {k for k in est if k.startswith("w_cell|")}
        for t in adsb_blend.TIERS:
            c[f"w_tier|{t}"] = el.filter((pl.col("adsb_tier") == t)
                                         & ~(pl.concat_str(pl.lit("w_cell|"), "ADEP_mvt", pl.lit("/"), "adsb_tier")
                                             .is_in(list(cell_keys)))).height
        for (t, ql), g in el.group_by("adsb_tier", "_q"):
            c[f"q|{t}/{ql}"] = g.height
        for (a,), g in pm.group_by("ADEP_mvt"):
            c[f"partial_a|{a}"] = g.height
        big = set(k.split("|")[1] for k in est if k.startswith("partial_a|") and k != "partial_a|pooled")
        c["partial_a|pooled"] = pm.filter(~pl.col("ADEP_mvt").is_in(list(big))).height
        c["partial_b|all"] = pm.height
        c["partial_w|near"] = pm.filter(pl.col("adsb_first_own_m") < adsb_partial.D_SPLIT).height
        c["partial_w|far"] = pm.filter(pl.col("adsb_first_own_m") >= adsb_partial.D_SPLIT).height
        return c

    n_fit = counts(ho)
    n_fit["w_tier|appear"] = ho.filter(adsb_blend.eligible_mask() & (pl.col("adsb_tier") == "appear")).height
    n_fit["w_tier|dwell"] = ho.filter(adsb_blend.eligible_mask() & (pl.col("adsb_tier") == "dwell")).height
    n_app = counts(rk)   # w_tier: applied only where no cell weight exists

    # ---------------------------------------------------------- cluster bootstrap
    section(f"BOOTSTRAP: {N_BOOT} resamples of (airport, day) clusters")
    hm = ho.filter(pl.col("adsb_matched").fill_null(False)).with_columns(
        cl=pl.concat_str("ADEP_mvt", pl.lit("|"), pl.col("day").cast(pl.Utf8)))
    rkm = rk.filter(pl.col("adsb_matched").fill_null(False))
    keys = hm["cl"].unique().sort().to_list()
    idx_by = {k: np.flatnonzero((hm["cl"] == k).to_numpy()) for k in keys}
    print(f"matched holdout rows {hm.height:,} in {len(keys)} clusters; matched ranking rows {rkm.height:,}")
    rng = np.random.default_rng(SEED)
    draws, preds = [], []
    t0 = dt.datetime.now()
    for b in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        ii = np.concatenate([idx_by[keys[k]] for k in pick])
        try:
            f = fit_all(hm[ii])
        except Exception as ex:  # noqa: BLE001  (a draw can lack a whole airport)
            print(f"  draw {b}: fit failed ({ex}); skipped")
            continue
        draws.append(flat(f))
        preds.append(final_pred(rkm, f))
        if b == 9:
            print(f"  10 draws in {(dt.datetime.now() - t0).seconds}s", flush=True)
    print(f"  {len(draws)} successful draws")

    rows = []
    for k, v in est.items():
        vals = np.array([d[k] for d in draws if k in d])
        kind = k.split("|")[0]
        se = float(vals.std(ddof=1)) if len(vals) > 2 else np.nan
        absolute = kind in ("lag", "partial_a")
        big_se = (se > 20.0) if absolute else (se / max(abs(v), 1e-9) > 0.25)
        rows.append(dict(param=k, est=v, se=se, draws_with=len(vals), n_fit=n_fit.get(k, 0),
                         n_apply_2026=n_app.get(k, 0),
                         flag_size=(n_fit.get(k, 0) < 500) and (n_app.get(k, 0) > 5000),
                         flag_se=bool(big_se)))
    t = pl.DataFrame(rows).with_columns(
        rel_se=pl.when(pl.col("param").str.starts_with("lag|") | pl.col("param").str.starts_with("partial_a|"))
        .then(None).otherwise(pl.col("se") / pl.col("est").abs()),
        ratio_apply_fit=pl.col("n_apply_2026") / pl.col("n_fit").clip(lower_bound=1))
    section("PARAMETERS: estimate, SE, fit rows (Jan/Jul 2025), applied rows (ranking 2026)")
    print(t.select("param", "est", "se", "rel_se", "n_fit", "n_apply_2026", "ratio_apply_fit",
                   "flag_size", "flag_se").sort("param"))
    print("\nFLAGGED:")
    print(t.filter(pl.col("flag_size") | pl.col("flag_se")).select(
        "param", "est", "se", "n_fit", "n_apply_2026", "flag_size", "flag_se").sort("param"))

    # ---------------------------------------------------------- 2026 prediction wobble
    section("LABEL-FREE: SD of the final 2026 prediction across bootstrap draws")
    P = np.vstack(preds)
    sd = P.std(axis=0, ddof=1)
    base_final = final_pred(rkm, full)
    changed = np.abs(base_final - rkm["pred"].to_numpy()) > 1e-6
    w26 = rkm.with_columns(sd=pl.Series(sd), touched=pl.Series(changed))
    print(w26.filter("touched").group_by("ADEP_mvt").agg(
        pl.len().alias("rows_touched"), pl.col("sd").pow(2).mean().sqrt().alias("rms_sd_s"),
        pl.col("sd").quantile(0.9).alias("p90_sd_s")).sort("ADEP_mvt"))
    tot = rk.height
    ssum = float((sd[changed] ** 2).sum())
    print(f"implied RMSE inflation on the whole ranking set from parameter noise alone "
          f"(sqrt(sum sd^2 / N)): {np.sqrt(ssum / tot):.2f} s  (added in quadrature to a ~270 s RMSE: "
          f"{np.sqrt(270 ** 2 + ssum / tot) - 270:.3f} s)")

    # ---------------------------------------------------------- pull sizing
    section("PULL SIZING: Sep-Dec 2025 extra fit rows, from the 2025-09-15 / 11-15 recovery rates")
    d9 = pl.concat([pl.read_parquet(DET / f"day={d}.parquet") for d in TRAIN_DAYS]).rename({"airport": "ADEP_mvt"})
    d9 = adsb_partial.with_inputs(d9.with_columns(mvt_ts=pl.lit(0.0)))
    rates = d9.group_by("ADEP_mvt").agg(
        (pl.len() / 2).alias("dep_per_day"),
        adsb_blend.eligible_mask().mean().alias("ad_rate"),
        adsb_partial.mask().mean().alias("partial_rate"))
    dep_sepdec = (pl.scan_parquet([str(p) for p in sorted(RAW.glob("training_2025-*.parquet"))
                                   if p.name[9:16] in ("2025-09", "2025-10", "2025-11", "2025-12")])
                  .filter(pl.col("PHASE_mvt") == "DEP").group_by("ADEP_mvt").agg(pl.len().alias("dep_sepdec"))
                  .collect())
    size = rates.join(dep_sepdec, on="ADEP_mvt").with_columns(
        extra_ad_full=pl.col("dep_sepdec") * pl.col("ad_rate"),
        extra_partial_full=pl.col("dep_sepdec") * pl.col("partial_rate")).with_columns(
        extra_ad_half=pl.col("extra_ad_full") / 2, extra_partial_half=pl.col("extra_partial_full") / 2)
    fit_ad = ho.filter(adsb_blend.eligible_mask()).group_by("ADEP_mvt").len().rename({"len": "fit_ad_now"})
    fit_pt = adsb_partial.with_inputs(ho).filter(adsb_partial.mask()).group_by("ADEP_mvt").len().rename(
        {"len": "fit_partial_now"})
    print(size.join(fit_ad, on="ADEP_mvt", how="left").join(fit_pt, on="ADEP_mvt", how="left")
          .select("ADEP_mvt", "dep_sepdec", "ad_rate", "partial_rate", "fit_ad_now", "extra_ad_full",
                  "extra_ad_half", "fit_partial_now", "extra_partial_full", "extra_partial_half")
          .sort("ADEP_mvt"))
    print("caveat: two days' recovery rates; the Jan/Jul 2025 rates were lower at several airports "
          "(EDDF 12% vs 41%, EGLL 2.6% vs 7.6%), so these are optimistic for some airports.")
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
