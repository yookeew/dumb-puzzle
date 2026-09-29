"""Pre-registered gate: ADS-B detector with empirical positions for the 19 stands
absent from Gateway (reports/adsb_missing_stand_preregistration.md, commit 5a091ab).

Step 1 (label-free): write the override detector output for all 126 days
(src/link/adsb_pushback.main(missing_stands=True) -> cache/adsb_pushback_ms/) and
check that every row not at the 19 stands is identical to the shipped output, except
rows whose matched run was re-assigned to a movement at one of the 19 stands.

Step 2 (gate): v21 stack (lgb + catcorr, NNLS cross-fit Jan<->Jul, identical in both
arms), then the three ADS-B stages exactly as stack_submit.py --cat-corrected fits
them (blend with per-airport cells, quality factors, partial on the q-blended base,
final clip), each fitted on one holdout month and applied to the other. Baseline arm =
shipped detector output, treatment = override output. Decision rule per §47.

Run:  .venv/Scripts/python.exe tests/adsb_missing_stand_test.py > logs/adsb_missing_stand_test.log
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from _harness import cluster_bootstrap  # noqa: E402
from post import adsb_blend, adsb_partial  # noqa: E402
from src.ingest.normalise_adsb import available_days  # noqa: E402
from src.ingest.stands import load_stands, missing_stand_overrides, stand_key, stand_positions  # noqa: E402
from src.link import adsb_pushback  # noqa: E402

EVAL = ROOT / "cache" / "eval"
RAW = ROOT / "data" / "raw"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
DET, DET_MS = adsb_pushback.OUT, adsb_pushback.OUT_MS
ENGINES = ["lgb", "catcorr"]
JAN, JUL = "2025-01", "2025-07"
TRAIN_DAYS = ("2025-09-15", "2025-11-15")
MONSTER_S, N_MONSTER = 5 * 3600, 31
CEIL, LIRF_CEIL = 10800, 140000          # stack_submit._clip
N_RES, SEED = 3000, 0
N_OVERRIDES = 19


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def stands_of_day(day: dt.date) -> pl.DataFrame:
    return adsb_pushback.departures(day).select("MVT_ID_mvt", "ADEP_mvt", key=stand_key(pl.col("STAND_mvt")))


# --------------------------------------------------------------------------- step 1
def step1() -> None:
    section("STEP 1: override detector output + identity check")
    ov = missing_stand_overrides()
    assert ov.height == N_OVERRIDES, f"expected {N_OVERRIDES} override stands, got {ov.height}"
    print(f"override stands: {ov.height}  {sorted((a, k) for a, k in ov.select('airport', 'key').iter_rows())}")
    t0 = dt.datetime.now()
    adsb_pushback.main(missing_stands=True)
    print(f"wrote {DET_MS.relative_to(ROOT)} ({len(list(DET_MS.glob('day=*.parquet')))} days) in "
          f"{(dt.datetime.now() - t0).seconds}s")
    pos = stand_positions(load_stands())
    ovkeys = ov.select(pl.col("airport").alias("ADEP_mvt"), "key").with_columns(at_override=pl.lit(True))
    cols = list(adsb_pushback.SCHEMA)
    tot = dict(rows=0, at_override=0, changed_override=0, reassigned=0, violations=0)
    reassigned_rows = []
    for day in available_days():
        new = pl.read_parquet(DET_MS / f"day={day.isoformat()}.parquet")
        old = (adsb_pushback.detect_day(day, pos) if day.isoformat() in TRAIN_DAYS
               else pl.read_parquet(DET / f"day={day.isoformat()}.parquet"))
        st = stands_of_day(day).join(ovkeys, on=["ADEP_mvt", "key"], how="left").with_columns(
            pl.col("at_override").fill_null(False))
        j = (old.join(new, on="MVT_ID_mvt", how="full", suffix="_n", coalesce=True)
             .join(st.select("MVT_ID_mvt", "at_override"), on="MVT_ID_mvt", how="left"))
        diff = pl.lit(False)
        for c in cols[1:]:
            diff = diff | pl.col(c).ne_missing(pl.col(f"{c}_n"))
        j = j.with_columns(diff=diff)
        new_ov_runs = set(new.join(st.filter("at_override").select("MVT_ID_mvt"), on="MVT_ID_mvt")
                          .filter("adsb_matched").select("airport", "adsb_first_ts", "adsb_n_pts").iter_rows())
        other = j.filter(~pl.col("at_override") & pl.col("diff"))
        ok_re = [(r["airport"], r["adsb_first_ts"], r["adsb_n_pts"]) in new_ov_runs for r in other.to_dicts()]
        n_re = int(sum(ok_re))
        viol = other.filter(~pl.Series(ok_re, dtype=pl.Boolean)) if other.height else other
        tot["rows"] += j.height
        tot["at_override"] += int(j["at_override"].sum())
        tot["changed_override"] += int(j.filter(pl.col("at_override") & pl.col("diff")).height)
        tot["reassigned"] += n_re
        tot["violations"] += viol.height
        if other.height:
            reassigned_rows.append(other.select("MVT_ID_mvt", "airport", pl.lit(day.isoformat()).alias("day"),
                                                "adsb_tier", pl.col("adsb_tier_n").alias("adsb_tier_new"),
                                                "adsb_matched", pl.col("adsb_matched_n").alias("adsb_matched_new"))
                                   .with_columns(explained=pl.Series(ok_re, dtype=pl.Boolean)))
        if viol.height:
            print(f"  VIOLATION {day}: {viol.height} rows differ outside the override stands, not re-assigned")
    print(f"identity check: {tot}")
    if reassigned_rows:
        r = pl.concat(reassigned_rows)
        with pl.Config(tbl_rows=60, tbl_formatting="ASCII_MARKDOWN"):
            print("non-override rows that changed (all must be 'explained' = run re-assigned to an override-stand movement):")
            print(r.group_by("airport", "adsb_tier", "adsb_tier_new", "explained").len().sort("airport"))
    if tot["violations"]:
        raise SystemExit("STOP (pre-registered): rows outside the 19 stands changed for another reason")
    print("identity check PASSED")


# --------------------------------------------------------------------------- step 2
def det(d: Path) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(d.glob("day=2025-0[17]-*.parquet"))]).select(
        "MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts", "adsb_matched", "adsb_first_ts",
        "adsb_first_own_m", "adsb_pb_gap_s", "adsb_pb_dist_m", "adsb_pb_gs")


def fit_all(f: pl.DataFrame) -> dict:
    p = adsb_blend.fit(f, per_airport=True)
    q = adsb_blend.fit_quality(f, p)
    fq = adsb_partial.with_inputs(f.with_columns(base=adsb_blend.apply(f, p, q)))
    return {"p": p, "q": q, "pp": adsb_partial.fit(fq)}


def apply_all(df: pl.DataFrame, f: dict) -> np.ndarray:
    d = adsb_partial.with_inputs(df.with_columns(base=adsb_blend.apply(df, f["p"], f["q"])))
    out = adsb_partial.apply(d, f["pp"])
    hi = np.where(df["ADEP_mvt"].to_numpy() == "LIRF", LIRF_CEIL, CEIL)
    return np.clip(out, 0, hi)


def flat(f: dict) -> dict[str, float]:
    p, q, pp = f["p"], f["q"], f["pp"]
    o = {f"lag|{a}": v for a, v in p["lags"].items()}
    o["lag|pooled"] = p["pooled"]
    o.update({f"w_tier|{t}": v for t, v in p["w"].items()})
    o.update({f"w_cell|{a}/{t}": v for (a, t), v in p["cells"].items()})
    o.update({f"q|{t}/{ql}": v for (t, ql), v in q.items()})
    o.update({f"partial_a|{a}": v for a, v in pp["a"].items()})
    o["partial_a|pooled"], o["partial_b"] = pp["a_pooled"], pp["b"]
    o.update({f"partial_w|{k}": v for k, v in pp["w"].items()})
    return o


def step2() -> None:
    section("STEP 2: GATE (v21 stack + ADS-B stages, cross-fit Jan<->Jul)")
    ho = None
    for e in ENGINES:
        d = pl.read_parquet(EVAL / f"{e}_mixed_holdout_ev.parquet").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", pl.col("ym").cast(pl.Utf8), "taxi", pl.col("pred").alias(e))
        ho = d if ho is None else ho.join(d.select("MVT_ID_mvt", e), on="MVT_ID_mvt", how="inner")
    stack = np.empty(ho.height)
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        f = ho.filter(pl.col("ym") == fm)
        w, _ = nnls(f.select(ENGINES).to_numpy(), f["taxi"].to_numpy().astype(float))
        m = (ho["ym"] == am).to_numpy()
        stack[m] = ho.filter(pl.col("ym") == am).select(ENGINES).to_numpy() @ w
        print(f"  stack weights fit {fm}: {dict(zip(ENGINES, np.round(w, 4)))}")
    mv = (pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col("MVT_ID_mvt").cast(pl.Int64), mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                  day=pl.col("MVT_TIME_UTC_mvt").dt.date()).collect())
    ho = ho.with_columns(pred=pl.Series(stack)).join(mv, on="MVT_ID_mvt", how="left")

    arms, fits = {}, {}
    for name, d in (("base", DET), ("treat", DET_MS)):
        a = ho.join(det(d), on="MVT_ID_mvt", how="left").with_columns(
            adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts"))
        out = np.empty(a.height)
        fits[name] = {}
        for fm, am in ((JAN, JUL), (JUL, JAN)):
            f = fit_all(a.filter(pl.col("ym") == fm))
            fits[name][fm] = flat(f)
            m = (a["ym"] == am).to_numpy()
            out[m] = apply_all(a, f)[m]
        fits[name]["full"] = flat(fit_all(a))
        arms[name] = (a, out)
    a_b, pb = arms["base"]
    a_t, pt = arms["treat"]
    assert a_b["MVT_ID_mvt"].equals(a_t["MVT_ID_mvt"])
    t = a_b["taxi"].to_numpy().astype(float)
    keep = t <= MONSTER_S
    assert int((~keep).sum()) == N_MONSTER, f"expected {N_MONSTER} monster rows, got {int((~keep).sum())}"
    df = a_b.select("MVT_ID_mvt", "ADEP_mvt", "ym", "day", "taxi").with_columns(
        base=pl.Series(pb), treat=pl.Series(pt),
        ad_b=a_b["adsb_tier"].is_in(["appear", "dwell"]).fill_null(False),
        ad_t=a_t["adsb_tier"].is_in(["appear", "dwell"]).fill_null(False))

    def score(mask: np.ndarray, label: str):
        d = df.with_columns(se_b=(pl.col("base") - pl.col("taxi")) ** 2,
                            se_t=(pl.col("treat") - pl.col("taxi")) ** 2).filter(pl.Series(mask))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        p_, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), N_RES, SEED)
        print(f"  {label:<22} base={np.sqrt(d['se_b'].mean()):8.3f}  treat={np.sqrt(d['se_t'].mean()):8.3f}  "
              f"delta={p_:+7.3f}  CI[{lo:+.3f},{hi:+.3f}]  P(worse)={pw:.3f}")
        return p_, pw

    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()
    print(f"holdout rows {df.height:,}; monster rows {N_MONSTER}; rows differing between arms "
          f"{int((np.abs(pb - pt) > 1e-9).sum()):,}")
    print(" TRIMMED (decides):")
    d_jan, _ = score(jan & keep, "Jan trimmed")
    d_jul, _ = score(jul & keep, "Jul trimmed")
    d_pool, p_pool = score(keep, "POOLED trimmed")
    print(" FULL (guard):")
    score(jan, "Jan full")
    score(jul, "Jul full")
    _, p_full = score(np.ones(df.height, bool), "POOLED full")
    adopt = d_pool < 0 and p_pool < 0.05 and d_jan < 0 and d_jul < 0 and p_full < 0.9
    print(f"\nrule: pooled trimmed delta<0 & P<0.05 [{d_pool:+.3f}, {p_pool:.3f}], both months<0 "
          f"[Jan {d_jan:+.3f}, Jul {d_jul:+.3f}], full guard P<0.9 [{p_full:.3f}]  -> "
          f"{'ADOPT (stage for v22)' if adopt else 'REJECT'}")

    section("SECONDARY (not decisive)")
    moved = df.filter(pl.col("ad_t") & ~pl.col("ad_b"))
    lost = df.filter(pl.col("ad_b") & ~pl.col("ad_t"))
    print(f"moved into appear/dwell: {moved.height:,}; moved out: {lost.height:,}")
    agg = lambda c: ((pl.col(c) - pl.col("taxi")) ** 2).mean().sqrt()  # noqa: E731
    with pl.Config(tbl_rows=40, tbl_formatting="ASCII_MARKDOWN", float_precision=2, tbl_cols=-1):
        for lbl, fr in (("moved rows", moved), ("moved-out rows", lost)):
            print(f"{lbl}, per airport and month:")
            print(fr.group_by("ADEP_mvt", "ym").agg(pl.len(), agg("base").alias("base_rmse"),
                                                     agg("treat").alias("treat_rmse"))
                  .with_columns(delta=pl.col("treat_rmse") - pl.col("base_rmse")).sort("ADEP_mvt", "ym"))
        print("per airport, all rows (trimmed):")
        print(df.filter(pl.Series(keep)).group_by("ADEP_mvt").agg(
            pl.len(), agg("base").alias("base"), agg("treat").alias("treat"),
            (pl.col("treat") - pl.col("base")).abs().gt(1e-9).sum().alias("rows_changed"))
            .with_columns(delta=pl.col("treat") - pl.col("base")).sort("ADEP_mvt"))
        unmoved = ~(df["ad_t"] & ~df["ad_b"]).to_numpy() & ~(df["ad_b"] & ~df["ad_t"]).to_numpy()
        ch = np.abs(pb - pt)[unmoved]
        print(f"rows NOT moving tier whose prediction changed (refit parameters): {int((ch > 1e-6).sum()):,}; "
              f"mean |change| on those {ch[ch > 1e-6].mean() if (ch > 1e-6).any() else 0:.2f} s; "
              f"max {ch.max():.2f} s")
        rows = []
        for k in sorted(set(fits["base"]["full"]) | set(fits["treat"]["full"])):
            r = dict(param=k)
            for s in ("full", JAN, JUL):
                b, tt = fits["base"][s].get(k), fits["treat"][s].get(k)
                r[f"base_{s}"], r[f"treat_{s}"] = b, tt
                r[f"d_{s}"] = None if b is None or tt is None else tt - b
            rows.append(r)
        pr = pl.DataFrame(rows)
        print("parameter shifts, treat - base (fit on full holdout / Jan / Jul); rows with any change:")
        print(pr.filter(pl.any_horizontal([pl.col(f"d_{s}").abs() > 1e-9 for s in ("full", JAN, JUL)])
                        | pl.any_horizontal([pl.col(f"base_{s}").is_null() != pl.col(f"treat_{s}").is_null()
                                             for s in ("full", JAN, JUL)]))
              .select("param", "base_full", "treat_full", "d_full", "d_2025-01", "d_2025-07"))


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"pre-registration: reports/adsb_missing_stand_preregistration.md")
    step1()
    step2()
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
