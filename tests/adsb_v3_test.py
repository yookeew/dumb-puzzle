"""ADS-B v3 (inferred stands + fallback match) gate, per reports/adsb_v3_preregistration.md.

Base: v21 cross-fit on the Jan+Jul 2025 holdout with the production detector
cache (tests/ltfm_egll_anatomy.py::v21_pred). Treatment: the same pipeline on
cache/adsb_pushback_v3/, then the gated fallback blend. Decision on trimmed RMSE.

Run:  .venv/Scripts/python.exe tests/adsb_v3_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402
from post import adsb_blend, adsb_partial  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

JAN, JUL = A.JAN, A.JUL
V1, V3 = ROOT / "cache" / "adsb_pushback", ROOT / "cache" / "adsb_pushback_v3"
FB_TIERS = ("fb_dwell", "fb_appear")
GATE_S = 300.0
TRIM = 18000
N_RES, SEED = 3000, 0
DET_COLS = ["adsb_day_coverage", "adsb_matched", "adsb_tier", "adsb_pushback_ts", "adsb_takeoff_gap_s",
            "adsb_n_pts", "adsb_first_ts", "adsb_first_gs", "adsb_first_own_m", "adsb_min_own_m",
            "adsb_pb_gap_s", "adsb_pb_dist_m", "adsb_pb_gs"]


def read_det(d: Path) -> pl.DataFrame:
    det = pl.concat([pl.read_parquet(p) for p in sorted(d.glob("day=2025-0[17]*.parquet"))], how="diagonal")
    if "adsb_fallback" not in det.columns:
        det = det.with_columns(adsb_fallback=pl.lit(False))
    return det.select("MVT_ID_mvt", *DET_COLS, "adsb_fallback")


def with_det(base: pl.DataFrame, det: pl.DataFrame) -> pl.DataFrame:
    return (base.drop(DET_COLS + ["adsb_taxi"]).join(det, on="MVT_ID_mvt", how="left")
            .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))


def fb_only(v1: pl.DataFrame, v3: pl.DataFrame) -> pl.DataFrame:
    """v1 detections, with the v3 fallback rows' fields swapped in."""
    fb = v3.filter(pl.col("adsb_fallback"))
    return pl.concat([v1.join(fb.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti"), fb])


def no_fb(v3: pl.DataFrame) -> pl.DataFrame:
    """v3 detections with the fallback rows reset to unmatched (inferred stands only)."""
    return v3.with_columns(*[pl.when(pl.col("adsb_fallback")).then(None).otherwise(pl.col(c)).alias(c)
                             for c in ("adsb_tier", "adsb_pushback_ts", "adsb_pb_gap_s", "adsb_pb_dist_m",
                                       "adsb_pb_gs")],
                           adsb_fallback=pl.lit(False))


def fb_mask(gate: bool) -> pl.Expr:
    m = pl.col("adsb_tier").is_in(FB_TIERS).fill_null(False) & (pl.col("ADEP_mvt") != "LIRF")
    if gate:
        m = m & ((pl.col("fb_c") - pl.col("base")).abs() <= GATE_S).fill_null(False)
    return m


def pipeline(df: pl.DataFrame, fb_stage: bool, gate: bool = True) -> tuple[np.ndarray, dict]:
    """v21 cross-fit (stack -> quality blend -> partial), plus the optional fallback blend."""
    eng = ["lgb", "catcorr"]
    out, info = np.empty(df.height), {}
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        f = df.filter(pl.col("ym") == fm)
        w, _ = nnls(f.select(eng).to_numpy(), f["taxi"].to_numpy())
        f = f.with_columns(pred=pl.Series(f.select(eng).to_numpy() @ w))
        p = adsb_blend.fit(f, per_airport=True)
        q = adsb_blend.fit_quality(f, p)
        f = adsb_partial.with_inputs(f.with_columns(base=adsb_blend.apply(f, p, q)))
        pp = adsb_partial.fit(f)
        d = df.with_columns(pred=pl.Series(df.select(eng).to_numpy() @ w))
        d = adsb_partial.with_inputs(d.with_columns(base=adsb_blend.apply(d, p, q)))
        b = adsb_partial.apply(d, pp)
        if fb_stage:
            lag = pl.col("ADEP_mvt").replace_strict(p["lags"], default=p["pooled"], return_dtype=pl.Float64)
            fpart = f.with_columns(base=pl.Series(adsb_partial.apply(f, pp)))
            fpart = fpart.with_columns(fb_c=pl.col("adsb_taxi") - lag).filter(fb_mask(gate))
            wt = {}
            for t in FB_TIERS:
                g = fpart.filter(pl.col("adsb_tier") == t)
                r, dl = (g["taxi"] - g["base"]).to_numpy(), (g["fb_c"] - g["base"]).to_numpy()
                wt[t] = float(np.clip((r * dl).sum() / max((dl * dl).sum(), 1e-9), 0, 1)) if g.height else 0.0
            dd = d.with_columns(base=pl.Series(b), fb_c=pl.col("adsb_taxi") - lag)
            m = dd.select(fb_mask(gate)).to_series().to_numpy()
            wv = dd["adsb_tier"].replace_strict(wt, default=0.0, return_dtype=pl.Float64).to_numpy()
            b = np.where(m, b + wv * (dd["fb_c"].to_numpy() - b), b)
            info[f"fit {fm}"] = {"w": {k: round(v, 3) for k, v in wt.items()}, "n_fit": fpart.height,
                                 "n_applied_other_month": int(m[(df["ym"] == am).to_numpy()].sum())}
        sel = (df["ym"] == am).to_numpy()
        out[sel] = b[sel]
    return out, info


def score(df: pl.DataFrame, base: np.ndarray, new: np.ndarray, label: str) -> tuple[float, float]:
    d = df.with_columns(se_b=pl.Series((base - df["taxi"].to_numpy()) ** 2),
                        se_t=pl.Series((new - df["taxi"].to_numpy()) ** 2))
    cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
    pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), N_RES, SEED)
    print(f"  {label:<28} base={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
          f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
    return pt, pw


def evaluate(df: pl.DataFrame, base: np.ndarray, new: np.ndarray, name: str, decisive: bool = False) -> None:
    print(f"\n=== {name} ===")
    tr = (df["taxi"] <= TRIM).to_numpy()
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()
    sub = lambda m: df.filter(pl.Series(m))  # noqa: E731
    print(" TRIMMED (decides):")
    ptJ, _ = score(sub(tr & jul), base[tr & jul], new[tr & jul], "A  Jul (fit Jan)")
    ptB, _ = score(sub(tr & jan), base[tr & jan], new[tr & jan], "B  Jan (fit Jul)")
    ptP, pwP = score(sub(tr), base[tr], new[tr], "POOLED trimmed")
    print(" FULL (guard):")
    _, pwF = score(df, base, new, "POOLED full")
    if decisive:
        ok = ptP < 0 and pwP < 0.05 and ptJ < 0 and ptB < 0 and pwF < 0.9
        print(f"\nPRIMARY DECISION: {'ADOPT -> build, board A/B' if ok else 'REJECT -> stop'}")
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12, float_precision=2):
            t = df.with_columns(b=pl.Series(base), n=pl.Series(new)).filter(pl.col("taxi") <= TRIM)
            print(t.group_by("ADEP_mvt").agg(
                ((pl.col("b") - pl.col("taxi")).pow(2).mean().sqrt()).alias("base"),
                ((pl.col("n") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new"),
                (pl.col("b") != pl.col("n")).sum().alias("changed"))
                .with_columns(delta=pl.col("new") - pl.col("base")).sort("ADEP_mvt"))


def main() -> None:
    df = A.load()
    base = A.v21_pred(df)
    v1, v3 = read_det(V1), read_det(V3)
    print(f"holdout rows {df.height:,}; v21 cross-fit trimmed RMSE "
          f"{np.sqrt(np.mean((base - df['taxi'].to_numpy())[(df['taxi'] <= TRIM).to_numpy()] ** 2)):.2f}")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12):
        print(v3.join(df.select("MVT_ID_mvt", "ym"), on="MVT_ID_mvt").group_by("ym", "adsb_tier")
              .len().pivot(on="adsb_tier", index="ym", values="len").sort("ym"))

    d3 = with_det(df, v3)
    new, info = pipeline(d3, fb_stage=True, gate=True)
    print("\nfallback weights:", info)
    evaluate(d3, base, new, "PRIMARY: v3 (inferred stands + gated fallback)", decisive=True)

    parts = {"inferred stands only": (with_det(df, no_fb(v3)), False, True),
             "fallback only (gated)": (with_det(df, fb_only(v1, v3)), True, True),
             "v3, fallback UNgated": (d3, True, False)}
    for name, (d, fb, gate) in parts.items():
        n2, inf2 = pipeline(d, fb_stage=fb, gate=gate)
        if fb:
            print(f"\n{name} weights: {inf2}")
        evaluate(d, base, n2, f"reported: {name}")


if __name__ == "__main__":
    main()
