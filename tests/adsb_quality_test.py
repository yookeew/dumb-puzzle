"""ADS-B blend quality modulation, per reports/adsb_quality_preregistration.md.

Run:  .venv/Scripts/python.exe tests/adsb_quality_test.py
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
from adsb_partial_test import load  # noqa: E402

DET = ROOT / "cache" / "adsb_pushback"
JAN, JUL = "2025-01", "2025-07"
D_OK, GS_OK, GAP_OK = 70.0, 5.0, 40.0
N_RES, SEED = 3000, 0


def quality() -> pl.Expr:
    good = (pl.col("adsb_pb_dist_m") <= D_OK) & pl.when(pl.col("adsb_tier") == "appear").then(
        pl.col("adsb_pb_gs") <= GS_OK).otherwise(pl.col("adsb_pb_gap_s") <= GAP_OK)
    return pl.when(good.fill_null(False)).then(pl.lit("good")).otherwise(pl.lit("poor"))


def weights(d: pl.DataFrame, p: dict) -> np.ndarray:
    """Production per-row blend weight (cell weight, else tier weight)."""
    wt = pl.col("adsb_tier").replace_strict(p["w"], default=0.0, return_dtype=pl.Float64)
    cell = {f"{a}|{t}": v for (a, t), v in p["cells"].items()}
    key = pl.concat_str("ADEP_mvt", pl.lit("|"), "adsb_tier")
    return d.select(key.replace_strict(cell, default=None, return_dtype=pl.Float64).fill_null(wt))\
        .to_series().to_numpy()


def main() -> None:
    df = load()
    q = pl.concat([pl.read_parquet(p) for p in sorted(DET.glob("day=2025-0[17]*.parquet"))]).select(
        "MVT_ID_mvt", "adsb_pb_gap_s", "adsb_pb_dist_m", "adsb_pb_gs")
    df = df.join(q, on="MVT_ID_mvt", how="left").with_columns(qual=quality())

    # cross-fit stack
    stack = np.empty(df.height)
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        f = df.filter(pl.col("ym") == fm)
        w, _ = nnls(f.select("lgb", "cat").to_numpy(), f["taxi"].to_numpy())
        m = (df["ym"] == am).to_numpy()
        stack[m] = df.filter(pl.col("ym") == am).select("lgb", "cat").to_numpy() @ w
    df = df.with_columns(pred=pl.Series(stack))

    elig = df.select(adsb_blend.eligible_mask()).to_series().to_numpy()
    base = np.empty(df.height)   # v17: stack -> blend -> partial
    new = np.empty(df.height)    # stack -> quality-modulated blend -> partial
    qs = {}
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        fmask = (df["ym"] == fm).to_numpy()
        p = adsb_blend.fit(df.filter(pl.col("ym") == fm), per_airport=True)
        blend = adsb_blend.apply(df, p).to_numpy()
        corr = df.pipe(lambda d: d.with_columns(adsb_c=pl.col("adsb_taxi") - pl.col("ADEP_mvt").replace_strict(
            p["lags"], default=p["pooled"], return_dtype=pl.Float64)))["adsb_c"].to_numpy()
        wrow = weights(df, p)
        u = wrow * (corr - stack)            # production step per row
        r = df["taxi"].to_numpy() - stack
        fac = {}
        for tier in ("appear", "dwell"):
            for ql in ("good", "poor"):
                g = fmask & elig & (df["adsb_tier"] == tier).to_numpy() & (df["qual"] == ql).to_numpy()
                fac[(tier, ql)] = float(max((r[g] * u[g]).sum() / max((u[g] ** 2).sum(), 1e-9), 0.0))
        qs[f"fit {fm}"] = fac
        mult = np.ones(df.height)
        for (tier, ql), v in fac.items():
            g = (df["adsb_tier"] == tier).to_numpy() & (df["qual"] == ql).to_numpy()
            mult[g] = v
        wq = np.clip(wrow * mult, 0, 1)
        qblend = np.where(elig, stack + wq * np.nan_to_num(corr - stack), stack)

        for arr, bl in ((base, blend), (new, qblend)):
            d = adsb_partial.with_inputs(df.with_columns(base=pl.Series(bl)))
            pp = adsb_partial.fit(d.filter(pl.col("ym") == fm))
            out = adsb_partial.apply(d, pp)
            m = (df["ym"] == am).to_numpy()
            arr[m] = out[m]
    for k, v in qs.items():
        print(f"q factors, {k}: " + "  ".join(f"{t}/{ql}={x:.2f}" for (t, ql), x in v.items()))

    def score(e: pl.Expr, label: str) -> tuple[float, float]:
        m = df.select(e).to_series().to_numpy()
        d = df.filter(e).with_columns(se_b=pl.Series((base[m] - df["taxi"].to_numpy()[m]) ** 2),
                                      se_t=pl.Series((new[m] - df["taxi"].to_numpy()[m]) ** 2))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(),
                                           cl["n"].to_numpy(), N_RES, SEED)
        print(f"  {label:<24} v17={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
              f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    print("\ncross-fit results vs v17:")
    ptA, _ = score(pl.col("ym") == JUL, "A  Jul (fit Jan)")
    ptB, _ = score(pl.col("ym") == JAN, "B  Jan (fit Jul)")
    _, pw = score(pl.lit(True), "POOLED")
    print(f"  rule: {'ADOPT' if ptA < 0 and ptB < 0 and pw < 0.05 else 'REJECT'}")
    t = df.with_columns(b=pl.Series(base), n=pl.Series(new), e=pl.Series(elig)).filter("e")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=1, tbl_rows=10):
        print(t.group_by("adsb_tier", "qual").agg(pl.len().alias("rows"),
              ((pl.col("b") - pl.col("taxi")).pow(2).mean().sqrt()).alias("v17"),
              ((pl.col("n") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new")).sort("adsb_tier", "qual"))


if __name__ == "__main__":
    main()
