"""ADS-B partial-track estimate on top of stack + ADS-B, per its pre-registration.

Implements reports/adsb_partial_preregistration.md. The base is the cross-fit
production pipeline on the Jan+Jul 2025 holdout (lgb+cat NNLS stack, then the
§39 ADS-B blend); the partial-track blend is cross-fit on top of it.

Run:  .venv/Scripts/python.exe tests/adsb_partial_test.py
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
from post import adsb_blend  # noqa: E402

EVAL = ROOT / "cache" / "eval"
DET = ROOT / "cache" / "adsb_pushback"
RAW = ROOT / "data" / "raw"
JAN, JUL = "2025-01", "2025-07"
D_MAX, D_SPLIT = 1000.0, 500.0
MIN_AP_ROWS = 100
N_RES, SEED = 3000, 0


def load() -> pl.DataFrame:
    ev = {e: pl.read_parquet(EVAL / f"{e}_mixed_holdout_ev.parquet") for e in ("lgb", "cat")}
    df = ev["lgb"].select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", pl.col("ym").cast(pl.Utf8),
                          "taxi", pl.col("pred").alias("lgb")).join(
        ev["cat"].select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias("cat")),
        on="MVT_ID_mvt", how="inner")
    mvt = (pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                   mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                   day=pl.col("MVT_TIME_UTC_mvt").dt.date()).collect())
    det = pl.concat([pl.read_parquet(p) for p in sorted(DET.glob("day=2025-0[17]*.parquet"))])
    df = df.join(mvt, on="MVT_ID_mvt", how="left").join(det, on="MVT_ID_mvt", how="left")
    return df.with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts"),
                           L=pl.col("mvt_ts") - pl.col("adsb_first_ts"),
                           dkm=pl.col("adsb_first_own_m") / 1000.0)


def partial_mask() -> pl.Expr:
    return (pl.col("adsb_matched").fill_null(False)
            & ~pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False)
            & (pl.col("adsb_first_own_m") < D_MAX).fill_null(False)
            & (pl.col("ADEP_mvt") != "LIRF"))


def base_pred(df: pl.DataFrame) -> np.ndarray:
    """Cross-fit stack, then cross-fit ADS-B blend (the current production pipeline)."""
    stack = np.empty(df.height)
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        f = df.filter(pl.col("ym") == fm)
        w, _ = nnls(f.select("lgb", "cat").to_numpy(), f["taxi"].to_numpy())
        m = (df["ym"] == am).to_numpy()
        stack[m] = df.filter(pl.col("ym") == am).select("lgb", "cat").to_numpy() @ w
    d = df.with_columns(pred=pl.Series(stack))
    out = np.empty(df.height)
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        p = adsb_blend.fit(d.filter(pl.col("ym") == fm), per_airport=True)
        b = adsb_blend.apply(d, p).to_numpy()
        m = (d["ym"] == am).to_numpy()
        out[m] = b[m]
    return out


def fit_partial(f: pl.DataFrame, split: bool) -> dict:
    f = f.filter(partial_mask())
    unseen = (f["taxi"] - f["L"]).to_numpy()
    lo, hi = np.percentile(unseen, [1, 99])
    y = np.clip(unseen, lo, hi)
    counts = f.group_by("ADEP_mvt").len()
    big = sorted(counts.filter(pl.col("len") >= MIN_AP_ROWS)["ADEP_mvt"].to_list())
    ap = f["ADEP_mvt"].to_numpy()
    X = np.column_stack([(ap == a).astype(float) for a in big]
                        + [(~np.isin(ap, big)).astype(float), f["dkm"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = dict(zip(big, coef[:len(big)]))
    p = {"a": a, "a_pooled": float(coef[len(big)]), "b": float(coef[-1])}
    est = _est(f, p)
    base = f["base"].to_numpy()
    r = f["taxi"].to_numpy() - base
    dl = est - base

    def w(m):
        return float(np.clip((r[m] * dl[m]).sum() / max((dl[m] ** 2).sum(), 1e-9), 0, 1))

    near = (f["adsb_first_own_m"] < D_SPLIT).to_numpy()
    p["w"] = {"near": w(near), "far": w(~near)} if split else {"all": w(np.ones(len(r), bool))}
    return p


def _est(df: pl.DataFrame, p: dict) -> np.ndarray:
    a = df["ADEP_mvt"].replace_strict(p["a"], default=p["a_pooled"], return_dtype=pl.Float64).to_numpy()
    return df["L"].to_numpy() + a + p["b"] * df["dkm"].to_numpy()


def apply_partial(df: pl.DataFrame, p: dict) -> np.ndarray:
    base = df["base"].to_numpy().copy()
    m = df.select(partial_mask()).to_series().to_numpy()
    sub = df.filter(partial_mask())
    est = _est(sub, p)
    if "all" in p["w"]:
        w = p["w"]["all"]
    else:
        w = np.where((sub["adsb_first_own_m"] < D_SPLIT).to_numpy(), p["w"]["near"], p["w"]["far"])
    base[m] = base[m] + w * (est - base[m])
    return base


def cross(df: pl.DataFrame, split: bool) -> tuple[np.ndarray, dict]:
    out, ps = np.empty(df.height), {}
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        p = fit_partial(df.filter(pl.col("ym") == fm), split)
        ps[f"fit {fm}"] = p
        b = apply_partial(df, p)
        m = (df["ym"] == am).to_numpy()
        out[m] = b[m]
    return out, ps


def score(df: pl.DataFrame, new: np.ndarray, label: str) -> tuple[float, float]:
    d = df.with_columns(se_b=(pl.col("base") - pl.col("taxi")).pow(2),
                        se_t=pl.Series((new - df["taxi"].to_numpy()) ** 2))
    cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
    pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(),
                                       cl["n"].to_numpy(), N_RES, SEED)
    print(f"  {label:<26} base={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
          f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
    return pt, pw


def main() -> None:
    df = load()
    df = df.with_columns(base=pl.Series(base_pred(df)))
    print(f"holdout rows {df.height:,}; base (stack + ADS-B, cross-fit) RMSE "
          f"{np.sqrt(((df['base'] - df['taxi']) ** 2).mean()):.2f}")
    print(f"partial-eligible rows: {df.select(partial_mask().sum()).item():,}")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20):
        print(df.filter(partial_mask()).group_by("ym", "ADEP_mvt").len()
              .pivot(on="ADEP_mvt", index="ym", values="len").sort("ym"))

    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()
    print("\nPRIMARY (one weight):")
    out, ps = cross(df, split=False)
    for k, p in ps.items():
        print(f"  {k}: b={p['b']:.0f} s/km  w={p['w']}  a={ {a: round(v) for a, v in p['a'].items()} }"
              f"  a_pooled={p['a_pooled']:.0f}")
    ptA, _ = score(df.filter(pl.col("ym") == JUL), out[jul], "A  Jul (fit Jan)")
    ptB, _ = score(df.filter(pl.col("ym") == JAN), out[jan], "B  Jan (fit Jul)")
    _, pwP = score(df, out, "POOLED")
    print(f"  rule: {'ADOPT' if ptA < 0 and ptB < 0 and pwP < 0.05 else 'REJECT'}")
    pm = df.select(partial_mask()).to_series().to_numpy()
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20, float_precision=1):
        t = df.with_columns(new=pl.Series(out), elig=pl.Series(pm))
        print("\n  per airport (all rows):")
        print(t.group_by("ADEP_mvt").agg(pl.col("elig").sum().alias("n_elig"),
              ((pl.col("base") - pl.col("taxi")).pow(2).mean().sqrt()).alias("base"),
              ((pl.col("new") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new"))
              .with_columns(delta=pl.col("new") - pl.col("base")).sort("ADEP_mvt"))
        print("  per distance band (eligible rows):")
        print(t.filter("elig").with_columns(band=pl.when(pl.col("adsb_first_own_m") < D_SPLIT)
                                            .then(pl.lit("<500m")).otherwise(pl.lit("500-1000m")))
              .group_by("band").agg(pl.len().alias("n"),
              ((pl.col("base") - pl.col("taxi")).pow(2).mean().sqrt()).alias("base"),
              ((pl.col("new") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new")).sort("band"))

    print("\nSECONDARY (two distance-band weights):")
    out2, ps2 = cross(df, split=True)
    for k, p in ps2.items():
        print(f"  {k}: w={ {kk: round(v, 3) for kk, v in p['w'].items()} }")
    rm = lambda x, m: float(np.sqrt(np.mean((x[m] - df["taxi"].to_numpy()[m]) ** 2)))  # noqa: E731
    wins = rm(out2, jul) < rm(out, jul) and rm(out2, jan) < rm(out, jan)
    print(f"  vs primary: Jul {rm(out, jul):.2f} -> {rm(out2, jul):.2f}  Jan {rm(out, jan):.2f} -> "
          f"{rm(out2, jan):.2f}  => {'ADOPT over primary' if wins else 'not adopted'}")


if __name__ == "__main__":
    main()
