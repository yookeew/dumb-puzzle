"""LIRF echo-aware ADS-B mixture, per reports/lirf_mixture_preregistration.md.

Within-July day-split cross-fit (Jan 2025 has no LIRF ADS-B coverage). The
nudge acts only on each engine's non-echo component m_i:
    new = base + w * sum_i s_i (1 - e_i) (adsb_c - m_i)

Run:  .venv/Scripts/python.exe tests/lirf_mixture_test.py
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
from post import adsb_partial  # noqa: E402
from adsb_partial_test import base_pred, load  # noqa: E402

EVAL = ROOT / "cache" / "eval"
JAN, JUL = "2025-01", "2025-07"
ECHO_CEIL = 140000
N_RES, SEED = 3000, 0


def engine_parts() -> pl.DataFrame:
    parts = None
    for e in ("lgb", "cat"):
        ev = pl.read_parquet(EVAL / f"{e}_mixed_holdout_ev.parquet").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64),
            pl.col("echo_prob").alias(f"e_{e}"),
            pl.col("taxi_model_raw").clip(0, ECHO_CEIL).alias(f"m_{e}"),
            pl.col("sched_takeoff_offset").alias("offset") if e == "lgb" else pl.lit(None).alias("_x"),
            pl.col("pred").alias(f"p_{e}"), "is_echo")
        ev = ev.drop([c for c in ev.columns if c == "_x"])
        parts = ev if parts is None else parts.join(ev.drop("is_echo"), on="MVT_ID_mvt")
    return parts


def main() -> None:
    df = load()
    df = df.with_columns(base=pl.Series(base_pred(df)))
    df = adsb_partial.with_inputs(df)
    out = np.empty(df.height)
    for fm, am in ((JAN, JUL), (JUL, JAN)):  # production base incl. §41, cross-fit
        p = adsb_partial.fit(df.filter(pl.col("ym") == fm))
        b = adsb_partial.apply(df, p)
        m = (df["ym"] == am).to_numpy()
        out[m] = b[m]
    df = df.with_columns(base=pl.Series(out)).join(engine_parts(), on="MVT_ID_mvt", how="left")

    # decomposition check on LIRF rows: pred_i == e_i*clip(offset) + (1-e_i)*m_i
    li = df.filter(pl.col("ADEP_mvt") == "LIRF")
    for e in ("lgb", "cat"):
        rec = li[f"e_{e}"] * li["offset"].clip(0, ECHO_CEIL) + (1 - li[f"e_{e}"]) * li[f"m_{e}"]
        err = float((rec - li[f"p_{e}"]).abs().max())
        print(f"decomposition check {e}: max |reconstructed - pred| = {err:.6f}")
        assert err < 1e-3, "echo decomposition does not match the engine's prediction"

    # stack weights for July rows: fit on Jan (production cross-fit)
    fj = df.filter(pl.col("ym") == JAN)
    s, _ = nnls(fj.select("lgb", "cat").to_numpy(), fj["taxi"].to_numpy())
    print(f"stack weights for Jul rows (fit Jan): lgb {s[0]:.3f} cat {s[1]:.3f}")

    jul = df.filter(pl.col("ym") == JUL).with_columns(
        odd=(pl.col("day").dt.day() % 2 == 1),
        elig=(pl.col("ADEP_mvt") == "LIRF") & pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False))
    print(f"July rows {jul.height:,}; LIRF eligible {jul['elig'].sum():,} "
          f"(echo {jul.filter('elig')['is_echo'].sum():.0f})")

    def fit(f: pl.DataFrame) -> dict:
        f = f.filter("elig")
        ne = f.filter(pl.col("is_echo") == 0)
        lag = float((ne["adsb_taxi"] - ne["taxi"]).median())
        c = _c(f, lag)
        r = (f["taxi"] - f["base"]).to_numpy()
        w = float(np.clip((r * c).sum() / max((c * c).sum(), 1e-9), 0, 1))
        return {"lag": lag, "w": w}

    def _c(f: pl.DataFrame, lag: float) -> np.ndarray:
        adsb_c = (f["adsb_taxi"] - lag).to_numpy()
        return sum(s[i] * (1 - f[f"e_{e}"].to_numpy()) * (adsb_c - f[f"m_{e}"].to_numpy())
                   for i, e in enumerate(("lgb", "cat")))

    def apply(d: pl.DataFrame, p: dict) -> np.ndarray:
        new = d["base"].to_numpy().copy()
        m = d["elig"].to_numpy()
        new[m] = new[m] + p["w"] * _c(d.filter("elig"), p["lag"])
        return new

    new = np.empty(jul.height)
    for fit_odd in (True, False):
        p = fit(jul.filter(pl.col("odd") == fit_odd))
        print(f"  fit on {'odd' if fit_odd else 'even'} days: lag={p['lag']:+.0f}s  w={p['w']:.3f}")
        m = (jul["odd"] != fit_odd).to_numpy()
        new[m] = apply(jul, p)[m]

    def score(d: pl.DataFrame, nv: np.ndarray, label: str) -> tuple[float, float]:
        x = d.with_columns(se_b=(pl.col("base") - pl.col("taxi")).pow(2),
                           se_t=pl.Series((nv - d["taxi"].to_numpy()) ** 2))
        cl = x.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(),
                                           cl["n"].to_numpy(), N_RES, SEED)
        print(f"  {label:<34} base={np.sqrt(x['se_b'].mean()):8.2f}  new={np.sqrt(x['se_t'].mean()):8.2f}  "
              f"delta={pt:+7.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    print("\nWithin-July day-split cross-fit (all July rows):")
    odd = jul["odd"].to_numpy()
    pt_e, _ = score(jul.filter(~pl.col("odd")), new[~odd], "even days (fit odd)")
    pt_o, _ = score(jul.filter(pl.col("odd")), new[odd], "odd days (fit even)")
    _, pw = score(jul, new, "POOLED July")
    lirf = (jul["ADEP_mvt"] == "LIRF").to_numpy()
    score(jul.filter(pl.col("ADEP_mvt") == "LIRF"), new[lirf], "LIRF rows only")

    per_day = jul.with_columns(new=pl.Series(new)).filter(pl.col("ADEP_mvt") == "LIRF").group_by("day").agg(
        ((pl.col("new") - pl.col("taxi")).pow(2) - (pl.col("base") - pl.col("taxi")).pow(2)).sum().alias("dse")
    ).sort("dse")
    top3 = per_day.head(3)["day"].to_list()
    keep = ~jul["day"].is_in(top3).to_numpy()
    pt3, _ = score(jul.filter(~pl.col("day").is_in(top3)), new[keep], f"minus best 3 days {[str(d) for d in top3]}")

    adopt = pt_e < 0 and pt_o < 0 and pw < 0.05 and pt3 < 0
    print(f"\nrule (both halves < 0, pooled P(worse) < 0.05, survives dropping best 3 days): "
          f"{'ADOPT' if adopt else 'REJECT'}")

    e_avg = s[0] * jul["e_lgb"] + s[1] * jul["e_cat"]
    t = jul.with_columns(new=pl.Series(new), e_band=pl.when(e_avg < 0.1).then(pl.lit("<0.1"))
                         .when(e_avg < 0.5).then(pl.lit("0.1-0.5")).otherwise(pl.lit(">=0.5"))).filter("elig")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=0, tbl_rows=10):
        print("\nLIRF eligible rows by true is_echo and by echo-probability band:")
        for by in ("is_echo", "e_band"):
            print(t.group_by(by).agg(pl.len().alias("n"),
                  ((pl.col("base") - pl.col("taxi")).pow(2).mean().sqrt()).alias("base"),
                  ((pl.col("new") - pl.col("taxi")).pow(2).mean().sqrt()).alias("new"),
                  ((pl.col("adsb_taxi") - pl.col("taxi")).pow(2).mean().sqrt()).alias("adsb_raw")).sort(by))


if __name__ == "__main__":
    main()
