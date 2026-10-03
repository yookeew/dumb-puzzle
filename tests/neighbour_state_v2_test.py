"""Neighbour-state v2 gate (within-day only), per reports/neighbour_state_v2_preregistration.md.

Rows with ADS-B information keep v27's method. Every other row gets s + c~, where c is the
part-(b) neighbour corrector of tests/neighbour_state_test.py and c~ removes its mean over the
applied rows of the same (airport, UTC day), so the stage cannot move an airport's level.

Run:  .venv/Scripts/python.exe tests/neighbour_state_v2_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_12m_test as T  # noqa: E402
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
import neighbour_state_test as N  # noqa: E402

JAN, JUL = A.JAN, A.JUL


def day_demean(c: np.ndarray, airport: np.ndarray, day: np.ndarray, app: np.ndarray) -> np.ndarray:
    """c minus its mean over the applied rows of the same (airport, day); non-applied rows untouched."""
    out = c.copy()
    g = pl.DataFrame({"a": airport[app], "d": day[app], "c": c[app]})
    out[app] = c[app] - g.select(pl.col("c").mean().over("a", "d")).to_series().to_numpy()
    return out


def main() -> None:
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    ex, lag = N.train_extra()
    base, _ = T.combiner_12m(d4, ex, v23, use_holdout=True)          # v27's method

    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    airports = sorted(d4["ADEP_mvt"].unique().to_list())
    nb_ex = N.nb_frame(ex, ex["s"].to_numpy(), lag, arr)
    ex_ok = (ex["taxi"] <= C.TRIM).to_numpy()
    yex = (ex["taxi"] - ex["s"]).to_numpy()
    iex = (ex["day"].dt.day() % 5 == 0).to_numpy()
    Xb_ex = N.x_b(ex, ex["s"].to_numpy(), nb_ex, airports)

    pop = d4.select(C2.has_adsb()).to_series().to_numpy()
    apt, day = d4["ADEP_mvt"].to_numpy(), d4["day"].to_numpy()
    new, raw = base.copy(), base.copy()
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = C.stack(d4, fm)
        Xb = N.x_b(d4, s, N.nb_frame(d4, s, lag, arr), airports)
        y = d4["taxi"].to_numpy() - s
        ih = (d4["day"].dt.day() % 5 == 0).to_numpy()
        fr = ((d4["ym"] == fm) & (d4["taxi"] <= C.TRIM)).to_numpy()
        mb, bb = N.fit(np.vstack([Xb_ex[ex_ok], Xb[fr]]), np.concatenate([yex[ex_ok], y[fr]]),
                       np.concatenate([iex[ex_ok], ih[fr]]), N.CAT_B)
        app = (d4["ym"] == am).to_numpy() & ~pop
        c = np.zeros(d4.height)
        c[app] = mb.predict(Xb[app])
        ct = day_demean(c, apt, day, app)
        raw[app] = np.clip(s[app] + c[app], 0, N.CEIL)
        new[app] = np.clip(s[app] + ct[app], 0, N.CEIL)
        print(f"fit {fm}: best_iter {bb}; applied to {am}: {app.sum():,} rows; correction RMS raw "
              f"{np.sqrt(np.mean(c[app] ** 2)):.1f}, demeaned {np.sqrt(np.mean(ct[app] ** 2)):.1f}", flush=True)
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=12, float_precision=1):
            print(pl.DataFrame({"a": apt[app], "c": c[app], "ct": ct[app]}).group_by("a").agg(
                pl.len(), mean_raw=pl.col("c").mean(), rms_raw=pl.col("c").pow(2).mean().sqrt(),
                rms_demeaned=pl.col("ct").pow(2).mean().sqrt()).sort("a"))

    V.evaluate(d4, base, new, "PRIMARY: neighbour state v2 (within-day, (b) only) vs v27 method", decisive=True)
    V.evaluate(d4, base, raw, "REPORTED: (b) only, not demeaned")


if __name__ == "__main__":
    main()
