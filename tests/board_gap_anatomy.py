"""Where could the leaderboard gap be? (PROGRESS.md §54)

v21 cross-fit on the Jan+Jul 2025 holdout (tests/ltfm_egll_anatomy.py::v21_pred):
how concentrated the squared error is in the top rows, the like-for-like
comparison with arnavhm13's reported holdout numbers, and the LIRF delayed-
departure cells (echo rate, prediction vs offset) that motivated
reports/lirf_hedge_preregistration.md. Also: what the board gap equals in rows.

Run:  .venv/Scripts/python.exe tests/board_gap_anatomy.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, polars as pl
import ltfm_egll_anatomy as A

US, LEAD = 270.2, 224.0
n_rk = pl.read_parquet(ROOT / "data" / "ranking" / "submitting.parquet").height
gap = (US ** 2 - LEAD ** 2) * n_rk
print(f"ranking rows {n_rk:,}; SSE gap {gap:.3e} = one row off by {np.sqrt(gap):,.0f} s, "
      f"10 rows by {np.sqrt(gap / 10):,.0f} s, 100 rows by {np.sqrt(gap / 100):,.0f} s; "
      f"uniform = -{100 * (1 - LEAD ** 2 / US ** 2):.0f}% MSE")
for e in (10000, 20000, 40000, 80000):
    print(f"  fixing one row off by {e:>6} s: {US - np.sqrt(US ** 2 - e ** 2 / n_rk):5.2f} points")

df = A.load(); df = df.with_columns(pred=pl.Series(A.v21_pred(df))).with_columns(r=pl.col("taxi") - pl.col("pred"))
tot = float((df["r"] ** 2).sum()); n = df.height
print(f"\nholdout full RMSE {np.sqrt(tot / n):.2f}")
se = np.sort((df["r"] ** 2).to_numpy())[::-1]
for k in (1, 5, 10, 30, 100, 1000):
    print(f"  drop top {k:>4} rows: RMSE {np.sqrt(se[k:].sum() / n):7.2f}  (they carry {se[:k].sum() / tot:.1%} of SSE)")
print("\nlike-for-like with arnavhm13 (Jan 357 / Jul 333; Jan ~218 with labels > 1 h removed):")
for ym in (A.JAN, A.JUL):
    s = df.filter(pl.col("ym") == ym); s1 = s.filter(pl.col("taxi") <= 3600)
    print(f"  {ym}: full {np.sqrt((s['r'] ** 2).mean()):.1f}   labels<=1h {np.sqrt((s1['r'] ** 2).mean()):.1f}")
L = df.filter(pl.col("ADEP_mvt") == "LIRF").with_columns(
    echo=(pl.col("taxi") - pl.col("offset")).abs() <= 600,
    ob=pl.col("offset").cut([3600, 3 * 3600, 6 * 3600, 12 * 3600], labels=["<1h", "1-3h", "3-6h", "6-12h", "12h+"]))
with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20, float_precision=3):
    print("\nLIRF by delay band x NM-unmatched:")
    print(L.group_by("ob", "nmu").agg(pl.len(), pl.col("echo").mean().alias("p_echo"),
          pl.col("offset").median().alias("off_p50"), pl.col("pred").median().alias("pred_p50"),
          (pl.col("r").pow(2).sum() / tot).alias("sse_share")).sort("ob", "nmu"))
