"""Follow-up to tests/ltfm_egll_anatomy.py (PROGRESS.md §53): for delayed departures
(T - SOBT > 1 h), how often is the label "pushed on time, then waited" (taxi within
30 min of T - SOBT), by airport and NM match; and does a cross-fit per-cell bias shift
on (delay band, NM-unmatched) help the v21 holdout prediction? LIRF excluded.

Run:  .venv/Scripts/python.exe tests/ltfm_egll_late_mix.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, polars as pl
import ltfm_egll_anatomy as A
df=A.load(); df=df.with_columns(pred=pl.Series(A.v21_pred(df))).with_columns(r=pl.col("taxi")-pl.col("pred"))
tot=float((df["r"]**2).sum())
df=df.filter(pl.col("ADEP_mvt")!="LIRF").with_columns(
    obin=pl.col("offset").cut([1800,3600,5400,7200,10800],labels=["<30m","30-60","60-90","90-120","2-3h","3h+"]),
    kind=pl.when(pl.col("taxi")<=2400).then(pl.lit("normal")).when(pl.col("offset")-pl.col("taxi")<=1800).then(pl.lit("waited")).otherwise(pl.lit("other")))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_float_precision(3); pl.Config.set_tbl_width_chars(220)
print("non-LIRF rows, by offset bin x NM-unmatched: share of ALL-airport SSE, label mix, v21 bias")
print(df.group_by("obin","nmu").agg(pl.len().alias("n"),(pl.col("r").pow(2).sum()/tot).alias("sse_share"),
   (pl.col("kind")=="waited").mean().alias("p_waited"),(pl.col("kind")=="normal").mean().alias("p_normal"),
   pl.col("taxi").mean().alias("taxi_mean"),pl.col("pred").mean().alias("pred_mean"),
   (pl.col("r").pow(2).mean().sqrt()).alias("rmse")).sort("obin","nmu"))
big=df.filter(pl.col("offset")>3600)
print("\noffset>1h, per airport x nmu:")
print(big.group_by("ADEP_mvt","nmu").agg(pl.len().alias("n"),(pl.col("r").pow(2).sum()/tot).alias("sse_share"),
   (pl.col("kind")=="waited").mean().alias("p_waited"),pl.col("taxi").mean().alias("taxi_mean"),pl.col("pred").mean().alias("pred_mean")).sort("ADEP_mvt","nmu"))
print("\nJan vs Jul stability of p_waited, offset>1h:")
print(big.group_by("ym","nmu").agg(pl.len(),(pl.col("kind")=="waited").mean().alias("p_waited")).sort("ym","nmu"))
# cross-fit hedge ceiling: per (obin, nmu) cell, pred' = pred + mean residual from other month
out=df["pred"].to_numpy().copy()
for fm,am in ((A.JAN,A.JUL),(A.JUL,A.JAN)):
    c=df.filter((pl.col("ym")==fm)&(pl.col("offset")>3600)).group_by("obin","nmu").agg(pl.col("r").mean().alias("adj"))
    t=df.with_row_index().filter((pl.col("ym")==am)&(pl.col("offset")>3600)).join(c,on=["obin","nmu"],how="left")
    out[t["index"].to_numpy()]+=t["adj"].fill_null(0).to_numpy()
t=df["taxi"].to_numpy()
for nm,mask in (("all non-LIRF",np.ones(len(t),bool)),("trimmed",t<=18000)):
    print(f"{nm}: v21 {np.sqrt(np.mean((df['pred'].to_numpy()[mask]-t[mask])**2)):.2f} -> cell-bias shift {np.sqrt(np.mean((out[mask]-t[mask])**2)):.2f}")
