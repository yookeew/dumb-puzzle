"""Scoping check for the ADS-B far-sighting idea (PROGRESS.md §50 idea, result §52).

Population: ADS-B-matched departures with no pushback tier, first seen >= 1 km
from the own stand, LIRF excluded (the complement of §41's partial stage).
Uses only the two non-holdout 2025 ADS-B days (09-15, 11-15), with an honest
base = mean of the month-wise OOF LightGBM and CatBoost predictions.
No holdout rows are read.

Run:  .venv/Scripts/python.exe tests/adsb_far_scope.py
"""
import os; os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR","load_as_storage")
import numpy as np, polars as pl
from pathlib import Path
R=str(Path(__file__).resolve().parents[1])+"/"
det=pl.concat([pl.read_parquet(R+f"cache/adsb_pushback/day={d}.parquet") for d in ("2025-09-15","2025-11-15")])
oof=None
for e in ("lgb","cat"):
    s=pl.concat([pl.read_parquet(R+f"cache/oof/{e}_mixed/fold={m}.parquet") for m in ("2025-09","2025-11")]).select(pl.col("MVT_ID_mvt").cast(pl.Int64),pl.col("pred").alias(e))
    oof=s if oof is None else oof.join(s,on="MVT_ID_mvt")
mvt=(pl.scan_parquet(R+"data/raw/training_*.parquet").filter(pl.col("PHASE_mvt")=="DEP")
     .select(pl.col("MVT_ID_mvt").cast(pl.Int64),"ADEP_mvt",taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
             mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms")/1000.0).collect())
df=det.join(mvt,on="MVT_ID_mvt").join(oof,on="MVT_ID_mvt").with_columns(
    base=(pl.col("lgb")+pl.col("cat"))/2, L=pl.col("mvt_ts")-pl.col("adsb_first_ts"), dkm=pl.col("adsb_first_own_m")/1000)
df=df.with_columns(unseen=pl.col("taxi")-pl.col("L"), r=pl.col("taxi")-pl.col("base"))
print("rows",df.height,"base rmse",np.sqrt((df["r"]**2).mean()))
far=df.filter(pl.col("adsb_matched").fill_null(False) & ~pl.col("adsb_tier").is_in(["appear","dwell"]).fill_null(False)
              & (pl.col("adsb_first_own_m")>=1000).fill_null(False) & (pl.col("ADEP_mvt")!="LIRF"))
print("far rows",far.height, "  of which gs>=5:",far.filter(pl.col("adsb_first_gs")>=5).height)
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_float_precision(1)
far=far.with_columns(dbin=pl.col("dkm").cut([1.5,2,3,4,6],labels=["1-1.5","1.5-2","2-3","3-4","4-6","6+"]))
q=lambda c:[pl.col(c).quantile(x).alias(f"{c}_p{int(x*100)}") for x in (0.1,0.5,0.9)]
print(far.group_by("ADEP_mvt").agg(pl.len(),*q("unseen"),(pl.col("taxi")<pl.col("L")-30).mean().alias("viol"),
      (pl.col("base")<pl.col("L")).mean().alias("base_below_L"),pl.col("taxi").std().alias("taxi_sd")).sort("ADEP_mvt"))
print(far.group_by("dbin").agg(pl.len(),*q("unseen"),(pl.col("taxi")<pl.col("L")-30).mean().alias("viol")).sort("dbin"))
# in-sample: OLS est, LS weight, gains
def gains(f,label):
    y=np.clip(f["unseen"].to_numpy(),*np.percentile(f["unseen"].to_numpy(),[1,99]))
    ap=f["ADEP_mvt"].to_numpy(); aps=sorted(set(ap))
    X=np.column_stack([(ap==a).astype(float) for a in aps]+[f["dkm"].to_numpy()])
    c,*_=np.linalg.lstsq(X,y,rcond=None); est=X@c+f["L"].to_numpy()
    b=f["base"].to_numpy(); t=f["taxi"].to_numpy(); r=t-b; dl=est-b
    w=np.clip((r*dl).sum()/(dl**2).sum(),0,1)
    rm=lambda p:np.sqrt(np.mean((p-t)**2))
    print(f"{label}: n={len(t)} b={c[-1]:.0f}s/km w={w:.3f}  base={rm(b):.1f} est={rm(est):.1f} blend={rm(b+w*dl):.1f}"
          f"  clipL={rm(np.maximum(b,f['L'].to_numpy())):.1f}  SSE share of all rows={((r**2).sum()/(df['r']**2).sum()):.3f}"
          f"  corr(r,dl)={np.corrcoef(r,dl)[0,1]:.3f}")
gains(far,"far all")
gains(far.filter(pl.col("adsb_first_gs")>=5),"far moving")
gains(far.filter(pl.col("ADEP_mvt")!="EHAM"),"far noEHAM")
for a in sorted(set(far["ADEP_mvt"])):
    s=far.filter(pl.col("ADEP_mvt")==a)
    if s.height>=50: gains(s,"  "+a)
print("\n--- cross-day: can residual r be predicted from track info? ---")
import lightgbm as lgb
far=far.with_columns(day=pl.from_epoch(pl.col("mvt_ts").cast(pl.Int64),"s").dt.date(),
                     Lrel=pl.col("L")-pl.col("base"))
cols=["L","dkm","adsb_first_gs","Lrel","base","adsb_n_pts","adsb_min_own_m"]
far=far.with_columns(apc=pl.col("ADEP_mvt").cast(pl.Categorical).to_physical())
days=sorted(set(far["day"]))
tot_b=tot_n=0
for tr,te in ((days[0],days[1]),(days[1],days[0])):
    a=far.filter(pl.col("day")==tr); b=far.filter(pl.col("day")==te)
    m=lgb.train(dict(objective="huber",alpha=300,num_leaves=7,min_data_in_leaf=40,learning_rate=0.05,verbose=-1,seed=0,deterministic=True,force_row_wise=True,num_threads=1),
                lgb.Dataset(a.select(cols+["apc"]).to_numpy(),a["r"].to_numpy()),200)
    p=m.predict(b.select(cols+["apc"]).to_numpy())
    rb=b["r"].to_numpy()
    for s in (1.0,0.5):
        print(f"fit {tr} -> {te}: n={b.height} base={np.sqrt(np.mean(rb**2)):.1f}  +{s}*gbm={np.sqrt(np.mean((rb-s*p)**2)):.1f}")
    # linear per-day check
    X=np.column_stack([np.ones(a.height),a["Lrel"].to_numpy()]); c,*_=np.linalg.lstsq(X,a["r"].to_numpy(),rcond=None)
    pl_=c[0]+c[1]*b["Lrel"].to_numpy()
    print(f"   linear on Lrel: coef={c.round(3)}  rmse={np.sqrt(np.mean((rb-pl_)**2)):.1f}")
