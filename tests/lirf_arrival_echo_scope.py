"""Scoping: does LIRF arrival echo (same hour / same operator same day) predict departure echo?
Non-holdout training months only. PROGRESS.md §63."""
import os; os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR","load_as_storage")
import numpy as np, polars as pl
m=pl.read_parquet("data/raw/training_*.parquet").filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1,7]))
L=m.filter(((pl.col("ADEP_mvt")=="LIRF")&(pl.col("PHASE_mvt")=="DEP"))|((pl.col("ADES_mvt")=="LIRF")&(pl.col("PHASE_mvt")=="ARR")))
L=L.with_columns(d=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                 off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                 t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), op=pl.col("AIRCRAFT_OPERATOR_flt"))
L=L.with_columns(echo=pl.col("d").abs()<30)
dep=L.filter(pl.col("PHASE_mvt")=="DEP").sort("t"); arr=L.filter((pl.col("PHASE_mvt")=="ARR")&pl.col("d").is_not_null()).sort("t")
print("LIRF arrivals echo rate:",round(arr["echo"].mean(),3)," departures:",round(dep["echo"].mean(),3))
at=arr["t"].to_numpy(); ae=arr["echo"].to_numpy().astype(float); cs=np.r_[0,np.cumsum(ae)]
def win_rate(t,W):
    lo=np.searchsorted(at,t-W); hi=np.searchsorted(at,t+W); n=hi-lo
    return np.where(n>0,(cs[hi]-cs[lo])/np.maximum(n,1),np.nan), n
dt_=dep["t"].to_numpy(); de=dep["echo"].to_numpy(); late=(dep["off"].to_numpy()>3600)
for W in (1800,3600,3*3600):
    r,n=win_rate(dt_,W)
    q=np.nanquantile(r,[0.5,0.9,0.99])
    print(f"\nW=±{W//60} min: arrival echo rate around each departure (p50/p90/p99 {q.round(2)})")
    for nm,mask in (("all deps",np.ones_like(de,bool)),(">1h late",late)):
        bins=[-0.01,0.1,0.2,0.3,0.5,1.01]; idx=np.digitize(r,bins)-1
        row=[]
        for b in range(len(bins)-1):
            k=mask&(idx==b)&~np.isnan(r)
            if k.sum(): row.append(f"[{bins[b]:.1f},{bins[b+1]:.1f}): n={k.sum():>6} echo={de[k].mean():.3f}")
        print(f"  {nm:<9} "+" | ".join(row))
# same operator, same day: echo rate of that operator's arrivals that day (excluding nothing; arrivals only)
dd=dep.with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date()); aa=arr.with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date())
g=aa.group_by("op","day").agg(pl.col("echo").mean().alias("op_arr_echo"),pl.len().alias("n_arr"))
j=dd.join(g,on=["op","day"],how="left").filter(pl.col("n_arr")>=3)
for nm,f in (("all deps",pl.lit(True)),(">1h late",pl.col("off")>3600)):
    t=j.filter(f).with_columns(b=pl.col("op_arr_echo").cut([0.1,0.3,0.6]))
    print(f"\nsame operator same day, {nm}:"); print(t.group_by("b").agg(pl.len(),pl.col("echo").mean()).sort("b"))
print("\n--- does same-day arrival echo add beyond the operator's long-run echo rate? ---")
# long-run rate from OTHER months (leave-month-out) to mimic an OOF feature
dd2=dd.with_columns(mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
lr=[]
for mo in sorted(set(dd2["mo"])):
    o=dd2.filter(pl.col("mo")!=mo).group_by("op").agg(pl.col("echo").mean().alias("op_lr"),pl.len().alias("n_lr"))
    lr.append(dd2.filter(pl.col("mo")==mo).join(o,on="op",how="left"))
k=pl.concat(lr).join(g,on=["op","day"],how="left").filter((pl.col("n_arr")>=3)&(pl.col("n_lr")>=50))
k=k.with_columns(lrb=pl.col("op_lr").cut([0.05,0.15,0.3,0.5]),ab=pl.col("op_arr_echo").cut([0.1,0.3,0.6]))
for nm,f in (("all deps",pl.lit(True)),(">1h late",pl.col("off")>3600)):
    t=k.filter(f)
    print(f"\n{nm}: echo rate by (operator long-run echo bin) x (same-day arrival echo bin)")
    print(t.group_by("lrb","ab").agg(pl.len(),pl.col("echo").mean().round(3)).pivot(on="ab",index="lrb",values="echo").sort("lrb"))
    print(t.group_by("lrb","ab").agg(pl.len()).pivot(on="ab",index="lrb",values="len").sort("lrb"))
# how much of LIRF delayed-echo could be flagged: share of late-echo rows with op_arr_echo>0.3
le=k.filter((pl.col("off")>3600)&pl.col("echo"))
print("\nlate echo rows:",le.height," with same-day op arrival echo>0.3:",le.filter(pl.col("op_arr_echo")>0.3).height)
