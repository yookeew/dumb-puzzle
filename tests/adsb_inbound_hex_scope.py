"""Scoping: pushback via the inbound arrival's ADS-B identity (hex). PROGRESS.md §61.
Non-holdout days 2025-09-15 / 11-15 only. Result: no gain (see §61)."""
import os,sys; os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR","load_as_storage")
from pathlib import Path as _P; sys.path.insert(0,str(_P(__file__).resolve().parents[1])); os.chdir(_P(__file__).resolve().parents[1])
import datetime as dt, numpy as np, polars as pl
from src.ingest.normalise_adsb import load_day
from src.link.adsb_pushback import to_xy, GAP_S, SURFACE_GS, R_STAND
from src.link.stand_link import build_stand_links
APS=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LSZH"]
def movements(days):
    srcs=sorted({f"data/raw/training_{d:%Y-%m}-01_*.parquet" for d in days})
    import glob
    files=[f for s in srcs for f in glob.glob(s)]
    m=pl.concat([pl.read_parquet(f) for f in files],how="diagonal_relaxed")
    lo=dt.datetime.combine(min(days),dt.time()); hi=dt.datetime.combine(max(days)+dt.timedelta(1),dt.time())
    m=m.filter(pl.col("MVT_TIME_UTC_mvt").is_between(pl.lit(lo).dt.replace_time_zone("UTC") if m["MVT_TIME_UTC_mvt"].dtype.time_zone else pl.lit(lo), pl.lit(hi).dt.replace_time_zone("UTC") if m["MVT_TIME_UTC_mvt"].dtype.time_zone else pl.lit(hi)))
    return m
def hex_tracks(adsb):
    out={}
    for (ap,h),g in adsb.sort("ts").group_by(["airport","hex"]):
        out[(ap,h)]=(g["ts"].to_numpy(),np.nan_to_num(g["gs"].to_numpy(),nan=SURFACE_GS),g["lat"].to_numpy(),g["lon"].to_numpy())
    return out
for D in (dt.date(2025,9,15),dt.date(2025,11,15)):
    days=[D-dt.timedelta(1),D]
    mv=movements(days)
    lab=mv.filter(pl.col("PHASE_mvt")=="DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64),taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    blind=mv.with_columns(pl.when(pl.col("PHASE_mvt")=="DEP").then(None).otherwise(pl.col("BLOCK_TIME_UTC_mvt")).alias("BLOCK_TIME_UTC_mvt"))
    links=build_stand_links(blind).select(pl.col("MVT_ID_mvt").cast(pl.Int64),pl.col("inbound_mvt_id").cast(pl.Int64),"link_confidence")
    arr=mv.filter(pl.col("PHASE_mvt")=="ARR").select(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("inbound_mvt_id"),ap=pl.col("ADES_mvt"),land=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms")/1000)
    dep=mv.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("MVT_TIME_UTC_mvt").dt.date()==D)).select(pl.col("MVT_ID_mvt").cast(pl.Int64),ap=pl.col("ADEP_mvt"),T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms")/1000)
    dep=dep.join(links,on="MVT_ID_mvt",how="left").join(arr,on=["inbound_mvt_id","ap"],how="left").join(lab,on="MVT_ID_mvt",how="left")
    v4=pl.read_parquet(f"cache/adsb_pushback_v4/day={D}.parquet").select("MVT_ID_mvt","adsb_tier")
    dep=dep.join(v4,on="MVT_ID_mvt",how="left").filter(pl.col("ap").is_in(APS))
    import pathlib
    adsb=pl.concat([load_day(d) for d in days if pathlib.Path(f"data/external/adsb/day={d}").exists()]); tr=hex_tracks(adsb)
    # landing events per airport
    land_ev={}
    for (ap,h),(ts,gs,la,lo) in tr.items():
        k=np.flatnonzero((gs[:-1]>=SURFACE_GS)&(gs[1:]<SURFACE_GS)&(np.diff(ts)<=120))+1
        for i in k: land_ev.setdefault(ap,[]).append((ts[i],h,i))
    rows=[]
    for ap in APS:
        ev=sorted(land_ev.get(ap,[])); 
        if not ev: continue
        et=np.array([e[0] for e in ev]); used=set()
        d=dep.filter((pl.col("ap")==ap)&pl.col("land").is_not_null()).sort("land")
        lat0=float(np.nanmean(adsb.filter(pl.col("airport")==ap)["lat"].to_numpy()))
        for r in d.iter_rows(named=True):
            lo_,hi_=np.searchsorted(et,[r["land"]-180,r["land"]+180])
            cand=[(abs(et[k]-r["land"]),k) for k in range(lo_,hi_) if k not in used]
            res=dict(MVT_ID_mvt=r["MVT_ID_mvt"],ap=ap,taxi=r["taxi"],tier=r["adsb_tier"],conf=r["link_confidence"],hex_found=False,confirmed=False,pb=None,kind=None)
            if cand:
                _,k=min(cand); used.add(k); t0,h,i=ev[k]
                ts,gs,la,lo=tr[(ap,h)]; xy=to_xy(la,lo,lat0)
                # arrival run: from i until gap > GAP_S
                j=i
                while j+1<len(ts) and ts[j+1]-ts[j]<=GAP_S and ts[j+1]<r["T"]: j+=1
                stat=[q for q in range(i,j+1) if gs[q]<=1]
                res["hex_found"]=True
                to=np.flatnonzero((gs[:-1]<SURFACE_GS)&(gs[1:]>=SURFACE_GS)&(np.diff(ts)<=120))+1
                res["confirmed"]=bool(np.any(np.abs(ts[to]-r["T"])<=180)) if len(to) else False
                if stat:
                    park=xy[stat[-1]]
                    after=np.flatnonzero((ts>ts[stat[-1]])&(ts<=r["T"]))
                    if len(after):
                        dpk=np.hypot(*(xy[after]-park).T)
                        near=after[dpk<R_STAND]
                        if len(near):
                            k=near[-1]
                            if k+1<len(ts) and ts[k+1]<=r["T"] and gs[k]<=1:
                                res.update(pb=float(ts[k+1]),kind="last_at_park_stat")
                            else:
                                res.update(pb=float(ts[k]),kind="last_at_park_moving")
                        else:
                            res.update(kind="first_away")
                    else: res.update(kind="no_dep_samples")
                else: res.update(kind="no_park")
            rows.append(res)
    R=pl.DataFrame(rows,infer_schema_length=None)
    R=R.join(dep.select("MVT_ID_mvt","T"),on="MVT_ID_mvt").with_columns(err=pl.col("T")-pl.col("pb")-pl.col("taxi"),
        newinfo=~pl.col("tier").is_in(["appear","dwell"]).fill_null(False))
    print(f"\n{D}: departures with linked inbound {R.height:,}; inbound hex found {R['hex_found'].mean():.2f}")
    pl.Config.set_tbl_rows(30); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_float_precision(2)
    print(R.group_by("kind").agg(pl.len(),pl.col("newinfo").mean().alias("not_tiered_now"),(pl.col("err").abs()<=120).mean().alias("w120"),(pl.col("err").abs()>600).mean().alias("bad600"),pl.col("err").median().alias("err_p50")).sort("kind"))
    print("confirmed identity (same hex takes off within 180 s of T):", R["confirmed"].sum())
    print(R.filter("confirmed").group_by("kind").agg(pl.len(),pl.col("newinfo").mean().alias("not_tiered_now"),(pl.col("err").abs()<=120).mean().alias("w120"),(pl.col("err").abs()>600).mean().alias("bad600"),pl.col("err").median().alias("err_p50")).sort("kind"))
    p=R.filter(pl.col("pb").is_not_null()&pl.col("confirmed"))
    print("pushback found, by airport (new = not appear/dwell in v4):")
    print(p.group_by("ap").agg(pl.len(),pl.col("newinfo").sum().alias("new"),(pl.col("err").abs()<=120).mean().alias("w120"),(pl.col("err").abs()>600).mean().alias("bad600")).sort("ap"))
    nn=p.filter("newinfo"); print(f"NEW pushbacks: {nn.height} of {dep.height} departures ({nn.height/dep.height:.1%}); within120 {(nn['err'].abs()<=120).mean():.2f}, bad600 {(nn['err'].abs()>600).mean():.2f}")
