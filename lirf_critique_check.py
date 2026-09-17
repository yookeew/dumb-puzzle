import polars as pl
import numpy as np

ev = pl.read_parquet("cache/lirf_diag_ev.parquet")
lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF")
e = pl.col("pred").cast(float) - pl.col("taxi").cast(float)

print("=== 0. Identity check: offset == taxi + d ===")
chk = lirf.select((pl.col("sched_takeoff_offset") - (pl.col("taxi") + pl.col("d"))).abs().max().alias("max_dev")).item()
print(f"max |offset - (taxi+d)| = {chk}")
print()

print("=== 1. Bias binned by PREDICTION decile (not truth) ===")
decp = lirf.with_columns(dq=(pl.col("pred").rank("ordinal") - 1) * 10 // pl.len())
g = (decp.group_by("dq").agg(
        pl.col("pred").mean().alias("mean_pred"),
        pl.col("taxi").mean().alias("mean_taxi"),
        e.mean().alias("bias"), e.pow(2).mean().sqrt().alias("rmse"),
        pl.len().alias("n")).sort("dq"))
for r in g.iter_rows(named=True):
    print(f"  d{r['dq']}  mean_pred={r['mean_pred']:.0f}  mean_taxi={r['mean_taxi']:.0f}  "
          f"bias={r['bias']:+.1f}  rmse={r['rmse']:.1f}  n={r['n']:,}")
print()

print("=== 2. Bias/corr binned by OFFSET decile (not truth) ===")
deco = lirf.with_columns(oq=(pl.col("sched_takeoff_offset").rank("ordinal") - 1) * 10 // pl.len())
g = (deco.group_by("oq").agg(
        pl.col("sched_takeoff_offset").mean().alias("mean_offset"),
        pl.col("sched_takeoff_offset").min().alias("lo"),
        pl.col("sched_takeoff_offset").max().alias("hi"),
        pl.col("taxi").mean().alias("mean_taxi"),
        pl.col("d").mean().alias("mean_d"),
        pl.col("d").std().alias("std_d"),
        pl.col("taxi").std().alias("std_taxi"),
        pl.corr("sched_takeoff_offset", "taxi").alias("corr_offset_taxi"),
        e.mean().alias("bias"), e.pow(2).mean().sqrt().alias("rmse"),
        pl.len().alias("n")).sort("oq"))
for r in g.iter_rows(named=True):
    print(f"  o{r['oq']}  offset[{r['lo']:.0f}-{r['hi']:.0f}] mean_taxi={r['mean_taxi']:.0f}  "
          f"mean_d={r['mean_d']:.0f}  std_d={r['std_d']:.0f}  std_taxi={r['std_taxi']:.0f}  "
          f"corr={r['corr_offset_taxi']:.3f}  bias={r['bias']:+.1f}  rmse={r['rmse']:.1f}  n={r['n']:,}")
print()

print("=== 3. Error budget: bias^2 share of MSE, per true-taxi decile ===")
dect = lirf.with_columns(dq=(pl.col("taxi").rank("ordinal") - 1) * 10 // pl.len())
g = (dect.group_by("dq").agg(
        e.pow(2).mean().alias("mse"), e.mean().pow(2).alias("bias2"),
        pl.len().alias("n")).sort("dq")
     .with_columns((pl.col("mse") * pl.col("n")).alias("sq_total"),
                   (pl.col("bias2") * pl.col("n")).alias("bias2_total")))
total_mse = g["sq_total"].sum()
total_bias2 = g["bias2_total"].sum()
print(f"total MSE={total_mse:.0f}  total bias^2 contribution={total_bias2:.0f}  "
      f"({100*total_bias2/total_mse:.1f}% of MSE)")
print(f"implied RMSE if all bias removed (upper bound, unreachable): "
      f"{((total_mse - total_bias2)/lirf.height)**0.5:.1f}")
for r in g.iter_rows(named=True):
    print(f"  d{r['dq']}  mse={r['mse']:.0f}  bias2={r['bias2']:.0f}  "
          f"%mse_from_bias={100*r['bias2']/r['mse']:.1f}%  n={r['n']:,}  "
          f"%_of_total_mse={100*r['sq_total']/total_mse:.1f}%")
print()

print("=== 4. Tail concentration: top-N rows' share of LIRF MSE ===")
lirf_e2 = lirf.with_columns(sq=e.pow(2))
total_sq = lirf_e2["sq"].sum()
srt = lirf_e2.sort("sq", descending=True)
for n in (1, 5, 10, 50, 100, 500):
    top_sq = srt.head(n)["sq"].sum()
    print(f"  top {n:>4} rows: {100*top_sq/total_sq:.1f}% of LIRF total squared error")
print()

print("=== 5. Extreme rows detail (top 10 by squared error) ===")
top10 = srt.head(10).select("MVT_ID_mvt", "taxi", "d", "sched_takeoff_offset", "pred",
                             "echo_prob", "is_echo", "has_aobt3")
for r in top10.iter_rows(named=True):
    print(f"  taxi={r['taxi']:.0f}  d={r['d']:.0f}  offset={r['sched_takeoff_offset']:.0f}  "
          f"pred={r['pred']:.0f}  echo_prob={r['echo_prob']:.3f}  is_echo={r['is_echo']}  "
          f"has_aobt3={r['has_aobt3']}")
print()

print("=== 6. Training filter reconciliation: rows with taxi outside [30,7200] ===")
print(f"LIRF rows with taxi > 7200s (excluded from main-regressor training): "
      f"{(lirf['taxi'] > 7200).sum():,} / {lirf.height:,} "
      f"({100*(lirf['taxi'] > 7200).sum()/lirf.height:.1f}%)")
print(f"LIRF rows with taxi > 14400s: {(lirf['taxi'] > 14400).sum():,}")
print(f"LIRF rows with taxi > 36*3600 (36h): {(lirf['taxi'] > 36*3600).sum():,}")
print(f"LIRF max taxi in holdout: {lirf['taxi'].max():,}")
print(f"LIRF max pred in holdout: {lirf['pred'].max():,.0f}")
