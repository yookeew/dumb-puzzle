import polars as pl
import numpy as np

ev = pl.read_parquet("cache/lirf_diag_ev.parquet")
lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF")

# median d for LIRF -- use the WHOLE LIRF holdout population as a stand-in
# "typical" d (directional check only, not a fitted OOF prior)
median_d_lirf = lirf["d"].median()
print(f"median d (LIRF, holdout pop): {median_d_lirf:.1f}")
print()

def rmse(pred, truth):
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.sqrt(np.mean((p - t) ** 2)))

def report(df, label):
    taxi = df["taxi"].to_numpy()
    offset = df["sched_takeoff_offset"].to_numpy()
    pred_model = df["pred"].to_numpy()
    naive_offset = offset  # d_hat = 0
    naive_median = offset - median_d_lirf
    print(f"--- {label} (n={df.height}) ---")
    print(f"  1. current model (full pipeline):     rmse={rmse(pred_model, taxi):.0f}")
    print(f"  2. naive taxi=offset (d_hat=0):        rmse={rmse(naive_offset, taxi):.0f}")
    print(f"  3. naive taxi=offset-median(d):        rmse={rmse(naive_median, taxi):.0f}")
    print(f"  mean true taxi: {taxi.mean():.0f}  mean offset: {offset.mean():.0f}  "
          f"mean d: {df['d'].mean():.0f}  std d: {df['d'].std():.0f}")
    print()

print("=== Top 50 LIRF holdout rows by OFFSET ===")
top50_offset = lirf.sort("sched_takeoff_offset", descending=True).head(50)
report(top50_offset, "top 50 by offset")

print("=== Top 10 LIRF holdout rows by SQUARED ERROR (current model) ===")
e = (pl.col("pred").cast(float) - pl.col("taxi").cast(float)).pow(2)
top10_err = lirf.with_columns(sq=e).sort("sq", descending=True).head(10)
report(top10_err, "top 10 by error")

print("=== All LIRF holdout rows (sanity: whole-population comparison) ===")
report(lirf, "all LIRF")

print("=== LIRF rows with offset > 7200s (excluded from training under current LABEL_HI) ===")
big_off = lirf.filter(pl.col("sched_takeoff_offset") > 7200)
report(big_off, "offset>7200")

print("=== LIRF rows with offset <= 7200s (normal population) ===")
small_off = lirf.filter(pl.col("sched_takeoff_offset") <= 7200)
report(small_off, "offset<=7200")
