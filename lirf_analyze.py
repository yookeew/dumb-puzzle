import polars as pl

ev = pl.read_parquet("cache/lirf_diag_ev.parquet")
lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF")
e = pl.col("pred").cast(float) - pl.col("taxi").cast(float)

print(f"LIRF n={lirf.height:,}  overall rmse={lirf.select(e.pow(2).mean().sqrt()).item():.1f}")
print()

print("=== LIRF by echo_pred (classifier verdict, P>0.5) ===")
g = (lirf.group_by("echo_pred").agg(
        e.pow(2).mean().sqrt().alias("rmse"), e.pow(2).mean().alias("sqerr"),
        pl.len().alias("n"))
     .with_columns((pl.col("sqerr") * pl.col("n")).alias("sq_total")))
total_sq = g["sq_total"].sum()
for r in g.sort("echo_pred", descending=True).iter_rows(named=True):
    print(f"  echo_pred={r['echo_pred']}  rmse={r['rmse']:.1f}  n={r['n']:,}  "
          f"%sqerr={100*r['sq_total']/total_sq:.1f}%")
print()

print("=== LIRF by true is_echo (ground truth) ===")
g = (lirf.group_by("is_echo").agg(
        e.pow(2).mean().sqrt().alias("rmse"), pl.len().alias("n")))
for r in g.sort("is_echo", descending=True).iter_rows(named=True):
    print(f"  is_echo={r['is_echo']}  rmse={r['rmse']:.1f}  n={r['n']:,}")
print()

print("=== Comparison: 'echo_pred=False' lane, LIRF vs rest ===")
g2 = (ev.filter(~pl.col("echo_pred")).group_by(pl.col("ADEP_mvt") == "LIRF").agg(
        e.pow(2).mean().sqrt().alias("rmse"), pl.len().alias("n")))
for r in g2.iter_rows(named=True):
    label = "LIRF" if r["ADEP_mvt"] else "rest"
    print(f"  {label}  rmse={r['rmse']:.1f}  n={r['n']:,}")
print()

print("=== Calibration: predicted echo_prob decile vs actual echo rate ===")
print("-- global --")
dec = ev.with_columns(
    qb=(pl.col("echo_prob").rank("ordinal") - 1) * 10 // pl.len())
g = (dec.group_by("qb").agg(
        pl.col("echo_prob").mean().alias("mean_pred"),
        pl.col("is_echo").mean().alias("actual_rate"),
        pl.len().alias("n")).sort("qb"))
for r in g.iter_rows(named=True):
    print(f"  bin{r['qb']}  mean_pred={r['mean_pred']:.3f}  actual_rate={r['actual_rate']:.3f}  n={r['n']:,}")

print("-- LIRF only --")
dec2 = lirf.with_columns(
    qb=(pl.col("echo_prob").rank("ordinal") - 1) * 10 // pl.len())
g2 = (dec2.group_by("qb").agg(
        pl.col("echo_prob").mean().alias("mean_pred"),
        pl.col("is_echo").mean().alias("actual_rate"),
        pl.len().alias("n")).sort("qb"))
for r in g2.iter_rows(named=True):
    print(f"  bin{r['qb']}  mean_pred={r['mean_pred']:.3f}  actual_rate={r['actual_rate']:.3f}  n={r['n']:,}")
print()

print("=== LIRF true-taxi deciles (where does the error concentrate) ===")
dec3 = lirf.with_columns(dq=(pl.col("taxi").rank("ordinal") - 1) * 10 // pl.len())
g3 = (dec3.group_by("dq").agg(
        e.pow(2).mean().sqrt().alias("rmse"), e.mean().alias("bias"),
        pl.col("taxi").min().alias("lo"), pl.col("taxi").max().alias("hi"),
        pl.col("is_echo").mean().alias("echo_rate"),
        pl.len().alias("n")).sort("dq"))
for r in g3.iter_rows(named=True):
    print(f"  d{r['dq']}  taxi[{r['lo']}-{r['hi']}]  rmse={r['rmse']:.1f}  bias={r['bias']:+.1f}  "
          f"echo_rate={r['echo_rate']:.2f}  n={r['n']:,}")
