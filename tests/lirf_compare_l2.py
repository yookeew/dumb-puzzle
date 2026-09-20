import polars as pl

huber = pl.read_parquet("cache/lirf_diag_ev.parquet").filter(pl.col("ADEP_mvt") == "LIRF")
l2 = pl.read_parquet("cache/lirf_diag_l2_ev.parquet").filter(pl.col("ADEP_mvt") == "LIRF")
e = pl.col("pred").cast(float) - pl.col("taxi").cast(float)

def deciles(df, label):
    dec = df.with_columns(dq=(pl.col("taxi").rank("ordinal") - 1) * 10 // pl.len())
    g = (dec.group_by("dq").agg(
            e.pow(2).mean().sqrt().alias("rmse"), e.mean().alias("bias"),
            pl.col("taxi").min().alias("lo"), pl.col("taxi").max().alias("hi"),
            pl.len().alias("n")).sort("dq"))
    print(f"=== {label} ===")
    for r in g.iter_rows(named=True):
        print(f"  d{r['dq']}  taxi[{r['lo']}-{r['hi']}]  rmse={r['rmse']:.1f}  bias={r['bias']:+.1f}  n={r['n']:,}")
    print()
    return g

gh = deciles(huber, "HUBER alpha=800, LIRF")
gl = deciles(l2, "L2, LIRF")

print("=== d9 (tail) comparison ===")
h9 = gh.filter(pl.col("dq") == 9).row(0, named=True)
l9 = gl.filter(pl.col("dq") == 9).row(0, named=True)
print(f"  huber d9: rmse={h9['rmse']:.1f} bias={h9['bias']:+.1f}")
print(f"  l2    d9: rmse={l9['rmse']:.1f} bias={l9['bias']:+.1f}")

print()
print("=== max predictions in LIRF (is huber capping the ceiling of predictions?) ===")
print("huber pred stats:", huber.select(pl.col("pred").min(), pl.col("pred").quantile(0.5),
                                         pl.col("pred").quantile(0.99), pl.col("pred").max()).row(0))
print("l2    pred stats:", l2.select(pl.col("pred").min(), pl.col("pred").quantile(0.5),
                                      pl.col("pred").quantile(0.99), pl.col("pred").max()).row(0))
