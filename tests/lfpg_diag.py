import polars as pl
import numpy as np

pl.Config.set_tbl_rows(30)
pl.Config.set_ascii_tables(True)

ev = pl.read_parquet("cache/lirf_ceilfix_mixed_ev.parquet")
feats = pl.read_parquet("cache/features/holdout_gap2025.parquet")

lfpg = ev.filter(pl.col("ADEP_mvt") == "LFPG").join(
    feats.select(
        "MVT_ID_mvt", "T", "SOBT", "hour", "dow", "month", "n_dep_30m_prev",
        "n_dep_60m_prev", "n_deprwy_30m_prev", "prev_gap", "sat_run",
        "dep_pressure", "dep_rwy_config", "mins_since_cfg_change",
        "sched_ground", "actual_ground", "inbound_arr_delay", "aobt3_taxi",
        "wake_match",
    ),
    on="MVT_ID_mvt", how="left",
)
print(f"LFPG rows: {lfpg.height}")

def rmse(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    return float(np.sqrt(np.mean((p - t) ** 2)))

e = lfpg["pred"] - lfpg["taxi"]
print(f"\noverall RMSE={rmse(lfpg['pred'], lfpg['taxi']):.1f}  bias={float(e.mean()):.1f}  mae={float(e.abs().mean()):.1f}")

print("\n--- CEIL binding check (is raw ever clipped?) ---")
print(lfpg.select(
    pl.col("taxi_model_raw").min().alias("raw_min"),
    pl.col("taxi_model_raw").max().alias("raw_max"),
    (pl.col("taxi_model_raw") > 10800).sum().alias("n_raw_over_10800"),
    (pl.col("taxi_model_raw") > 7200).sum().alias("n_raw_over_7200"),
))

print("\n--- echo_pred lane ---")
print(lfpg.with_columns(sq=e.pow(2)).group_by("echo_pred").agg(
    pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n"),
    (pl.col("sq").sum() / e.pow(2).sum()).alias("pct_sse"),
).sort("echo_pred"))

print("\n--- is_echo (true label) lane ---")
print(lfpg.with_columns(sq=e.pow(2)).group_by("is_echo").agg(
    pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n"),
).sort("is_echo"))

print("\n--- has_aobt3 lane ---")
print(lfpg.with_columns(sq=e.pow(2)).group_by("has_aobt3").agg(
    pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n"),
    (pl.col("sq").sum() / e.pow(2).sum()).alias("pct_sse"),
).sort("has_aobt3"))

print("\n--- use_prior lane ---")
print(lfpg.with_columns(sq=e.pow(2)).group_by("use_prior").agg(
    pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n"),
).sort("use_prior"))

print("\n--- by true-taxi decile ---")
lfpg2 = lfpg.with_columns(decile=(pl.col("taxi").rank() / lfpg.height * 10).floor().clip(0, 9).cast(pl.Int32))
print(lfpg2.with_columns(sq=e.pow(2), err=e).group_by("decile").agg(
    pl.col("taxi").min().alias("taxi_lo"), pl.col("taxi").max().alias("taxi_hi"),
    pl.col("sq").mean().sqrt().alias("rmse"), pl.col("err").mean().alias("bias"),
    pl.len().alias("n"),
).sort("decile"))

print("\n--- top 15 worst rows by squared error ---")
top = lfpg.with_columns(sq=e.pow(2), err=e).sort("sq", descending=True).head(15)
print(top.select("MVT_ID_mvt", "taxi", "pred", "err", "sched_takeoff_offset", "has_aobt3",
                  "echo_pred", "echo_prob", "use_prior", "dep_rwy_config", "prev_gap", "sat_run", "hour"))

print("\n--- error budget: top-10 worst rows' share of total SSE ---")
sse_total = float(e.pow(2).sum())
sse_top10 = float(top.head(10)["sq"].sum())
print(f"top10 SSE = {sse_top10:.0f} / total {sse_total:.0f} = {sse_top10/sse_total*100:.1f}%")

print("\n--- by month (both must be checked) ---")
print(lfpg.with_columns(sq=e.pow(2)).group_by("ym").agg(
    pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n"),
).sort("ym"))

print("\n--- runway config ---")
print(lfpg.with_columns(sq=e.pow(2)).group_by("dep_rwy_config").agg(
    pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n"),
).sort("n", descending=True).head(10))

print("\n--- by hour ---")
print(lfpg.with_columns(sq=e.pow(2)).group_by("hour").agg(
    pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n"),
).sort("hour"))
