import polars as pl
import numpy as np

ev = pl.read_parquet("cache/lirf_diag_ev.parquet")
lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF")

def rmse(pred, truth):
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.sqrt(np.mean((p - t) ** 2)))

print("=== A. Oracle-bound equivalence check ===")
# _reconstruct_taxi for LIRF rows always has use_prior=False (not_lirf_ho excludes
# LIRF), so w_model=(1-echo_prob). Replacing taxi_model with taxi_echo (~=offset,
# since offset << ECHO_WIDE_CEIL=140000 for virtually all rows) in the w_model term
# gives: echo_prob*taxi_echo + (1-echo_prob)*taxi_echo = taxi_echo = offset (clipped).
big = lirf.filter(pl.col("sched_takeoff_offset") > 7200)
taxi_echo = big["sched_takeoff_offset"].clip(0, 140000).to_numpy()
oracle_pred = taxi_echo  # p*taxi_echo + (1-p)*taxi_echo algebraically
naive_offset = big["sched_takeoff_offset"].to_numpy()
print(f"  oracle (1-p branch -> offset) RMSE on offset>7200: {rmse(oracle_pred, big['taxi']):.0f}")
print(f"  naive offset RMSE on offset>7200 (from earlier check): {rmse(naive_offset, big['taxi']):.0f}")
print(f"  current model RMSE on offset>7200 (from earlier check): "
      f"{rmse(big['pred'], big['taxi']):.0f}")
print("  -> confirms oracle == naive offset algebraically (ECHO_WIDE_CEIL rarely binds)")
print()

print("=== B. Conditional calibration: P(echo) within offset>7200 bin ===")
print(f"  n={big.height}")
print(f"  mean predicted echo_prob: {big['echo_prob'].mean():.3f}")
print(f"  actual is_echo rate (|d|<30s): {big['is_echo'].mean():.3f}")
print(f"  actual 'small |d|' rate (|d|<300s, looser): "
      f"{(big['d'].abs() < 300).cast(pl.Float64).mean():.3f}")
print(f"  actual 'small |d|' rate (|d|<600s): {(big['d'].abs() < 600).cast(pl.Float64).mean():.3f}")
# finer calibration within this bin
dec = big.with_columns(qb=(pl.col("echo_prob").rank("ordinal") - 1) * 5 // pl.len())
g = dec.group_by("qb").agg(
    pl.col("echo_prob").mean().alias("mean_pred"),
    pl.col("is_echo").mean().alias("actual_echo_rate"),
    pl.len().alias("n")).sort("qb")
for r in g.iter_rows(named=True):
    print(f"    bin{r['qb']}  mean_pred={r['mean_pred']:.3f}  actual_rate={r['actual_echo_rate']:.3f}  n={r['n']}")
print()

print("=== C. AOBT_3-based discriminator in the offset>7200 bin ===")
feats = pl.read_parquet("cache/features/holdout_gap2025.parquet")
j = big.join(feats.select("MVT_ID_mvt", "aobt3_taxi", "aobt3_vs_eobt"), on="MVT_ID_mvt", how="left")
miss_rate = j["aobt3_taxi"].is_null().mean()
print(f"  aobt3_taxi missingness in offset>7200 bin: {miss_rate:.3f}")
avail = j.filter(pl.col("aobt3_taxi").is_not_null())
print(f"  n with aobt3_taxi available: {avail.height}")
print(f"  RMSE(aobt3_taxi, taxi) where available: {rmse(avail['aobt3_taxi'], avail['taxi']):.0f}")
print(f"  RMSE(current model pred, taxi) on that same subset: {rmse(avail['pred'], avail['taxi']):.0f}")
print(f"  corr(aobt3_taxi, taxi) where available: {avail.select(pl.corr('aobt3_taxi','taxi')).item():.3f}")
# does availability of aobt3 correlate with is_echo?
print(f"  is_echo rate WHEN aobt3 missing: {j.filter(pl.col('aobt3_taxi').is_null())['is_echo'].mean():.3f}")
print(f"  is_echo rate WHEN aobt3 present: {avail['is_echo'].mean():.3f}")
