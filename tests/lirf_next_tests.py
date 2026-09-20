import polars as pl
import numpy as np

FLOOR, ECHO_WIDE_CEIL = 0, 140000
CEIL, LIRF_RAW_CEIL = 10800, 140000
D_CLIP_LO, D_CLIP_HI = -3600, 10800  # reuse encode.py's existing D_CLIP

def rmse(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    return float(np.sqrt(np.mean((p - t) ** 2)))

def report_by_airport(df, pred_col, label):
    print(f"--- {label} ---")
    e = (df[pred_col].cast(pl.Float64) - df["taxi"].cast(pl.Float64))
    print(f"  overall rmse={float((e.pow(2).mean())**0.5):.1f}")
    g = df.with_columns(sq=e.pow(2)).group_by("ADEP_mvt").agg(
        pl.col("sq").mean().sqrt().alias("rmse"), pl.len().alias("n")).sort("ADEP_mvt")
    print("  " + "  ".join(f"{r['ADEP_mvt']}:{r['rmse']:.0f}" for r in g.iter_rows(named=True)))

# ======================================================================
print("=" * 70)
print("TEST 1: AOBT_3-availability blend, on top of the SHIPPED v12 state")
print("=" * 70)
ev = pl.read_parquet("cache/lirf_ceilfix_mixed_ev.parquet")
feats = pl.read_parquet("cache/features/holdout_gap2025.parquet")
ev = ev.join(feats.select("MVT_ID_mvt", "aobt3_taxi"), on="MVT_ID_mvt", how="left")

report_by_airport(ev.filter(pl.col("ADEP_mvt") == "LIRF"), "pred", "baseline (shipped v12, LIRF)")

# gate matching validated evidence exactly: LIRF & offset>7200 & aobt3 present
gate_narrow = ((pl.col("ADEP_mvt") == "LIRF") & (pl.col("sched_takeoff_offset") > 7200)
               & pl.col("aobt3_taxi").is_not_null())
taxi_echo = pl.col("sched_takeoff_offset").clip(0, ECHO_WIDE_CEIL)
new_component = pl.col("echo_prob") * taxi_echo + (1 - pl.col("echo_prob")) * pl.col("aobt3_taxi")
ev = ev.with_columns(
    pred_t1_narrow=pl.when(gate_narrow).then(new_component).otherwise(pl.col("pred")),
)
report_by_airport(ev.filter(pl.col("ADEP_mvt") == "LIRF"), "pred_t1_narrow",
                   "TEST1-narrow (LIRF & offset>7200 & aobt3 present)")

# broader variant: all LIRF rows with aobt3 present, regardless of offset
gate_broad = (pl.col("ADEP_mvt") == "LIRF") & pl.col("aobt3_taxi").is_not_null()
ev = ev.with_columns(
    pred_t1_broad=pl.when(gate_broad).then(new_component).otherwise(pl.col("pred")),
)
report_by_airport(ev.filter(pl.col("ADEP_mvt") == "LIRF"), "pred_t1_broad",
                   "TEST1-broad (LIRF & aobt3 present, any offset)")
n_narrow = ev.filter(gate_narrow).height
n_broad = ev.filter(gate_broad).height
print(f"  rows affected: narrow={n_narrow}  broad={n_broad}")
print()

# ======================================================================
print("=" * 70)
print("TEST 2: bound d-hat directly (uniform, no airport name) vs shipped fix")
print("=" * 70)
flip = pl.read_parquet("cache/lirf_diag_flip_ev.parquet")  # pure flip target, ALL airports, PRE-fix pred
not_lirf = flip["ADEP_mvt"] != "LIRF"
d_hat = flip["sched_takeoff_offset"] - flip["taxi_model_raw"]

# (a) OLD pre-fix baseline: already stored as 'pred' (uniform CEIL=10800)
report_by_airport(flip, "pred", "(a) OLD pre-fix (uniform CEIL=10800)")

# (b) SHIPPED (airport-gated ceil, reproduce on this dataset for a clean comparison)
ceil_shipped = np.where(not_lirf.to_numpy(), CEIL, LIRF_RAW_CEIL)
taxi_model_shipped = np.clip(flip["taxi_model_raw"].to_numpy(), FLOOR, ceil_shipped)
taxi_echo_np = np.clip(flip["sched_takeoff_offset"].to_numpy(), 0, ECHO_WIDE_CEIL)
p = flip["echo_prob"].to_numpy()
use_prior = flip["use_prior"].to_numpy()
# reconstruct prior_taxi implicitly is unnecessary: use_prior rows are untouched by
# either test, so just keep old 'pred' for them (see reasoning in chat)
old_pred = flip["pred"].to_numpy()
pred_b = np.where(use_prior, old_pred, p * taxi_echo_np + (1 - p) * taxi_model_shipped)
flip = flip.with_columns(pred_shipped_repro=pl.Series(pred_b))
report_by_airport(flip, "pred_shipped_repro", "(b) SHIPPED repro (airport-gated ceil)")

# (c) IDEA 2: clip d_hat to D_CLIP, reconstruct, then a UNIFORM wide ceil (no airport name)
d_hat_clipped = d_hat.clip(D_CLIP_LO, D_CLIP_HI)
raw_taxi_new = flip["sched_takeoff_offset"] - d_hat_clipped
taxi_model_new = raw_taxi_new.clip(FLOOR, ECHO_WIDE_CEIL).to_numpy()
pred_c = np.where(use_prior, old_pred, p * taxi_echo_np + (1 - p) * taxi_model_new)
flip = flip.with_columns(pred_idea2=pl.Series(pred_c))
report_by_airport(flip, "pred_idea2", "(c) IDEA2 (clip d_hat to [-3600,10800], uniform wide ceil, NO airport gate)")
