# Colab model-training driver.
#
# Feature building stays local (Polars/CPU, ~30 s). Only the LightGBM/XGBoost
# fits move here — the local CPU fit is ~4 min each and the full pipeline
# (holdout + submission) is ~25 min, too slow to iterate on.
#
# ---------------------------------------------------------------------------
# SETUP (run once per Colab session)
# ---------------------------------------------------------------------------
# 1. Locally:  .venv/Scripts/python.exe src/features/export_model_inputs.py
#    -> writes cache/features/{train2025,ranking,labels2025}.parquet  (~90 MB)
# 2. Upload those 3 files + data/ranking/submitting.parquet + the src/ tree
#    to Google Drive (or `git clone` the repo and copy the parquets in).
# 3. In Colab:
#
#     !pip -q install polars==1.44.1 lightgbm==4.7.0 xgboost
#     from google.colab import drive; drive.mount('/content/drive')
#     import sys; sys.path.insert(0, '/content/drive/MyDrive/smart-jigsaw/src')
#     import os; os.chdir('/content/drive/MyDrive/smart-jigsaw')
#
# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
# %%
from models.fit import run

# LightGBM (CPU on Colab is still faster than the laptop; ~5 min total).
# Default loss is huber (alpha=800) -- beat plain L2 on the gap-realistic
# holdout (303.1s vs 309.2s), see reports/eval/lgb_colab_huber_a800.md. It
# hadn't fully plateaued at the local ROUNDS ceiling (8000); worth pushing
# rounds higher here where it's cheap to check whether it improves further.
model, ev = run(engine="lgb", name="lgb_colab")

# %%
# XGBoost on the Colab T4 GPU (~90 s per fit). Set Runtime > Change runtime
# type > T4 GPU first.
model_x, ev_x = run(engine="xgb", name="xgb_colab")

# %%
# ev / ev_x are Polars frames (MVT_ID_mvt, taxi, pred, ADEP_mvt, ym) for the
# Jan+Jul 2025 holdout — slice them for per-airport / per-decile analysis.
import polars as pl
print(ev.group_by("ADEP_mvt").agg(
    ((pl.col("pred") - pl.col("taxi")).pow(2).mean().sqrt()).alias("rmse"),
    pl.len(),
).sort("rmse", descending=True))

# %%
# OOF CatBoost predictions for the corrector (reports/oof_corrector_preregistration.md).
# T4 runtime, ~3 h for 10 folds x 3 fits. Each fold is saved to
# cache/oof/cat_mixed/fold=<month>.parquet as it finishes; if the session drops,
# rerun this cell and finished folds are skipped. Keep cache/ on Drive.
import os; os.environ["PRC_MEMLOG"] = "1"
from models.fit import run_oof
run_oof(engine="cat", target="mixed", eta=0.05)

# %%
# CatBoost rerun for v21 (reports/oof_corrector_v2_preregistration.md): the
# production CatBoost run, also saving the ranking rows' pred/echo_prob, which
# the corrector needs. T4 runtime, ~40 min (3 holdout fits + 3 refits).
# Named cat_mixed_rerun so data/submissions/cat_mixed.parquet (v19's input) is
# left alone. Copy back: cache/eval/cat_mixed_rerun_holdout_ev.parquet,
# cache/eval/cat_mixed_rank.parquet (and data/submissions/cat_mixed_rerun.parquet).
import os; os.environ["PRC_MEMLOG"] = "1"
from models.fit import run
run(engine="cat", target="mixed", eta=0.05, name="cat_mixed_rerun", submit=True,
    ev_out="cache/eval/cat_mixed_rerun_holdout_ev.parquet",
    rank_out="cache/eval/cat_mixed_rank.parquet")

# %%
# Stage-1 screen: CatBoost with categorical pairs (reports/cat_ctr2_preregistration.md).
# T4 runtime. Holdout fits only (submit=False, no refit): 2 regressors + echo classifier.
# The log prints cumulative seconds per fit ("holdout fit ...s") -- note them: if the
# mean fit time exceeds 25 min, stage 2 (30 OOF fits) is not feasible and we stop.
# Copy back: cache/eval/cat_ctr2_mixed_holdout_ev.parquet (+ the cell output as a log).
import os; os.environ["PRC_MEMLOG"] = "1"
from models.fit import run
run(engine="cat_ctr2", target="mixed", eta=0.05, seed=42, name="cat_ctr2_mixed", submit=False,
    ev_out="cache/eval/cat_ctr2_mixed_holdout_ev.parquet")

# %%
# Submissions land in data/submissions/<name>.parquet — download and upload to
# the challenge portal as <team>_v<n>.parquet.
