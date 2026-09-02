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

# LightGBM (CPU on Colab is still faster than the laptop; ~5 min total)
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
# Submissions land in data/submissions/<name>.parquet — download and upload to
# the challenge portal as <team>_v<n>.parquet.
