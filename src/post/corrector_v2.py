"""Corrected-CatBoost inputs for the v21 stack (reports/oof_corrector_v2_preregistration.md).

Writes the two files src/post/stack_submit.py --cat-corrected reads in place of
CatBoost's:
  cache/eval/catcorr_mixed_holdout_ev.parquet  holdout: the gate's corrector (fit on
                                               the 10 training months' OOF rows)
                                               applied to CatBoost's holdout predictions
  data/submissions/catcorr_mixed.parquet       ranking: the corrector refit on all 12
                                               months, applied to the CatBoost rerun's
                                               ranking predictions

Inputs:
  cache/oof/cat_mixed/fold=*.parquet            OOF folds (run_oof)
  cache/oof/corrector_v2_meta.json              round count from the gate
  cache/eval/cat_mixed_holdout_ev.parquet       CatBoost holdout predictions (gate baseline)
  cache/eval/cat_mixed_rerun_holdout_ev.parquet the Colab rerun's holdout predictions
  cache/eval/cat_mixed_rank.parquet             the Colab rerun's ranking rows (run(rank_out=))

Per the pre-registration: if the rerun's holdout predictions differ from the
gate baseline by more than 1 s RMS, the rerun's holdout file is used throughout.

Run:  .venv/Scripts/python.exe src/post/corrector_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import oof_corrector_test as v1  # noqa: E402
import oof_corrector_v2_test as v2  # noqa: E402
from features.encode import (  # noqa: E402
    NMU_COLS,
    add_nmu_echo_rate_oof,
    apply_group_encodings,
    apply_nmu_echo_rate,
    apply_priors,
    add_group_encodings_oof,
    fit_group_encodings,
    fit_nmu_echo_rate,
    fit_priors,
)
from models.fit import _matrix  # noqa: E402

EVAL = ROOT / "cache" / "eval"
SUB = ROOT / "data" / "submissions"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
RERUN_EV = EVAL / "cat_mixed_rerun_holdout_ev.parquet"
RANK_IN = EVAL / "cat_mixed_rank.parquet"
RERUN_TOL_RMS = 1.0
# fitted correctors are cached so a missing ranking file doesn't cost the ~20 min of fits
M10_PATH = ROOT / "cache" / "oof" / "corrector_v2_m10.txt"
M12_PATH = ROOT / "cache" / "oof" / "corrector_v2_m12.txt"


def fit_fixed(X, y, rounds: int, cache: Path | None = None):
    import lightgbm as lgb

    if cache is not None and cache.exists():
        print(f"loading cached corrector {cache}")
        return lgb.Booster(model_file=str(cache))

    params = dict(v1.CORR_PARAMS, objective="huber", alpha=v2.HUBER_ALPHA)
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    m = lgb.train(params, lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False),
                  num_boost_round=rounds)
    if cache is not None:
        m.save_model(str(cache))
    return m


def add_extra(X, frame: pl.DataFrame) -> None:
    X["base_pred"], X["base_echo_prob"] = frame["pred"].to_numpy(), frame["echo_prob"].to_numpy()
    for c in NMU_COLS:
        X[c] = frame[c].cast(pl.Float64).to_numpy()


def corrected(model, X, base: np.ndarray) -> np.ndarray:
    out = base.copy()
    m = base <= v2.PRED_MAX
    out[m] = np.maximum(0.0, base[m] + model.predict(X[m]))
    return out


def main() -> None:
    meta = json.loads(v2.META.read_text())
    rounds = int(meta["full_rounds"])
    print(f"corrector rounds (from the gate's 10-month refit): {rounds}")

    # ---- which CatBoost holdout predictions (pre-registered 1 s RMS rule)
    base_ev = pl.read_parquet(v1.CAT_EV)
    rerun = pl.read_parquet(RERUN_EV)
    j = base_ev.select("MVT_ID_mvt", "pred").join(rerun.select("MVT_ID_mvt", pl.col("pred").alias("p2")),
                                                  on="MVT_ID_mvt")
    rms = float(np.sqrt(((j["pred"] - j["p2"]) ** 2).mean()))
    print(f"rerun vs gate holdout predictions: n={j.height:,} of {base_ev.height:,}, RMS diff {rms:.3f} s, "
          f"max |diff| {(j['pred'] - j['p2']).abs().max():.1f} s")
    ev = rerun if rms > RERUN_TOL_RMS else base_ev
    print(f"-> using {'the RERUN' if rms > RERUN_TOL_RMS else 'the gate'} holdout predictions")
    ev = ev.select("MVT_ID_mvt", "ADEP_mvt", pl.col("ym").cast(pl.Utf8), "taxi", "pred", "echo_prob")

    oof = v1.load_oof()
    f_tr, f_ho, lab, _ = v1.build_inputs()
    nmu_tr, nmu_ho = v2.nmu_inputs(f_tr, f_ho, lab)
    Xtr_all, _, _, categories = _matrix(f_tr)
    Xho_all, _, _, _ = _matrix(f_ho, categories)
    train_cols = list(Xtr_all.columns)
    Xho_all = Xho_all[train_cols]  # align by name, as run() does

    tr = (f_tr.select("MVT_ID_mvt").with_row_index("_i")
          .join(oof.select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt")
          .join(lab.select("MVT_ID_mvt", "taxi", "ym"), on="MVT_ID_mvt"))
    ho = (f_ho.select("MVT_ID_mvt").with_row_index("_i")
          .join(ev.select("MVT_ID_mvt", "pred", "echo_prob", "taxi", "ym"), on="MVT_ID_mvt"))

    # ---- holdout: the gate's corrector (10 training months), applied to the holdout
    tr10 = tr.join(nmu_tr, on="MVT_ID_mvt", how="left").filter(
        (pl.col("taxi") >= 0) & (pl.col("pred") <= v2.PRED_MAX))
    X10 = Xtr_all.iloc[tr10["_i"].to_numpy()].reset_index(drop=True)
    add_extra(X10, tr10)
    m10 = fit_fixed(X10, tr10["taxi"].to_numpy() - tr10["pred"].to_numpy(), rounds, M10_PATH)
    del X10
    ho_g = ho.join(nmu_ho, on="MVT_ID_mvt", how="left")
    Xho = Xho_all.iloc[ho_g["_i"].to_numpy()].reset_index(drop=True)
    add_extra(Xho, ho_g)
    ho_corr = corrected(m10, Xho, ho_g["pred"].to_numpy())
    t = ho_g["taxi"].to_numpy()
    keep = t <= v1.MONSTER_S
    print(f"holdout: CatBoost trimmed {v1.rmse(ho_g['pred'].to_numpy(), t, keep):.2f} -> corrected "
          f"{v1.rmse(ho_corr, t, keep):.2f}; full {v1.rmse(ho_g['pred'].to_numpy(), t):.2f} -> "
          f"{v1.rmse(ho_corr, t):.2f}  (gate: 273.80 -> 270.55, 337.48 -> 334.72)")
    (ev.select("MVT_ID_mvt", "ADEP_mvt", "ym", "taxi")
     .join(ho_g.select("MVT_ID_mvt").with_columns(pred=pl.Series(ho_corr)), on="MVT_ID_mvt")
     .write_parquet(EVAL / "catcorr_mixed_holdout_ev.parquet"))
    print(f"wrote {EVAL / 'catcorr_mixed_holdout_ev.parquet'}")

    # ---- ranking corrector: refit on all 12 months (training OOF rows + holdout rows).
    # NMU for every labelled row: leave-one-month-out across all 12 months.
    flag = pl.concat([f.select("MVT_ID_mvt", nm_unmatched=pl.col("aobt3_taxi").is_null())
                      for f in (f_tr, f_ho)])
    lab_n = lab.join(flag, on="MVT_ID_mvt", how="inner")
    keys = pl.concat([f.select("MVT_ID_mvt", "ADEP_mvt", "AIRCRAFT_OPERATOR_flt") for f in (f_tr, f_ho)])
    nmu12 = add_nmu_echo_rate_oof(keys, lab_n).select("MVT_ID_mvt", *NMU_COLS)
    tr12 = tr.join(nmu12, on="MVT_ID_mvt", how="left").filter(
        (pl.col("taxi") >= 0) & (pl.col("pred") <= v2.PRED_MAX))
    ho12 = ho.join(nmu12, on="MVT_ID_mvt", how="left").filter(
        (pl.col("taxi") >= 0) & (pl.col("pred") <= v2.PRED_MAX))
    Xa = Xtr_all.iloc[tr12["_i"].to_numpy()].reset_index(drop=True)
    Xb = Xho_all.iloc[ho12["_i"].to_numpy()].reset_index(drop=True)
    add_extra(Xa, tr12)
    add_extra(Xb, ho12)
    import pandas as pd

    X12 = pd.concat([Xa, Xb], ignore_index=True)
    y12 = np.concatenate([tr12["taxi"].to_numpy() - tr12["pred"].to_numpy(),
                          ho12["taxi"].to_numpy() - ho12["pred"].to_numpy()])
    del Xa, Xb, Xtr_all, Xho_all
    print(f"ranking corrector training rows {len(y12):,} ({tr12.height:,} OOF + {ho12.height:,} holdout)")
    m12 = fit_fixed(X12, y12, rounds, M12_PATH)
    del X12
    if not RANK_IN.exists():
        raise SystemExit(f"{RANK_IN} not found; correctors are cached, rerun once it is in place")

    # ---- ranking inputs: priors / group encodings as run()'s all-2025 refit builds them
    feats = pl.read_parquet(v1.FEAT / "train2025.parquet")
    lab_all = pl.read_parquet(v1.FEAT / "labels2025.parquet").join(
        feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"), on="MVT_ID_mvt", how="left")
    priors_a = fit_priors(lab_all)
    lab_a = apply_priors(feats, priors_a).select("MVT_ID_mvt").join(lab_all, on="MVT_ID_mvt")
    genc_a = fit_group_encodings(lab_a)
    f_r = apply_group_encodings(apply_priors(pl.read_parquet(v1.FEAT / "ranking.parquet"), priors_a), genc_a)
    f_r = apply_nmu_echo_rate(f_r, fit_nmu_echo_rate(lab_n))
    rk = f_r.select("MVT_ID_mvt", *NMU_COLS).with_row_index("_i").join(
        pl.read_parquet(RANK_IN).select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt")
    if rk.height != f_r.height:
        raise SystemExit(f"ranking rows: features {f_r.height} vs rerun {rk.height}")
    Xr_all, _, _, _ = _matrix(f_r.drop(NMU_COLS), categories)
    Xr_all = Xr_all[train_cols]  # align by name, as run() does
    Xr = Xr_all.iloc[rk["_i"].to_numpy()].reset_index(drop=True)
    add_extra(Xr, rk)
    base_r = rk["pred"].to_numpy()
    rk_corr = corrected(m12, Xr, base_r)
    applied = base_r <= v2.PRED_MAX
    print(f"ranking: corrected {applied.mean() * 100:.2f}% of rows; mean shift "
          f"{(rk_corr - base_r).mean():+.1f} s, mean |shift| {np.abs(rk_corr - base_r).mean():.1f} s")

    tpl = pl.read_parquet(TEMPLATE)
    sub = tpl.select("MVT_ID_mvt").join(
        rk.select("MVT_ID_mvt").with_columns(
            TAXITIME_SEC_mvt=pl.Series(np.round(rk_corr).astype("int32"))), on="MVT_ID_mvt", how="left")
    assert sub.height == tpl.height and sub["TAXITIME_SEC_mvt"].null_count() == 0
    assert (sub["TAXITIME_SEC_mvt"] >= 0).all()
    sub.write_parquet(SUB / "catcorr_mixed.parquet")
    print(f"wrote {SUB / 'catcorr_mixed.parquet'}")


if __name__ == "__main__":
    main()
