"""Pre-registered holdout gate: OOF residual corrector v2 on CatBoost.

reports/oof_corrector_v2_preregistration.md. Differences from v1
(tests/oof_corrector_test.py): Huber loss (alpha 800), training rows and
application selected on the base prediction (<= 7,200 s) instead of the true
label, the NM-unmatched carrier echo rate as an extra input, and a 5,000-round
cap. Writes the corrector's refit round count to cache/oof/corrector_v2_meta.json
(the v21 ranking refit reuses it).

Run:  .venv/Scripts/python.exe tests/oof_corrector_v2_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oof_corrector_test as v1  # noqa: E402
from _harness import cluster_bootstrap  # noqa: E402
from features.encode import (  # noqa: E402
    NMU_COLS,
    add_nmu_echo_rate_oof,
    apply_nmu_echo_rate,
    fit_nmu_echo_rate,
)
from models.fit import HOLDOUT_MONTHS, TRAIN_MONTHS, _matrix  # noqa: E402

ROOT = v1.ROOT
JAN, JUL = HOLDOUT_MONTHS
PRED_MAX = 7200.0
HUBER_ALPHA = 800.0
CORR_ROUNDS, CORR_ES = 5000, 100
META = ROOT / "cache" / "oof" / "corrector_v2_meta.json"


def nmu_inputs(f_tr: pl.DataFrame, f_ho: pl.DataFrame, lab: pl.DataFrame):
    """NMU_COLS keyed on MVT_ID_mvt: leave-one-month-out over the training
    months for training rows, fit on the 10 training months for holdout rows."""
    flag = pl.concat([f.select("MVT_ID_mvt", nm_unmatched=pl.col("aobt3_taxi").is_null())
                      for f in (f_tr, f_ho)])
    lab_n = lab.join(flag, on="MVT_ID_mvt", how="inner")
    tr_lab = lab_n.filter(pl.col("ym").is_in(TRAIN_MONTHS))
    tr = add_nmu_echo_rate_oof(f_tr.select("MVT_ID_mvt", "ADEP_mvt", "AIRCRAFT_OPERATOR_flt"), tr_lab)
    ho = apply_nmu_echo_rate(f_ho.select("MVT_ID_mvt", "ADEP_mvt", "AIRCRAFT_OPERATOR_flt"),
                             fit_nmu_echo_rate(tr_lab))
    return tr.select("MVT_ID_mvt", *NMU_COLS), ho.select("MVT_ID_mvt", *NMU_COLS)


def fit_corrector(X, y, ym):
    import lightgbm as lgb

    params = dict(v1.CORR_PARAMS, objective="huber", alpha=HUBER_ALPHA)
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    va = ym == v1.ES_MONTH
    ds = lgb.Dataset(X[~va], label=y[~va], categorical_feature=cats, free_raw_data=False)
    vds = lgb.Dataset(X[va], label=y[va], reference=ds, categorical_feature=cats, free_raw_data=False)
    m = lgb.train(params, ds, num_boost_round=CORR_ROUNDS, valid_sets=[vds],
                  callbacks=[lgb.early_stopping(CORR_ES, verbose=False)])
    best = m.best_iteration or CORR_ROUNDS
    r0 = float(np.sqrt(np.mean(y[va] ** 2)))
    r1 = float(np.sqrt(np.mean((y[va] - m.predict(X[va], num_iteration=best)) ** 2)))
    full_rounds = max(50, int(round(1.1 * best)))
    m_all = lgb.train(params, lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False),
                      num_boost_round=full_rounds)
    print(f"  corrector v2 best_iter={best} full_rounds={full_rounds}  "
          f"{v1.ES_MONTH} residual RMSE {r0:.2f} -> {r1:.2f} (rows with base pred <= {PRED_MAX:.0f})")
    return m_all, best, full_rounds


def main() -> None:
    oof = v1.load_oof()
    ev = pl.read_parquet(v1.CAT_EV).select("MVT_ID_mvt", "ADEP_mvt", pl.col("ym").cast(pl.Utf8),
                                           "taxi", "pred", "echo_prob")
    f_tr, f_ho, lab, ho_day = v1.build_inputs()
    nmu_tr, nmu_ho = nmu_inputs(f_tr, f_ho, lab)

    tr = (f_tr.select("MVT_ID_mvt").with_row_index("_i")
          .join(oof.select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt")
          .join(lab.select("MVT_ID_mvt", "taxi", "ym"), on="MVT_ID_mvt")
          .join(nmu_tr, on="MVT_ID_mvt", how="left")
          .filter((pl.col("taxi") >= 0) & (pl.col("pred") <= PRED_MAX)))
    ho = (f_ho.select("MVT_ID_mvt").with_row_index("_i")
          .join(ev.select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt")
          .join(nmu_ho, on="MVT_ID_mvt", how="left"))
    if ho.height != ev.height:
        raise SystemExit(f"holdout rows: features {ho.height} vs cat ev {ev.height}")
    Xtr_all, names, cats, categories = _matrix(f_tr)
    Xho_all, _, _, _ = _matrix(f_ho, categories)
    Xtr = Xtr_all.iloc[tr["_i"].to_numpy()].reset_index(drop=True)
    Xho = Xho_all.iloc[ho["_i"].to_numpy()].reset_index(drop=True)
    del Xtr_all, Xho_all
    for X, f in ((Xtr, tr), (Xho, ho)):
        X["base_pred"], X["base_echo_prob"] = f["pred"].to_numpy(), f["echo_prob"].to_numpy()
        for c in NMU_COLS:
            X[c] = f[c].cast(pl.Float64).to_numpy()
    y = tr["taxi"].to_numpy() - tr["pred"].to_numpy()
    ym = tr["ym"].to_numpy()
    print(f"corrector training rows {len(y):,} (training months, taxi >= 0, base pred <= {PRED_MAX:.0f}); "
          f"nmu_op_echo_rate null: train {int(np.isnan(Xtr['nmu_op_echo_rate']).sum())}, "
          f"holdout {int(np.isnan(Xho['nmu_op_echo_rate']).sum())}")

    m, best, full_rounds = fit_corrector(Xtr, y, ym)
    META.write_text(json.dumps(dict(best_iter=best, full_rounds=full_rounds, pred_max=PRED_MAX,
                                    huber_alpha=HUBER_ALPHA), indent=2))

    base = ho["pred"].to_numpy()
    apply_mask = base <= PRED_MAX
    corr = base.copy()
    corr[apply_mask] = np.maximum(0.0, base[apply_mask] + m.predict(Xho[apply_mask]))
    df = (ho.select("MVT_ID_mvt", "pred").join(ev.select("MVT_ID_mvt", "ADEP_mvt", "ym", "taxi"), on="MVT_ID_mvt")
          .join(ho_day, on="MVT_ID_mvt", how="left")
          .with_columns(corr=pl.Series(corr), applied=pl.Series(apply_mask),
                        nm_unmatched=pl.Series(Xho["aobt3_taxi"].isna().to_numpy())))
    t = df["taxi"].to_numpy()
    keep = t <= v1.MONSTER_S
    n_monster = int((~keep).sum())
    print(f"holdout rows {df.height:,}; monster labels (> 5 h): {n_monster}; "
          f"corrector applied to {apply_mask.mean() * 100:.2f}% of rows")
    if n_monster != v1.N_MONSTER_EXPECTED:
        raise SystemExit(f"expected {v1.N_MONSTER_EXPECTED} monster rows, got {n_monster}")
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()

    def score(mask: np.ndarray, label: str):
        d = df.with_columns(se_b=(pl.col("pred") - pl.col("taxi")) ** 2,
                            se_t=(pl.col("corr") - pl.col("taxi")) ** 2).filter(pl.Series(mask))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), 3000, 0)
        print(f"  {label:<24} base={np.sqrt(d['se_b'].mean()):7.2f}  corr={np.sqrt(d['se_t'].mean()):7.2f}  "
              f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    print("\nGATE (v2 corrector) vs uncorrected CatBoost")
    print(" TRIMMED (decides):")
    d_jul, _ = score(jul & keep, "Jul trimmed")
    d_jan, _ = score(jan & keep, "Jan trimmed")
    d_pool, p_pool = score(keep, "POOLED trimmed")
    print(" FULL (guard):")
    score(jul, "Jul full")
    score(jan, "Jan full")
    _, p_full = score(np.ones(df.height, bool), "POOLED full")
    passed = d_pool < 0 and p_pool < 0.05 and d_jan < 0 and d_jul < 0 and p_full < 0.9
    print(f"\nrule: pooled trimmed delta<0 & P<0.05 [{d_pool:+.2f}, {p_pool:.3f}], "
          f"both months trimmed<0 [Jan {d_jan:+.2f}, Jul {d_jul:+.2f}], full guard P<0.9 [{p_full:.3f}]"
          f"\n  -> {'PASS (go to v21)' if passed else 'FAIL (stop)'}")

    print("\nDIAGNOSTICS (not decisive)")
    score(keep & (df["ADEP_mvt"] != "LIRF").to_numpy(), "pooled trimmed, no LIRF")
    shift = df["corr"] - df["pred"]
    for flag in (True, False):
        s = shift.filter(df["nm_unmatched"] == flag)
        print(f"  mean correction, NM-{'unmatched' if flag else 'matched'}: {s.mean():+.1f} s "
              f"(mean |.| {s.abs().mean():.1f} s, n={s.len():,})")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=2, tbl_rows=20):
        agg = lambda c: ((pl.col(c) - pl.col("taxi")) ** 2).mean().sqrt()  # noqa: E731
        for lbl, frame in (("trimmed", df.filter(pl.Series(keep))), ("full", df)):
            print(f"per airport, {lbl}:")
            print(frame.group_by("ADEP_mvt").agg(agg("pred").alias("base"), agg("corr").alias("corr"), pl.len())
                  .with_columns(delta=pl.col("corr") - pl.col("base")).sort("ADEP_mvt"))
        print("per decile of true taxi (all rows):")
        print(df.with_columns(dec=pl.col("taxi").qcut(10, labels=[str(i) for i in range(1, 11)]))
              .group_by("dec").agg(pl.col("taxi").max().alias("taxi_max"), agg("pred").alias("base"),
                                   agg("corr").alias("corr"), pl.len())
              .with_columns(delta=pl.col("corr") - pl.col("base")).sort("taxi_max"))


if __name__ == "__main__":
    main()
