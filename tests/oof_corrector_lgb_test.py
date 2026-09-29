"""Pre-registered holdout gate: OOF residual corrector v2 on LightGBM.

reports/oof_corrector_lgb_preregistration.md. The corrector is exactly v2
(tests/oof_corrector_v2_test.py) with the stack's LightGBM as the base. The gate
is decided on the stack: cross-fit NNLS lgbcorr + catcorr vs lgb + catcorr (v21).
Needs:
  cache/oof/lgb_mixed/fold=<m>.parquet        (run_oof(engine="lgb", target="mixed"))
  cache/eval/lgb_mixed_holdout_ev.parquet     (the stack's LightGBM)
  cache/eval/catcorr_mixed_holdout_ev.parquet (src/post/corrector_v2.py, v21)
Writes cache/eval/lgbcorr_mixed_holdout_ev.parquet, the fitted corrector
(cache/oof/corrector_lgb_m10.txt) and its round count
(cache/oof/corrector_lgb_meta.json) for the v22 ranking refit.

Run:  .venv/Scripts/python.exe tests/oof_corrector_lgb_test.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "cache" / "eval"
# v1 reads its base-model paths at import time
os.environ.setdefault("PRC_OOF_DIR", str(ROOT / "cache" / "oof" / "lgb_mixed"))
os.environ.setdefault("PRC_CAT_EV", str(EVAL / "lgb_mixed_holdout_ev.parquet"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oof_corrector_test as v1  # noqa: E402
import oof_corrector_v2_test as v2  # noqa: E402
from _harness import cluster_bootstrap  # noqa: E402
from features.encode import NMU_COLS  # noqa: E402
from models.fit import HOLDOUT_MONTHS, _matrix  # noqa: E402
from stack_r20k_test import cross  # noqa: E402

JAN, JUL = HOLDOUT_MONTHS
CATCORR_EV = EVAL / "catcorr_mixed_holdout_ev.parquet"
OUT_EV = EVAL / "lgbcorr_mixed_holdout_ev.parquet"
META = ROOT / "cache" / "oof" / "corrector_lgb_meta.json"
M10_PATH = ROOT / "cache" / "oof" / "corrector_lgb_m10.txt"


def corrected_holdout() -> pl.DataFrame:
    """v2's corrector on the LightGBM base: fit on the 10 training months' OOF
    rows, applied to LightGBM's holdout predictions (rows with pred <= PRED_MAX)."""
    oof = v1.load_oof()
    ev = pl.read_parquet(v1.CAT_EV)
    f_tr, f_ho, lab, ho_day = v1.build_inputs()
    nmu_tr, nmu_ho = v2.nmu_inputs(f_tr, f_ho, lab)

    tr = (f_tr.select("MVT_ID_mvt").with_row_index("_i")
          .join(oof.select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt")
          .join(lab.select("MVT_ID_mvt", "taxi", "ym"), on="MVT_ID_mvt")
          .join(nmu_tr, on="MVT_ID_mvt", how="left")
          .filter((pl.col("taxi") >= 0) & (pl.col("pred") <= v2.PRED_MAX)))
    ho = (f_ho.select("MVT_ID_mvt").with_row_index("_i")
          .join(ev.select("MVT_ID_mvt", "pred", "echo_prob"), on="MVT_ID_mvt")
          .join(nmu_ho, on="MVT_ID_mvt", how="left"))
    if ho.height != ev.height:
        raise SystemExit(f"holdout rows: features {ho.height} vs lgb ev {ev.height}")
    Xtr_all, _, _, categories = _matrix(f_tr)
    Xho_all, _, _, _ = _matrix(f_ho, categories)
    Xtr = Xtr_all.iloc[tr["_i"].to_numpy()].reset_index(drop=True)
    Xho = Xho_all.iloc[ho["_i"].to_numpy()].reset_index(drop=True)
    del Xtr_all, Xho_all
    for X, f in ((Xtr, tr), (Xho, ho)):
        X["base_pred"], X["base_echo_prob"] = f["pred"].to_numpy(), f["echo_prob"].to_numpy()
        for c in NMU_COLS:
            X[c] = f[c].cast(pl.Float64).to_numpy()
    y = tr["taxi"].to_numpy() - tr["pred"].to_numpy()
    print(f"corrector training rows {len(y):,} (training months, taxi >= 0, "
          f"base pred <= {v2.PRED_MAX:.0f})")

    m, best, full_rounds = v2.fit_corrector(Xtr, y, tr["ym"].to_numpy())
    m.save_model(str(M10_PATH))
    META.write_text(json.dumps(dict(best_iter=best, full_rounds=full_rounds, cap=v2.CORR_ROUNDS,
                                    pred_max=v2.PRED_MAX, huber_alpha=v2.HUBER_ALPHA), indent=2))
    print(f"  best_iter {best} of cap {v2.CORR_ROUNDS}"
          f"{'  (HIT THE CAP)' if best >= v2.CORR_ROUNDS else ''}")

    base = ho["pred"].to_numpy()
    apply_mask = base <= v2.PRED_MAX
    corr = base.copy()
    corr[apply_mask] = np.maximum(0.0, base[apply_mask] + m.predict(Xho[apply_mask]))
    out = (ho.select("MVT_ID_mvt")
           .with_columns(pred=pl.Series(corr), applied=pl.Series(apply_mask),
                         nm_unmatched=pl.Series(Xho["aobt3_taxi"].isna().to_numpy()))
           .join(ev.select("MVT_ID_mvt", "ADEP_mvt", "ym", "taxi"), on="MVT_ID_mvt"))
    out.drop("applied", "nm_unmatched").write_parquet(OUT_EV)
    print(f"wrote {OUT_EV.name}; corrector applied to {apply_mask.mean() * 100:.2f}% of holdout rows")
    return out.join(ho_day, on="MVT_ID_mvt", how="left")


def main() -> None:
    corr = corrected_holdout()
    pick = lambda p, c: pl.read_parquet(p).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias(c))  # noqa: E731
    df = (corr.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ym").cast(pl.Utf8))
          .rename({"pred": "lgbcorr"})
          .join(pick(v1.CAT_EV, "lgb"), on="MVT_ID_mvt")
          .join(pick(CATCORR_EV, "catcorr"), on="MVT_ID_mvt"))
    if df.height != corr.height:
        raise SystemExit(f"holdout join lost rows: {corr.height} -> {df.height}")
    t = df["taxi"].to_numpy()
    keep = t <= v1.MONSTER_S
    n_monster = int((~keep).sum())
    print(f"holdout rows {df.height:,}; monster labels (> 5 h): {n_monster}")
    if n_monster != v1.N_MONSTER_EXPECTED:
        raise SystemExit(f"expected {v1.N_MONSTER_EXPECTED} monster rows, got {n_monster}")
    jan, jul = (df["ym"] == JAN).to_numpy(), (df["ym"] == JUL).to_numpy()

    old, wo = cross(df, ["lgb", "catcorr"])
    new, wn = cross(df, ["lgbcorr", "catcorr"])
    print(f"stack weights  control   {wo}\n               treatment {wn}")

    def score(a: np.ndarray, b: np.ndarray, mask: np.ndarray, label: str):
        d = df.with_columns(se_b=pl.Series((a - t) ** 2), se_t=pl.Series((b - t) ** 2)).filter(pl.Series(mask))
        cl = d.group_by("ADEP_mvt", "day").agg(pl.col("se_b").sum(), pl.col("se_t").sum(), pl.len().alias("n"))
        pt, lo, hi, pw = cluster_bootstrap(cl["se_b"].to_numpy(), cl["se_t"].to_numpy(), cl["n"].to_numpy(), 3000, 0)
        print(f"  {label:<24} base={np.sqrt(d['se_b'].mean()):7.2f}  new={np.sqrt(d['se_t'].mean()):7.2f}  "
              f"delta={pt:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  P(worse)={pw:.3f}")
        return pt, pw

    alln = np.ones(df.height, bool)
    print("\nGATE: stack lgbcorr+catcorr vs lgb+catcorr (cross-fit)")
    print(" TRIMMED (decides):")
    d_jul, _ = score(old, new, jul & keep, "Jul trimmed (fit Jan)")
    d_jan, _ = score(old, new, jan & keep, "Jan trimmed (fit Jul)")
    d_pool, p_pool = score(old, new, keep, "POOLED trimmed")
    print(" FULL (guard):")
    score(old, new, jul, "Jul full")
    score(old, new, jan, "Jan full")
    _, p_full = score(old, new, alln, "POOLED full")
    passed = d_pool < 0 and p_pool < 0.05 and d_jan < 0 and d_jul < 0 and p_full < 0.9
    print(f"\nrule: pooled trimmed delta<0 & P<0.05 [{d_pool:+.2f}, {p_pool:.3f}], "
          f"both months trimmed<0 [Jan {d_jan:+.2f}, Jul {d_jul:+.2f}], full guard P<0.9 [{p_full:.3f}]"
          f"\n  -> {'PASS (go to v22)' if passed else 'FAIL (stop)'}")

    print("\nDIAGNOSTICS (not decisive)")
    lg, lc = df["lgb"].to_numpy(), df["lgbcorr"].to_numpy()
    print(" lgbcorr vs lgb, single model:")
    score(lg, lc, keep, "pooled trimmed")
    score(lg, lc, alln, "pooled full")
    print(" stack:")
    score(old, new, keep & (df["ADEP_mvt"] != "LIRF").to_numpy(), "pooled trimmed, no LIRF")
    shift = df["lgbcorr"] - df["lgb"]
    for flag in (True, False):
        s = shift.filter(df["nm_unmatched"] == flag)
        print(f"  mean correction, NM-{'unmatched' if flag else 'matched'}: {s.mean():+.1f} s "
              f"(mean |.| {s.abs().mean():.1f} s, n={s.len():,})")
    d2 = df.with_columns(old=pl.Series(old), new=pl.Series(new))
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", float_precision=2, tbl_rows=20):
        agg = lambda c: ((pl.col(c) - pl.col("taxi")) ** 2).mean().sqrt()  # noqa: E731
        for lbl, frame in (("trimmed", d2.filter(pl.Series(keep))), ("full", d2)):
            print(f"stack per airport, {lbl}:")
            print(frame.group_by("ADEP_mvt").agg(agg("old").alias("control"), agg("new").alias("treat"), pl.len())
                  .with_columns(delta=pl.col("treat") - pl.col("control")).sort("ADEP_mvt"))
        print("stack per decile of true taxi (all rows):")
        print(d2.with_columns(dec=pl.col("taxi").qcut(10, labels=[str(i) for i in range(1, 11)]))
              .group_by("dec").agg(pl.col("taxi").max().alias("taxi_max"), agg("old").alias("control"),
                                   agg("new").alias("treat"), pl.len())
              .with_columns(delta=pl.col("treat") - pl.col("control")).sort("taxi_max"))


if __name__ == "__main__":
    main()
