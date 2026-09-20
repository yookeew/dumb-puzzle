"""Direct test of the overfitting hypothesis: train vs valid vs holdout RMSE.

If the model is overfitting the training data, in-sample RMSE will sit far
below holdout RMSE. If they're close, the model is at an information limit
and new features fail for lack of signal, not lack of capacity.

Mirrors the first half of models.fit.run() (same priors, same OOF group
encodings, same matrix, same fitter) and then scores the SAME model on
training rows, the inner-validation month, and the holdout.

Run:  .venv/Scripts/python.exe tests/train_gap.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.encode import (  # noqa: E402
    add_group_encodings_oof, apply_group_encodings, apply_priors,
    fit_group_encodings, fit_priors,
)
from models.fit import (  # noqa: E402
    ENGINES, ETA, EARLY_STOP, HOLDOUT_MONTHS, HUBER_ALPHA, LABEL_HI, LABEL_LO,
    LOSS_OBJECTIVE, ROUNDS, VALID_MONTH, _matrix,
)

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
RULE = "=" * 78


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main() -> None:
    t0 = time.time()
    feats = pl.read_parquet(FEAT_DIR / "train2025.parquet")
    feats_ho = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet")
    lab = pl.read_parquet(FEAT_DIR / "labels2025.parquet").join(
        feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"), on="MVT_ID_mvt", how="left")

    is_ho = pl.col("ym").is_in(HOLDOUT_MONTHS)
    tr_lab, ho_lab = lab.filter(~is_ho), lab.filter(is_ho)
    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    f_ho = apply_priors(feats_ho.join(ho_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    ho_lab = f_ho.select("MVT_ID_mvt").join(ho_lab, on="MVT_ID_mvt")
    f_tr = add_group_encodings_oof(f_tr, tr_lab)
    f_ho = apply_group_encodings(f_ho, fit_group_encodings(tr_lab))

    Xtr, names, cats, categories = _matrix(f_tr)
    taxi_tr = tr_lab["taxi"].to_numpy()
    ym_tr = tr_lab["ym"].to_numpy()
    keep = (taxi_tr >= LABEL_LO) & (taxi_tr <= LABEL_HI)
    iv = ym_tr == VALID_MONTH
    tr_mask, va_mask = keep & ~iv, keep & iv

    model, pf, bi = ENGINES["lgb"](
        Xtr[tr_mask], taxi_tr[tr_mask], cats, eta=ETA, rounds=ROUNDS,
        es=EARLY_STOP, valid=(Xtr[va_mask], taxi_tr[va_mask]),
        objective=LOSS_OBJECTIVE["huber"]["lgb"], alpha=HUBER_ALPHA, seed=42)
    print(f"fit {time.time() - t0:.0f}s  best_iter={bi}  (target=direct, huber)")

    Xho, _, _, _ = _matrix(f_ho, categories)
    taxi_ho = ho_lab["taxi"].to_numpy()
    ho_keep = (taxi_ho >= LABEL_LO) & (taxi_ho <= LABEL_HI)

    r_tr = _rmse(pf(model, Xtr[tr_mask]), taxi_tr[tr_mask])
    r_va = _rmse(pf(model, Xtr[va_mask]), taxi_tr[va_mask])
    r_ho = _rmse(pf(model, Xho[names][ho_keep]), taxi_ho[ho_keep])

    print(f"\n{RULE}\nGENERALIZATION GAP (raw model output, same label window)\n{RULE}")
    print(f"\n  train rows      (fitted, n={tr_mask.sum():,})      RMSE = {r_tr:7.1f}s")
    print(f"  inner-valid     (early stop, n={va_mask.sum():,})   RMSE = {r_va:7.1f}s")
    print(f"  holdout Jan+Jul (unseen, n={ho_keep.sum():,})     RMSE = {r_ho:7.1f}s")
    print(f"\n  train -> holdout gap: {r_ho - r_tr:+.1f}s  "
          f"({(r_ho / r_tr - 1) * 100:+.1f}%)")
    print(f"  valid -> holdout gap: {r_ho - r_va:+.1f}s  "
          f"({(r_ho / r_va - 1) * 100:+.1f}%)")
    print("\n  A large train->holdout gap = overfitting (capacity problem).")
    print("  A small gap = the model is at an information limit.")


if __name__ == "__main__":
    main()
