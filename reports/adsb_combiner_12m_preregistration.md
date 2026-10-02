# ADS-B combiner retrained with the other 2025 months — pre-registration

Written 2026-10-02, **before the extra ADS-B days were downloaded** (the
Colab pull to `adsb_restofyear/` started today). No combiner has been fit
on them. Changes after results exist go under Amendments, dated.

## Why

The v25 combiner (§58) is trained only on the Jan/Jul 2025 holdout. Each
cross-fit direction sees one month (~70–80k ADS-B rows).
- Its fits are unstable (best_iter 4,614 vs 1,008).
- Its coverage mix differs from 2026. EGLL detected pushbacks are ~3% in
  Jan/Jul 2025 but 22–75% in 2026, and the network grew through 2025.

The other 10 months have honest month-wise out-of-fold base predictions
(`cache/oof/{lgb,cat}_mixed/`), so their ADS-B days can be added as
training rows.

## Data

- **New days:** days 1–10 of Feb–Jun and Aug–Dec 2025 (as many as the
  pull delivers; the count is reported), plus the existing 2025-09-15 and
  2025-11-15.
- **Pipeline:** normalised by `src/ingest/normalise_adsb.py`, detected by
  `src/link/adsb_pushback.py --v4` (the detector v25 uses).
- **Stack prediction `s`:**
  - On these OOF-month rows: NNLS(lgb OOF, cat OOF), weights fit on those
    rows.
  - On the holdout: NNLS(lgb, catcorr) from the fit month, as in v24/v25.
  - The OOF months have no corrected CatBoost; that mismatch is accepted
    and noted.
- **Rows:** labels ≤ 5 h, LIRF excluded, as for v24.

## Treatment (`tests/adsb_combiner_12m_test.py`)

The v24 combiner exactly: same inputs (`tests/adsb_combiner_test.py`
FEATS), hyperparameters, day-of-month-divisible-by-5 inner early stopping
and refit at the best iteration. Only the training rows change:
- **Direction A** (scores Jul): train on all OOF-month ADS-B-day rows plus
  the Jan holdout rows.
- **Direction B** (scores Jan): the OOF-month rows plus the Jul holdout
  rows.
- **Application:** restricted as in v25. On rows with ADS-B information
  (matched or a fallback tier), non-LIRF, of the scored month; the v23
  pipeline value elsewhere.

**Base:** v25's method cross-fit (the v24 combiner restricted), as in
`tests/adsb_v5_test.py`.

## Rule

Trimmed RMSE decides (labels ≤ 5 h, the fixed 31-row set). Adopt iff all
of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- the number of new days and training rows per direction;
- best iterations;
- per airport;
- the variant trained on OOF months only (no holdout month).

**If it passes:** build the next submission. The combiner is fit on all
OOF-month rows plus both holdout months, rounds from the same inner
split, and applied as in v25 (`src/post/adsb_combiner.py`, restricted).
**Keep iff the board beats the best score at build time (v25 264.39 now).**

**If it fails: stop.** No reweighting of months, no hyperparameter
changes.

## Amendments

(none)
