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

**2026-10-02, before any extra-month day reached the local repo and before
any gate run:**
- **The training months are fixed to Sep–Dec 2025:** days 1–10 of each,
  as many as the pull delivers, plus 2025-09-15 and 2025-11-15.
- **The gate runs once, on those months only.**
- **Reasons:** time (the Feb–Aug pull would finish days later), and those
  months' receiver coverage is closest to 2026.
- **Months that arrive later are not added to this test.** Using them
  would need a new pre-registration.
- `tests/adsb_combiner_12m_test.py` and `src/post/adsb_combiner.py --m12`
  are restricted accordingly (`MONTHS` in the test).

**2026-10-02, second amendment, again before any extra-month day reached
the local repo:**
- **The days are changed from 1–10 to days 01, 02, 03, 04, 07, 10, 13, 16,
  19, 22, 25 and 28 of each of Sep–Dec** (as many as the pull delivers).
- **Reason:** ADS-B coverage, weather and traffic swing from day to day.
  Consecutive days 1–10 are a correlated sample. Spreading the same budget
  over the month is more representative. Days 01–03 were already fetched,
  so they're kept.
- Everything else in the first amendment stands: Sep–Dec only, one gate
  run.

**2026-10-02, third amendment, before any extra-month day reached the
local repo:**
- 2025-09-15 and 2025-11-15 stay in the training rows, but **from a fresh
  download** in the `adsb_restofyear` pull.
- The two older Colab extracts of those days are corrupted.
  `normalise_adsb.py` reads `external-data/adsb-restofyear/` last, so the
  fresh files replace them.

**2026-10-02, fourth amendment, before any extra-month day reached the
local repo (label-free):** a thin-day rule.
- **The rule:** a listed day is used only if its last row in
  `external-data/adsb-restofyear/manifest.csv` has status `ok` or
  `tolerant` **and** its point count is ≥ 50% of the median point count of
  that month's listed days (statuses ok/tolerant).
- **Reason:** a day read with many damaged archive members skipped has
  truncated tracks. Those create artificial "first seen far from the stand"
  patterns.
- The test prints the dropped days. The rule is applied as written,
  whatever it drops.


## Results (2026-10-02, `tests/adsb_combiner_12m_test.py`, log `logs/adsb_combiner_12m_test.log`; run after commit f185b51)

- **Data:** 50 days pulled. The thin-day rule dropped 6: 10-04 (empty),
  10-07, 11-19, 11-22, 12-10, 12-28 (all tolerant reads with
  < 50% of the month median). That leaves 44 days and 253,302 labelled
  OOF-month DEP rows. OOF stack weights lgb 0.81 / cat 0.20.

**Primary: ADOPT.**

| trimmed RMSE | base (v25 method) | 12-month combiner | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan + OOF) | 288.41 | 287.22 | −1.19 | [−1.47, −0.95] | 0.000 |
| B: Jan (fit Jul + OOF) | 216.84 | 215.54 | −1.31 | [−1.81, −0.94] | 0.000 |
| pooled | 258.93 | 257.70 | −1.22 | [−1.46, −1.02] | 0.000 |
| guard: pooled full | 324.01 | 323.04 | −0.98 | [−1.30, −0.75] | 0.000 |

- **Every airport improves or holds:** EHAM −5.74, EDDM −3.58,
  LSZH −2.55, EDDF −1.83, LEBL −1.78, EGLL −1.22, LEMD −0.99, LFPG −0.66.
- **Best iterations 1,996 / 1,386** (v24: 4,614 / 1,008), so the fits are
  much more stable.
- **Reported, OOF months only** (no holdout month in training): pooled
  −1.08, Jan −1.08, Jul −1.10. Later-2025 ADS-B alone transfers to
  Jan/Jul.

**Build** (`src/post/adsb_combiner.py --m12`, log
`logs/adsb_combiner_m12_build.log`).
- Fit on both holdout months plus the 44 OOF-month days (552,057 rows);
  best_iter 2,752.
- `data/submissions/smart-jigsaw_v26.parquet` (= `smart-jigsaw_m12.parquet`).
  196,880 rows differ from v25, RMS 23.1 s.
- Keep iff the board < 264.39.

**Board: v26 = 262.70 (v25 264.39, −1.69). ADOPTED; new best.**
