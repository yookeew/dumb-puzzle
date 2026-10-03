# ADS-B combiner retrained with all of 2025 — pre-registration

Written 2026-10-03, after v26 (Sep–Dec, board 262.70) and **before any
Feb–Aug 2025 ADS-B day was normalised, detected or used**. Seven such days
(04-01, 05-01, 06-01, 06-02, 08-01, 08-02, 08-03) arrived early. They sit
untouched in `external-data/adsb-later/`. Changes after results go under
Amendments, dated.

## Why

Adding Sep–Dec training rows to the combiner gave −1.22 trimmed on the
holdout and −1.69 on the board (§64), with no airport worse. The fits also
became more stable. The other six months (Feb–Jun, Aug) have the same
honest month-wise OOF base predictions, so their ADS-B is the obvious
next increment.

## Data

- **Days:** the same sample as v26, applied to Feb, Mar, Apr, May, Jun and
  Aug 2025: days 01, 02, 03, 04, 07, 10, 13, 16, 19, 22, 25 and 28, as many
  as the Colab pull delivers.
- **Pipeline:** fetched into `adsb_restofyear/` by the existing Colab cell.
  The `adsb-later/` days are moved there too. Then
  `src/ingest/normalise_adsb.py` and `src/link/adsb_pushback.py --v4`.
- **The same thin-day rule** (fourth amendment of
  `reports/adsb_combiner_12m_preregistration.md`): status ok/tolerant and
  ≥ 50% of the month's median points.
- **The months, the day sample and the rule are fixed now.** The gate runs
  once, on whatever the pull has delivered when it's run. The number of
  days per month is reported.

## Treatment and base (`tests/adsb_combiner_fullyear_test.py`)

- **Base = v26's method, cross-fit:** the restricted combiner trained on
  the other holdout month plus the Sep–Dec days.
- **Treatment:** identical, but trained on the other holdout month plus
  Feb–Jun, Aug and Sep–Dec days.
- **Same for both:** hyperparameters, inputs, early stopping, OOF stack
  and restricted application.

## Rule

Trimmed RMSE decides (labels ≤ 5 h, the fixed 31-row set). Adopt iff all
of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- days kept per month;
- training rows;
- best iterations;
- per airport.

**If it passes:** build v27 = `src/post/adsb_combiner.py` trained on both
holdout months plus all the kept days. **Keep iff the board < 262.70
(v26).**

**If it fails: stop.** No month subsets or weighting.

## Amendments

(none)

## Results (2026-10-03, `tests/adsb_combiner_fullyear_test.py`, log `logs/adsb_combiner_fullyear_test.log`; run after commit d866ed8)

- **Data:** all 122 planned days arrived. The thin-day rule dropped 7: the
  six Sep–Dec days from §64 plus 08-10 (tolerant read, 211k points). That
  leaves 115 days and 660,242 labelled OOF-month rows (Sep–Dec alone:
  253,302). OOF stack weights 0.56 / 0.45.
- The base reproduces v26's holdout numbers exactly (Jul 287.22, Jan
  215.54).

**Primary: ADOPT.**

| trimmed RMSE | base (v26 method) | full year | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan + OOF) | 287.22 | 286.82 | −0.40 | [−0.52, −0.28] | 0.000 |
| B: Jan (fit Jul + OOF) | 215.54 | 214.90 | −0.63 | [−0.97, −0.37] | 0.000 |
| pooled | 257.70 | 257.22 | −0.48 | [−0.63, −0.35] | 0.000 |
| guard: pooled full | 323.04 | 322.65 | −0.38 | [−0.54, −0.27] | 0.000 |

- **Per airport:** EHAM −2.66, EDDM −1.66, LSZH −1.66, LEBL −1.05,
  EDDF −0.68, EGLL −0.25, LEMD −0.18, LFPG +0.05.
- **Best iterations 4,024 / 2,191** (v26 method: 1,996 / 1,386).
- **Diminishing returns:** Sep–Dec gave −1.22; the other six months add
  −0.48.

**v27 build** (`src/post/adsb_combiner.py --full`, log
`logs/adsb_combiner_fullyear_build.log`).
- Fit on both holdout months plus the 115 days (927,045 rows); best_iter
  3,055.
- `data/submissions/smart-jigsaw_v27.parquet`: 194,616 rows differ from
  v26, RMS 15.8 s.
- Keep iff the board < 262.70.

**Board: v27 = 262.13 (v26 262.70, −0.57). ADOPTED; new best.**
