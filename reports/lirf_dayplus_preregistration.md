# LIRF "+24 h" label adjustment — pre-registration

Written and committed 2026-09-27, before computing any holdout effect of the
adjustment. The exploration so far is described below; the holdout was used
only to *look at* current predictions on the segment, not to fit anything.

## Finding

The LIRF block-time feed sometimes stamps the real pushback clock time onto
the **scheduled** date. For flights that take off many hours after their
scheduled time, the label then becomes real taxi + 86,400 s.

- All 14 labels in the 24 h − 10 min to 24 h + 90 min band in 2025 are
  NM-unmatched; 13 are at LIRF.
- For 86% of them the block date equals the scheduled date, and block +
  1 day falls a median 15 min before takeoff.
- On the 10 training months, LIRF NM-unmatched rows delayed ≥ 12 h have
  labels that are: schedule echo 12, +24 h 6, normal 0.
- At other airports the equivalent rows are all normal (38 of 38).

The production model already predicts the echo value well for long LIRF
delays, but it under-predicts the +24 h rows by up to ~35,000 s.

## Definition

- **Segment:** `ADEP_mvt == LIRF` **and** NM-unmatched (`has_aobt3 ==
  False`; in ranking, `AOBT_3_flt` null) **and** `T − SOBT ≥ 12 h`. All of
  these are available at ranking time.
- **p:** the share of +24 h labels (taxi within 86,400 − 600 … 86,400 +
  5,400 s) among segment rows in the **10 training months** (all of 2025
  except Jan and Jul): 6 / 18 = 0.333.
- **Typical real taxi `t0`:** the median label of LIRF NM-unmatched rows
  with taxi ≤ 2 h in the training months.
- **Adjustment:** `new = (1 − p) · pred + p · (86,400 + t0)` on segment rows.
  All other rows are unchanged.

## Evaluation

- **Base:** the v19 pipeline's holdout prediction (cross-fit stack; LIRF
  rows are untouched by the ADS-B stages).
- **Rows affected:** holdout segment rows only (expected ~12). Nothing is
  fitted on the holdout.
- **Rule:** adopt iff segment squared error falls in **both** holdout months
  that contain segment rows, **and** overall holdout RMSE falls. With ~12
  rows no bootstrap is meaningful; the decision rests on the mechanism, and
  the leaderboard A/B is the confirmation.
- **Also reported:** segment rows individually, and the p sensitivity
  (0.25 / 0.33 / 0.40) as information only; p stays 0.333.

## Amendments

(none)
