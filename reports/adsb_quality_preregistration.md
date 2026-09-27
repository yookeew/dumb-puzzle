# ADS-B blend quality modulation — pre-registration

Written and committed 2026-09-27, before any holdout result of this design.
Thresholds come from exploration on 2025-09-15 and 2025-11-15 (training
months, outside the holdout) only.

## Exploration (training days, ~2,900 appear/dwell rows, per-airport-tier lag removed)

ADS-B pushback error spread (RMSE) by per-row quality:

| tier | good | poor |
|---|---|---|
| appear | first sighting ≤ 70 m from stand: 39–105 s | > 70 m: 159 s |
| appear | first-sighting speed ≤ 5 kt: 77–102 s | > 5 kt: 188 s |
| dwell | gap after last stationary sample ≤ 40 s: 121–162 s | > 40 s: ~204 s |
| dwell | pushback sample ≤ 70 m from stand: 132–148 s | > 70 m: 231 s |

## Definition

Quality, from detector fields (`src/link/adsb_pushback.py`):

- **appear good** iff `adsb_pb_dist_m ≤ 70` **and** `adsb_pb_gs ≤ 5`
- **dwell good** iff `adsb_pb_dist_m ≤ 70` **and** `adsb_pb_gap_s ≤ 40`
  (a null gap counts as poor)

The production blend (§39: per-airport lag, per-airport-tier cell weights,
LIRF excluded) is kept as it is. Each eligible row's weight is multiplied
by a factor `q[tier, quality]` (4 factors), fit by least squares on the fit
rows given the production weights. The product is clipped to [0, 1].

## Evaluation

- **Base:** v17 on the holdout, cross-fit (stack → blend → partial). The
  quality change touches only appear/dwell rows. Partial-track rows are
  disjoint and unchanged.
- **Cross-fit:** A fits on Jan and scores Jul; B fits on Jul and scores
  Jan. Both the blend parameters and the q factors come from the fit month.
- **Rule:** adopt iff both months' point deltas < 0 and pooled (airport,
  day) cluster-bootstrap P(worse) < 0.05 (3000 resamples, seed 0).
- **Reported:** the q factors per direction, and rows / RMSE by tier ×
  quality.

## Amendments

(none)
