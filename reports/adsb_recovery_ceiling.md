# ADS-B pushback recovery — ceiling check and flight-level validation

Two-step check, both run against data already on disk (production 2025
holdout + one day of ADS-B probe data, `external-data/adsb/adsb_20250115.parquet`,
2025-01-15, 2.1 GB via adsb.lol), before committing to a full 400 GB
historical ADS-B download. No further data pulled.

Context: `AOBT_3_flt` (already a feature, `aobt3_taxi`) is a noisy read on
the same off-block event as `BLOCK_TIME_UTC_mvt` — per
`reports/step0_audit.md` §4, 21.0% within ±1 min, 74.6% within ±5 min, naive
taxi from it alone RMSE 384.9s vs true-taxi sd 417.5s. ADS-B's only possible
value is denoising that channel further via the true pushback moment, not
adding a new signal from nothing.

## 1. Best-case ceiling (`tests/adsb_ceiling_test.py`)

One-day ground-coverage probe found ADS-B ground positions at only 6 of 10
departure airports — EHAM, EGLL, EDDM, EDDF, LSZH, LEBL. LFPG, LEMD, LIRF,
LTFM had **zero** aircraft with a ground position that day, not thin
coverage. Against the current production model (`target="mixed"`, holdout
RMSE 335.6s):

- Covered airports: 56.1% of ranking DEP rows, but only **25.0%** of
  holdout squared error (they're the easier airports, RMSE 187–289s each).
- Oracle ceiling (perfect recovery, zero error, on covered rows only):
  335.6s → 290.6s, **-45.0s (-13.4%)**.
- Sensitivity: 50% coverage → -21.7s (-6.5%); 25% coverage → -10.7s (-3.2%).
- LIRF (29.1% of total squared error, the single largest contributor) has
  zero ADS-B coverage — structurally out of reach regardless.

This established the *best possible* case assuming perfect recovery. The
50%/25% coverage figures were a guess pending an actual measurement — step 2
replaces the guess with a number.

## 2. Flight-level recovery rate and accuracy (`tests/adsb_recovery_test.py`)

Neither dataset carries a callsign/tail-to-flight join key in the extracted
columns (ADS-B probe: `hex, reg, ts, lat, lon, alt, gs, airport` — no
callsign message was kept), so matching uses the one anchor both sides
share and neither blanks: **movement time**. Per (airport, hex) aircraft:

1. `alt == -1` is a categorical ADS-B surface-position flag, not a computed
   altitude threshold — so ground/airborne segmentation is exact and
   airport-elevation-independent. A ground run immediately followed by an
   airborne run is a candidate departure; its last ground sample is the
   ADS-B takeoff-time proxy.
2. Within that run, the **last** sub-run of ≥3 min with ground speed ≤2 kt
   is treated as the gate/stand dwell; the first sample after it ends is
   the pushback estimate. No such dwell (track picked up already moving) →
   `censored` (a takeoff was seen but no usable pushback estimate).
3. Candidate ADS-B takeoffs are matched 1:1 to true DEP movements by
   nearest `MVT_TIME_UTC_mvt`, same airport, ±180s tolerance (greedy,
   closest pairs first).

### Recovery rate

| airport | true DEP | ADS-B candidates | matched | censored | **usable** | **recovery%** |
|---|---|---|---|---|---|---|
| EHAM | 512 | 522 | 485 | 342 | 143 | 27.9% |
| EGLL | 632 | 536 | 465 | 400 | 65 | 10.3% |
| EDDM | 361 | 364 | 358 | 80 | 278 | 77.0% |
| EDDF | 513 | 509 | 448 | 398 | 50 | 9.7% |
| LSZH | 292 | 293 | 265 | 163 | 102 | 34.9% |
| LEBL | 319 | 338 | 312 | 280 | 32 | 10.0% |
| **TOTAL** | 2,629 | | | | 670 | **25.5%** |

Recovery lands at the **pessimistic** end of the 25–50% sensitivity range
used in step 1, not the middle or optimistic end. Coverage is dominated by
whether an aircraft shows a clean ≥3-minute near-zero-speed dwell in ADS-B
at all — most departures at EGLL/EDDF/LEBL never do (~90% censored),
consistent with the known gate/apron ADS-B reception problem (buildings,
multipath) rather than a detector bug specific to those airports.

**EDDM's 77% is an outlier and looks like a different mechanism, not
better coverage** — see accuracy below.

### Accuracy (matched + uncensored only, n=670)

| | value |
|---|---|
| median (adsb_pushback − true block) | **+7.84 min** |
| p10 / p90 | −0.22 / +31.84 min |
| \|d\|≤60s | 18.2% |
| \|d\|≤300s | 31.8% |
| RMSE | **2,179.6 s** |

Compare to `AOBT_3_flt` (full year, all 10 airports, `reports/step0_audit.md`
§4): median −0.8 min, \|d\|≤60s 21.0%, \|d\|≤300s 74.6%. **The naive ADS-B
detector is worse on every axis** — less accurate at 1 minute, dramatically
less accurate at 5 minutes, and with a real, non-trivial positive bias
(detected pushback lags true off-block by 8 minutes on median, not merely
noisier around zero).

Concretely: true taxi sd for these 2,629 covered-airport departures that day
is 538.8s (predicting the mean beats a wild guess by this margin). The
naive-ADS-B taxi estimate's error equals `diff_sec` exactly (since
`taxi = MVT_TIME − pushback` and MVT_TIME is exact), so its RMSE is
**2,179.6s — over 4x worse than just predicting the mean**, before any
model even sees it.

### EDDM anomaly — plausible mechanism, not chased further this session

EDDM's ground-row count (165,677) is ~4-5x every other covered airport, its
censored rate is by far the lowest (22% vs 62-88% elsewhere), and its
median offset is the second-largest positive bias (+21.9 min). Read
together, this looks like **de-icing pad holds**, not gate dwells: Munich in
January routinely has aircraft push back, taxi to a de-icing pad, and sit
there stationary for several minutes before proceeding — a second ≥3-minute
near-zero-speed dwell downstream of the real pushback that the detector's
"last dwell before takeoff" rule picks up instead of the gate departure.
That would explain all three observations at once (more usable-looking
dwells, lower censoring, larger positive bias) without needing a
per-airport code difference. **Not verified further this session** — flagged
so a future pass doesn't mistake EDDM's high recovery rate for the detector
working better there; it may be finding the wrong event.

## Conclusion

This is a first-pass, unrefined detector (simple ground-speed threshold +
longest-dwell heuristic on categorical surface data), not a mature pushback
algorithm — a stand/gate-position-aware version, or one that explicitly
excludes de-icing-pad dwells, would likely do better. But two structural
findings don't depend on refining the detector:

1. **Coverage is the harder constraint.** 4 of 10 airports have zero ADS-B
   ground data at all (LIRF included — 29.1% of total error). Among the 6
   covered airports, on this one day, only 25.5% of true departures yield
   any usable pushback signal — this is a real ground-truth measurement,
   not a guess, and it sits at the bottom of the sensitivity range already
   modeled in step 1 (§1: 25% coverage → -10.7s/-3.2% ceiling, and that
   assumed *perfect* accuracy at that coverage).
2. **Where recovered, the naive signal is currently worse than useless** —
   RMSE 2,179.6s vs a 538.8s do-nothing baseline, and worse than the
   `AOBT_3_flt` feature already in the model on every accuracy axis. Feeding
   this into the model as-is would hurt, not help; it would need
   substantial detector engineering (stand-position matching, de-icing-hold
   exclusion, tighter dwell criteria) before it could even match the
   feature already in production, let alone denoise it.

Net: even the disk-only, zero-marginal-cost version of this check overturns
the optimistic framing. Recommend **not** proceeding to the full 400 GB
download without first validating that a materially better pushback
detector is achievable on this one day of data — the current one doesn't
clear the bar the existing feature already sets.
