# Stage 2 (features, Option A) + Stage 3a (LightGBM on d) — first real model

Run 2026-09-02. `src/features/build_features.py`, `src/models/train_lgbm.py`.
Target: d = AOBT − SOBT (pushback delay); recover taxi = sched_takeoff_offset − d̂, clip [60, 5400].

## Holdout (train on 10 months, score Jan + Jul 2025)

| | RMSE (s) |
|---|---|
| **LightGBM on d** | **300.9** |
| group-mean baseline (baseline_v1) | 400.3 |
| per-airport mean taxi | 436.4 |

### Per airport
| airport | RMSE | n |
|---|---|---|
| LEMD | 197.0 | 35,328 |
| EDDM | 198.5 | 27,091 |
| LSZH | 221.7 | 22,289 |
| LEBL | 246.1 | 28,985 |
| EHAM | 262.0 | 40,949 |
| EDDF | 262.2 | 36,830 |
| LTFM | 281.5 | 46,539 |
| LFPG | 296.7 | 39,587 |
| EGLL | 307.4 | 40,210 |
| **LIRF** | **601.3** | 26,486 |

### Per month
Jan 254.2 · Jul 333.9  (summer traffic heavier)

### Per true-taxi decile
d0–d8 RMSE 184–261; **d9 (taxi > 1505 s) RMSE 709** — the long tail dominates squared error.

## Top features by gain
1. sched_takeoff_offset (the T − SOBT offset)
2. eobt_delay, iobt_delay, lobt_delay — planned-pushback deltas vs schedule
3. aobt3_vs_eobt, aobt3_taxi — the toggleable AOBT_3 block
4. STAND_mvt, median_taxi_prior, unimpeded_taxi — physical baseline
5. U_sec, sched_ground, inbound_arr_delay — rotation
6. arr_rwy_config, dep_rwy_config, prev_gap, hour — congestion / config

Congestion & config features contribute but are not dominant yet.

## Submission
`data/submissions/lgbm_v1.parquet` — 215,876 rows, validated (exact MVT_ID match).
Pred median 974 s, p01–p99 = 326–2605 s. 644 rows pinned at floor (60 s), 110 at ceil.

## Known issues / next
- **LIRF 2× everyone else** — investigate (stand coding? config inference? towing?).
- **d9 tail** — d has heavy outliers (diverted / data-error flights); L2 chases them.
  Try clipping/huber on d, or a dedicated long-taxi treatment.
- Floor pinning (644 rows at 60 s) — model overpredicts d for short-offset flights;
  raise floor or add a min-taxi prior.
- AOBT_3 block is always on — make it a real toggle and measure its isolated contribution.
- Not yet done: per-airport residual model (3a-b), queue refinement (3b),
  distributional head (3c), ensemble.
