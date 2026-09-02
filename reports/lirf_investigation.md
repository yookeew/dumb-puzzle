# Why LIRF (Rome–Fiumicino) is the worst airport

Holdout RMSE by airport (Stage 3a global model): LIRF **601 s** vs 197–307 s for
the other nine. This is not a modelling bug — LIRF's target is genuinely harder,
for two compounding reasons.

## 1. LIRF's taxi-out distribution has a huge right tail

| airport | mean taxi | **std** | p99 |
|---|---|---|---|
| **LIRF** | 1195 | **1332** | **4019** |
| EGLL | 1364 | 422 | 2701 |
| LFPG | 1020 | 453 | 2409 |
| LEMD | 1015 | 313 | 1932 |
| … all others | — | 300–450 | 1700–2700 |

LIRF's taxi sd is **3–4× every other airport**. 3.2 % of LIRF departures have
recorded taxi > 40 min (max 131 167 s ≈ 36 h). These long values cluster in
**July** (worst month, and a ranking month), hours 08–14, runway 25 (single-runway
ops → queueing).

## 2. ~18 % of LIRF rows: the block-time feed echoes the schedule

`d = BLOCK_TIME_UTC_mvt − SCHED_TIME_UTC_mvt` (off-block minus scheduled):

| airport | fraction with \|d\| < 30 s ("echo") |
|---|---|
| **LIRF** | **18.1 %** |
| LEBL | 5.6 % |
| every other airport | 2.9–5.1 % |

For LIRF "echo" rows, `BLOCK_TIME_UTC_mvt` is a near-exact copy of
`SCHED_TIME_UTC_mvt` (off by seconds). The recorded taxi then equals
**takeoff − scheduled**, i.e. it absorbs the entire departure delay:

- LIRF echo rows: `RMSE(taxi, takeoff − SCHED)` = **3 s** (taxi ≡ the schedule→takeoff gap)
- LIRF echo rows: `RMSE(taxi, takeoff − AOBT_3)` = 922 s (AOBT_3 reflects the *real* pushback, which is *not* what the label records)
- LIRF non-echo rows: `RMSE(taxi, takeoff − AOBT_3)` = 440 s (normal)

So for ~1 in 5 LIRF departures the "taxi-out" label is really "schedule delay +
taxi", and AOBT_3 (the actual off-block) disagrees with it by design.

### The echo rows are partially predictable

| signal | echo rate |
|---|---|
| IOBT delay > 60 min | **0 %** |
| IOBT delay ≤ 0 | 22 % |
| AOBT_3 / EOBT_1 / FLIGHT_ID missing | 48 % |
| night hours (00–01, 22–23) | 26–48 % |
| midday (12–16) | 14–16 % |
| operator (anonymised) | ranges **1.7 % → 54 %** across operators |
| stand prefix "2" vs "4" | 32 % vs 11 % |

`AIRCRAFT_OPERATOR_flt`, `STAND_mvt`, `IOBT`/`EOBT` deltas and hour are already
model features, so the structure is learnable — but a single pooled model spends
its capacity on the 9 well-behaved airports. LIRF's `d` distribution (sd ≈ 1780 s
vs 300–500 s elsewhere) is simply a different problem.

## What this implies for the model

- **The ranking set has the same feed behaviour** (Jan/Jul 2026 LIRF), so RMSE
  *rewards* predicting the inflated values. We must model them, not discard them.
- **Raise the training label ceiling** from 5400 s so the tail is learned
  (done: [0, 14400] s; only the ~36 h poison rows dropped).
- **Per-airport residual head** — give LIRF its own correction on top of the
  global `d` model, trained on out-of-fold global predictions. This is the
  targeted fix; results in `reports/stage3a_resid.md`.
- Longer term: an explicit "echo probability" feature (per airport×operator×
  stand historical echo rate, out-of-fold) and possibly a two-part model
  (P(echo) × schedule-delay-taxi + (1−P) × normal-taxi).
