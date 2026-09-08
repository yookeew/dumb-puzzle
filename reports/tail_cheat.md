# Tail cheat test — Jan+Jul 2025 holdout

## A. Real model error by taxi band

_Blocked — `reports\eval\lgb_v2_holdout.parquet` not found. It is written by fit.py's run() now; produce it with one Colab run, then re-run this script._

## B. The cheat — ceiling on a two-part model

Flagged rows get a part-2 prediction; the rest get an oracle (true taxi). Overall RMSE = the best a two-part model could do with that flag + that part-2. Reference outcome: taxi > 2400s.

No-cheat baselines: honest median **426.7**, per-airport median **442.1**, oracle body **0.0**. Current lgb model ≈ **311**.

### Flag coverage vs taxi>2400s

| flag | n flagged | precision | recall |
| --- | --- | --- | --- |
| echo (BLOCK≈SCHED, <30s) | 17,308 | 4% | 15% |
| label >> real taxi (gap>600s, AOBT_3 present) | 14,897 | 22% | 69% |
| AOBT_3 null | 5,280 | 5% | 5% |
| AOBT_3 null OR echo | 22,419 | 4% | 18% |
| ORACLE outcome: taxi>2400s | 4,791 | 100% | 100% |

### Overall RMSE: oracle body + (flag → part-2)

| flag \ part-2 | offset (MVT−SCHED) | MVT−AOBT_3 (clip 0,4h) | airport median | honest baseline |
| --- | --- | --- | --- | --- |
| echo (BLOCK≈SCHED, <30s) | 1.7 | 183.6 | 183.1 | 182.0 |
| label >> real taxi (gap>600s, AOBT_3 present) | 608.0 | 286.9 | 277.6 | 272.8 |
| AOBT_3 null | 805.9 | 147.2 | 147.2 | 146.3 |
| AOBT_3 null OR echo | 805.9 | 205.3 | 204.8 | 202.9 |
| ORACLE outcome: taxi>2400s | 356.3 | 267.4 | 295.0 | 291.4 |

- Every cell assumes a **perfect body** and **perfect flag detection** — it is an upper bound.
- If the best cell is ~290 the two-part model is not the fix; ~250 and it is. Compare to the current ≈311.

## C. AOBT_3 — input already? nullness a good detector?

`AOBT_3_flt` **is** a model input: build_features emits `aobt3_taxi = MVT − AOBT_3` and `aobt3_vs_eobt = AOBT_3 − EOBT_1`. It does **not** emit `AOBT_3 − SCHED` (the real schedule deviation) or an explicit `aobt3_missing` flag — both are cheap adds.

| quantity | value |
| --- | --- |
| AOBT_3 null | all rows | 1.5%  (n=344,294) |
| AOBT_3 null | echo | 1.0%  (n=17,308) |
| AOBT_3 null | taxi>2400 | 5.1%  (n=4,791) |
| AOBT_3 null | taxi>7200 | 65.4%  (n=104) |
| echo | AOBT_3 null | 3.2%  (n=5,280) |
| echo | AOBT_3 present | 5.1%  (n=339,014) |
| taxi>2400 | AOBT_3 null | 4.6%  (n=5,280) |
| taxi>2400 | AOBT_3 present | 1.3%  (n=339,014) |

### C (ranking) — does the feed behave the same in 2026?

215,876 ranking DEP rows. BLOCK blanked: 100.0%  (echo flag unavailable here).

- AOBT_3 null overall: **1.5%** (holdout DEP was in part C above)

| airport | n | AOBT_3 null % | offset>2400 % | AOBT_3 null & offset>1800 % |
| --- | --- | --- | --- | --- |
| EHAM | 38,182 | 3.2% | 25.7% | 2.5% |
| LSZH | 9,995 | 3.0% | 26.2% | 2.3% |
| LFPG | 18,129 | 1.8% | 33.6% | 1.7% |
| LTFM | 22,396 | 1.5% | 27.6% | 0.6% |
| EDDF | 36,317 | 1.2% | 25.1% | 1.1% |
| EDDM | 10,953 | 1.1% | 22.6% | 1.1% |
| LIRF | 11,061 | 1.0% | 17.8% | 0.8% |
| LEBL | 12,201 | 0.6% | 17.9% | 0.6% |
| EGLL | 39,840 | 0.6% | 25.2% | 0.6% |
| LEMD | 16,802 | 0.4% | 23.9% | 0.4% |

Last column ~ the share of ranking rows a null-based classifier would flag as inflated. Compare `offset>2400 %` to the holdout tail rate (1.4%) to see if 2026 is heavier.

## D. Is the inflation `AOBT_3 − BLOCK` predictable?

AOBT_3 present: 1,723,531 train / 339,014 holdout rows (98.5% of holdout).

- inflation p1/p50/p90/p99 = -837 / -53 / 357 / 1382 s;  |infl|<60s for **19%**, infl>600s for **4%**

- predict inflation with 0: RMSE 427 s;  with (airport,hour) median: RMSE 411 s  → NOT meaningfully predictable from coarse history

- taxi RMSE on the AOBT_3-present holdout: `MVT−AOBT_3` alone **409.4**;  `MVT−AOBT_3 + inflation_median` **392.0**  (current lgb ≈ 311 on all rows)
