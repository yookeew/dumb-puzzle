# Holdout eval — queue_treatment_mixed

- engine: **lgb**   |   generated: 2026-09-20 14:45
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7994/direct=3590 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 336.1 | 154.4 | +2.6 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 658.4 | +24.7 | 1274 | 26,528 |
| LFPG | 578.8 | +1.3 | 1052 | 39,590 |
| EGLL | 286.3 | +10.1 | 1382 | 40,210 |
| LTFM | 268.1 | -13.0 | 1070 | 46,539 |
| LEBL | 224.2 | -0.8 | 957 | 28,985 |
| LSZH | 209.1 | -0.8 | 759 | 22,289 |
| EHAM | 199.6 | +1.1 | 798 | 40,949 |
| EDDF | 190.8 | +1.8 | 875 | 36,830 |
| EDDM | 187.0 | -3.2 | 832 | 27,091 |
| LEMD | 179.9 | +11.2 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.0 | 153,660 |
| 2025-07 | 321.0 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 239.4 | 339,016 | 49.9% |
| False | 1912.6 | 5,323 | 50.1% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1263.2 | 2,391 | 9.8% |
| False | 320.3 | 341,948 | 90.2% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 197.5 | +110.9 | 34,434 |
| d1 | 569–671 | 178.0 | +83.9 | 34,434 |
| d2 | 671–770 | 178.2 | +62.3 | 34,434 |
| d3 | 770–842 | 175.5 | +57.5 | 34,434 |
| d4 | 842–923 | 191.0 | +42.7 | 34,434 |
| d5 | 923–1020 | 188.7 | +25.4 | 34,434 |
| d6 | 1020–1134 | 202.1 | +10.0 | 34,434 |
| d7 | 1134–1266 | 230.3 | -12.3 | 34,434 |
| d8 | 1266–1505 | 256.2 | -61.8 | 34,434 |
| d9 | 1505–88132 | 874.6 | -292.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 613 | 947 | 1463 | 90723 | 5 | 41 |
