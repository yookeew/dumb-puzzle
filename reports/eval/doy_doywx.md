# Holdout eval — doy_doywx

- engine: **lgb**   |   generated: 2026-09-22 13:49
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7997/direct=2758 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 337.0 | 155.0 | +0.1 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 661.0 | +24.7 | 1274 | 26,528 |
| LFPG | 579.8 | +0.5 | 1052 | 39,590 |
| EGLL | 288.4 | +2.4 | 1382 | 40,210 |
| LTFM | 268.1 | -17.0 | 1070 | 46,539 |
| LEBL | 224.1 | +1.2 | 957 | 28,985 |
| LSZH | 211.4 | -2.9 | 759 | 22,289 |
| EHAM | 199.6 | -3.0 | 798 | 40,949 |
| EDDF | 191.8 | -2.2 | 875 | 36,830 |
| EDDM | 186.7 | -1.5 | 832 | 27,091 |
| LEMD | 179.6 | +9.5 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.6 | 153,660 |
| 2025-07 | 322.2 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 240.3 | 339,016 | 50.0% |
| False | 1916.1 | 5,323 | 50.0% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1471.4 | 2,451 | 13.6% |
| False | 314.5 | 341,888 | 86.4% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.1 | +110.7 | 34,434 |
| d1 | 569–671 | 179.2 | +83.5 | 34,434 |
| d2 | 671–770 | 177.1 | +61.4 | 34,434 |
| d3 | 770–842 | 177.1 | +56.7 | 34,434 |
| d4 | 842–923 | 193.2 | +41.4 | 34,434 |
| d5 | 923–1020 | 191.1 | +23.5 | 34,434 |
| d6 | 1020–1134 | 202.3 | +7.1 | 34,434 |
| d7 | 1134–1266 | 230.6 | -16.3 | 34,434 |
| d8 | 1266–1505 | 259.1 | -67.3 | 34,434 |
| d9 | 1505–88132 | 875.6 | -299.7 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 612 | 945 | 1458 | 90707 | 5 | 42 |
