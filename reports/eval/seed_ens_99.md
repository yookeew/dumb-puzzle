# Holdout eval — seed_ens_99

- engine: **lgb**   |   generated: 2026-09-19 21:25
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=3236 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 379.3 | 156.4 | +3.3 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 908.2 | +9.1 | 1274 | 26,528 |
| LFPG | 580.9 | +5.3 | 1052 | 39,590 |
| EGLL | 288.8 | +9.4 | 1382 | 40,210 |
| LTFM | 267.8 | +0.4 | 1070 | 46,539 |
| LEBL | 224.4 | +1.4 | 957 | 28,985 |
| LSZH | 211.7 | -0.3 | 759 | 22,289 |
| EHAM | 200.6 | -2.9 | 798 | 40,949 |
| EDDF | 192.1 | +3.2 | 875 | 36,830 |
| EDDM | 190.1 | -4.4 | 832 | 27,091 |
| LEMD | 180.8 | +10.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 366.5 | 153,660 |
| 2025-07 | 389.2 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 252.9 | 339,016 | 43.8% |
| False | 2287.3 | 5,323 | 56.2% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1995.0 | 2,406 | 19.3% |
| False | 341.8 | 341,933 | 80.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.3 | +112.7 | 34,434 |
| d1 | 569–671 | 178.6 | +86.6 | 34,434 |
| d2 | 671–770 | 173.5 | +64.3 | 34,434 |
| d3 | 770–842 | 176.7 | +60.4 | 34,434 |
| d4 | 842–923 | 194.0 | +45.7 | 34,434 |
| d5 | 923–1020 | 188.4 | +27.8 | 34,434 |
| d6 | 1020–1134 | 201.5 | +12.3 | 34,434 |
| d7 | 1134–1266 | 231.4 | -10.0 | 34,434 |
| d8 | 1266–1505 | 249.9 | -60.7 | 34,434 |
| d9 | 1505–88132 | 1037.4 | -306.2 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 28 | 615 | 950 | 1465 | 75754 | 0 | 42 |
