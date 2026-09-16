# Holdout eval — test_direct

- engine: **lgb**   |   generated: 2026-09-16 17:54
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 2811 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 383.9 | 156.0 | +5.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 938.5 | +8.9 | 1274 | 26,528 |
| LFPG | 579.7 | +8.5 | 1052 | 39,590 |
| EGLL | 286.0 | +12.5 | 1382 | 40,210 |
| LTFM | 267.2 | +2.1 | 1070 | 46,539 |
| LEBL | 223.9 | +2.9 | 957 | 28,985 |
| LSZH | 209.8 | +0.6 | 759 | 22,289 |
| EHAM | 197.1 | -1.8 | 798 | 40,949 |
| EDDF | 191.4 | +4.7 | 875 | 36,830 |
| EDDM | 187.7 | -3.1 | 832 | 27,091 |
| LEMD | 180.1 | +12.0 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 364.0 | 153,660 |
| 2025-07 | 399.2 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 260.6 | 339,016 | 45.4% |
| False | 2282.1 | 5,323 | 54.6% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 2118.2 | 2,406 | 21.3% |
| False | 341.9 | 341,933 | 78.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 199.2 | +113.1 | 34,434 |
| d1 | 569–671 | 178.5 | +86.7 | 34,434 |
| d2 | 671–770 | 174.4 | +64.7 | 34,434 |
| d3 | 770–842 | 178.4 | +61.1 | 34,434 |
| d4 | 842–923 | 196.9 | +46.8 | 34,434 |
| d5 | 923–1020 | 191.4 | +29.4 | 34,434 |
| d6 | 1020–1134 | 202.0 | +14.3 | 34,434 |
| d7 | 1134–1266 | 231.2 | -7.6 | 34,434 |
| d8 | 1266–1505 | 248.5 | -57.4 | 34,434 |
| d9 | 1505–88132 | 1053.0 | -301.5 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 39 | 615 | 951 | 1469 | 72546 | 0 | 43 |
