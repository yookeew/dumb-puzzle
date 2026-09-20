# Holdout eval — echoclf_hour_test

- engine: **lgb**   |   generated: 2026-09-18 12:09
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=3310 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 379.3 | 156.4 | +3.5 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 908.0 | +10.0 | 1274 | 26,528 |
| LFPG | 581.1 | +6.1 | 1052 | 39,590 |
| EGLL | 289.0 | +10.4 | 1382 | 40,210 |
| LTFM | 267.6 | +0.0 | 1070 | 46,539 |
| LEBL | 224.4 | +1.1 | 957 | 28,985 |
| LSZH | 212.0 | -0.5 | 759 | 22,289 |
| EHAM | 200.5 | -3.0 | 798 | 40,949 |
| EDDF | 192.5 | +3.3 | 875 | 36,830 |
| EDDM | 189.8 | -4.4 | 832 | 27,091 |
| LEMD | 180.6 | +11.1 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 366.6 | 153,660 |
| 2025-07 | 389.2 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 253.1 | 339,016 | 43.8% |
| False | 2286.2 | 5,323 | 56.2% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1993.8 | 2,406 | 19.3% |
| False | 341.9 | 341,933 | 80.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.4 | +112.8 | 34,434 |
| d1 | 569–671 | 178.4 | +86.5 | 34,434 |
| d2 | 671–770 | 173.4 | +64.2 | 34,434 |
| d3 | 770–842 | 176.9 | +60.5 | 34,434 |
| d4 | 842–923 | 194.3 | +45.7 | 34,434 |
| d5 | 923–1020 | 189.0 | +28.0 | 34,434 |
| d6 | 1020–1134 | 202.3 | +12.5 | 34,434 |
| d7 | 1134–1266 | 231.6 | -9.8 | 34,434 |
| d8 | 1266–1505 | 249.7 | -60.0 | 34,434 |
| d9 | 1505–88132 | 1037.1 | -304.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 615 | 950 | 1467 | 75767 | 0 | 42 |
