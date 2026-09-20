# Holdout eval — echoclf_hour_test

- engine: **lgb**   |   generated: 2026-09-18 13:34
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=2816 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 382.4 | 156.0 | +3.4 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 926.6 | +10.7 | 1274 | 26,528 |
| LFPG | 581.0 | +6.2 | 1052 | 39,590 |
| EGLL | 288.1 | +9.2 | 1382 | 40,210 |
| LTFM | 266.0 | -0.0 | 1070 | 46,539 |
| LEBL | 223.6 | +1.4 | 957 | 28,985 |
| LSZH | 211.6 | -0.8 | 759 | 22,289 |
| EHAM | 200.6 | -2.7 | 798 | 40,949 |
| EDDF | 192.1 | +2.6 | 875 | 36,830 |
| EDDM | 189.8 | -3.7 | 832 | 27,091 |
| LEMD | 180.4 | +10.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 363.1 | 153,660 |
| 2025-07 | 397.3 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 258.2 | 339,016 | 44.9% |
| False | 2283.5 | 5,323 | 55.1% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1989.1 | 2,412 | 19.0% |
| False | 345.5 | 341,927 | 81.0% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 197.0 | +112.8 | 34,434 |
| d1 | 569–671 | 178.1 | +86.6 | 34,434 |
| d2 | 671–770 | 173.4 | +64.1 | 34,434 |
| d3 | 770–842 | 176.7 | +60.4 | 34,434 |
| d4 | 842–923 | 193.0 | +45.5 | 34,434 |
| d5 | 923–1020 | 191.0 | +28.0 | 34,434 |
| d6 | 1020–1134 | 201.2 | +12.4 | 34,434 |
| d7 | 1134–1266 | 231.6 | -10.1 | 34,434 |
| d8 | 1266–1505 | 249.9 | -60.5 | 34,434 |
| d9 | 1505–88132 | 1049.0 | -305.1 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 615 | 950 | 1466 | 70257 | 0 | 43 |
