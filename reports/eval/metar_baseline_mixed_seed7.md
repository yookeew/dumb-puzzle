# Holdout eval — metar_baseline_mixed_seed7

- engine: **lgb**   |   generated: 2026-09-20 16:10
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=8000/direct=2405 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 337.6 | 156.0 | +5.2 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 662.0 | +26.3 | 1274 | 26,528 |
| LFPG | 580.9 | +6.6 | 1052 | 39,590 |
| EGLL | 288.5 | +10.0 | 1382 | 40,210 |
| LTFM | 267.4 | +0.6 | 1070 | 46,539 |
| LEBL | 224.0 | +1.8 | 957 | 28,985 |
| LSZH | 211.6 | -0.3 | 759 | 22,289 |
| EHAM | 200.4 | -1.6 | 798 | 40,949 |
| EDDF | 192.3 | +3.2 | 875 | 36,830 |
| EDDM | 190.6 | -4.1 | 832 | 27,091 |
| LEMD | 180.6 | +11.4 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 356.1 | 153,660 |
| 2025-07 | 322.0 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 241.0 | 339,016 | 50.2% |
| False | 1916.8 | 5,323 | 49.8% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1472.5 | 2,406 | 13.3% |
| False | 315.5 | 341,933 | 86.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 201.3 | +114.1 | 34,434 |
| d1 | 569–671 | 182.7 | +87.5 | 34,434 |
| d2 | 671–770 | 178.8 | +65.1 | 34,434 |
| d3 | 770–842 | 179.6 | +61.2 | 34,434 |
| d4 | 842–923 | 197.3 | +46.4 | 34,434 |
| d5 | 923–1020 | 192.5 | +28.9 | 34,434 |
| d6 | 1020–1134 | 204.5 | +12.9 | 34,434 |
| d7 | 1134–1266 | 235.0 | -9.4 | 34,434 |
| d8 | 1266–1505 | 258.6 | -59.8 | 34,434 |
| d9 | 1505–88132 | 872.9 | -295.1 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 616 | 950 | 1466 | 91459 | 5 | 44 |
