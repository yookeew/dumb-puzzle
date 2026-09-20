# Holdout eval — atfm_v1_treatment

- engine: **lgb**   |   generated: 2026-09-19 18:55
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=3414 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 378.9 | 155.1 | +2.2 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 910.2 | +21.5 | 1274 | 26,528 |
| LFPG | 580.0 | +0.4 | 1052 | 39,590 |
| EGLL | 288.5 | +7.8 | 1382 | 40,210 |
| LTFM | 262.1 | -1.5 | 1070 | 46,539 |
| LEBL | 223.4 | -1.9 | 957 | 28,985 |
| LSZH | 211.5 | -4.1 | 759 | 22,289 |
| EHAM | 201.6 | -5.5 | 798 | 40,949 |
| EDDF | 191.6 | +1.6 | 875 | 36,830 |
| EDDM | 189.2 | -3.3 | 832 | 27,091 |
| LEMD | 181.5 | +9.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 361.1 | 153,660 |
| 2025-07 | 392.6 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 254.3 | 339,016 | 44.3% |
| False | 2273.3 | 5,323 | 55.7% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1847.6 | 2,423 | 16.7% |
| False | 347.0 | 341,916 | 83.3% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 195.7 | +111.0 | 34,434 |
| d1 | 569–671 | 174.1 | +84.3 | 34,434 |
| d2 | 671–770 | 162.5 | +61.7 | 34,434 |
| d3 | 770–842 | 174.6 | +58.0 | 34,434 |
| d4 | 842–923 | 196.8 | +43.5 | 34,434 |
| d5 | 923–1020 | 191.6 | +26.4 | 34,434 |
| d6 | 1020–1134 | 201.5 | +10.7 | 34,434 |
| d7 | 1134–1266 | 235.1 | -11.2 | 34,434 |
| d8 | 1266–1505 | 244.9 | -60.9 | 34,434 |
| d9 | 1505–88132 | 1038.7 | -301.2 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 613 | 948 | 1466 | 75033 | 0 | 38 |
