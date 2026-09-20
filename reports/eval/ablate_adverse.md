# Holdout eval — ablate_adverse

- engine: **lgb**   |   generated: 2026-09-20 17:23
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7997/direct=3169 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 336.8 | 155.2 | +2.1 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 658.9 | +26.0 | 1274 | 26,528 |
| LFPG | 579.1 | -0.0 | 1052 | 39,590 |
| EGLL | 289.1 | +9.7 | 1382 | 40,210 |
| LTFM | 267.9 | -8.4 | 1070 | 46,539 |
| LEBL | 224.4 | -2.3 | 957 | 28,985 |
| LSZH | 209.8 | -0.8 | 759 | 22,289 |
| EHAM | 200.9 | -1.7 | 798 | 40,949 |
| EDDF | 192.0 | +0.3 | 875 | 36,830 |
| EDDM | 188.7 | -5.8 | 832 | 27,091 |
| LEMD | 179.9 | +9.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.9 | 153,660 |
| 2025-07 | 321.5 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 240.1 | 339,016 | 50.1% |
| False | 1914.2 | 5,323 | 49.9% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1493.5 | 2,395 | 13.7% |
| False | 314.0 | 341,944 | 86.3% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 199.4 | +111.0 | 34,434 |
| d1 | 569–671 | 178.6 | +83.7 | 34,434 |
| d2 | 671–770 | 176.4 | +61.9 | 34,434 |
| d3 | 770–842 | 177.6 | +57.3 | 34,434 |
| d4 | 842–923 | 195.0 | +42.7 | 34,434 |
| d5 | 923–1020 | 192.0 | +25.4 | 34,434 |
| d6 | 1020–1134 | 204.4 | +9.8 | 34,434 |
| d7 | 1134–1266 | 234.9 | -12.2 | 34,434 |
| d8 | 1266–1505 | 260.7 | -61.8 | 34,434 |
| d9 | 1505–88132 | 871.8 | -296.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 613 | 946 | 1464 | 91373 | 5 | 45 |
