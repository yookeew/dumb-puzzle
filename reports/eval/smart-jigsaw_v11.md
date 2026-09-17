# Holdout eval — smart-jigsaw_v11

- engine: **lgb**   |   generated: 2026-09-17 09:57
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7998/direct=3310 (inner-valid 2025-06)  |  full-refit rounds flip=8798/direct=3641
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 370.1 | 156.6 | +4.2 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 857.4 | +17.9 | 1274 | 26,528 |
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
| 2025-01 | 364.9 | 153,660 |
| 2025-07 | 374.2 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 250.9 | 339,016 | 45.2% |
| False | 2202.8 | 5,323 | 54.8% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1877.9 | 2,406 | 18.0% |
| False | 336.3 | 341,933 | 82.0% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 201.1 | +112.9 | 34,434 |
| d1 | 569–671 | 182.3 | +86.5 | 34,434 |
| d2 | 671–770 | 177.9 | +64.3 | 34,434 |
| d3 | 770–842 | 180.2 | +60.7 | 34,434 |
| d4 | 842–923 | 197.3 | +45.8 | 34,434 |
| d5 | 923–1020 | 192.3 | +28.2 | 34,434 |
| d6 | 1020–1134 | 204.5 | +12.6 | 34,434 |
| d7 | 1134–1266 | 234.3 | -9.7 | 34,434 |
| d8 | 1266–1505 | 259.0 | -59.7 | 34,434 |
| d9 | 1505–88132 | 996.1 | -300.1 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 614 | 950 | 1467 | 77517 | 5 | 45 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 634 | 960 | 1035 | 1489 | 97895 | 4 | 42 |
