# Holdout eval — lgb_colab_echoclf

- engine: **lgb**   |   generated: 2026-09-16 22:53
- loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 7998 (inner-valid 2025-06)  |  full-refit rounds 8798
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 376.0 | 161.3 | +4.8 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 857.4 | +17.9 | 1274 | 26,528 |
| LFPG | 581.7 | +7.8 | 1052 | 39,590 |
| EGLL | 287.5 | +8.3 | 1382 | 40,210 |
| LTFM | 275.4 | +0.2 | 1070 | 46,539 |
| LEBL | 231.2 | +2.9 | 957 | 28,985 |
| EHAM | 230.7 | -0.0 | 798 | 40,949 |
| EDDF | 230.6 | +5.3 | 875 | 36,830 |
| LSZH | 212.2 | -0.2 | 759 | 22,289 |
| EDDM | 193.3 | -4.0 | 832 | 27,091 |
| LEMD | 186.2 | +10.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 372.3 | 153,660 |
| 2025-07 | 379.0 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 259.6 | 339,016 | 46.9% |
| False | 2202.8 | 5,323 | 53.1% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1877.9 | 2,406 | 17.4% |
| False | 342.9 | 341,933 | 82.6% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 227.8 | +114.9 | 34,434 |
| d1 | 569–671 | 193.1 | +86.6 | 34,434 |
| d2 | 671–770 | 186.7 | +64.8 | 34,434 |
| d3 | 770–842 | 196.3 | +61.4 | 34,434 |
| d4 | 842–923 | 206.2 | +45.4 | 34,434 |
| d5 | 923–1020 | 204.1 | +29.3 | 34,434 |
| d6 | 1020–1134 | 219.1 | +12.4 | 34,434 |
| d7 | 1134–1266 | 242.1 | -9.6 | 34,434 |
| d8 | 1266–1505 | 277.1 | -59.5 | 34,434 |
| d9 | 1505–88132 | 991.7 | -297.2 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 610 | 950 | 1471 | 77517 | 71 | 49 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 630 | 960 | 1037 | 1496 | 97895 | 56 | 48 |
