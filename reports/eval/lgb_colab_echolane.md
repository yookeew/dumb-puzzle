# Holdout eval — lgb_colab_echolane

- engine: **lgb**   |   generated: 2026-09-16 12:21
- loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 7998 (inner-valid 2025-06)  |  full-refit rounds 8798
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 484.6 | 165.7 | +2.7 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 1320.1 | -55.3 | 1274 | 26,528 |
| LFPG | 605.9 | +15.2 | 1052 | 39,590 |
| EGLL | 302.7 | +12.4 | 1382 | 40,210 |
| EDDF | 302.5 | +11.3 | 875 | 36,830 |
| EHAM | 291.7 | +5.3 | 798 | 40,949 |
| LTFM | 279.1 | +1.6 | 1070 | 46,539 |
| LEBL | 274.2 | +4.6 | 957 | 28,985 |
| LSZH | 218.2 | +1.8 | 759 | 22,289 |
| EDDM | 209.3 | -1.5 | 832 | 27,091 |
| LEMD | 206.4 | +12.5 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 429.4 | 153,660 |
| 2025-07 | 524.8 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 283.1 | 339,016 | 33.6% |
| False | 3176.0 | 5,323 | 66.4% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 233.2 | +114.7 | 34,434 |
| d1 | 569–671 | 205.6 | +85.9 | 34,434 |
| d2 | 671–770 | 210.5 | +64.8 | 34,434 |
| d3 | 770–842 | 195.6 | +60.6 | 34,434 |
| d4 | 842–923 | 226.6 | +46.4 | 34,434 |
| d5 | 923–1020 | 219.9 | +29.8 | 34,434 |
| d6 | 1020–1134 | 259.1 | +14.4 | 34,434 |
| d7 | 1134–1266 | 273.0 | -8.3 | 34,434 |
| d8 | 1266–1505 | 318.0 | -56.9 | 34,434 |
| d9 | 1505–88132 | 1351.5 | -324.9 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 608 | 949 | 1472 | 25998 | 222 | 58 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 628 | 960 | 1037 | 1498 | 29056 | 179 | 82 |
