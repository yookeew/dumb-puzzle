# Holdout eval — lgb_colab_ceilfix

- engine: **lgb**   |   generated: 2026-09-16 00:11
- loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 7998 (inner-valid 2025-06)  |  full-refit rounds 8798
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 140000] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 560.7 | 165.9 | +2.2 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| EDDF | 1036.0 | +18.7 | 875 | 36,830 |
| LIRF | 788.0 | -50.8 | 1274 | 26,528 |
| LFPG | 709.0 | +12.1 | 1052 | 39,590 |
| EHAM | 670.2 | +7.5 | 798 | 40,949 |
| LEBL | 312.7 | -0.3 | 957 | 28,985 |
| EGLL | 309.5 | +9.1 | 1382 | 40,210 |
| LTFM | 277.9 | +0.5 | 1070 | 46,539 |
| LSZH | 217.5 | -0.2 | 759 | 22,289 |
| EDDM | 208.8 | -3.7 | 832 | 27,091 |
| LEMD | 202.7 | +10.2 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 618.5 | 153,660 |
| 2025-07 | 509.4 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 376.2 | 339,016 | 44.3% |
| False | 3365.6 | 5,323 | 55.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 226.8 | +112.3 | 34,434 |
| d1 | 569–671 | 200.3 | +83.8 | 34,434 |
| d2 | 671–770 | 251.0 | +63.1 | 34,434 |
| d3 | 770–842 | 210.1 | +58.2 | 34,434 |
| d4 | 842–923 | 304.9 | +44.5 | 34,434 |
| d5 | 923–1020 | 242.0 | +27.0 | 34,434 |
| d6 | 1020–1134 | 435.5 | +12.5 | 34,434 |
| d7 | 1134–1266 | 623.3 | -8.6 | 34,434 |
| d8 | 1266–1505 | 1029.7 | -52.4 | 34,434 |
| d9 | 1505–88132 | 1074.9 | -318.3 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 607 | 948 | 1467 | 140000 | 384 | 1 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 627 | 959 | 1038 | 1494 | 102217 | 430 | 0 |
