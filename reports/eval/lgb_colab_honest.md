# Holdout eval — lgb_colab_honest

- engine: **lgb**   |   generated: 2026-09-16 00:33
- loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 7998 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 522.6 | 165.5 | -1.6 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 1509.0 | -73.6 | 1274 | 26,528 |
| LFPG | 602.3 | +10.2 | 1052 | 39,590 |
| EGLL | 303.5 | +9.0 | 1382 | 40,210 |
| EDDF | 290.4 | +8.0 | 875 | 36,830 |
| EHAM | 280.0 | +2.1 | 798 | 40,949 |
| LTFM | 277.9 | +0.5 | 1070 | 46,539 |
| LEBL | 264.0 | -1.0 | 957 | 28,985 |
| LSZH | 217.5 | -0.2 | 759 | 22,289 |
| EDDM | 205.7 | -3.8 | 832 | 27,091 |
| LEMD | 202.7 | +10.2 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 447.3 | 153,660 |
| 2025-07 | 576.2 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 283.1 | 339,016 | 28.9% |
| False | 3544.9 | 5,323 | 71.1% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 226.8 | +112.3 | 34,434 |
| d1 | 569–671 | 196.3 | +83.8 | 34,434 |
| d2 | 671–770 | 198.1 | +62.5 | 34,434 |
| d3 | 770–842 | 189.4 | +58.0 | 34,434 |
| d4 | 842–923 | 212.6 | +43.3 | 34,434 |
| d5 | 923–1020 | 210.5 | +26.5 | 34,434 |
| d6 | 1020–1134 | 245.8 | +10.4 | 34,434 |
| d7 | 1134–1266 | 255.7 | -12.9 | 34,434 |
| d8 | 1266–1505 | 304.2 | -61.8 | 34,434 |
| d9 | 1505–88132 | 1502.9 | -338.4 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 607 | 948 | 1467 | 10800 | 384 | 50 |
