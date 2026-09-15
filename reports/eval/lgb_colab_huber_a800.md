# Holdout eval — lgb_colab_huber_a800

- engine: **lgb**   |   generated: 2026-09-15 18:18
- loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 7998 (inner-valid 2025-06)  |  full-refit rounds 8798
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,294 (true taxi clipped to [0, 4h] for reporting)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 303.1 | 161.9 | +2.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 561.9 | -31.5 | 1220 | 26,486 |
| LFPG | 317.2 | +13.9 | 1048 | 39,587 |
| EGLL | 303.5 | +9.0 | 1382 | 40,210 |
| EDDF | 290.4 | +8.0 | 875 | 36,830 |
| EHAM | 280.0 | +2.1 | 798 | 40,949 |
| LTFM | 277.9 | +0.5 | 1070 | 46,539 |
| LEBL | 264.0 | -1.0 | 957 | 28,985 |
| LSZH | 217.5 | -0.2 | 759 | 22,289 |
| EDDM | 205.7 | -3.8 | 832 | 27,091 |
| LEMD | 202.7 | +10.2 | 1011 | 35,328 |

## Per month

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 260.2 | 153,652 |
| 2025-07 | 333.7 | 190,642 |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 226.8 | +112.3 | 34,430 |
| d1 | 569–671 | 196.3 | +83.7 | 34,429 |
| d2 | 671–770 | 198.1 | +62.5 | 34,430 |
| d3 | 770–842 | 189.3 | +57.9 | 34,429 |
| d4 | 842–923 | 212.7 | +43.3 | 34,429 |
| d5 | 923–1020 | 210.4 | +26.5 | 34,430 |
| d6 | 1020–1134 | 245.9 | +10.4 | 34,429 |
| d7 | 1134–1266 | 255.7 | -13.0 | 34,430 |
| d8 | 1266–1505 | 304.2 | -61.6 | 34,429 |
| d9 | 1505–13849 | 667.7 | -301.7 | 34,429 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 607 | 947 | 1467 | 10800 | 384 | 28 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 627 | 959 | 1033 | 1494 | 10800 | 430 | 79 |
