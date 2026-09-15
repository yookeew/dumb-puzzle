# Holdout eval — lgb_colab_huber2

- engine: **lgb**   |   generated: 2026-09-15 14:50
- loss huber (alpha=400.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 8000 (inner-valid 2025-06)  |  full-refit rounds 8800
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,294 (true taxi clipped to [0, 4h] for reporting)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 315.6 | 161.5 | -0.6 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 560.6 | -43.9 | 1220 | 26,486 |
| EDDF | 371.0 | +12.6 | 875 | 36,830 |
| LFPG | 323.6 | +8.8 | 1048 | 39,587 |
| EGLL | 304.6 | +6.5 | 1382 | 40,210 |
| EHAM | 292.8 | +1.9 | 798 | 40,949 |
| LTFM | 278.7 | -2.8 | 1070 | 46,539 |
| LEBL | 268.6 | -5.8 | 957 | 28,985 |
| LSZH | 217.0 | -2.1 | 759 | 22,289 |
| EDDM | 216.8 | -5.8 | 832 | 27,091 |
| LEMD | 204.6 | +8.8 | 1011 | 35,328 |

## Per month

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 271.1 | 153,652 |
| 2025-07 | 347.3 | 190,642 |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 224.7 | +110.5 | 34,430 |
| d1 | 569–671 | 217.1 | +82.5 | 34,429 |
| d2 | 671–770 | 218.4 | +61.1 | 34,430 |
| d3 | 770–842 | 187.7 | +55.1 | 34,429 |
| d4 | 842–923 | 245.7 | +42.6 | 34,429 |
| d5 | 923–1020 | 228.1 | +24.6 | 34,430 |
| d6 | 1020–1134 | 253.4 | +8.6 | 34,429 |
| d7 | 1134–1266 | 268.6 | -15.7 | 34,430 |
| d8 | 1266–1505 | 323.4 | -64.4 | 34,429 |
| d9 | 1505–13849 | 679.8 | -310.9 | 34,429 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 606 | 945 | 1460 | 10800 | 308 | 52 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 627 | 957 | 1031 | 1486 | 10800 | 333 | 110 |
