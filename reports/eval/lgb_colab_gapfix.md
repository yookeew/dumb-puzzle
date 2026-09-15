# Holdout eval — lgb_colab_gapfix

- engine: **lgb**   |   generated: 2026-09-15 12:25
- eta 0.02  |  rounds ceiling 8000  |  early-stopped at 2192 (inner-valid 2025-06)  |  full-refit rounds 2411
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,294 (true taxi clipped to [0, 4h] for reporting)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 309.2 | 164.7 | +4.7 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 575.6 | -23.4 | 1220 | 26,486 |
| LFPG | 312.0 | +14.5 | 1048 | 39,587 |
| EGLL | 310.6 | +13.9 | 1382 | 40,210 |
| EDDF | 303.7 | +10.3 | 875 | 36,830 |
| EHAM | 295.2 | +6.9 | 798 | 40,949 |
| LTFM | 289.2 | +4.0 | 1070 | 46,539 |
| LEBL | 259.0 | -0.7 | 957 | 28,985 |
| LSZH | 222.1 | -0.6 | 759 | 22,289 |
| LEMD | 200.9 | +11.0 | 1011 | 35,328 |
| EDDM | 200.7 | -3.6 | 832 | 27,091 |

## Per month

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 266.2 | 153,652 |
| 2025-07 | 340.0 | 190,642 |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 266.9 | +119.9 | 34,430 |
| d1 | 569–671 | 199.7 | +87.1 | 34,429 |
| d2 | 671–770 | 214.6 | +66.3 | 34,430 |
| d3 | 770–842 | 197.6 | +61.0 | 34,429 |
| d4 | 842–923 | 213.5 | +45.3 | 34,429 |
| d5 | 923–1020 | 211.8 | +28.8 | 34,430 |
| d6 | 1020–1134 | 237.7 | +12.3 | 34,429 |
| d7 | 1134–1266 | 260.0 | -10.7 | 34,430 |
| d8 | 1266–1505 | 295.3 | -60.5 | 34,429 |
| d9 | 1505–13849 | 677.0 | -302.4 | 34,429 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 609 | 949 | 1472 | 10800 | 596 | 21 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 626 | 959 | 1032 | 1494 | 10800 | 717 | 67 |
