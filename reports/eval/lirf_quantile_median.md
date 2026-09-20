# Holdout eval — lirf_quantile_median

- engine: **lgb**   |   generated: 2026-09-17 17:40
- target flip  |  loss quantile  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=8000 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 345.6 | 159.0 | -7.9 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 657.8 | -4.4 | 1274 | 26,528 |
| LFPG | 584.0 | -16.2 | 1052 | 39,590 |
| EGLL | 288.7 | +7.4 | 1382 | 40,210 |
| LTFM | 280.3 | -15.8 | 1070 | 46,539 |
| EDDF | 240.3 | -9.3 | 875 | 36,830 |
| LEBL | 232.7 | -16.7 | 957 | 28,985 |
| EHAM | 228.7 | +4.5 | 798 | 40,949 |
| LSZH | 212.9 | -14.9 | 759 | 22,289 |
| EDDM | 196.2 | -15.2 | 832 | 27,091 |
| LEMD | 187.8 | -3.5 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 364.4 | 153,660 |
| 2025-07 | 329.6 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 253.9 | 339,016 | 53.1% |
| False | 1903.0 | 5,323 | 46.9% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1453.9 | 2,406 | 12.4% |
| False | 324.6 | 341,933 | 87.6% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 218.4 | +108.2 | 34,434 |
| d1 | 569–671 | 184.7 | +77.9 | 34,434 |
| d2 | 671–770 | 176.7 | +54.8 | 34,434 |
| d3 | 770–842 | 187.3 | +50.7 | 34,434 |
| d4 | 842–923 | 199.2 | +34.6 | 34,434 |
| d5 | 923–1020 | 197.8 | +18.0 | 34,434 |
| d6 | 1020–1134 | 214.5 | +0.4 | 34,434 |
| d7 | 1134–1266 | 238.6 | -22.9 | 34,434 |
| d8 | 1266–1505 | 276.4 | -74.0 | 34,434 |
| d9 | 1505–88132 | 887.7 | -326.2 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 606 | 938 | 1451 | 91488 | 88 | 49 |
