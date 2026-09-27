# Holdout eval — lgb_colab

- engine: **lgb**   |   generated: 2026-09-26 20:34
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7996/direct=3181 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 335.6 | 154.8 | +1.9 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 652.7 | +22.6 | 1274 | 26,528 |
| LFPG | 579.5 | +1.0 | 1052 | 39,590 |
| EGLL | 288.6 | +8.6 | 1382 | 40,210 |
| LTFM | 268.3 | -15.0 | 1070 | 46,539 |
| LEBL | 224.1 | -0.8 | 957 | 28,985 |
| LSZH | 209.5 | -0.6 | 759 | 22,289 |
| EHAM | 199.8 | +0.4 | 798 | 40,949 |
| EDDF | 190.9 | +1.4 | 875 | 36,830 |
| EDDM | 186.5 | -3.4 | 832 | 27,091 |
| LEMD | 179.9 | +11.9 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.3 | 153,660 |
| 2025-07 | 319.8 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 239.8 | 339,016 | 50.3% |
| False | 1903.5 | 5,323 | 49.7% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1415.1 | 2,396 | 12.4% |
| False | 315.3 | 341,943 | 87.6% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 197.7 | +111.1 | 34,434 |
| d1 | 569–671 | 177.7 | +84.0 | 34,434 |
| d2 | 671–770 | 176.5 | +62.2 | 34,434 |
| d3 | 770–842 | 175.9 | +57.5 | 34,434 |
| d4 | 842–923 | 190.9 | +42.4 | 34,434 |
| d5 | 923–1020 | 188.2 | +25.2 | 34,434 |
| d6 | 1020–1134 | 199.2 | +9.2 | 34,434 |
| d7 | 1134–1266 | 228.7 | -13.1 | 34,434 |
| d8 | 1266–1505 | 256.2 | -63.0 | 34,434 |
| d9 | 1505–88132 | 874.2 | -296.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 613 | 947 | 1462 | 90506 | 5 | 41 |
