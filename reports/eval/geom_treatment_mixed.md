# Holdout eval — geom_treatment_mixed

- engine: **lgb**   |   generated: 2026-09-27 02:20
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7996/direct=3309 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 336.2 | 154.7 | +2.2 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 657.9 | +25.3 | 1274 | 26,528 |
| LFPG | 579.0 | +1.4 | 1052 | 39,590 |
| EGLL | 288.6 | +9.0 | 1382 | 40,210 |
| LTFM | 267.9 | -14.0 | 1070 | 46,539 |
| LEBL | 224.3 | -1.0 | 957 | 28,985 |
| LSZH | 209.1 | -0.8 | 759 | 22,289 |
| EHAM | 199.4 | +0.9 | 798 | 40,949 |
| EDDF | 190.9 | +1.5 | 875 | 36,830 |
| EDDM | 186.4 | -4.2 | 832 | 27,091 |
| LEMD | 179.4 | +10.7 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.2 | 153,660 |
| 2025-07 | 321.0 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 239.6 | 339,016 | 50.0% |
| False | 1912.1 | 5,323 | 50.0% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1486.9 | 2,375 | 13.5% |
| False | 313.8 | 341,964 | 86.5% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 197.8 | +110.7 | 34,434 |
| d1 | 569–671 | 177.5 | +83.6 | 34,434 |
| d2 | 671–770 | 177.1 | +62.2 | 34,434 |
| d3 | 770–842 | 176.2 | +57.7 | 34,434 |
| d4 | 842–923 | 192.0 | +42.7 | 34,434 |
| d5 | 923–1020 | 190.7 | +25.6 | 34,434 |
| d6 | 1020–1134 | 201.3 | +9.9 | 34,434 |
| d7 | 1134–1266 | 231.2 | -12.4 | 34,434 |
| d8 | 1266–1505 | 256.9 | -62.5 | 34,434 |
| d9 | 1505–88132 | 874.2 | -295.9 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 612 | 947 | 1463 | 90588 | 5 | 42 |
