# Holdout eval — ades_cat_treatment_mixed

- engine: **lgb**   |   generated: 2026-09-20 15:28
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7994/direct=2485 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 335.9 | 154.8 | +2.4 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 651.4 | +21.8 | 1274 | 26,528 |
| LFPG | 579.9 | +2.0 | 1052 | 39,590 |
| EGLL | 290.7 | +9.8 | 1382 | 40,210 |
| LTFM | 267.6 | -11.7 | 1070 | 46,539 |
| LEBL | 224.3 | -2.2 | 957 | 28,985 |
| LSZH | 210.6 | -1.1 | 759 | 22,289 |
| EHAM | 200.5 | +1.6 | 798 | 40,949 |
| EDDF | 191.6 | +2.2 | 875 | 36,830 |
| EDDM | 187.4 | -3.7 | 832 | 27,091 |
| LEMD | 180.2 | +9.9 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.6 | 153,660 |
| 2025-07 | 319.9 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 239.0 | 339,016 | 49.9% |
| False | 1912.8 | 5,323 | 50.1% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 983.5 | 2,441 | 6.1% |
| False | 326.7 | 341,898 | 93.9% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 196.4 | +111.8 | 34,434 |
| d1 | 569–671 | 174.2 | +84.3 | 34,434 |
| d2 | 671–770 | 170.2 | +62.3 | 34,434 |
| d3 | 770–842 | 175.9 | +57.6 | 34,434 |
| d4 | 842–923 | 183.0 | +42.5 | 34,434 |
| d5 | 923–1020 | 190.2 | +25.3 | 34,434 |
| d6 | 1020–1134 | 196.8 | +9.1 | 34,434 |
| d7 | 1134–1266 | 226.0 | -13.1 | 34,434 |
| d8 | 1266–1505 | 262.0 | -62.0 | 34,434 |
| d9 | 1505–88132 | 878.2 | -294.4 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 613 | 947 | 1464 | 89176 | 5 | 41 |
