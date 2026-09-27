# Holdout eval — ades_cat_baseline_mixed

- engine: **lgb**   |   generated: 2026-09-20 15:07
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7997/direct=3169 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 335.9 | 154.9 | +2.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 655.1 | +23.5 | 1274 | 26,528 |
| LFPG | 579.2 | +1.4 | 1052 | 39,590 |
| EGLL | 288.6 | +8.9 | 1382 | 40,210 |
| LTFM | 268.2 | -15.7 | 1070 | 46,539 |
| LEBL | 224.0 | -0.9 | 957 | 28,985 |
| LSZH | 209.3 | -0.4 | 759 | 22,289 |
| EHAM | 199.8 | +1.0 | 798 | 40,949 |
| EDDF | 190.9 | +1.8 | 875 | 36,830 |
| EDDM | 186.7 | -3.9 | 832 | 27,091 |
| LEMD | 180.1 | +11.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 353.9 | 153,660 |
| 2025-07 | 320.7 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 239.9 | 339,016 | 50.2% |
| False | 1906.7 | 5,323 | 49.8% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1269.2 | 2,372 | 9.8% |
| False | 320.1 | 341,967 | 90.2% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 197.9 | +111.3 | 34,434 |
| d1 | 569–671 | 177.7 | +84.0 | 34,434 |
| d2 | 671–770 | 177.1 | +62.3 | 34,434 |
| d3 | 770–842 | 175.8 | +57.5 | 34,434 |
| d4 | 842–923 | 192.0 | +42.5 | 34,434 |
| d5 | 923–1020 | 189.2 | +25.4 | 34,434 |
| d6 | 1020–1134 | 199.6 | +9.3 | 34,434 |
| d7 | 1134–1266 | 229.5 | -12.9 | 34,434 |
| d8 | 1266–1505 | 256.4 | -62.9 | 34,434 |
| d9 | 1505–88132 | 874.5 | -296.6 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 613 | 947 | 1462 | 90434 | 5 | 42 |
