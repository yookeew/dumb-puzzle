# Holdout eval — test_flip

- engine: **lgb**   |   generated: 2026-09-16 17:50
- target flip  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 8000 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 397.3 | 164.5 | +8.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 883.4 | +17.4 | 1274 | 26,528 |
| LFPG | 602.3 | +13.8 | 1052 | 39,590 |
| EGLL | 303.0 | +10.2 | 1382 | 40,210 |
| EDDF | 290.8 | +10.1 | 875 | 36,830 |
| EHAM | 280.0 | +4.1 | 798 | 40,949 |
| LTFM | 278.1 | +2.9 | 1070 | 46,539 |
| LEBL | 264.2 | +6.8 | 957 | 28,985 |
| LSZH | 217.1 | +1.5 | 759 | 22,289 |
| EDDM | 205.8 | -1.6 | 832 | 27,091 |
| LEMD | 202.7 | +13.0 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 383.6 | 153,660 |
| 2025-07 | 408.1 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 265.7 | 339,016 | 44.0% |
| False | 2390.7 | 5,323 | 56.0% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1976.5 | 2,406 | 17.3% |
| False | 362.6 | 341,933 | 82.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 237.3 | +115.8 | 34,434 |
| d1 | 569–671 | 214.2 | +87.7 | 34,434 |
| d2 | 671–770 | 219.0 | +66.6 | 34,434 |
| d3 | 770–842 | 203.7 | +62.3 | 34,434 |
| d4 | 842–923 | 240.5 | +48.6 | 34,434 |
| d5 | 923–1020 | 230.2 | +31.8 | 34,434 |
| d6 | 1020–1134 | 264.4 | +16.8 | 34,434 |
| d7 | 1134–1266 | 284.6 | -5.1 | 34,434 |
| d8 | 1266–1505 | 323.0 | -53.6 | 34,434 |
| d9 | 1505–88132 | 1010.4 | -291.4 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 609 | 951 | 1478 | 74603 | 98 | 73 |
