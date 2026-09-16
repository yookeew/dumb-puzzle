# Holdout eval — test_flip_seed7

- engine: **lgb**   |   generated: 2026-09-16 18:07
- target flip  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 8000 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 399.5 | 164.7 | +8.7 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 885.1 | +18.5 | 1274 | 26,528 |
| LFPG | 602.4 | +13.3 | 1052 | 39,590 |
| EDDF | 305.1 | +11.2 | 875 | 36,830 |
| EGLL | 304.8 | +11.3 | 1382 | 40,210 |
| EHAM | 280.3 | +4.4 | 798 | 40,949 |
| LTFM | 276.9 | +4.2 | 1070 | 46,539 |
| LEBL | 266.2 | +8.0 | 957 | 28,985 |
| LSZH | 217.6 | +1.5 | 759 | 22,289 |
| EDDM | 216.5 | -1.1 | 832 | 27,091 |
| LEMD | 204.4 | +13.8 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 384.0 | 153,660 |
| 2025-07 | 411.5 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 268.2 | 339,016 | 44.4% |
| False | 2396.5 | 5,323 | 55.6% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1981.2 | 2,406 | 17.2% |
| False | 364.8 | 341,933 | 82.8% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 235.6 | +115.0 | 34,434 |
| d1 | 569–671 | 220.3 | +88.3 | 34,434 |
| d2 | 671–770 | 231.7 | +67.7 | 34,434 |
| d3 | 770–842 | 204.1 | +62.5 | 34,434 |
| d4 | 842–923 | 245.6 | +49.4 | 34,434 |
| d5 | 923–1020 | 233.8 | +32.4 | 34,434 |
| d6 | 1020–1134 | 266.5 | +18.0 | 34,434 |
| d7 | 1134–1266 | 289.3 | -3.9 | 34,434 |
| d8 | 1266–1505 | 326.7 | -52.0 | 34,434 |
| d9 | 1505–88132 | 1010.1 | -290.7 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 609 | 952 | 1480 | 74603 | 104 | 76 |
