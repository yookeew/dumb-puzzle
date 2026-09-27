# Holdout eval — doy_doy

- engine: **lgb**   |   generated: 2026-09-22 13:19
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=8000/direct=2475 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 337.9 | 155.6 | +1.2 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 663.5 | +26.0 | 1274 | 26,528 |
| LFPG | 581.2 | +0.6 | 1052 | 39,590 |
| EGLL | 288.6 | +3.6 | 1382 | 40,210 |
| LTFM | 268.0 | -8.5 | 1070 | 46,539 |
| LEBL | 224.4 | +2.3 | 957 | 28,985 |
| LSZH | 211.7 | -2.0 | 759 | 22,289 |
| EHAM | 200.0 | -7.0 | 798 | 40,949 |
| EDDF | 192.3 | -1.8 | 875 | 36,830 |
| EDDM | 190.0 | -0.7 | 832 | 27,091 |
| LEMD | 180.0 | +8.9 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 355.2 | 153,660 |
| 2025-07 | 323.3 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 241.0 | 339,016 | 50.1% |
| False | 1920.7 | 5,323 | 49.9% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1476.3 | 2,571 | 14.3% |
| False | 314.1 | 341,768 | 85.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 199.7 | +111.6 | 34,434 |
| d1 | 569–671 | 181.9 | +85.0 | 34,434 |
| d2 | 671–770 | 176.7 | +62.4 | 34,434 |
| d3 | 770–842 | 179.6 | +58.3 | 34,434 |
| d4 | 842–923 | 198.9 | +43.0 | 34,434 |
| d5 | 923–1020 | 193.9 | +24.8 | 34,434 |
| d6 | 1020–1134 | 204.7 | +8.3 | 34,434 |
| d7 | 1134–1266 | 235.2 | -14.8 | 34,434 |
| d8 | 1266–1505 | 261.5 | -65.8 | 34,434 |
| d9 | 1505–88132 | 873.4 | -300.5 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 613 | 947 | 1459 | 91012 | 5 | 47 |
