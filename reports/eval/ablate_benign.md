# Holdout eval — ablate_benign

- engine: **lgb**   |   generated: 2026-09-20 17:44
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7999/direct=2810 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 336.5 | 155.2 | +3.3 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 657.3 | +25.2 | 1274 | 26,528 |
| LFPG | 579.9 | +5.1 | 1052 | 39,590 |
| EGLL | 288.6 | +9.1 | 1382 | 40,210 |
| LTFM | 268.3 | -11.3 | 1070 | 46,539 |
| LEBL | 223.9 | +0.3 | 957 | 28,985 |
| LSZH | 211.3 | -1.1 | 759 | 22,289 |
| EHAM | 199.6 | +0.1 | 798 | 40,949 |
| EDDF | 191.7 | +3.2 | 875 | 36,830 |
| EDDM | 186.8 | -1.9 | 832 | 27,091 |
| LEMD | 179.9 | +10.7 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.7 | 153,660 |
| 2025-07 | 321.1 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 240.3 | 339,016 | 50.2% |
| False | 1910.1 | 5,323 | 49.8% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1458.2 | 2,392 | 13.0% |
| False | 314.9 | 341,947 | 87.0% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.7 | +112.4 | 34,434 |
| d1 | 569–671 | 179.3 | +85.1 | 34,434 |
| d2 | 671–770 | 177.8 | +63.4 | 34,434 |
| d3 | 770–842 | 176.9 | +59.1 | 34,434 |
| d4 | 842–923 | 192.1 | +44.0 | 34,434 |
| d5 | 923–1020 | 189.5 | +26.6 | 34,434 |
| d6 | 1020–1134 | 200.8 | +10.7 | 34,434 |
| d7 | 1134–1266 | 231.2 | -11.6 | 34,434 |
| d8 | 1266–1505 | 257.1 | -61.7 | 34,434 |
| d9 | 1505–88132 | 874.8 | -294.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 614 | 948 | 1464 | 90314 | 5 | 40 |
