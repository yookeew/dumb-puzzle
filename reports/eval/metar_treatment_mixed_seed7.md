# Holdout eval — metar_treatment_mixed_seed7

- engine: **lgb**   |   generated: 2026-09-20 16:30
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=8000/direct=3005 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 335.9 | 154.9 | +2.3 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 655.2 | +23.7 | 1274 | 26,528 |
| LFPG | 579.1 | +2.4 | 1052 | 39,590 |
| EGLL | 288.4 | +9.8 | 1382 | 40,210 |
| LTFM | 268.2 | -14.6 | 1070 | 46,539 |
| LEBL | 223.9 | -0.8 | 957 | 28,985 |
| LSZH | 209.6 | -0.8 | 759 | 22,289 |
| EHAM | 199.8 | +1.3 | 798 | 40,949 |
| EDDF | 191.1 | +1.9 | 875 | 36,830 |
| EDDM | 186.6 | -3.9 | 832 | 27,091 |
| LEMD | 179.8 | +10.9 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.0 | 153,660 |
| 2025-07 | 320.6 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 240.0 | 339,016 | 50.2% |
| False | 1905.7 | 5,323 | 49.8% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1272.3 | 2,372 | 9.9% |
| False | 320.0 | 341,967 | 90.1% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 197.8 | +111.5 | 34,434 |
| d1 | 569–671 | 177.9 | +84.4 | 34,434 |
| d2 | 671–770 | 178.3 | +62.9 | 34,434 |
| d3 | 770–842 | 175.6 | +57.9 | 34,434 |
| d4 | 842–923 | 192.3 | +42.9 | 34,434 |
| d5 | 923–1020 | 189.5 | +25.7 | 34,434 |
| d6 | 1020–1134 | 199.5 | +9.8 | 34,434 |
| d7 | 1134–1266 | 230.2 | -12.5 | 34,434 |
| d8 | 1266–1505 | 256.5 | -62.8 | 34,434 |
| d9 | 1505–88132 | 873.8 | -296.4 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 614 | 947 | 1463 | 90361 | 5 | 43 |
