# Holdout eval — loss_l2_treatment

- engine: **lgb**   |   generated: 2026-09-19 19:55
- target direct  |  loss l2  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=1783 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 380.1 | 157.3 | +5.9 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 912.4 | +24.4 | 1274 | 26,528 |
| LFPG | 579.8 | +9.5 | 1052 | 39,590 |
| EGLL | 290.2 | +12.6 | 1382 | 40,210 |
| LTFM | 267.3 | +0.7 | 1070 | 46,539 |
| LEBL | 224.6 | +3.1 | 957 | 28,985 |
| LSZH | 212.4 | -0.6 | 759 | 22,289 |
| EHAM | 201.6 | -2.1 | 798 | 40,949 |
| EDDF | 192.1 | +4.8 | 875 | 36,830 |
| EDDM | 191.5 | -4.0 | 832 | 27,091 |
| LEMD | 180.5 | +11.3 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 367.6 | 153,660 |
| 2025-07 | 389.9 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 253.1 | 339,016 | 43.7% |
| False | 2294.1 | 5,323 | 56.3% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 2000.4 | 2,406 | 19.4% |
| False | 342.5 | 341,933 | 80.6% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 201.7 | +115.3 | 34,434 |
| d1 | 569–671 | 181.5 | +88.5 | 34,434 |
| d2 | 671–770 | 177.5 | +66.1 | 34,434 |
| d3 | 770–842 | 181.1 | +62.5 | 34,434 |
| d4 | 842–923 | 197.9 | +47.6 | 34,434 |
| d5 | 923–1020 | 192.7 | +29.7 | 34,434 |
| d6 | 1020–1134 | 205.6 | +13.9 | 34,434 |
| d7 | 1134–1266 | 237.7 | -7.9 | 34,434 |
| d8 | 1266–1505 | 255.9 | -58.1 | 34,434 |
| d9 | 1505–88132 | 1032.5 | -298.9 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 616 | 951 | 1470 | 75871 | 0 | 44 |
