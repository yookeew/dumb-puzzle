# Holdout eval — test_direct_seed7

- engine: **lgb**   |   generated: 2026-09-16 18:12
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 2405 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 383.7 | 155.9 | +5.5 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 937.9 | +10.4 | 1274 | 26,528 |
| LFPG | 579.4 | +9.3 | 1052 | 39,590 |
| EGLL | 285.6 | +11.7 | 1382 | 40,210 |
| LTFM | 266.9 | +2.9 | 1070 | 46,539 |
| LEBL | 223.5 | +3.5 | 957 | 28,985 |
| LSZH | 209.3 | +0.8 | 759 | 22,289 |
| EHAM | 197.1 | -0.5 | 798 | 40,949 |
| EDDF | 191.0 | +4.6 | 875 | 36,830 |
| EDDM | 188.4 | -2.5 | 832 | 27,091 |
| LEMD | 180.2 | +12.5 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 363.7 | 153,660 |
| 2025-07 | 399.0 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 260.5 | 339,016 | 45.4% |
| False | 2281.0 | 5,323 | 54.6% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 2117.1 | 2,406 | 21.3% |
| False | 341.6 | 341,933 | 78.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 199.7 | +114.1 | 34,434 |
| d1 | 569–671 | 179.0 | +87.7 | 34,434 |
| d2 | 671–770 | 174.7 | +65.4 | 34,434 |
| d3 | 770–842 | 178.4 | +61.8 | 34,434 |
| d4 | 842–923 | 197.0 | +47.4 | 34,434 |
| d5 | 923–1020 | 191.4 | +30.0 | 34,434 |
| d6 | 1020–1134 | 201.3 | +14.7 | 34,434 |
| d7 | 1134–1266 | 230.7 | -7.4 | 34,434 |
| d8 | 1266–1505 | 248.0 | -57.3 | 34,434 |
| d9 | 1505–88132 | 1052.2 | -301.7 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 44 | 616 | 952 | 1469 | 72559 | 0 | 44 |
