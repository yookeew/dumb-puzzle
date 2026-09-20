# Holdout eval — ades_cat_treatment

- engine: **lgb**   |   generated: 2026-09-19 19:22
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=2165 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 394.3 | 156.3 | +3.3 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 986.1 | +4.6 | 1274 | 26,528 |
| LFPG | 581.5 | +5.6 | 1052 | 39,590 |
| EGLL | 290.2 | +10.5 | 1382 | 40,210 |
| LTFM | 267.1 | +1.0 | 1070 | 46,539 |
| LEBL | 224.6 | +1.1 | 957 | 28,985 |
| LSZH | 212.6 | -1.0 | 759 | 22,289 |
| EHAM | 201.0 | -2.1 | 798 | 40,949 |
| EDDF | 192.9 | +3.6 | 875 | 36,830 |
| EDDM | 191.1 | -4.8 | 832 | 27,091 |
| LEMD | 181.4 | +11.7 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 379.8 | 153,660 |
| 2025-07 | 405.6 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 254.0 | 339,016 | 40.8% |
| False | 2439.3 | 5,323 | 59.2% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1398.2 | 2,444 | 8.9% |
| False | 377.6 | 341,895 | 91.1% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 196.5 | +113.9 | 34,434 |
| d1 | 569–671 | 173.6 | +87.2 | 34,434 |
| d2 | 671–770 | 165.6 | +64.5 | 34,434 |
| d3 | 770–842 | 175.3 | +60.5 | 34,434 |
| d4 | 842–923 | 182.3 | +45.6 | 34,434 |
| d5 | 923–1020 | 188.6 | +28.0 | 34,434 |
| d6 | 1020–1134 | 196.6 | +11.9 | 34,434 |
| d7 | 1134–1266 | 225.6 | -10.7 | 34,434 |
| d8 | 1266–1505 | 256.8 | -60.5 | 34,434 |
| d9 | 1505–88132 | 1097.1 | -306.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 75 | 616 | 950 | 1466 | 68031 | 0 | 43 |
