# Holdout eval — seed_ens_7

- engine: **lgb**   |   generated: 2026-09-19 21:11
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=2405 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 379.1 | 156.2 | +4.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 907.6 | +10.6 | 1274 | 26,528 |
| LFPG | 580.9 | +6.6 | 1052 | 39,590 |
| EGLL | 288.5 | +10.0 | 1382 | 40,210 |
| LTFM | 267.4 | +0.6 | 1070 | 46,539 |
| LEBL | 224.0 | +1.8 | 957 | 28,985 |
| LSZH | 211.6 | -0.3 | 759 | 22,289 |
| EHAM | 200.4 | -1.6 | 798 | 40,949 |
| EDDF | 192.3 | +3.2 | 875 | 36,830 |
| EDDM | 190.6 | -4.1 | 832 | 27,091 |
| LEMD | 180.6 | +11.4 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 366.3 | 153,660 |
| 2025-07 | 389.1 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 252.9 | 339,016 | 43.8% |
| False | 2285.4 | 5,323 | 56.2% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1992.8 | 2,406 | 19.3% |
| False | 341.7 | 341,933 | 80.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.7 | +114.1 | 34,434 |
| d1 | 569–671 | 178.9 | +87.5 | 34,434 |
| d2 | 671–770 | 173.5 | +65.0 | 34,434 |
| d3 | 770–842 | 176.8 | +61.2 | 34,434 |
| d4 | 842–923 | 194.1 | +46.3 | 34,434 |
| d5 | 923–1020 | 188.7 | +28.7 | 34,434 |
| d6 | 1020–1134 | 201.1 | +12.8 | 34,434 |
| d7 | 1134–1266 | 230.7 | -9.7 | 34,434 |
| d8 | 1266–1505 | 248.7 | -60.2 | 34,434 |
| d9 | 1505–88132 | 1037.0 | -306.0 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 47 | 616 | 950 | 1466 | 75789 | 0 | 42 |
