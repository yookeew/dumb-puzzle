# Holdout eval — seed_ens_13

- engine: **lgb**   |   generated: 2026-09-19 21:18
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=3116 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 379.4 | 156.5 | +4.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 908.7 | +11.1 | 1274 | 26,528 |
| LFPG | 581.1 | +6.4 | 1052 | 39,590 |
| EGLL | 288.7 | +10.2 | 1382 | 40,210 |
| LTFM | 267.6 | +1.4 | 1070 | 46,539 |
| LEBL | 224.2 | +2.1 | 957 | 28,985 |
| LSZH | 211.6 | -0.1 | 759 | 22,289 |
| EHAM | 200.3 | -2.5 | 798 | 40,949 |
| EDDF | 192.5 | +3.2 | 875 | 36,830 |
| EDDM | 189.9 | -3.7 | 832 | 27,091 |
| LEMD | 181.1 | +11.3 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 366.5 | 153,660 |
| 2025-07 | 389.4 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 253.1 | 339,016 | 43.8% |
| False | 2287.3 | 5,323 | 56.2% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1994.4 | 2,406 | 19.3% |
| False | 342.0 | 341,933 | 80.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 199.0 | +113.3 | 34,434 |
| d1 | 569–671 | 179.1 | +87.2 | 34,434 |
| d2 | 671–770 | 173.7 | +64.7 | 34,434 |
| d3 | 770–842 | 177.2 | +61.1 | 34,434 |
| d4 | 842–923 | 194.4 | +46.2 | 34,434 |
| d5 | 923–1020 | 189.1 | +28.6 | 34,434 |
| d6 | 1020–1134 | 201.7 | +12.9 | 34,434 |
| d7 | 1134–1266 | 231.4 | -9.3 | 34,434 |
| d8 | 1266–1505 | 249.4 | -59.5 | 34,434 |
| d9 | 1505–88132 | 1037.3 | -304.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 43 | 615 | 950 | 1467 | 75719 | 0 | 42 |
