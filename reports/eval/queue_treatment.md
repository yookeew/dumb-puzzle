# Holdout eval — queue_treatment

- engine: **lgb**   |   generated: 2026-09-19 21:53
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=3539 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 381.1 | 155.7 | +3.2 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 920.1 | +11.3 | 1274 | 26,528 |
| LFPG | 580.1 | +3.0 | 1052 | 39,590 |
| EGLL | 286.8 | +11.3 | 1382 | 40,210 |
| LTFM | 267.4 | +0.4 | 1070 | 46,539 |
| LEBL | 224.4 | +0.7 | 957 | 28,985 |
| LSZH | 211.4 | -0.3 | 759 | 22,289 |
| EHAM | 200.3 | -2.9 | 798 | 40,949 |
| EDDF | 192.4 | +2.3 | 875 | 36,830 |
| EDDM | 189.7 | -4.7 | 832 | 27,091 |
| LEMD | 180.6 | +10.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 363.8 | 153,660 |
| 2025-07 | 394.5 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 253.5 | 339,016 | 43.6% |
| False | 2302.4 | 5,323 | 56.4% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1969.7 | 2,419 | 18.8% |
| False | 344.7 | 341,920 | 81.2% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 197.8 | +111.9 | 34,434 |
| d1 | 569–671 | 176.9 | +85.3 | 34,434 |
| d2 | 671–770 | 172.0 | +63.2 | 34,434 |
| d3 | 770–842 | 176.2 | +59.6 | 34,434 |
| d4 | 842–923 | 192.8 | +44.8 | 34,434 |
| d5 | 923–1020 | 189.3 | +27.4 | 34,434 |
| d6 | 1020–1134 | 201.8 | +12.0 | 34,434 |
| d7 | 1134–1266 | 230.9 | -10.1 | 34,434 |
| d8 | 1266–1505 | 249.3 | -60.1 | 34,434 |
| d9 | 1505–88132 | 1045.0 | -301.5 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 614 | 949 | 1467 | 72844 | 0 | 44 |
