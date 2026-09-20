# Holdout eval — metar_treatment_mixed

- engine: **lgb**   |   generated: 2026-09-20 12:02
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7997/direct=2669 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 336.4 | 155.1 | +2.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 655.7 | +22.5 | 1274 | 26,528 |
| LFPG | 579.1 | +1.6 | 1052 | 39,590 |
| EGLL | 289.6 | +10.5 | 1382 | 40,210 |
| LTFM | 269.8 | -14.5 | 1070 | 46,539 |
| LEBL | 224.0 | -2.0 | 957 | 28,985 |
| LSZH | 210.0 | -0.6 | 759 | 22,289 |
| EHAM | 199.8 | +1.6 | 798 | 40,949 |
| EDDF | 191.6 | +1.5 | 875 | 36,830 |
| EDDM | 186.9 | -4.0 | 832 | 27,091 |
| LEMD | 180.1 | +10.0 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 354.5 | 153,660 |
| 2025-07 | 321.0 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 240.4 | 339,016 | 50.3% |
| False | 1907.0 | 5,323 | 49.7% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1445.2 | 2,413 | 12.9% |
| False | 315.0 | 341,926 | 87.1% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.5 | +112.1 | 34,434 |
| d1 | 569–671 | 177.1 | +84.4 | 34,434 |
| d2 | 671–770 | 177.3 | +62.5 | 34,434 |
| d3 | 770–842 | 175.1 | +57.6 | 34,434 |
| d4 | 842–923 | 191.5 | +42.7 | 34,434 |
| d5 | 923–1020 | 188.6 | +25.2 | 34,434 |
| d6 | 1020–1134 | 201.8 | +9.5 | 34,434 |
| d7 | 1134–1266 | 231.5 | -13.1 | 34,434 |
| d8 | 1266–1505 | 257.8 | -62.9 | 34,434 |
| d9 | 1505–88132 | 874.9 | -297.6 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 614 | 946 | 1462 | 90114 | 5 | 40 |
