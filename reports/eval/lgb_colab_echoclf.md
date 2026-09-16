# Holdout eval — lgb_colab_echoclf

- engine: **lgb**   |   generated: 2026-09-16 14:53
- loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at 7998 (inner-valid 2025-06)  |  full-refit rounds 8798
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 397.3 | 164.4 | +7.9 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 883.1 | +17.7 | 1274 | 26,528 |
| LFPG | 602.5 | +13.0 | 1052 | 39,590 |
| EGLL | 303.3 | +10.5 | 1382 | 40,210 |
| EDDF | 290.7 | +10.0 | 875 | 36,830 |
| EHAM | 280.1 | +4.5 | 798 | 40,949 |
| LTFM | 277.9 | +2.9 | 1070 | 46,539 |
| LEBL | 264.4 | +5.7 | 957 | 28,985 |
| LSZH | 217.3 | +1.5 | 759 | 22,289 |
| EDDM | 205.8 | -1.5 | 832 | 27,091 |
| LEMD | 202.9 | +12.6 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 383.6 | 153,660 |
| 2025-07 | 408.0 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 265.6 | 339,016 | 44.0% |
| False | 2391.3 | 5,323 | 56.0% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1976.1 | 2,406 | 17.3% |
| False | 362.6 | 341,933 | 82.7% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 237.2 | +115.7 | 34,434 |
| d1 | 569–671 | 214.0 | +87.6 | 34,434 |
| d2 | 671–770 | 218.7 | +66.5 | 34,434 |
| d3 | 770–842 | 203.7 | +62.3 | 34,434 |
| d4 | 842–923 | 240.2 | +48.4 | 34,434 |
| d5 | 923–1020 | 230.6 | +31.6 | 34,434 |
| d6 | 1020–1134 | 264.3 | +16.6 | 34,434 |
| d7 | 1134–1266 | 284.6 | -5.1 | 34,434 |
| d8 | 1266–1505 | 323.2 | -53.6 | 34,434 |
| d9 | 1505–88132 | 1010.5 | -291.4 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 609 | 951 | 1478 | 74603 | 102 | 74 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 628 | 962 | 1042 | 1504 | 98618 | 82 | 95 |
