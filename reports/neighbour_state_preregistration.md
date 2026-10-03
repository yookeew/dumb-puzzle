# Neighbour-state stage — pre-registration

Written 2026-10-03, after scoping on the 10 OOF training months and
**before any holdout (Jan/Jul 2025) row was read for this idea**. Changes
made after results exist go under Amendments, dated.

## Why (scoping, OOF months only; PROGRESS.md §70)

- **The errors are shared in time.** The OOF stack's residual correlates
  with the mean *true* residual of the other departures at the same
  airport within ±15 min at 0.14 (LIRF) to 0.44 (LTFM); most airports sit
  at 0.30-0.40. Within the same runway it's higher: LTFM 0.48, EDDF 0.37,
  LSZH 0.38, LFPG 0.35.
  - So a shared "how the airport/runway is running right now" component
    is missing from the model.
  - Knowing it would remove ~10-20% of residual variance.
- **Label-free proxies recover part of it**:
  - **Neighbours' NM pseudo-residual** (T - AOBT_3 - s) correlates with
    the true residual at |0.05-0.17|, mostly negatively. When NM
    off-blocks run early or late for a stretch, the model over-trusts
    each flight's own value.
  - **Neighbours' ADS-B pseudo-residual** (observed neighbours only),
    measured on rows without their own observed pushback, on the 115
    OOF ADS-B days: corr 0.13-0.27 at the seven ADS-B airports (oracle
    0.19-0.39).
    - §56 called this dead on 2 days; that sample was too small.
  - **Cross-fit screens** (LightGBM on r = taxi - s, even <-> odd
    months):
    - NM, offset and arrival neighbours over a control model: -1.4% /
      -1.5% residual RMS (all 10 OOF months);
    - NM + ADS-B neighbours on the ADS-B days: -1.1% / -1.2% overall,
      -1.2% / -1.3% on rows without their own observed pushback.
- **Expected:** ~1-1.5% RMSE, i.e. 2-4 board points. Possibly more in
  2026, where observed pushback coverage is higher (EGLL Jul 75%, EHAM
  and EDDM ~75%).

## Neighbour features (label-free; one builder for holdout, OOF days and ranking)

For each departure i at airport A with takeoff T_i: means over the
*other* departures j (or arrivals) with time in [T_i - W, T_i + W], for
W ∈ {900, 1800, 3600} s. Each is computed twice: same airport, and same
airport and runway (departures only). The mean is null when fewer than 2
contributors.

1. `nbp_*`: mean of p_j = (T_j - AOBT_3_j) - s_j over neighbours with
   0 < T_j - AOBT_3_j < 5,400.
2. `nbq_*`: mean of q_j = clip(adsb_taxi_j + lag_A - s_j, ±1,800) over
   neighbours with an observed pushback (tier appear/dwell). lag_A is the
   airport median of (taxi - adsb_taxi) on the training rows. Also
   `nbq_n_*`, the number of observed neighbours.
3. `nboff_*`: mean of clip(T_j - SOBT_j, -3,600, 14,400).
4. `arrx_*`: mean arrival taxi-in excess (in-block - landing - the 2025
   median for that (airport, stand, runway)) over arrivals landing in the
   window. Plus `arrx_prev`, arrivals landing in [T_i - 1,800, T_i - 900].
   Airport only.

`s` is the stack prediction the treatment is trained against: the OOF
stack on training days, the cross-fit NNLS stack on the holdout, and the
production stack on the ranking. Neighbour sets come from the same frame
(the same day's departures), never from labels.

## Treatment (`tests/neighbour_state_test.py`)

Two parts, both LightGBM on r = taxi - s with the combiner's PARAMS,
early stopping and day-of-month inner split:
- **(a) Rows with ADS-B information** (`has_adsb`): the full-year
  combiner (v27 method) with the neighbour features added to its FEATS.
- **(b) All other rows:** a neighbour corrector on [airport, hour, s, p,
  T - SOBT] plus the neighbour features.
- **Training, both parts:** the same rows as v27's combiner, i.e. the
  115 OOF-month ADS-B days plus the other holdout month (cross-fit).
  Labels ≤ 5 h; LIRF rows are excluded from (a)'s training, as in v27.
  (b) includes all airports.
- **Prediction** = s + model, clipped to [0, 140,000] at LIRF and [0,
  10,800] elsewhere.

**Base:** v27's method cross-fit (`tests/adsb_combiner_fullyear_test.py`
treatment).

## Rule

Trimmed RMSE (labels ≤ 5 h, the fixed 31-row set) decides. Adopt iff all
of:
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

P(worse) from the paired (airport, day) cluster bootstrap, 3,000
resamples, seed 0.

**Reported, not decisive:**
- parts (a) and (b) separately (each applied alone on top of the base);
- per airport;
- best iterations;
- neighbour-feature gain share.

**If it passes:** build v29 with `src/post/neighbour_state.py`, trained on
both holdout months plus the 115 days, with the production stack and
2026 neighbour features. **Keep iff the board < 262.13 (v27).**

**If it fails: stop.** No window, feature-subset or part-only variants on
the holdout.

## Amendments

- **2026-10-03, before any result:** the treatment's clip is changed to
  [0, 140,000] at every airport (from 10,800 outside LIRF).
  - v27's combiner doesn't clip, and the stack legitimately exceeds
    10,800 s on echo-hedged rows outside LIRF.
  - A 10,800 cap would change those rows for reasons unrelated to
    neighbours and contaminate the comparison.

## Results (2026-10-03, `tests/neighbour_state_test.py`, log `logs/neighbour_state_test.log`; run after the amendment commit)

**Primary: ADOPT.**
- **Training:** 660,242 OOF-month rows on 115 days, plus the other holdout
  month.
- **Best iterations:** (a) 1,037 / 951, (b) 696 / 1,059.
- **Neighbour-feature gain share:** (a) 0.30 / 0.35, (b) 0.46 / 0.54. The
  top neighbour inputs in (b) are `nbq_rwy_900` and `nbp_rwy_900`.

| trimmed RMSE | base (v27 method) | neighbour state | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 286.82 | 282.53 | −4.29 | [−7.23, −1.90] | 0.000 |
| B: Jan (fit Jul) | 214.90 | 211.86 | −3.04 | [−5.23, −0.56] | 0.011 |
| pooled | 257.22 | 253.44 | −3.78 | [−5.81, −2.07] | 0.000 |
| guard: pooled full | 322.65 | 320.45 | −2.20 | [−4.33, −0.28] | 0.013 |

- **Per airport (trimmed):** LTFM −11.94, EGLL −6.95, LIRF −4.81, LEMD
  −4.79, LEBL −1.65, LFPG −1.18, LSZH −0.23, EDDF +0.43, EHAM +0.79,
  EDDM +0.90.
- **Reported, not decisive:**
  - part (a) alone: pooled trimmed +0.29 (P(worse) 0.987), Jan +0.66,
    Jul +0.08;
  - part (b) alone: pooled trimmed −4.08 (P(worse) 0.000), Jul −4.37,
    Jan −3.71, full −2.44.
  - So the gain is all in (b). Adding neighbours to the ADS-B combiner
    doesn't help. Per the rule, v29 is built as pre-registered, (a)+(b).
    A (b)-only build would need its own pre-registration.

**Board: v29 = 262.33 (v27 262.13, +0.20). REJECTED; v27 stays.**
- From the two scores and the change vector c (no row-level reading):
  - Σc² = 1.96e9 and ΔSSE = +3.7e7, so Σc·(y − v27) = 9.6e8;
  - the SSE-optimal scale of c is ~0.49 (≈ 259.5 at that scale).
- So the changes point the right way in 2026 but are about 2× too large.
- **Diagnosis (label-free, holdout vs ranking features):** the neighbour
  features carry the airport's period-level NM bias and delay level, and
  those drift more between years than between Jan and Jul 2025.
  - nbp at LTFM Jan: +262 -> +164; LFPG Jul: −87 -> +44; EDDM Jan: −133
    -> −224.
  - nboff at LTFM and EDDF in Jan: about +500 s.
  - v29 shifted whole airports in 2026 (EGLL −21 s, LEMD (b) −21 s, LTFM
    (b) +14 s), and the Jan↔Jul cross-fit couldn't see that risk.
