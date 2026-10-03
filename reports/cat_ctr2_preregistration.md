# CatBoost with categorical pairs (`max_ctr_complexity=2`) — pre-registration

Written 2026-09-30, before any `cat_ctr2` fit existed. Changes made after
results exist go under Amendments, dated.

## Why

Production CatBoost (`engine="cat"`) uses `max_ctr_complexity=1`: no
combinations of categorical features. We chose that only for speed
(PROGRESS.md §40): CatBoost's default combination search took ~70 min per
fit on a T4, against 5-6 min without it. It was never tested for
accuracy.

- The categoricals include stand, runway, operator, aircraft type and
  airport. Pairs such as stand × runway or operator × airport are exactly
  the interactions taxi time should depend on. Target statistics on pairs
  are what CatBoost does that LightGBM can't.
- The competitors' chat says the teams ahead of us gained from a stronger
  base model, not from external data (§46).
- **Evidence against:** in §40, the switch to `max_ctr_complexity=1`
  together with lr 0.05 scored *better* on the inner validation (flip 493
  vs 503). The two changes were confounded, so this is weak evidence, but
  it's why a cheap screen comes first.

## The one change

Engine `cat_ctr2` (`src/models/fit.py`): identical to `cat` (depth 8,
border_count 127, Huber 800, lr 0.05, 8,000-round cap, early stopping
patience 150, seed 42, `target="mixed"`) except `max_ctr_complexity=2`. The
echo classifier and both regressors use it.

## Stage 1 — screen (Colab T4, one `run()` without refit)

```python
run(engine="cat_ctr2", target="mixed", eta=0.05, seed=42, name="cat_ctr2_mixed",
    submit=False, ev_out="cache/eval/cat_ctr2_mixed_holdout_ev.parquet")
```

**Feasibility rule (a timing check, decided before any holdout number):**
Stage 2 needs 30 more fits (10 OOF folds × 3). So if the mean fit time in
stage 1 exceeds **25 min**, stage 2 isn't feasible by 2026-10-11 on Colab.
In that case stop and record the timing, whatever the screen says.

**Screen (`tests/cat_ctr2_screen_test.py`), uncorrected models:**
- Control: cross-fit NNLS(`lgb`, `cat`), where `cat` =
  `cat_mixed_rerun_holdout_ev.parquet` (the CatBoost v21 is built on).
- Treatment: cross-fit NNLS(`lgb`, `cat_ctr2`).
- Proceed to stage 2 iff all of:
  1. pooled trimmed delta < 0 with P(worse) < 0.05;
  2. trimmed delta < 0 in both Jan and Jul;
  3. full-RMSE guard P(worse) < 0.9.
- Trimmed = holdout labels ≤ 5 h (31 rows excluded, checked). P(worse) from
  the paired (airport, day) cluster bootstrap, 3,000 resamples, seed 0.
- Reported, not decisive: `cat_ctr2` vs `cat` as single models; per
  airport; per decile; stack weights; best_iter per fit; fit times.

**If the screen fails: stop.** No other `ctr` setting or rescue variant.

## Stage 2 — gate (only if stage 1 passes and is feasible)

1. **Colab:** `run_oof(engine="cat_ctr2", target="mixed", eta=0.05)` →
   `cache/oof/cat_ctr2_mixed/`. Then the production run with ranking rows:
   `run(engine="cat_ctr2", target="mixed", eta=0.05, name="cat_ctr2_mixed",
   submit=True, ev_out=..., rank_out="cache/eval/cat_ctr2_mixed_rank.parquet")`.
   Its holdout predictions are used throughout (the stage-1 file is only
   the screen).
2. **Corrector:** exactly v2 as in v21 (§49): Huber 800, selection on the
   base prediction ≤ 7,200 s, NM-unmatched echo rate, 5,000-round cap (§51:
   the cap doesn't matter). Fit on the `cat_ctr2` OOF folds → `catcorr_ctr2`.
3. **Gate:** cross-fit NNLS(`lgb`, `catcorr_ctr2`) vs v21's NNLS(`lgb`,
   `catcorr`), with the same three-part rule as stage 1.
4. **If it passes:** build v23 the same way as v21 (ranking corrector refit
   on 12 months at the gate's round count; stack; ADS-B stages refit on
   top). **Keep iff its leaderboard score < 270.2.**

If the gate fails: stop and record it. v21 stays.

## Amendments

(none)
