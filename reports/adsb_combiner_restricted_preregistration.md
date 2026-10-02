# ADS-B combiner restricted to rows with ADS-B information — pre-registration

Written 2026-10-02, after the v24 combiner's holdout and board results
(PROGRESS.md §58). Changes after the board result go under Amendments.

## Why

In the v24 gate, the combiner made rows with **no** ADS-B information
worse (trimmed RMSE 274.4 → 276.4; LTFM +8.0, LFPG +0.9). Besides the
ADS-B fields it sees airport, hour and the stack prediction. On rows with
nothing else to go on, those let it learn month-specific biases. LTFM's
residual bias is −39 s in Jan and +46 s in Jul (§53), which don't transfer
between months.

In v24, rows at airports with no ADS-B moved by 36 s (LTFM) and 49 s
(LFPG) RMS.

## The change

Same fitted combiner as v24, applied only where the detector has
something:
- `adsb_matched`, or
- a fallback tier (`fb_dwell` / `fb_appear`).

Every other row keeps v23's value, i.e. the stack prediction, clipped,
because no ADS-B stage touched it in v23. LIRF keeps v23's value as in
v24.

## Evaluation — the board only

**The holdout can't decide this.** The restriction reverts exactly the
"no ADS-B" group, whose holdout numbers were already seen in the v24 gate
(that group: 276.4 → 274.4). The holdout delta is therefore known in
advance, and a gate on it would be circular. It's reported for the record
only.

**Rule: build v25 = v24 restricted. Keep iff the board < 265.66 (v24).**
Otherwise v24 stays. No other restriction set (e.g. by airport) will be
tried.

## Amendments

(none)
