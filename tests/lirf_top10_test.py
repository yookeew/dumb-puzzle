"""Read LIRF's top-10 worst holdout rows individually: what are they, is the
model's current prediction close or wildly off, and what would a hedge-
upward strategy recover.

Follow-up to reports/lirf_investigation.md SS4 ("top 10 rows = 58.4% of
LIRF's SSE") and SS7 (today's Q1-Q4). This pulls every raw field for each
of those 10 rows (not just the feature-matrix view) to actually read them:
takeoff/block/scheduled times, AOBT_3_flt (NM's independent off-block
read), destination-filed-vs-actual (diversion check), flight type/market
segment, plus the model's own diagnostic columns (echo_prob, taxi_model_raw,
has_aobt3) -- then quantifies how much of LIRF's and the WHOLE holdout's
squared error these 10 rows carry, and what a range of hedge strategies
(oracle, offset-anchored, clipped-raw) would recover.

Uses cache/lirf_ceilfix_mixed_ev.parquet (production v12) joined back to
the raw monthly training files for the holdout months (Jan+Jul 2025) --
the ev cache alone doesn't carry AOBT_3_flt, ADES_FILED_flt, FLIGHT_TYPE_flt
etc.

Run:  .venv/Scripts/python.exe tests/lirf_top10_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EV_PATH = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"
RAW_JAN = ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet"
RAW_JUL = ROOT / "data" / "raw" / "training_2025-07-01_2025-08-01.parquet"
RULE = "=" * 78
N_TOP = 10


def main() -> None:
    ev = pl.read_parquet(EV_PATH)
    total_se = float((ev["pred"] - ev["taxi"]).pow(2).sum())
    lirf = ev.filter(pl.col("ADEP_mvt") == "LIRF").with_columns(
        se=(pl.col("pred") - pl.col("taxi")).pow(2))
    lirf_se = float(lirf["se"].sum())

    top = lirf.sort("se", descending=True).head(N_TOP)
    top_se = float(top["se"].sum())

    print(f"{RULE}\nScale of the problem\n{RULE}")
    print(f"  total holdout SSE (all 10 airports, n={ev.height:,}): {total_se:,.0f}")
    print(f"  LIRF SSE (n={lirf.height:,}): {lirf_se:,.0f}  "
          f"({lirf_se / total_se * 100:.1f}% of total)")
    print(f"  top {N_TOP} LIRF rows' SSE: {top_se:,.0f}  "
          f"({top_se / lirf_se * 100:.1f}% of LIRF, "
          f"{top_se / total_se * 100:.1f}% of EVERYTHING scored)")

    ids = top["MVT_ID_mvt"].to_list()
    raw = pl.concat([
        pl.scan_parquet(RAW_JAN).filter(pl.col("MVT_ID_mvt").is_in(ids)),
        pl.scan_parquet(RAW_JUL).filter(pl.col("MVT_ID_mvt").is_in(ids)),
    ]).collect()

    joined = top.join(raw, on="MVT_ID_mvt", how="left", suffix="_raw").sort(
        "se", descending=True)

    print(f"\n{RULE}\nThe {N_TOP} rows, individually\n{RULE}")
    for i, r in enumerate(joined.iter_rows(named=True), 1):
        err = r["pred"] - r["taxi"]
        aobt3_gap = None
        if r.get("AOBT_3_flt") is not None and r.get("BLOCK_TIME_UTC_mvt") is not None:
            aobt3_gap = (r["AOBT_3_flt"] - r["BLOCK_TIME_UTC_mvt"]).total_seconds()
        diverted = (r.get("ADES_mvt") != r.get("ADES_FILED_flt")
                    if r.get("ADES_FILED_flt") is not None else None)
        print(f"\n--- #{i}  MVT_ID_mvt={r['MVT_ID_mvt']}  "
              f"({r['se'] / lirf_se * 100:.1f}% of LIRF SSE, "
              f"{r['se'] / total_se * 100:.2f}% of total) ---")
        print(f"  true taxi      = {r['taxi']:>8,.0f}s ({r['taxi']/3600:.1f}h)")
        print(f"  model pred     = {r['pred']:>8,.0f}s ({r['pred']/3600:.1f}h)   "
              f"error = {err:+,.0f}s")
        print(f"  taxi_model_raw = {r['taxi_model_raw']:>8,.0f}s   "
              f"(pre-blend/pre-clip regressor output)")
        print(f"  echo_prob={r['echo_prob']:.3f}  is_echo={r['is_echo']}  "
              f"has_aobt3={r['has_aobt3']}  use_prior={r['use_prior']}")
        print(f"  sched_takeoff_offset (MVT-SCHED) = {r['sched_takeoff_offset']:,.0f}s   "
              f"d (BLOCK-SCHED) = {r['d']:,.0f}s")
        print(f"  SCHED={r.get('SCHED_TIME_UTC_mvt')}  BLOCK={r.get('BLOCK_TIME_UTC_mvt')}  "
              f"MVT(takeoff)={r.get('MVT_TIME_UTC_mvt')}")
        if aobt3_gap is not None:
            print(f"  AOBT_3_flt={r.get('AOBT_3_flt')}  "
                  f"(AOBT_3 - BLOCK_TIME = {aobt3_gap:+,.0f}s)")
        else:
            print(f"  AOBT_3_flt = null (not NM-matched)")
        print(f"  ADEP->ADES = {r.get('ADEP_mvt_raw', r.get('ADEP_mvt'))}->"
              f"{r.get('ADES_mvt')}   ADES_FILED={r.get('ADES_FILED_flt')}"
              f"{'  <-- DIVERTED (filed != actual)' if diverted else ''}")
        print(f"  FLIGHT_TYPE={r.get('FLIGHT_TYPE_flt')}  "
              f"MARKET_SEGMENT={r.get('MARKET_SEGMENT_flt')}  "
              f"FLIGHT_RULE(mvt/flt)={r.get('FLIGHT_RULE_mvt')}/{r.get('FLIGHT_RULE_flt')}")
        print(f"  RUNWAY={r.get('RUNWAY_mvt')}  STAND={r.get('STAND_mvt')}  "
              f"AIRCRAFT_TYPE={r.get('AIRCRAFT_TYPE_mvt')}  OPERATOR={r['AIRCRAFT_OPERATOR_flt']}")

    # --- hedge-strategy what-ifs, applied ONLY to these 10 rows ------------
    print(f"\n{RULE}\nHedge-strategy what-ifs (override prediction on JUST these "
          f"{N_TOP} rows, recompute overall RMSE)\n{RULE}")
    base_rmse = (total_se / ev.height) ** 0.5
    print(f"  baseline overall RMSE: {base_rmse:.2f}s (n={ev.height:,})")

    taxi_v = top["taxi"].to_numpy()
    pred_v = top["pred"].to_numpy()
    raw_v = top["taxi_model_raw"].to_numpy()
    offset_v = top["sched_takeoff_offset"].to_numpy()

    strategies = {
        "current (production pred)": pred_v,
        "oracle (predict true taxi exactly)": taxi_v,
        "offset-anchored (predict sched_takeoff_offset)": offset_v,
        "clipped raw regressor, no echo blend (clip(raw,0,140000))":
            np.clip(raw_v, 0, 140000),
        "half-hedge toward offset (avg of pred and offset)":
            (pred_v + offset_v) / 2,
    }
    for name, alt_pred in strategies.items():
        alt_se_top = np.sum((alt_pred - taxi_v) ** 2)
        new_total_se = total_se - top_se + alt_se_top
        new_rmse = (new_total_se / ev.height) ** 0.5
        recovered_pct = (base_rmse - new_rmse) / base_rmse * 100
        print(f"\n  {name}:")
        print(f"    these {N_TOP} rows' SSE: {alt_se_top:,.0f}  "
              f"(was {top_se:,.0f})")
        print(f"    new overall RMSE: {new_rmse:.2f}s  "
              f"(delta {new_rmse - base_rmse:+.2f}s, {recovered_pct:+.2f}% "
              f"of baseline RMSE)")


if __name__ == "__main__":
    main()
