"""Pre-screen METAR/weather at (airport, day, hour) granularity, before
building anything -- same discipline as tests/atfm_ceiling.py's part_b
(PROGRESS.md S20), reused here since it's granularity-generic and doesn't
depend on any ATFM column.

Weather is genuinely hourly (unlike the EUROCONTROL daily feeds that capped
out at ~2.07s), so this should reproduce close to the documented ~10.82s
oracle ceiling at (airport, day, hour) on the current production residual
structure. If it doesn't, the production model has moved enough since S20
that the number needs updating before citing it.

Run:  .venv/Scripts/python.exe tests/metar_ceiling.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "cache" / "features"
PROD_EV = ROOT / "cache" / "lirf_ceilfix_mixed_ev.parquet"


def _decomp(ev: pl.DataFrame, keys: list[str], label: str, base_var: float) -> float:
    g = ev.group_by(keys).agg(pl.col("resid").mean().alias("mu"), pl.len().alias("n"))
    grand = ev.select(pl.col("resid").mean()).item()
    between = g.select(
        (pl.col("n") * (pl.col("mu") - grand).pow(2)).sum() / pl.col("n").sum()
    ).item()
    k, N = g.height, ev.height
    noise = base_var * (k - 1) / N
    adj = max(between - noise, 0.0)
    rmse0 = base_var**0.5
    gain_raw = rmse0 - max(base_var - between, 1e-9) ** 0.5
    gain_adj = rmse0 - max(base_var - adj, 1e-9) ** 0.5
    print(f"\n  {label}")
    print(
        f"    cells={k:,}  between={between:,.0f} ({between / base_var * 100:.1f}%)"
        f"  noise-expected={noise:,.0f}"
    )
    print(
        f"    naive oracle RMSE gain={gain_raw:.2f}s   noise-corrected={gain_adj:.2f}s"
    )
    return gain_adj


def main() -> None:
    ev = pl.read_parquet(PROD_EV).with_columns(resid=pl.col("taxi") - pl.col("pred"))
    f = pl.read_parquet(FEAT_DIR / "holdout_gap2025.parquet").select(
        "MVT_ID_mvt", hour="hour", day=pl.col("T").dt.date()
    )
    ev = ev.join(f, on="MVT_ID_mvt", how="left")
    base_var = ev.select(pl.col("resid").pow(2).mean()).item()
    print(f"production holdout RMSE = {base_var ** 0.5:.2f}s  (n={ev.height:,})")

    _decomp(ev, ["ADEP_mvt", "day"], "(airport, day) -- daily-granularity ceiling", base_var)
    hourly = _decomp(
        ev, ["ADEP_mvt", "day", "hour"],
        "(airport, day, hour) -- what HOURLY weather could reach", base_var,
    )
    print(f"\nHourly oracle ceiling: {hourly:.2f}s")
    print("(compare to the documented ~10.82s from PROGRESS.md S20; if close,")
    print("the bound still holds for the current production residual structure)")


if __name__ == "__main__":
    main()
