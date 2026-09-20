"""Family 5 — external ATFM/traffic context (EUROCONTROL Aviation Intelligence Unit).

Airport-day aggregates, not per-flight -- every departure at an airport on a
given day sees the same value, unlike families 2/3's per-flight rolling
windows. The genuinely new information here isn't volume (PROGRESS.md S18-19
found an internal-only disruption-volume proxy came back null -- the GBM
already captures congestion volume via existing interactions), it's *cause*:
nothing derivable from the movements/flight tables can say whether a bad day
was weather, staffing, or capacity. Same minutes, opposite meaning.

Source: https://ansperformance.eu (EUROCONTROL Aviation Intelligence Unit).
Licence: free to copy with attribution, non-commercial use only -- see
external-data/LICENSE. Admin-approved for this challenge, Discord ruling
2026-09-17. Full row-level detail in DATA_SOURCES.md.

    atfm_slot_adherence_{2025,2026}.csv  -- departure slot regulation/compliance
    apt_dly_{2025,2026}.csv.bz2          -- arrival ATFM delay by cause
    airport_traffic_{2025,2026}.csv      -- daily movement counts; also the
                                             "panel" (~377 ICAOs) used for
                                             ades_in_panel

All three are keyed (APT_ICAO, FLT_DATE) and never touch a blanked column --
safe to join onto train and ranking rows alike, no leak risk by construction.
"""

from __future__ import annotations

import bz2
import io
from functools import lru_cache
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "external-data"


def _read_csv_robust(path: Path) -> pl.DataFrame:
    """Some EUROCONTROL CSVs carry latin-1 bytes (accented airport/state
    names, e.g. Istanbul/Zurich) that choke polars' UTF-8 reader with a
    charmap error. Decode explicitly and re-encode rather than guessing."""
    raw = path.read_bytes()
    if path.suffix == ".bz2":
        raw = bz2.decompress(raw)
    text = raw.decode("latin-1")
    return pl.read_csv(io.BytesIO(text.encode("utf-8")))


def _read_years(prefix: str, ext: str, external_dir: Path) -> pl.DataFrame:
    frames = [
        _read_csv_robust(p)
        for year in (2025, 2026)
        if (p := external_dir / f"{prefix}_{year}.{ext}").exists()
    ]
    if not frames:
        raise FileNotFoundError(f"no {prefix}_*.{ext} under {external_dir}")
    return pl.concat(frames, how="vertical_relaxed")


def _parse_date(df: pl.DataFrame) -> pl.DataFrame:
    # apt_dly's FLT_DATE carries a "T00:00:00Z" suffix; slot_adherence/traffic
    # don't -- slicing the first 10 chars handles both.
    return df.with_columns(date=pl.col("FLT_DATE").str.slice(0, 10).str.to_date("%Y-%m-%d"))


def _slot_adherence(external_dir: Path) -> pl.DataFrame:
    """Was this airport's departure traffic under ATFM regulation today, and
    did regulated flights make their slot."""
    d = _parse_date(_read_years("atfm_slot_adherence", "csv", external_dir))
    reg = pl.col("FLT_DEP_REG_1")
    return d.select(
        "APT_ICAO", "date",
        reg_share=pl.when(pl.col("FLT_DEP_1") > 0)
        .then(reg / pl.col("FLT_DEP_1")).otherwise(None),
        in_slot_share=pl.when(reg > 0)
        .then(pl.col("FLT_DEP_IN_1") / reg).otherwise(None),
        late_share=pl.when(reg > 0)
        .then(pl.col("FLT_DEP_OUT_LATE_1") / reg).otherwise(None),
    )


def _apt_dly(external_dir: Path) -> pl.DataFrame:
    """Arrival-side ATFM delay by cause -- the "why" behind a regulated day.
    Only W(eather) and S(taffing) are named explicitly here; the remaining
    cause codes are folded into a residual share rather than guessed at."""
    d = _parse_date(_read_years("apt_dly", "csv.bz2", external_dir))
    total = pl.col("DLY_APT_ARR_1")
    return d.select(
        "APT_ICAO", "date",
        dly_min_per_flight=pl.when(pl.col("FLT_ARR_1") > 0)
        .then(total / pl.col("FLT_ARR_1")).otherwise(None),
        dly_weather_share=pl.when(total > 0)
        .then(pl.col("DLY_APT_ARR_W_1") / total).otherwise(None),
        dly_staffing_share=pl.when(total > 0)
        .then(pl.col("DLY_APT_ARR_S_1") / total).otherwise(None),
    )


def _airport_traffic(external_dir: Path) -> pl.DataFrame:
    """Daily movement counts -- the denominator that turns slot_adherence's
    raw counts into rates, and the ~377-ICAO panel `panel_icaos()` reads."""
    d = _parse_date(_read_years("airport_traffic", "csv", external_dir))
    return d.select(
        "APT_ICAO", "date",
        traffic_dep=pl.col("FLT_DEP_1"),
        traffic_arr=pl.col("FLT_ARR_1"),
        traffic_tot=pl.col("FLT_TOT_1"),
    )


@lru_cache(maxsize=1)
def load_atfm_daily(external_dir: str = str(EXTERNAL_DIR)) -> pl.DataFrame:
    """One row per (APT_ICAO, date) with all three sources full-outer-joined.

    A row missing from one source (e.g. an airport too small for
    slot_adherence but present in airport_traffic) keeps the columns from the
    sources that do have it and leaves the rest null -- never imputed here;
    `build_features._atfm_context` decides how nulls surface to the model.
    """
    ext = Path(external_dir)
    sa = _slot_adherence(ext)
    ad = _apt_dly(ext)
    tr = _airport_traffic(ext)
    return (
        sa.join(ad, on=["APT_ICAO", "date"], how="full", coalesce=True)
        .join(tr, on=["APT_ICAO", "date"], how="full", coalesce=True)
    )


@lru_cache(maxsize=1)
def panel_icaos(external_dir: str = str(EXTERNAL_DIR)) -> frozenset[str]:
    """ICAOs covered by airport_traffic -- the "~100 airport" (actually ~377)
    panel referenced by `ades_in_panel`. A destination outside this set gets
    no ATFM context at all, not a silently-imputed zero."""
    tr = _airport_traffic(Path(external_dir))
    return frozenset(tr.select("APT_ICAO").unique().to_series().to_list())
