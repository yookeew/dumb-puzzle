"""§51 Step 1: stand-resolution audit, 2025 vs 2026 (label-free).

Question: did 2026 appear/dwell recovery (esp. EDDF: 36-48% on 2025-09/11 days,
2-8% on 2026 days, §37) drop because STAND_mvt stopped resolving to Gateway
coordinates (a naming/normalisation bug), or for another reason?

A. Resolution: % of DEP rows whose STAND_mvt resolves via src/ingest/stands.py,
   per airport and month (Jan/Jul 2025, Jan/Jul 2026); top-20 unresolved strings
   per airport and year.
B. Recovery vs resolution, from the detector output (cache/adsb_pushback/):
   per airport-year, appear/dwell % of DEP, and the same conditioned on
   coverage day + matched run + resolved stand; where the matched run is
   first seen relative to the resolved stand.
C. Per-stand persistence: stands with appear/dwell in 2025, their 2026 rate.
   A renamed apron would show as specific stand families dropping to ~0 while
   others hold; a receiver change drops everything together.
D. (needs raw ADS-B points in data/external/adsb/, i.e. Colab) Screen-1
   population: stationary prefix (gs <= 1 kt, >= 3 min) before first motion,
   its nearest Gateway stand and that stand's name vs STAND_mvt.
   Skipped with a message when the points aren't present.

Reads no labels. No production code changed.

Run:  .venv/Scripts/python.exe tests/step51_stand_audit.py > logs/step51_stand_audit.log
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from ingest.stands import attach_stand_coords, load_stands, stand_key, stand_positions  # noqa: E402
from post import adsb_partial  # noqa: E402

RAW = ROOT / "data" / "raw"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
DET = ROOT / "cache" / "adsb_pushback"
ADSB = ROOT / "data" / "external" / "adsb"
RUNWAYS = ROOT / "data" / "external" / "runways.csv"
STAT_GS, STAT_MIN_S = 1.0, 180.0


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def deps() -> pl.DataFrame:
    cols = dict(MVT_ID_mvt=pl.col("MVT_ID_mvt").cast(pl.Int64), ADEP_mvt=pl.col("ADEP_mvt"),
                STAND_mvt=pl.col("STAND_mvt"), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
                day=pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8),
                mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0)
    tr = pl.scan_parquet(str(RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP").select(**cols)
    rk = pl.scan_parquet(str(RANKING)).filter(pl.col("PHASE_mvt") == "DEP").select(**cols)
    return pl.concat([tr.collect(), rk.collect()]).with_columns(year=pl.col("ym").str.slice(0, 4))


def main() -> None:
    print(f"run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    files = sorted(DET.glob("day=*.parquet"))
    print(f"detector files {len(files)}, newest mtime "
          f"{dt.datetime.fromtimestamp(max(f.stat().st_mtime for f in files)):%Y-%m-%d %H:%M}")
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(1)
    pl.Config.set_fmt_str_lengths(60)

    stands = load_stands()
    pos = stand_positions(stands)
    d = attach_stand_coords(deps(), pos).with_columns(res=pl.col("stand_lat").is_not_null())

    # ------------------------------------------------------------------ A
    section("A. STAND_mvt -> Gateway resolution, % of DEP rows")
    jj = d.filter(pl.col("ym").str.slice(5, 2).is_in(["01", "07"]))
    print(jj.group_by("ADEP_mvt", "ym").agg((100 * pl.col("res").mean()).alias("pct"))
          .pivot(on="ym", index="ADEP_mvt", values="pct").select("ADEP_mvt", "2025-01", "2025-07",
                                                                 "2026-01", "2026-07").sort("ADEP_mvt"))
    print("all 2025 months (resolution by month, %):")
    print(d.filter(pl.col("year") == "2025").group_by("ADEP_mvt", "ym").agg((100 * pl.col("res").mean()).alias("p"))
          .pivot(on="ym", index="ADEP_mvt", values="p").sort("ADEP_mvt").select(
              "ADEP_mvt", *[f"2025-{m:02d}" for m in range(1, 13)]))
    print("top-20 unresolved STAND_mvt per airport and year (Jan/Jul months), n and % of that airport-year's DEP:")
    un = jj.filter(~pl.col("res"))
    tot = jj.group_by("ADEP_mvt", "year").len().rename({"len": "tot"})
    for ap in sorted(jj["ADEP_mvt"].unique().to_list()):
        g = (un.filter(pl.col("ADEP_mvt") == ap).group_by("year", "STAND_mvt").len()
             .join(tot, on=["year"], how="left").filter(pl.col("ADEP_mvt") == ap)
             .with_columns(pct=100 * pl.col("len") / pl.col("tot")))
        side = []
        for y in ("2025", "2026"):
            s = g.filter(pl.col("year") == y).sort("len", descending=True).head(20)
            side.append(s.select(pl.format("{} ({})", pl.col("STAND_mvt"), pl.col("len")).alias(f"unresolved {y}"))
                        .with_row_index())
        print(f"\n--- {ap}: unresolved share 2025 "
              f"{g.filter(pl.col('year') == '2025')['pct'].sum():.1f}% / 2026 "
              f"{g.filter(pl.col('year') == '2026')['pct'].sum():.1f}%")
        print(side[0].join(side[1], on="index", how="full", coalesce=True).drop("index"))

    # ------------------------------------------------------------------ B
    section("B. RECOVERY vs RESOLUTION (detector output, all ADS-B days)")
    det = pl.concat([pl.read_parquet(f).with_columns(day=pl.lit(f.stem.split("=", 1)[1])) for f in files])
    x = (det.drop("airport").join(d.select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "res", "year", "ym"),
                                  on="MVT_ID_mvt", how="left")
         .with_columns(ad=pl.col("adsb_tier").is_in(["appear", "dwell"]).fill_null(False),
                       matched=pl.col("adsb_matched").fill_null(False),
                       grp=pl.when(pl.col("day").is_in(["2025-09-15", "2025-11-15"])).then(pl.lit("2025 Sep/Nov"))
                       .otherwise(pl.col("year") + pl.lit(" Jan/Jul"))))
    miss = x["ADEP_mvt"].null_count()
    print(f"detector rows {x.height:,}; without a movement row {miss}")
    agg = x.group_by("ADEP_mvt", "grp").agg(
        pl.len().alias("dep"),
        (100 * pl.col("adsb_day_coverage").mean()).alias("cov_day%"),
        (100 * pl.col("res").mean()).alias("resolved%"),
        (100 * pl.col("matched").mean()).alias("matched%"),
        (100 * pl.col("ad").mean()).alias("AD%dep"),
        (100 * pl.col("ad").filter(pl.col("matched") & pl.col("res")).mean()).alias("AD%|matched&res"),
        (100 * pl.col("res").filter(pl.col("matched") & ~pl.col("ad")).mean()).alias("res%|matched&noAD"),
        pl.col("adsb_first_own_m").filter(pl.col("matched")).median().alias("med_first_own_m"),
        pl.col("adsb_min_own_m").filter(pl.col("matched")).median().alias("med_min_own_m"),
    ).sort("ADEP_mvt", "grp")
    print(agg)

    print("\nfirst-sighting distance to own (resolved) stand, matched runs: share by band (%)")
    band = pl.col("adsb_first_own_m").cut([100, 300, 1000, 3000]).cast(pl.Utf8)
    print(x.filter(pl.col("matched") & pl.col("res")).with_columns(b=band).group_by("ADEP_mvt", "grp", "b").len()
          .with_columns(pct=100 * pl.col("len") / pl.col("len").sum().over("ADEP_mvt", "grp"))
          .pivot(on="b", index=["ADEP_mvt", "grp"], values="pct").sort("ADEP_mvt", "grp")
          .select("ADEP_mvt", "grp", "(-inf, 100]", "(100, 300]", "(300, 1000]", "(1000, 3000]", "(3000, inf]"))

    print("\nEDDF by day (resolved%, matched%, AD%, median first-sighting distance):")
    print(x.filter(pl.col("ADEP_mvt") == "EDDF").group_by("day").agg(
        pl.len().alias("dep"), pl.col("adsb_day_coverage").first().alias("cov"),
        (100 * pl.col("res").mean()).alias("res%"), (100 * pl.col("matched").mean()).alias("matched%"),
        (100 * pl.col("ad").mean()).alias("AD%"),
        pl.col("adsb_first_own_m").filter(pl.col("matched")).median().alias("med_first_m"),
        pl.col("adsb_n_pts").filter(pl.col("matched")).median().alias("med_pts"))
        .sort("day").with_columns(month=pl.col("day").str.slice(0, 7))
        .group_by("month").agg(pl.col("dep").sum(), pl.col("cov").mean().alias("cov_days_frac"),
                               pl.col("res%").mean(), pl.col("matched%").mean(), pl.col("AD%").mean(),
                               pl.col("AD%").min().alias("AD%_min"), pl.col("AD%").max().alias("AD%_max"),
                               pl.col("med_first_m").median(), pl.col("med_pts").median()).sort("month"))

    # ------------------------------------------------------------------ C
    section("C. PER-STAND PERSISTENCE (stand families = leading letters of the key)")
    fam = x.filter(pl.col("matched") & pl.col("res")).with_columns(
        fam=stand_key(pl.col("STAND_mvt")).str.extract(r"^([A-Z]*)").fill_null(""))
    for ap in ("EDDF", "LEMD", "LSZH", "EGLL", "EDDM", "EHAM"):
        t = (fam.filter(pl.col("ADEP_mvt") == ap).group_by("fam", "grp")
             .agg(pl.len().alias("n"), (100 * pl.col("ad").mean()).alias("AD%"))
             .pivot(on="grp", index="fam", values=["n", "AD%"]))
        ncols = [c for c in t.columns if c.startswith("n_")]
        t = t.with_columns(tot=pl.sum_horizontal(ncols)).sort("tot", descending=True).head(12).drop("tot")
        print(f"\n--- {ap} (matched runs with resolved stand; AD% = appear/dwell share)")
        print(t)

    # ------------------------------------------------------------------ D
    section("D. STATIONARY PREFIX OF SCREEN-1 POPULATION vs NEAREST GATEWAY STAND")
    if not ADSB.exists():
        print(f"SKIPPED: raw ADS-B points not present at {ADSB.relative_to(ROOT)} "
              "(they live on Colab/Drive; run this script there for section D).")
    else:
        stationary_prefix(x, stands)
    print(f"run finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


def stationary_prefix(x: pl.DataFrame, stands: pl.DataFrame) -> None:
    """Locate each Screen-1 row's run in the raw points (by first sample ts and hex-free
    run matching on first_ts/n_pts), take its stationary prefix, compare to Gateway stands."""
    from src.ingest.normalise_adsb import load_day
    from src.link.adsb_pushback import GAP_S, SURFACE_GS, surface_runs, to_xy

    pop = x.filter(pl.col("matched") & ~pl.col("ad")).filter(~adsb_partial.mask())
    rw = pl.read_csv(RUNWAYS)
    out = []
    for day in sorted(pop["day"].unique().to_list()):
        p = pop.filter(pl.col("day") == day)
        pts = load_day(dt.date.fromisoformat(day))
        for (ap,), pa in p.group_by("ADEP_mvt"):
            st = stands.filter(pl.col("airport") == ap)
            if st.height == 0:
                continue
            lat0 = st["lat"].mean()
            sxy = to_xy(st["lat"].to_numpy(), st["lon"].to_numpy(), lat0)
            rxy = to_xy(rw.filter(pl.col("airport") == ap)["lat"].to_numpy(),
                        rw.filter(pl.col("airport") == ap)["lon"].to_numpy(), lat0)
            want = {(round(r["adsb_first_ts"], 3), int(r["adsb_n_pts"])): r for r in pa.to_dicts()}
            for _, g in pts.filter(pl.col("airport") == ap).group_by("hex"):
                g = g.sort("ts")
                ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
                gs = np.where(np.isnan(gs), SURFACE_GS, gs)
                xy = to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0)
                for i, j in surface_runs(ts, gs):
                    r = want.get((round(float(ts[i]), 3), j - i))
                    if r is None:
                        continue
                    k = i
                    while k < j and gs[k] <= STAT_GS:
                        k += 1
                    dur = float(ts[k - 1] - ts[i]) if k > i else 0.0
                    rec = dict(day=day, ADEP_mvt=ap, STAND_mvt=r["STAND_mvt"], res=r["res"],
                               prefix_s=dur, has_prefix=dur >= STAT_MIN_S)
                    if dur >= STAT_MIN_S:
                        c = xy[i:k].mean(axis=0)
                        dd = np.hypot(*(sxy - c).T)
                        n = int(dd.argmin())
                        rec.update(near_m=float(dd[n]), near_name=st["stand_name"][n],
                                   near_type=st["stand_type"][n], near_key=st["key"][n],
                                   rwy_m=float(np.hypot(*(rxy - c).T).min()) if len(rxy) else np.nan)
                    out.append(rec)
    o = pl.DataFrame(out)
    print(f"Screen-1 rows located in raw points: {o.height:,} of {pop.height:,}")
    o = o.with_columns(own_key=stand_key(pl.col("STAND_mvt")),
                       year=pl.col("day").str.slice(0, 4))
    print(o.group_by("ADEP_mvt", "year").agg(
        pl.len().alias("n"), (100 * pl.col("has_prefix").mean()).alias("prefix>=3min%"),
        pl.col("near_m").median().alias("med_near_m"),
        (100 * (pl.col("near_m") < 60).mean()).alias("near<60m%"),
        (100 * (pl.col("near_key") == pl.col("own_key")).mean()).alias("near==own%"))
        .sort("ADEP_mvt", "year"))
    print("most common (STAND_mvt -> nearest Gateway stand) pairs, prefix within 60 m of a stand:")
    print(o.filter(pl.col("near_m") < 60).group_by("ADEP_mvt", "STAND_mvt", "near_name", "near_type").len()
          .sort("len", descending=True).head(60))


if __name__ == "__main__":
    main()
