"""Tail audit — how much of the holdout RMSE is unpredictable / garbage tail?

No model. Reads the raw training parquets, reconstructs the Jan+Jul 2025 DEP
holdout, and answers:

  1. What does the taxi-out label distribution look like in the tail, and how
     self-consistent is it (TAXITIME vs MVT-BLOCK vs MVT-AOBT_3 vs MVT-SCHED)?
  2. If a perfect model nailed every row with taxi <= C and could only predict
     a constant for taxi > C, what overall RMSE remains? (the "noise floor")
  3. What is the tail made of — which airports, months, echo rows, diversions?
  4. A dumb-but-honest baseline (per airport/stand-group/hour median, fit on the
     10 train months): its RMSE and the share each taxi band contributes.

Optionally, if reports/eval/<name>_holdout.parquet exists (MVT_ID_mvt, taxi,
pred), the real model residuals are decomposed by band too.

Run:  .venv/Scripts/python.exe src/eval/tail_audit.py [model_name]
Writes reports/tail_audit.md and prints the same to stdout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

try:  # Windows consoles are often cp1252; the report uses −, Σ, ≤, →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
TRAIN_GLOB = str(ROOT / "data" / "raw" / "training_2025-*.parquet")
OUT = ROOT / "reports" / "tail_audit.md"

HOLDOUT_MONTHS = ("2025-01", "2025-07")
REPORT_CEIL = 4 * 3600           # fit.py clips true taxi to [0, 4h] for reporting
ECHO_ABS_D = 30                  # |BLOCK - SCHED| below this == feed echoed schedule
BANDS = [0, 900, 1200, 1500, 1800, 2400, 3600, 5400, 7200, 10800, 10**9]
CUTOFFS = [1500, 1800, 2100, 2400, 3000, 3600, 5400, 7200]

COLS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "ADES_FILED_flt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt",
    "STAND_mvt", "AIRCRAFT_TYPE_mvt",
]

_L: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _L.append(s)


def md_table(rows, headers) -> None:
    say("| " + " | ".join(headers) + " |")
    say("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        say("| " + " | ".join(str(c) for c in r) + " |")
    say("")


def _secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = (
        pl.scan_parquet(TRAIN_GLOB)
        .select(COLS)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .with_columns(
            ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
            hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
            taxi=pl.col("TAXITIME_SEC_mvt").cast(pl.Int64),
            stand_group=pl.col("STAND_mvt").str.extract(r"^([A-Za-z]+)").fill_null("_"),
        )
        .with_columns(
            taxi_block=_secs("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"),
            taxi_aobt3=_secs("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
            taxi_sched=_secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
            d_block=_secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
            eobt_delay=_secs("EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
            iobt_delay=_secs("IOBT_flt", "SCHED_TIME_UTC_mvt"),
            aobt3_dev=_secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt"),
            diverted=(pl.col("ADES_FILED_flt").is_not_null()
                      & (pl.col("ADES_FILED_flt") != pl.col("ADES_mvt"))),
        )
        .with_columns(is_echo=pl.col("d_block").abs() < ECHO_ABS_D)
    )
    full = lf.collect()
    train = full.filter(~pl.col("ym").is_in(HOLDOUT_MONTHS))
    ho = full.filter(pl.col("ym").is_in(HOLDOUT_MONTHS)
                     & pl.col("taxi").is_between(0, REPORT_CEIL))
    return train, ho


def band_label(lo: int, hi: int) -> str:
    hs = "inf" if hi >= 10**9 else str(hi)
    return f"{lo}-{hs}"


def section_distribution(ho: pl.DataFrame) -> None:
    say("## 1. Holdout taxi distribution by band")
    say(f"\n{ho.height:,} DEP rows, Jan+Jul 2025, true taxi clipped to [0, 4h] "
        f"(same as the eval report).\n")
    med = ho["taxi"].median()
    tot_ss = float(ho.select(((pl.col("taxi") - med) ** 2).sum()).item())
    rows = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        b = ho.filter(pl.col("taxi").is_between(lo, hi - 1))
        if not b.height:
            continue
        ss = float(b.select(((pl.col("taxi") - med) ** 2).sum()).item())
        rows.append((
            band_label(lo, hi), f"{b.height:,}", f"{100 * b.height / ho.height:.2f}%",
            f"{b['taxi'].mean():.0f}", f"{b['taxi'].max()}",
            f"{100 * ss / tot_ss:.1f}%",
        ))
    md_table(rows, ["taxi band (s)", "n", "% rows", "mean", "max",
                    "% of Σ(taxi−median)²"])
    say(f"Median taxi {med:.0f} s. The last column is each band's share of the "
        f"total variance a constant predictor must explain — if a band holds a "
        f"big share from few rows, that RMSE is coming from outliers.\n")


def section_consistency(ho: pl.DataFrame) -> None:
    say("## 2. Is the label self-consistent?")
    n = ho.height
    exact = ho.filter(
        (pl.col("taxi") - pl.col("taxi_block")).abs() <= 2
    ).height
    say(f"\n- `TAXITIME_SEC_mvt == MVT_TIME − BLOCK_TIME` (±2 s): "
        f"**{100 * exact / n:.1f}%** of rows. The label is the block-derived taxi "
        f"by construction; a long taxi means BLOCK_TIME is long before takeoff.\n")

    have3 = ho.filter(pl.col("taxi_aobt3").is_not_null())
    for lo in (0, 1500, 2400, 3600):
        s = have3.filter(pl.col("taxi") >= lo)
        if not s.height:
            continue
        gap = s.select((pl.col("taxi") - pl.col("taxi_aobt3")).median()).item()
        neg = s.filter(pl.col("taxi_aobt3") < 0).height
        say(f"- taxi ≥ {lo:>4} s (n={s.height:,}, AOBT_3 present): median "
            f"`taxi − (MVT−AOBT_3)` = **{gap:.0f} s**"
            + (f"; {neg:,} rows have MVT−AOBT_3 < 0" if neg else ""))
    say("\nA large positive gap means the recorded taxi is inflated relative to "
        "the NM actual off-block — the block feed pushed the off-block earlier "
        "(schedule echo / ATFM absorption), it is not extra time on the tarmac.\n")

    echo = ho.filter(pl.col("is_echo"))
    say(f"- echo rows (|BLOCK−SCHED| < {ECHO_ABS_D} s): {echo.height:,} "
        f"({100 * echo.height / n:.1f}%), mean taxi {echo['taxi'].mean():.0f} s "
        f"vs {ho.filter(~pl.col('is_echo'))['taxi'].mean():.0f} s non-echo.\n")


def _floor(ho: pl.DataFrame, cutoff: int, tail_pred: str) -> tuple[float, float, int]:
    """RMSE if taxi<=cutoff is predicted exactly and taxi>cutoff gets `tail_pred`.

    tail_pred: 'airport_median' | 'global_median' | 'aobt3'
    Returns (rmse_floor, tail_rmse_contribution, n_tail).
    """
    n = ho.height
    tail = ho.filter(pl.col("taxi") > cutoff)
    if tail_pred == "global_median":
        p = pl.lit(ho["taxi"].median())
    elif tail_pred == "aobt3":
        p = pl.col("taxi_aobt3").clip(0, REPORT_CEIL).fill_null(ho["taxi"].median())
    else:
        amed = ho.group_by("ADEP_mvt").agg(pl.col("taxi").median().alias("_am"))
        tail = tail.join(amed, on="ADEP_mvt", how="left")
        p = pl.col("_am")
    sse = float(tail.select(((pl.col("taxi") - p) ** 2).sum()).item())
    return (sse / n) ** 0.5, (sse / n) ** 0.5, tail.height


def section_floor(ho: pl.DataFrame) -> None:
    say("## 3. Noise floor — RMSE that survives a perfect body model")
    say("\nAssume an oracle predicts every row with taxi ≤ C exactly. The rows "
        "above C are treated as unpredictable and get a constant. The remaining "
        "overall RMSE is a lower bound on what any model scores if the tail is "
        "truly not predictable from pre-pushback features.\n")
    rows = []
    for c in CUTOFFS:
        f_am, _, nt = _floor(ho, c, "airport_median")
        f_a3, _, _ = _floor(ho, c, "aobt3")
        rows.append((
            f"{c}", f"{nt:,}", f"{100 * nt / ho.height:.2f}%",
            f"{f_am:.1f}", f"{f_a3:.1f}",
        ))
    md_table(rows, ["cutoff C (s)", "n tail", "% rows",
                    "RMSE floor (airport median)", "RMSE floor (if AOBT_3 known)"])
    say("- Column 4: the model's best case if it cannot tell tail rows apart.\n"
        "- Column 5: the model's best case if it *could* recover the true "
        "off-block for tail rows (i.e. how much of the tail is echo/feed noise "
        "vs genuinely long taxi).\n")


def section_composition(ho: pl.DataFrame) -> None:
    say("## 4. What is the tail made of?  (taxi > 2400 s)")
    tail = ho.filter(pl.col("taxi") > 2400)
    say(f"\n{tail.height:,} rows ({100 * tail.height / ho.height:.2f}% of holdout).\n")
    for key in ("ADEP_mvt", "ym"):
        g = (tail.group_by(key).agg(pl.len().alias("n"))
             .with_columns(share=pl.col("n") / tail.height).sort("n", descending=True))
        md_table(
            [(r[key], f"{r['n']:,}", f"{100 * r['share']:.1f}%")
             for r in g.iter_rows(named=True)],
            [key, "n", "% of tail"],
        )
    flags = [
        ("echo (|BLOCK−SCHED|<30s)", pl.col("is_echo")),
        ("AOBT_3 missing", pl.col("taxi_aobt3").is_null()),
        ("MVT−AOBT_3 < 600 s (real taxi short)", pl.col("taxi_aobt3") < 600),
        ("diverted (ADES_FILED≠ADES)", pl.col("diverted")),
    ]
    md_table(
        [(lbl, f"{tail.filter(e).height:,}",
          f"{100 * tail.filter(e).height / tail.height:.1f}%") for lbl, e in flags],
        ["tail flag", "n", "% of tail"],
    )


def section_absurd(ho: pl.DataFrame) -> None:
    say("## 5. Physically implausible rows (taxi > 7200 s = 2 h)")
    ab = ho.filter(pl.col("taxi") > 7200).sort("taxi", descending=True)
    say(f"\n{ab.height:,} rows. A handful with the raw time columns:\n")
    cols = ["ADEP_mvt", "ym", "taxi", "taxi_block", "taxi_aobt3", "taxi_sched",
            "d_block", "is_echo", "diverted"]
    md_table(
        [tuple(r[c] if not isinstance(r[c], float) else round(r[c])
               for c in cols) for r in ab.head(15).iter_rows(named=True)],
        cols,
    )
    if ab.height:
        keep_ss = float(ho.filter(pl.col("taxi") <= 7200)
                        .select(((pl.col("taxi") - ho["taxi"].median()) ** 2).sum()).item())
        ab_ss = float(ab.select(((pl.col("taxi") - ho["taxi"].median()) ** 2).sum()).item())
        say(f"\nThese {ab.height:,} rows ({100 * ab.height / ho.height:.3f}% of the "
            f"holdout) are {100 * ab_ss / (ab_ss + keep_ss):.1f}% of the total "
            f"Σ(taxi−median)². Dropping them is not an option (the ranking set "
            f"has them too) but predicting their conditional mean is the only "
            f"sane play.\n")


def _honest_baseline(train: pl.DataFrame, ho: pl.DataFrame) -> pl.DataFrame:
    keys = [
        (["ADEP_mvt", "stand_group", "hour"], "p_asg"),
        (["ADEP_mvt", "hour"], "p_ah"),
        (["ADEP_mvt"], "p_a"),
    ]
    t = train.filter(pl.col("taxi").is_between(30, REPORT_CEIL))
    out = ho
    for gk, nm in keys:
        g = t.group_by(gk).agg(pl.col("taxi").median().alias(nm))
        out = out.join(g, on=gk, how="left")
    return out.with_columns(pred=pl.coalesce("p_asg", "p_ah", "p_a"))


def section_baseline(train: pl.DataFrame, ho: pl.DataFrame) -> None:
    say("## 6. Dumb-but-honest baseline — where does ITS error live?")
    b = _honest_baseline(train, ho)
    e = pl.col("pred") - pl.col("taxi")
    overall = float(b.select(e.pow(2).mean()).item() ** 0.5)
    say(f"\nPredictor: median taxi per (airport, stand-group, hour) from the 10 "
        f"train months. Overall RMSE **{overall:.1f} s** "
        f"(the lgb model is ~311; this is the no-skill reference).\n")
    rows = []
    tot_mse = float(b.select(e.pow(2).sum()).item())
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        s = b.filter(pl.col("taxi").is_between(lo, hi - 1))
        if not s.height:
            continue
        mse = float(s.select(e.pow(2).sum()).item())
        rows.append((
            band_label(lo, hi), f"{s.height:,}",
            f"{float(s.select(e.pow(2).mean()).item()) ** 0.5:.0f}",
            f"{float(s.select(e.mean()).item()):+.0f}",
            f"{100 * mse / tot_mse:.1f}%",
        ))
    md_table(rows, ["taxi band (s)", "n", "band RMSE", "band bias (pred−true)",
                    "% of total MSE"])


def section_model(ho: pl.DataFrame, name: str) -> None:
    p = ROOT / "reports" / "eval" / f"{name}_holdout.parquet"
    if not p.exists():
        say("## 7. Real model residuals by band")
        say(f"\n_Skipped — {p.relative_to(ROOT)} not found. Add "
            f"`ev.write_parquet(REPORT_DIR / f\"{{name}}_holdout.parquet\")` to "
            f"fit.py's run() to produce it, then re-run this audit._\n")
        return
    ev = pl.read_parquet(p).join(
        ho.select("MVT_ID_mvt", "is_echo", "taxi_aobt3", "diverted"),
        on="MVT_ID_mvt", how="inner",
    )
    say("## 7. Real model residuals by band")
    e = pl.col("pred") - pl.col("taxi")
    overall = float(ev.select(e.pow(2).mean()).item() ** 0.5)
    tot_mse = float(ev.select(e.pow(2).sum()).item())
    say(f"\n{name}: overall holdout RMSE {overall:.1f} s (n={ev.height:,}).\n")
    rows = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        s = ev.filter(pl.col("taxi").is_between(lo, hi - 1))
        if not s.height:
            continue
        mse = float(s.select(e.pow(2).sum()).item())
        rows.append((
            band_label(lo, hi), f"{s.height:,}",
            f"{float(s.select(e.pow(2).mean()).item()) ** 0.5:.0f}",
            f"{float(s.select(e.mean()).item()):+.0f}",
            f"{100 * mse / tot_mse:.1f}%",
        ))
    md_table(rows, ["taxi band (s)", "n", "band RMSE", "band bias", "% of total MSE"])
    for lbl, cond in [("echo", pl.col("is_echo")), ("non-echo", ~pl.col("is_echo"))]:
        s = ev.filter(cond)
        say(f"- {lbl}: RMSE {float(s.select(e.pow(2).mean()).item()) ** 0.5:.1f} s, "
            f"bias {float(s.select(e.mean()).item()):+.1f}, n={s.height:,}")
    say("")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "lgb_v2"
    say(f"# Tail audit — Jan+Jul 2025 holdout\n")
    say(f"_Generated by src/eval/tail_audit.py; model hook = `{name}`_\n")
    train, ho = load()
    section_distribution(ho)
    section_consistency(ho)
    section_floor(ho)
    section_composition(ho)
    section_absurd(ho)
    section_baseline(train, ho)
    section_model(ho, name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_L), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
