"""Tail cheat test — set the ceiling on a two-part model before building it.

Three reads off the raw parquets, no model:

  A. Real model error by taxi band — needs reports/eval/<name>_holdout.parquet
     (the `ev` dump fit.py now writes). Replaces the section-6 estimate in
     tail_audit. Skipped with a note if the file is not there yet.

  B. The cheat: take a flag for "inflated / echo" rows, set those predictions to
     the schedule->takeoff offset (MVT - SCHED, a known input), leave the rest
     alone, and score. That is the best a two-part model could ever do. Run with
     several flags (oracle and realistic) x several baselines for "the rest".

  C. AOBT_3: is it a model input already, and is its nullness a good echo
     detector? Both directions on the holdout, plus the null rate on the 2026
     ranking set (where BLOCK is blanked, so the echo flag itself is unavailable
     and a ranking-computable proxy is what a real classifier must use).

Run:  .venv/Scripts/python.exe src/eval/tail_cheat.py [model_name]
Writes reports/tail_cheat.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.tail_audit import (  # noqa: E402
    BANDS,
    REPORT_CEIL,
    _honest_baseline,
    band_label,
    load,
)

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
OUT = ROOT / "reports" / "tail_cheat.md"
TAIL = 2400  # "inflated" reference outcome: taxi > TAIL

_L: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _L.append(s)


def md(rows, headers) -> None:
    say("| " + " | ".join(headers) + " |")
    say("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        say("| " + " | ".join(str(c) for c in r) + " |")
    say("")


def _rmse(df: pl.DataFrame, pred: pl.Expr) -> float:
    return float(df.select((pred - pl.col("taxi")).pow(2).mean()).item() ** 0.5)


# --------------------------------------------------------------------------- A
def part_a(ho: pl.DataFrame, name: str) -> None:
    say("## A. Real model error by taxi band")
    p = ROOT / "reports" / "eval" / f"{name}_holdout.parquet"
    if not p.exists():
        say(f"\n_Blocked — `{p.relative_to(ROOT)}` not found. It is written by "
            f"fit.py's run() now; produce it with one Colab run, then re-run "
            f"this script._\n")
        return
    ev = pl.read_parquet(p)
    e = pl.col("pred") - pl.col("taxi")
    tot = float(ev.select(e.pow(2).sum()).item())
    overall = float(ev.select(e.pow(2).mean()).item() ** 0.5)
    say(f"\n{name}: overall holdout RMSE **{overall:.1f}** (n={ev.height:,}).\n")
    rows = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        s = ev.filter(pl.col("taxi").is_between(lo, hi - 1))
        if not s.height:
            continue
        sse = float(s.select(e.pow(2).sum()).item())
        rows.append((
            band_label(lo, hi), f"{s.height:,}",
            f"{float(s.select(e.pow(2).mean()).item()) ** 0.5:.0f}",
            f"{float(s.select(e.mean()).item()):+.0f}",
            f"{100 * sse / tot:.1f}%",
        ))
    md(rows, ["taxi band (s)", "n", "band RMSE", "band bias", "% of total MSE"])


# --------------------------------------------------------------------------- B
def _flags(ho: pl.DataFrame) -> dict[str, pl.Expr]:
    aobt3_gap = pl.col("taxi") - pl.col("taxi_aobt3")  # label - real tarmac time
    return {
        "echo (BLOCK≈SCHED, <30s)": pl.col("is_echo"),
        "label >> real taxi (gap>600s, AOBT_3 present)":
            pl.col("taxi_aobt3").is_not_null() & (aobt3_gap > 600),
        "AOBT_3 null": pl.col("taxi_aobt3").is_null(),
        "AOBT_3 null OR echo": pl.col("taxi_aobt3").is_null() | pl.col("is_echo"),
        "ORACLE outcome: taxi>2400s": pl.col("taxi") > TAIL,
    }


def _parts(ho: pl.DataFrame) -> dict[str, pl.Expr]:
    """Candidate part-2 predictions for the flagged rows."""
    return {
        "offset (MVT−SCHED)": pl.col("taxi_sched"),
        "MVT−AOBT_3 (clip 0,4h)":
            pl.col("taxi_aobt3").clip(0, REPORT_CEIL)
            .fill_null(pl.col("apmed")),
        "airport median": pl.col("apmed"),
        "honest baseline": pl.col("honest"),
    }


def part_b(train: pl.DataFrame, ho: pl.DataFrame) -> None:
    say("## B. The cheat — ceiling on a two-part model")
    say(f"\nFlagged rows get a part-2 prediction; the rest get an oracle "
        f"(true taxi). Overall RMSE = the best a two-part model could do with "
        f"that flag + that part-2. Reference outcome: taxi > {TAIL}s.\n")

    hb = _honest_baseline(train, ho).select("MVT_ID_mvt", "pred")
    ho = ho.join(hb.rename({"pred": "honest"}), on="MVT_ID_mvt", how="left")
    amed = ho.group_by("ADEP_mvt").agg(pl.col("taxi").median().alias("apmed"))
    ho = ho.join(amed, on="ADEP_mvt", how="left")

    say(f"No-cheat baselines: honest median **{_rmse(ho, pl.col('honest')):.1f}**, "
        f"per-airport median **{_rmse(ho, pl.col('apmed')):.1f}**, "
        f"oracle body **0.0**. Current lgb model ≈ **311**.\n")

    n_tail = ho.filter(pl.col("taxi") > TAIL).height
    parts = _parts(ho)

    say("### Flag coverage vs taxi>2400s\n")
    cov = []
    for lbl, flag in _flags(ho).items():
        f = ho.with_columns(_f=flag.fill_null(False))
        nf = f.filter(pl.col("_f")).height
        hit = f.filter(pl.col("_f") & (pl.col("taxi") > TAIL)).height
        cov.append((lbl, f"{nf:,}", f"{(hit / nf if nf else 0):.0%}",
                    f"{(hit / n_tail if n_tail else 0):.0%}"))
    md(cov, ["flag", "n flagged", "precision", "recall"])

    say("### Overall RMSE: oracle body + (flag → part-2)\n")
    rows = []
    for lbl, flag in _flags(ho).items():
        f = ho.with_columns(_f=flag.fill_null(False))
        rows.append((lbl, *[
            f"{_rmse(f, pl.when(pl.col('_f')).then(p).otherwise(pl.col('taxi'))):.1f}"
            for p in parts.values()
        ]))
    md(rows, ["flag \\ part-2", *parts.keys()])
    say("- Every cell assumes a **perfect body** and **perfect flag detection** "
        "— it is an upper bound.\n"
        "- If the best cell is ~290 the two-part model is not the fix; "
        "~250 and it is. Compare to the current ≈311.\n")


# --------------------------------------------------------------------------- C
def part_d(train: pl.DataFrame, ho: pl.DataFrame) -> None:
    """Is the feed inflation predictable? taxi = (MVT−AOBT_3) + (AOBT_3−BLOCK).
    First term is a known feature; test whether the second (the inflation) can
    be predicted from group history, which would make a target reframe worth it.
    """
    say("## D. Is the inflation `AOBT_3 − BLOCK` predictable?")
    infl = (pl.col("aobt3_dev") - pl.col("d_block"))  # = AOBT_3 − BLOCK
    tr = train.filter(pl.col("taxi_aobt3").is_not_null()).with_columns(infl=infl)
    h = ho.filter(pl.col("taxi_aobt3").is_not_null()).with_columns(infl=infl)
    say(f"\nAOBT_3 present: {tr.height:,} train / {h.height:,} holdout rows "
        f"({100 * h.height / ho.height:.1f}% of holdout).\n")

    q = h.select(
        p1=pl.col("infl").quantile(0.01), p50=pl.col("infl").quantile(0.5),
        p90=pl.col("infl").quantile(0.9), p99=pl.col("infl").quantile(0.99),
        near0=(pl.col("infl").abs() < 60).mean(), big=(pl.col("infl") > 600).mean(),
    ).row(0, named=True)
    say(f"- inflation p1/p50/p90/p99 = {q['p1']:.0f} / {q['p50']:.0f} / "
        f"{q['p90']:.0f} / {q['p99']:.0f} s;  |infl|<60s for "
        f"**{q['near0']:.0%}**, infl>600s for **{q['big']:.0%}**\n")

    # cheap group-median predictor of the inflation, fit on train
    gk = ["ADEP_mvt", "hour"]
    gm = tr.group_by(gk).agg(pl.col("infl").median().alias("infl_hat"))
    h2 = h.join(gm, on=gk, how="left").with_columns(
        pl.col("infl_hat").fill_null(tr["infl"].median())
    )
    r0 = float(h2.select(pl.col("infl").pow(2).mean().sqrt()).item())
    rg = float(h2.select((pl.col("infl") - pl.col("infl_hat")).pow(2)
                         .mean().sqrt()).item())
    say(f"- predict inflation with 0: RMSE {r0:.0f} s;  with (airport,hour) "
        f"median: RMSE {rg:.0f} s  → {'' if rg < r0 - 20 else 'NOT '}"
        f"meaningfully predictable from coarse history\n")

    # what would taxi RMSE be with known-real-taxi + group-median inflation?
    tt = h2.with_columns(
        pred=pl.col("taxi_aobt3").clip(0, REPORT_CEIL) + pl.col("infl_hat")
    )
    r_reframe = _rmse(tt, pl.col("pred"))
    r_aobt3_only = _rmse(h2, pl.col("taxi_aobt3").clip(0, REPORT_CEIL))
    say(f"- taxi RMSE on the AOBT_3-present holdout: `MVT−AOBT_3` alone "
        f"**{r_aobt3_only:.1f}**;  `MVT−AOBT_3 + inflation_median` "
        f"**{r_reframe:.1f}**  (current lgb ≈ 311 on all rows)\n")


def _rate(df: pl.DataFrame, cond: pl.Expr, given: pl.Expr | None = None) -> str:
    d = df.filter(given) if given is not None else df
    if not d.height:
        return "n/a"
    return f"{100 * d.filter(cond).height / d.height:.1f}%  (n={d.height:,})"


def part_c(ho: pl.DataFrame) -> None:
    say("## C. AOBT_3 — input already? nullness a good detector?")
    say("\n`AOBT_3_flt` **is** a model input: build_features emits "
        "`aobt3_taxi = MVT − AOBT_3` and `aobt3_vs_eobt = AOBT_3 − EOBT_1`. "
        "It does **not** emit `AOBT_3 − SCHED` (the real schedule deviation) or "
        "an explicit `aobt3_missing` flag — both are cheap adds.\n")

    null = pl.col("taxi_aobt3").is_null()
    echo = pl.col("is_echo")
    big = pl.col("taxi") > TAIL
    huge = pl.col("taxi") > 7200
    md(
        [
            ("AOBT_3 null | all rows", _rate(ho, null)),
            ("AOBT_3 null | echo", _rate(ho, null, echo)),
            ("AOBT_3 null | taxi>2400", _rate(ho, null, big)),
            ("AOBT_3 null | taxi>7200", _rate(ho, null, huge)),
            ("echo | AOBT_3 null", _rate(ho, echo, null)),
            ("echo | AOBT_3 present", _rate(ho, echo, ~null)),
            ("taxi>2400 | AOBT_3 null", _rate(ho, big, null)),
            ("taxi>2400 | AOBT_3 present", _rate(ho, big, ~null)),
        ],
        ["quantity", "value"],
    )


def part_c_ranking() -> None:
    say("### C (ranking) — does the feed behave the same in 2026?")
    if not RANKING.exists():
        say(f"\n_Skipped — {RANKING.relative_to(ROOT)} not found._\n")
        return
    r = (
        pl.scan_parquet(RANKING)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt",
        )
        .with_columns(
            ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
            offset=(pl.col("MVT_TIME_UTC_mvt")
                    - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
            aobt3_null=pl.col("AOBT_3_flt").is_null(),
            block_null=pl.col("BLOCK_TIME_UTC_mvt").is_null(),
        )
        .collect()
    )
    say(f"\n{r.height:,} ranking DEP rows. BLOCK blanked: "
        f"{100 * r['block_null'].mean():.1f}%  (echo flag unavailable here).\n")
    say(f"- AOBT_3 null overall: **{100 * r['aobt3_null'].mean():.1f}%** "
        f"(holdout DEP was in part C above)\n")
    g = (
        r.group_by("ADEP_mvt")
        .agg(
            pl.len().alias("n"),
            (100 * pl.col("aobt3_null").mean()).alias("aobt3_null_%"),
            (100 * (pl.col("offset") > TAIL).mean()).alias("offset>2400_%"),
            (100 * (pl.col("aobt3_null") & (pl.col("offset") > 1800)).mean())
            .alias("null&off>1800_%"),
        )
        .sort("aobt3_null_%", descending=True)
    )
    md(
        [(row["ADEP_mvt"], f"{row['n']:,}", f"{row['aobt3_null_%']:.1f}%",
          f"{row['offset>2400_%']:.1f}%", f"{row['null&off>1800_%']:.1f}%")
         for row in g.iter_rows(named=True)],
        ["airport", "n", "AOBT_3 null %", "offset>2400 %",
         "AOBT_3 null & offset>1800 %"],
    )
    say("Last column ~ the share of ranking rows a null-based classifier would "
        "flag as inflated. Compare `offset>2400 %` to the holdout tail rate "
        "(1.4%) to see if 2026 is heavier.\n")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "lgb_v2"
    say("# Tail cheat test — Jan+Jul 2025 holdout\n")
    train, ho = load()
    part_a(ho, name)
    part_b(train, ho)
    part_c(ho)
    part_c_ranking()
    part_d(train, ho)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_L), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
