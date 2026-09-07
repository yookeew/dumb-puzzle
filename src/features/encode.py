"""Label-free encoding helpers shared by the feature builder and the model stage.

Kept separate from build_features.py so the model stage (and the Colab notebook)
can import just these — no dependency on the Polars-heavy family builders or the
stand linker.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

CAT_COLS = [
    "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "stand_group", "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_mvt",
    "AIRCRAFT_OPERATOR_flt", "inbound_actype", "link_confidence",
    "dep_rwy_config", "arr_rwy_config",
]

# columns that must never enter the model matrix: ids, raw datetimes, and
# anything derived from the DEP off-block time (d, taxi) or split bookkeeping (ym)
NON_FEATURES = {
    "MVT_ID_mvt", "T", "SOBT", "inbound_mvt_id", "sched_to_takeoff",
    "d", "taxi", "ym",
}

_STAND_GROUP = pl.col("STAND_mvt").str.extract(r"^([A-Za-z]+)").fill_null("_")


def fit_priors(dep_labeled: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Unimpeded-taxi percentiles from labelled training departures.

    dep_labeled needs: ADEP_mvt, RUNWAY_mvt, STAND_mvt, taxi (true seconds).
    Fit on the training split only — never the holdout / ranking rows.
    """
    d = dep_labeled.filter(pl.col("taxi").is_between(60, 5400)).with_columns(
        stand_group=_STAND_GROUP
    )
    q = 0.2
    return {
        "stand": d.group_by("ADEP_mvt", "STAND_mvt").agg(
            pl.col("taxi").quantile(q).alias("unimp_stand"),
            pl.col("taxi").median().alias("med_stand"),
            pl.len().alias("n_stand"),
        ),
        "sg_rwy": d.group_by("ADEP_mvt", "stand_group", "RUNWAY_mvt").agg(
            pl.col("taxi").quantile(q).alias("unimp_sg_rwy"),
            pl.len().alias("n_sg_rwy"),
        ),
        "rwy": d.group_by("ADEP_mvt", "RUNWAY_mvt").agg(
            pl.col("taxi").quantile(q).alias("unimp_rwy"),
            pl.col("taxi").median().alias("med_rwy"),
        ),
        "airport": d.group_by("ADEP_mvt").agg(
            pl.col("taxi").quantile(q).alias("unimp_airport"),
            pl.col("taxi").median().alias("med_airport"),
        ),
    }


def apply_priors(dep: pl.DataFrame, priors: dict[str, pl.DataFrame]) -> pl.DataFrame:
    d = dep.join(priors["stand"], on=["ADEP_mvt", "STAND_mvt"], how="left")
    d = d.join(priors["sg_rwy"], on=["ADEP_mvt", "stand_group", "RUNWAY_mvt"], how="left")
    d = d.join(priors["rwy"], on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
    d = d.join(priors["airport"], on="ADEP_mvt", how="left")
    return d.with_columns(
        unimpeded_taxi=pl.coalesce("unimp_stand", "unimp_sg_rwy", "unimp_rwy", "unimp_airport"),
        median_taxi_prior=pl.coalesce("med_stand", "med_rwy", "med_airport"),
        n_stand=pl.col("n_stand").fill_null(0),
        n_sg_rwy=pl.col("n_sg_rwy").fill_null(0),
    ).drop("unimp_stand", "unimp_sg_rwy", "unimp_rwy", "unimp_airport",
           "med_stand", "med_rwy", "med_airport")


# --------------------------------------------------------------------------- #
# Echo / pushback-delay group encodings (label-derived — same discipline as the
# taxi priors: fit on the training split, out-of-fold for the training rows).
#
# An "echo" row is one where the block-time feed just copies the schedule
# (|d| = |BLOCK − SCHED| < 30 s), so the recorded taxi absorbs the whole
# departure delay. That behaviour is stable per operator and per stand — some
# operators echo 2 % of the time, others 54 % (see reports/lirf_investigation.md).
# Giving the pooled model a historical echo-rate and mean-d per (airport,
# operator) and per (airport, stand) lets it recognise an echo row and predict
# the inflated value instead of guessing.
# --------------------------------------------------------------------------- #

GROUP_ENC_KEYS = {
    "op": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
    "stand": ["ADEP_mvt", "STAND_mvt"],
}
GROUP_ENC_COLS = [
    f"{p}_{s}" for p in GROUP_ENC_KEYS for s in ("echo_rate", "mean_d", "n")
]
_ECHO_ABS_D = 30          # |d| below this -> the feed echoed the schedule
_ENC_SMOOTH = 30.0        # pseudo-count pulling a group toward its airport mean
_D_CLIP = (-3600, 10800)  # clip d before averaging so the ~36 h poison tail
                          # (reports/lirf_investigation.md) can't wreck mean_d
_ENC_FOLD = "ym"          # out-of-fold unit for the training rows: calendar month


def _enc_prep(lab: pl.DataFrame) -> pl.DataFrame:
    """Labelled departures with the echo flag and a clipped d, d-nulls dropped.

    lab needs: ADEP_mvt, AIRCRAFT_OPERATOR_flt, STAND_mvt, d, and `_ENC_FOLD`.
    """
    return lab.filter(pl.col("d").is_not_null()).with_columns(
        _is_echo=(pl.col("d").abs() < _ECHO_ABS_D).cast(pl.Float64),
        _dc=pl.col("d").clip(*_D_CLIP).cast(pl.Float64),
    )


def _enc_airport(prepped: pl.DataFrame) -> pl.DataFrame:
    return prepped.group_by("ADEP_mvt").agg(
        pl.col("_is_echo").mean().alias("_a_echo"),
        pl.col("_dc").mean().alias("_a_meand"),
    )


def _enc_group(prepped: pl.DataFrame, ap: pl.DataFrame,
               keys: list[str], prefix: str) -> pl.DataFrame:
    raw = prepped.group_by(keys).agg(
        pl.col("_is_echo").mean().alias("_rate"),
        pl.col("_dc").mean().alias("_meand"),
        pl.len().alias("_n"),
    )
    j = raw.join(ap, on="ADEP_mvt", how="left")
    n, m = pl.col("_n"), _ENC_SMOOTH
    return j.with_columns(
        ((n * pl.col("_rate") + m * pl.col("_a_echo")) / (n + m)).alias(f"{prefix}_echo_rate"),
        ((n * pl.col("_meand") + m * pl.col("_a_meand")) / (n + m)).alias(f"{prefix}_mean_d"),
        pl.col("_n").cast(pl.Int32).alias(f"{prefix}_n"),
    ).select(keys + [f"{prefix}_echo_rate", f"{prefix}_mean_d", f"{prefix}_n"])


def fit_group_encodings(lab: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Echo-rate / mean-d group tables from labelled departures.

    Fit on the training split only, then `apply_group_encodings` onto the
    holdout / ranking frames. For the training frame itself use
    `add_group_encodings_oof` so a group's own rows don't leak into its estimate.
    """
    p = _enc_prep(lab)
    ap = _enc_airport(p)
    out: dict[str, pl.DataFrame] = {"_airport": ap}
    for name, keys in GROUP_ENC_KEYS.items():
        out[name] = _enc_group(p, ap, keys, name)
    return out


def apply_group_encodings(df: pl.DataFrame, enc: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join `fit_group_encodings` tables onto `df`; unseen groups fall back to
    the airport-level echo rate / mean d (and n = 0)."""
    d = df
    for name, keys in GROUP_ENC_KEYS.items():
        d = d.join(enc[name], on=keys, how="left")
    d = d.join(enc["_airport"], on="ADEP_mvt", how="left")
    for name in GROUP_ENC_KEYS:
        d = d.with_columns(
            pl.col(f"{name}_echo_rate").fill_null(pl.col("_a_echo")),
            pl.col(f"{name}_mean_d").fill_null(pl.col("_a_meand")),
            pl.col(f"{name}_n").fill_null(0),
        )
    return d.drop("_a_echo", "_a_meand")


def add_group_encodings_oof(f_train: pl.DataFrame, lab: pl.DataFrame) -> pl.DataFrame:
    """Out-of-fold group encodings for the training frame.

    Each row's echo-rate / mean-d come from labelled rows in every OTHER
    calendar month, so an operator (or stand) never sees its own rows in its
    estimate. Row order and columns of `f_train` are preserved; the six
    `GROUP_ENC_COLS` are added.

    `lab` needs MVT_ID_mvt plus the columns `_enc_prep` requires and `_ENC_FOLD`.
    """
    fold = lab.select("MVT_ID_mvt", _ENC_FOLD)
    keyed = f_train.join(fold, on="MVT_ID_mvt", how="left")
    p = _enc_prep(lab)
    parts = []
    for fo in p.select(_ENC_FOLD).unique().to_series().to_list():
        enc = fit_group_encodings(p.filter(pl.col(_ENC_FOLD) != fo))
        parts.append(apply_group_encodings(keyed.filter(pl.col(_ENC_FOLD) == fo), enc))
    covered = pl.concat(parts, how="vertical_relaxed").drop(_ENC_FOLD)
    # rows whose month is absent from `p` (all-null d that month) — rare; give
    # them the full-fit encoding rather than dropping them.
    missing = keyed.filter(~pl.col(_ENC_FOLD).is_in(p.select(_ENC_FOLD).unique().to_series()))
    if missing.height:
        covered = pl.concat(
            [covered, apply_group_encodings(missing.drop(_ENC_FOLD), fit_group_encodings(lab))],
            how="vertical_relaxed",
        )
    return f_train.select("MVT_ID_mvt").join(covered, on="MVT_ID_mvt", how="left")


def feature_matrix(
    feats: pl.DataFrame, categories: dict[str, list[str]] | None = None
) -> tuple[pl.DataFrame, list[str], list[str], dict[str, list[str]]]:
    """Encode categoricals to stable Int32 codes.

    Returns (X, feature_names, categorical_names, categories). Pass `categories`
    back on the eval/ranking frame so codes match the training fit; unseen values
    map to -1.
    """
    cats = [c for c in CAT_COLS if c in feats.columns]
    if categories is None:
        categories = {
            c: feats.select(pl.col(c).cast(pl.Utf8)).drop_nulls().unique()
            .to_series().sort().to_list()
            for c in cats
        }
    enc = [
        pl.col(c).cast(pl.Utf8)
        .replace_strict({v: i for i, v in enumerate(categories[c])},
                        default=-1, return_dtype=pl.Int32)
        .alias(c)
        for c in cats
    ]
    X = feats.with_columns(enc)
    names = [c for c in X.columns if c not in NON_FEATURES and X[c].dtype != pl.Datetime]
    X = X.with_columns([pl.col(c).cast(pl.Int8) for c in names if X[c].dtype == pl.Boolean])
    return X.select(names), names, cats, categories
