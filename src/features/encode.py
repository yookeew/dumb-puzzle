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
