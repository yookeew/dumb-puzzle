The 2026 edition of the PRC Data Challenge invites participants to predict the taxi-out time of flights from 11 major European airports.
Scope
The learning dataset will consist of all 11 airport movements for the full year of 2025.
Ranking
The applied ranking will use Root Mean Square Error (RMSE) for submitted predictions for the movements in January and July 2026.

A model to estimate taxi-out time can be used in post-operations analysis to identify constrained operations intervals and measure excess fuel burnt/CO2 production compared to unconstrained status.
The reasons for variability of taxi-out time are various and interlinked. These include airline-specific constraints, airport procedures and load, and Air Traffic Flow Management (ATFM)-related reasons.
To help you solve this year’s Data challenge, we provide the following (logical) datasets:
Movements (arrival and departure) information for 2025 at 11 major European airports including
taxi-out time for each movement at the airport and
scheduled and actual times
This is related to departing (PHASE = DEP) or arriving (PHASE = ARR) airport.
Flight information for each movement. This is what is matched in the Network Manager (NM) records.
The movements table rows have been (left) joined with the flight table ones.
There are a total of 4,167,797 movements at the 11 airports as shown in in Table 1.
Table 1: List of airports for which movements data is provided
ICAO
IATA
Name
EDDF
FRA
Frankfurt Main
EDDM
MUC
Munich
EGLL
LHR
London Heathrow
EHAM
AMS
Amsterdam Schiphol
LEBL
BCN
Josep Tarradellas Barcelona-El Prat
LEMD
MAD
Adolfo Suárez Madrid–Barajas
LFPG
CDG
Charles de Gaulle
LIRF
FCO
Rome–Fiumicino Leonardo da Vinci
LTAI
AYT
Antalya
LTFM
IST
İstanbul
LSZH
ZRH
Zürich

Note
Note 1
We have removed Military, Head of State, and Sensitive movements. These should not have a disruptive impact on the possibility to model and predict taxi-out time.
Note
Note 2
Real world data is messy. You will find some inconsistencies between movement and flight information. We haven’t tried to reconcile them because they are probably due to some limitations of the underlying processes trying to match different sources of data.
As per all our data challenges, we do not play with synthetic/artificial datasets.
Buckets
The 2024 PRC Data Challenge website has some instructions about how to use Minio to access OpenSky buckets at
https://ansperformance.eu/study/data-challenge/dc2024/data.html#using-minio-client
The Training Dataset
The training dataset is split into monthly files. Each file contains the movements for the airports of the challenge:
training_2025-01-01_2025-02-01.parquet (21M)
training_2025-02-01_2025-03-01.parquet (19M)
training_2025-03-01_2025-04-01.parquet (22M)
training_2025-04-01_2025-05-01.parquet (23M)
training_2025-05-01_2025-06-01.parquet (25M)
training_2025-06-01_2025-07-01.parquet (24M)
training_2025-07-01_2025-08-01.parquet (25M)
training_2025-08-01_2025-09-01.parquet (25M)
training_2025-09-01_2025-10-01.parquet (24M)
training_2025-10-01_2025-11-01.parquet (25M)
training_2025-11-01_2025-12-01.parquet (22M)
training_2025-12-01_2026-01-01.parquet (22M)
The Ranking Dataset
The ranking dataset contains the same columns as the training data with:
Movements for Jan and Jul 2026.
BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt for departures (movements with PHASE = "DEP") have been blanked out.
It consists of a single file:
ranking.parquet (27M)
The Submitting Dataset
The submitting dataset is the template to be filled with the predicted taxi-out times for submission. It only contains the columns MVT_ID_mvt and TAXITIME_SEC_mvt where MVT_ID_mvt values match those of the departures in the ranking dataset. It consists of only one file:
submitting.parquet (1.1M)
See Submission Instructions.
Column Description
Columns postfixed with _mvt are associated with the reporting airport.
Columns postfixed with _flt are associated with information from NM flight list.
Movement
MVT_ID_mvt: unique identifier for the movement record
FLIGHT_ID_mvt: unique identifier for the NM flight (if matched)
FLIGHT_mvt: flight number, i.e. what you see on your boarding pass
FLIGHT_RULE_mvt: which sets of regulations the flight is operated under. Possible values are:
I for Instrument Flight Rules (IFR)
V for Visual Flight Rules (VFR)
NA if unknown
ADEP_mvt: the (ICAO code of the) Aerodrome of DEParture (ADEP).
ADES_mvt: the (ICAO code of the) Aerodrome of DEStination (ADES).
PHASE_mvt: movement phase. DEP=departure, ARR=arrival.
MVT_TIME_UTC_mvt: (best available) movement time (takeoff if PHASE = DEP, landing if PHASE = ARR.)
BLOCK_TIME_UTC_mvt: block time (off-block if PHASE = DEP, in-block if PHASE = ARR.)
SCHED_TIME_UTC_mvt: scheduled time (of departure if PHASE = DEP, of arrival if PHASE = ARR.)
AIRCRAFT_TYPE_mvt: the ICAO code for the aircraft type, for example A21N for Airbus A321neo.
RUNWAY_mvt: the Runway (RWY) ID (of departure if PHASE = DEP, of arrival if PHASE = ARR.)
STAND_mvt: the stand ID (of departure if PHASE = DEP, of arrival if PHASE = ARR.)
TAXITIME_SEC_mvt: amount of seconds for taxi time (taxi-out if PHASE = DEP, taxi-in if PHASE = ARR.)
Flight
The following columns come from data recorded by NM.
LOBT_flt: last known off-block time
CALLSIGN_flt: the callsign of the relevant flight, e.g. BAW6VB.
ADEP_flt: the ICAO code of the ADEP.
ADES_flt: the ICAO code of the ADES.
ADES_FILED_flt: the ADES initially filed in the Flight Plan (FPL). If different from ADES_flt, then the flight has been diverted.
MARKET_SEGMENT_flt: the market segment type as defined in the Market Segment Rules, it can be:
“Mainline”
“Regional”
“Low-Cost”
“Business Aviation”
“All-Cargo”
“Charter” (Non-Scheduled)
“Military”
“Other”
“Not classified”
IOBT_flt: Initial Off-Block Time (IOBT).
FLIGHT_RULE_flt: which sets of regulations the flight is operated under (see FPL Item 8). Possible values are:
I for IFR
V for VFR
Y first IFR thereafter VFR
Z first VFR thereafter IFR
FLIGHT_TYPE_flt: flight type (see FPL Item 8). Possible values are:
S for scheduled air service
N for non-scheduled air service
G for general aviation
M for military (note: filtered out)
X for other than the preceding categories
AIRCRAFT_TYPE_flt: the ICAO code for the aircraft type, for example A30B for an Airbus A-300B2-200.
WK_TBL_CAT_flt: wake turbulence category (see FPL Item 9), can be:
L LIGHT, i.e. maximum certificated takeoff mass of 7000 kg (15_500 lbs) or less.
M MEDIUM, i.e. maximum certificated takeoff mass less than 136_000 kg (300_000 lbs), but more than 7_000 kg (15_500 lbs).
H HEAVY, i.e. maximum certificated takeoff mass of 136_000 kg (300_000 lbs) or more (except those specified as J).
J SUPER, presently only the AIRBUS A-380-800.
AIRCRAFT_OPERATOR_flt: the (anonymized) ICAO Airline Designator.
EOBT_1_flt: the Estimated Off-Block Time (EOBT) for FPL-based (M1) trajectory.
ARVT_1_flt: the arrival time for FPL-based (M1) trajectory.
AOBT_3_flt: the Actual Off-Block Time (AOBT) for flown (M3) trajectory.
ARVT_3_flt: the arrival time for flown (M3) trajectory.
List of Acronyms
ADEP: Aerodrome of DEParture
ADES: Aerodrome of DEStination
AOBT: Actual Off-Block Time
EOBT: Estimated Off-Block Time

RULES:
All used external datasets are openly accessible/usable and documented.
All produced source code are made openly available on GitHub under the GNU GPLv3 license. Note: It will then be forked by the Challenge GitHub account for organisational purposes.
All additional datasets used are openly available under an open source license.
Sufficient documentation is provided to understand and reproduce the results.
The solution is original: Teams must use their own original solutions. Re-using any existing implementation is only allowed if the original authors grant you the rights to use their solution and if you made significant modifications to the algorithm or model. In particular, simply re-using existing code and rewriting the data input and output mechanism is not sufficient. Adding parameters to the model and modifying filters to match the specific peculiarities of the data, however, can be considered sufficient.

Summary
EUROCONTROL PRC / OpenSky Network challenge. Predict TAXITIME_SEC_mvt for departures at 11 major European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTAI, LTFM, LSZH).
Train: all movements (arrivals + departures) for full-year 2025, ~4.17M rows, monthly parquet files.
Test: ranking.parquet, Jan + Jul 2026. For PHASE == "DEP" rows, BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are blanked. Nothing else is.
Metric: RMSE. Best score across all submissions counts.
Submission: fill TAXITIME_SEC_mvt in submitting.parquet, exact MVT_ID_mvt match, no missing or extra rows. Upload as <team-name>_v<n>.parquet.
Licence notes: LightGBM (MIT), XGBoost/CatBoost/PyArrow (Apache 2.0), Polars (MIT), scikit-learn/NetworkX (BSD) are all GPLv3-compatible. OSM data is ODbL — derived tables in data/external/ carry their own LICENSE and "© OpenStreetMap contributors" attribution. Prefer NOAA ISD for weather (public domain); avoid Meteostat (non-commercial clauses).
Core insight
Takeoff time (MVT_TIME_UTC_mvt) is not blanked, and the entire arrival side of the test set is intact. This is a retrospective reconstruction problem, not the forward forecasting problem the literature solves. Three consequences drive the whole design:
Flip the target. Predict pushback delay d = AOBT − SOBT, then recover taxi = (T − SOBT) − d. Identical loss, much better inductive bias. Note inbound delay is a negative predictor of taxi-out here.
Link departures to inbound aircraft via stand occupancy. Arrival in-block times are known exactly, so each departure can be paired with its inbound.
That pairing gives hard interval bounds. Pushback must fall between own arrival in-block (A_own) and next occupant's in-block (A_next), so L = max(0, T − A_next) ≤ taxi ≤ T − A_own = U. Under squared loss the optimal estimate is the truncated conditional mean.
Hard invariant
The feature builder never sees a departure off-block time. Null BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt on all DEP rows before any feature code runs, in train and test alike. Labels live in a separate table keyed on MVT_ID_mvt, joined only at fit time. build_features() takes one dataframe in ranking schema and cannot leak by construction.
Architecture
Stage 0 — ingest. Concatenate train + ranking into one frame per airport, arrivals and departures together, with a split column. Times as int64 epoch seconds, floats as float32, categoricals as codes with persisted mappings. Emit hours_of_available_context (July 2026 has no prior history; training rows never look like that).
Stage 1 — stand linking. Per (airport, stand), align arrival and departure sequences, reconcile counts over rolling 24h windows, accept links with plausible ground time and compatible aircraft type. Output inbound_mvt_id, link_confidence, A_own, A_next, L, U, bound_binding.
Stage 2 — features (~180 cols, six families): physical baseline (unimpeded percentile per stand-group/runway, OSM routed taxi distance) · takeoff-anchored congestion (exact rolling counts, inter-departure gaps, saturation runs) · runway configuration (inferred per 5-min bin, time since change) · rotation and schedule (bounds, inbound delay, ground time, EOBT/IOBT deltas) · weather and de-icing regime · categorical and calendar.
Stage 3 — models. Target d, objective L2, no transforms. (a) global LightGBM + per-airport residual model. (b) queue refinement: rebuild N(t)-style features from out-of-fold predicted off-blocks, never true ones, then refit. (c) distributional head: LightGBM quantiles 0.05–0.95, or binned empirical residuals as the cheap fallback.
Stage 3d — reference model (not scored; this is the deliverable PRC actually described). The full model predicts observed (constrained) taxi. The use case needs the unconstrained counterfactual to subtract from it. These are different models and must be fit separately — once takeoff time is a feature, the model can't answer "what would this have been on a quiet day", because the congestion is baked into the input. (IF TIME ALLOWS)
Fit taxi_unimpeded_hat on a restricted feature set: stand, runway, aircraft type, wake category, OSM routed distance, airport configuration. No congestion features, no takeoff time. Train on low-traffic periods only, or on all data with a low-quantile objective (~q10–q20). Then excess = taxi_observed − taxi_unimpeded_hat. Sanity-check against PRU's published reference taxi times; explain divergences (routed distance vs stand-group averages, type conditioning, config awareness). Costs one extra fit on a feature subset we already have, changes nothing about the submission.
Stage 4 — constraints. Truncate the predictive distribution to [L, U] where linking is confident, integrate for the truncated mean, clip to physical floor, convert back to taxi.
Stage 5 — eval. Holdout = Jan 2025 + Jul 2025, train on the other ten months. Report RMSE overall, per airport, per month, per decile of true taxi. Never random k-fold — neighbouring flights share congestion state.
Ensemble: LightGBM on d, CatBoost on d, LightGBM on taxi directly (decorrelated), turnaround estimator. Non-negative ridge blend on the holdout.
Compute
Colab free tier. Polars lazy/streaming, max_bin=127, cache every stage to cache/{stage}/{airport}.parquet. LightGBM CPU ≈ 10–15 min per fit on 2M × 180; XGBoost device="cuda" ≈ 90s if a T4 is available. No neural networks.
Eventual final layout before submission
src/
  ingest/      loaders, METAR fetch, OSM extract
  link/        stand alignment + bounds
  features/    build_features.py   <- single split-blind entry point
  models/      train, oof, quantile
  post/        truncate, clip, submit
  eval/        harness, per-slice reports
data/external/ routed_distances.parquet, metar.parquet, LICENSE (ODbL)
DATA_SOURCES.md
REPRODUCE.md

Planning:
Stage 0–1. Leak audit on AOBT_3_flt, LOBT_flt, EOBT_1_flt. Trivial baseline to prove submission path.
Stage 2 families 1–4, stage 3a. Real baseline.
Weather ingest, stage 3b, per-airport residual model.
Stage 3c + constraint layer. Measure truncation gain in isolation.
Ensemble, seasonal weighting, candidate submissions. Stage 3d reference model.
Freeze 4 Oct. Docs, reproduction script, JOAS draft.
Paper framing: the leaderboard scores the sum; the contribution is the split. Report the unimpeded/excess decomposition and, if we end up with both regimes, quantify what knowledge of the realised departure sequence is worth. State plainly that the model is for post-ops reconstruction and does not transfer to tactical pre-pushback prediction — say it before a reviewer does.
Open items
Do stage 1 validation first. On 2025 truth, measure per airport: link rate, binding rate, median U − L vs marginal sd, and violation rate. Violation rate above ~10% means towing has corrupted the alignment there; demote bounds to soft features for that airport. This determines whether the plan holds.
Leak audit. If AOBT_3_flt or LOBT_flt survive in ranking.parquet and track BLOCK_TIME_UTC_mvt closely, the problem changes shape. Build the leak columns as a toggleable feature block so a reissued ranking file costs a retrain, not a rewrite.
Verify MVT_TIME_UTC_mvt is populated for DEP rows in ranking.parquet before building anything on it. Two-minute null-count check. Sanity-check that MVT_TIME − SCHED_TIME looks like a plausible schedule-to-takeoff gap (~15–25 min centre, long right tail). If it's fully null the docs are stale and we fall back to a conventional forecasting architecture — hence the toggleable feature block.
Ask on OSN Discord (low urgency now, but free): confirm MVT_TIME_UTC_mvt availability for departures; whether OpenSky ADS-B ground trajectories are permitted for pushback detection; whether developing in a private repo before making it public is acceptable.
Conventions
Fixed seeds; LightGBM deterministic=true, force_row_wise=true.
Commit fetch scripts and small derived tables; caches are local only and must be rebuildable from source.
Every external source gets a row in DATA_SOURCES.md: name, URL, licence, date accessed, derived artefact, fetch script.

NOTE run stand-linking validation before building out the full 180 feature multimodal architection and keep a minimal working pipeline working end to end as an early safety net so that distributional head, queue refinement, reference model, and other fancy stuff don't prevent us from having a submittable, competitive entry
