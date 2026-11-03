# AVAIL ML Backend Proof of Concept

AVAIL ML Backend PoC is a fully executable Python project demonstrating the Phase 1 athlete training-load pipeline:

```text
Synthetic athlete sessions
        ↓
Validation and preprocessing
        ↓
Prior-only individual baselines
        ↓
Pre-training Readiness Score + completed Training Load Score
        ↓
Coach-reviewed Decision Support
        ↓
FastAPI JSON endpoints
```

The project demonstrates the data flow, calculation boundaries, API contract, and test strategy needed for an early backend integration without adding a database, cloud services, wearable integrations, or unsupported predictive claims.

> **Important:** This is a technical proof of concept, not a scientifically validated athlete-monitoring product. Every athlete and record is synthetic. The calculations and score weights are illustrative engineering choices. `Readiness Score` is a provisional context indicator, not a validated recovery measure. No output is an injury-risk estimate, diagnosis, medical recommendation, or autonomous training instruction.

## Contents

- [MVP capabilities](#mvp-capabilities)
- [What counts as ML](#what-counts-as-ml)
- [Project structure](#project-structure)
- [Requirements and quick start](#requirements-and-quick-start)
- [Data model](#data-model)
- [Processing pipeline](#processing-pipeline)
- [Load Score methodology](#load-score-methodology)
- [API reference](#api-reference)
- [Testing](#testing)
- [Design decisions](#design-decisions)
- [Known limitations](#known-limitations)
- [Production readiness](#production-readiness)
- [Suggested next phases](#suggested-next-phases)

## MVP capabilities

The PoC implements the following end-to-end capabilities:

1. Generates 30 chronological days of fictional training data for 20 athletes.
2. Validates the schema, converts types, rejects missing or implausible values, and removes duplicate athlete/date records.
3. Calculates historical individual baselines using strictly earlier observations.
4. Calculates simple internal and external training-demand proxies.
5. Normalizes both demand components against each athlete's own historical baseline.
6. Combines the normalized components into AVAIL Load Score V1.
7. Calculates a separate pre-training Readiness Score from HRV, resting HR, and sleep.
8. Combines readiness, recent completed load, and an explicitly submitted planned session in Decision Support.
9. Exposes separate readiness, training-load, decision-support, and history endpoints through FastAPI.
10. Returns provenance, validation, history-sufficiency, and data-quality fields.
11. Tests reproducibility, validation, formulas, separation, API behavior, score boundaries, and leakage prevention.

The system deliberately separates three concepts:

- **Readiness Score:** provisional training-preparation context calculated before a session.
- **Training Load Score:** demand created by a completed session.
- **Decision Support:** a review layer combining readiness, recent completed load, and planned training. It raises attention points but does not prescribe a training change.

## What counts as ML

This version implements the statistical and feature-engineering foundation of an ML system:

- Athlete-specific longitudinal feature processing
- Prior-only expanding historical baselines
- Personalized normalization
- Internal and external load feature engineering
- Chronological leakage prevention
- A deterministic, interpretable scoring layer
- A dataset and service structure that can support later model training

It does **not** contain a trained supervised-learning model. Load Score V1 is a deterministic weighted formula, not a model learned from outcome labels. This distinction is intentional: synthetic data can demonstrate software execution, but it cannot establish real-world predictive accuracy.

A later demonstration could train a chronological Random Forest or gradient-boosting model to predict an explicitly synthetic next-session load target. That would demonstrate training and inference mechanics only and would remain unsuitable for real athlete decisions without appropriate data and validation.

## Project structure

```text
avail-ml-poc/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI application and endpoints
├── data/
│   └── synthetic_athletes.csv # Generated fictional session data
├── ml/
│   ├── __init__.py
│   ├── baseline.py             # Prior-only baselines and normalization
│   ├── generate_data.py        # Reproducible data generator
│   ├── load.py                 # Internal and external load proxies
│   ├── preprocessing.py        # Schema, range, duplicate and order checks
│   └── scoring.py              # End-to-end scoring orchestration
├── tests/
│   ├── test_api.py
│   └── test_pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements and quick start

### Requirements

- Python 3.11 or later
- NumPy and Pandas
- FastAPI and Uvicorn
- Pytest and HTTPX

No database, Docker runtime, GPU, cloud account, or external API credentials are required.

Run all commands from the `avail-ml-poc` directory.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ml.generate_data
uvicorn api.main:app --reload
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ml.generate_data
uvicorn api.main:app --reload
```

After startup:

- Base URL: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

The API automatically generates the synthetic CSV if it is missing. Running the generator explicitly is still recommended because it makes the generation step visible and reproducible.

## Data model

The default generator uses random seed `42`, produces athlete IDs from `ATH-001` to `ATH-020`, and creates 30 daily observations per athlete beginning on 1 January 2026. The result contains 600 rows.

| Field | Type | Meaning | Generated range |
|---|---:|---|---:|
| `athlete_id` | string | Fictional athlete identifier | `ATH-001` to `ATH-020` |
| `date` | date | Session date | 30 chronological days |
| `duration_min` | integer | Session duration | 20-120 minutes |
| `avg_hr` | integer | Simulated average session heart rate | 105-190 bpm |
| `resting_hr` | float | Simulated resting heart rate | 40-85 bpm |
| `hrv` | float | Simulated HRV value | 20-130 ms |
| `distance_km` | float | Simulated session distance | 1-18 km |
| `sleep_hours` | float | Simulated sleep duration | 4.5-10 hours |
| `perceived_exertion` | integer | Simulated session RPE | 1-10 |
| `data_source` | string | Provenance marker | Always `synthetic` |

These are plausible demonstration ranges, not population norms or clinical thresholds.

### Regenerating the data

```bash
python -m ml.generate_data
```

Expected result:

```text
Wrote 600 synthetic records to .../data/synthetic_athletes.csv
```

Because the seed and logic are fixed, repeated runs produce the same data unless the generator parameters or implementation change.

## Processing pipeline

### 1. Validation and preprocessing

`ml/preprocessing.py`:

- Confirms every required column exists.
- Parses `date` values.
- Converts numeric fields.
- Rejects missing, unparseable, and out-of-range values.
- Rejects empty athlete identifiers.
- Resolves duplicate `athlete_id` and `date` pairs by retaining the first row.
- Sorts records by athlete and date using stable ordering.
- Adds `data_quality: "ok"` and `data_source: "synthetic"`.

The PoC fails fast rather than silently imputing invalid values. Production will require a more detailed policy covering sensor dropout, delayed sync, user non-response, correction, and quarantine.

### 2. Raw internal load

The illustrative internal-load proxy uses duration and relative heart-rate elevation:

```text
internal_load_raw = duration_min × max((avg_hr - resting_hr) / 100, 0)
```

Example:

```text
duration = 60 minutes
average heart rate = 150 bpm
resting heart rate = 50 bpm

internal_load_raw = 60 × ((150 - 50) / 100) = 60
```

This is an arbitrary-unit engineering proxy. It is not TRIMP, a clinical measure, or a validated physiological-load equation.

### 3. Raw external load

The current external-load proxy is simply:

```text
external_load_raw = distance_km
```

It can later be expanded with sport-appropriate GPS or IMU features such as high-speed running, sprint distance, accelerations, decelerations, explosive efforts, and player load.

### 4. Individual baselines

For every athlete and signal, the pipeline calculates an expanding mean from strictly earlier observations:

```text
baseline at session t = mean(session 1 ... session t-1)
```

The current observation is shifted out. Future observations cannot affect an earlier baseline or score.

Prior-only baselines are produced for:

- Raw internal load
- Raw external load
- Resting heart rate
- HRV

At least three earlier sessions are required before a baseline is considered available.

### 5. Personalized normalization

Each raw load is normalized relative to that athlete's historical baseline:

```text
normalized_component = 50 × raw_load / historical_baseline
```

The result is clipped to 0-100.

| Raw value relative to baseline | Normalized component |
|---:|---:|
| 0% | 0 |
| 50% | 25 |
| 100% | 50 |
| 150% | 75 |
| 200% or more | 100 |

This scale is an engineering convention, not a validated intensity classification.

### Warm-up behavior

During the first three sessions, there is insufficient prior history. The pipeline returns neutral component values and an explicit flag:

```json
{
  "internal_load": 50.0,
  "external_load": 50.0,
  "insufficient_history": true
}
```

Here, `50` means **baseline unavailable**. It must not be interpreted as average recovery, safe load, or a recommendation to maintain training.

## Load Score methodology

After normalization, AVAIL Load Score V1 applies the provisional architecture weights:

```text
Load Score = 0.55 × Internal Load + 0.45 × External Load
```

The result remains between 0 and 100. The API returns both components next to the combined score so consumers can inspect why the result is high or low.

The weights are provisional and unvalidated. Load Score V1 must not be described as:

- An injury-risk probability
- A recovery or readiness score
- A recommendation to increase or decrease training
- A menstrual-cycle inference
- A fatigue diagnosis
- A medical or return-to-play decision

## Readiness Score methodology

Readiness is calculated only from signals that can be available before training:

- HRV relative to the athlete's prior-only HRV baseline: 40%
- Resting heart rate relative to the prior-only resting-HR baseline: 30%
- Sleep duration relative to the prior-only sleep baseline: 30%

Each component maps the athlete's personal baseline to 50. Higher HRV and sleep increase their components; higher resting HR decreases its component. The provisional tolerance bands are ±20% for HRV, ±15% for resting HR, and ±25% for sleep, with every component clipped to 0-100.

```text
Readiness Score = 0.40 × HRV component
                + 0.30 × Resting-HR component
                + 0.30 × Sleep component
```

Completed-session fields such as duration, average session HR, and distance are not inputs to readiness. During baseline warm-up, components return neutral 50 with `insufficient_history: true`.

This score is provisional and unvalidated. It describes input context only and must not be presented as proof that an athlete is recovered, safe to train, or medically cleared.

## Decision Support methodology

Decision Support accepts coach-provided planned internal and external components on the same provisional 0-100 scale. It calculates the planned score with the existing 55/45 training-load weights and compares it with:

- Current pre-training readiness
- Mean completed Training Load Score from the latest seven sessions
- Availability of sufficient individual history

It can return transparent review codes such as `insufficient_individual_history`, `low_readiness_with_non_reduced_planned_demand`, `planned_demand_materially_above_recent_load`, and `review_pre_training_readiness_signals`. These are attention flags, not automated prescriptions.

## API reference

### `GET /health`

Checks that the application process and routing layer are available.

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "data_source": "synthetic",
  "validated": false
}
```

### `GET /athletes`

Returns available athlete IDs.

```bash
curl http://127.0.0.1:8000/athletes
```

Abbreviated example:

```json
{
  "athletes": ["ATH-001", "ATH-002", "ATH-003"],
  "count": 20,
  "data_source": "synthetic",
  "validated": false
}
```

The live response contains all 20 IDs.

### `GET /athletes/{athlete_id}/latest`

Returns the most recent record with readiness and completed load in separate objects.

```bash
curl http://127.0.0.1:8000/athletes/ATH-001/latest
```

Deterministic example from the default dataset:

```json
{
  "athlete_id": "ATH-001",
  "date": "2026-01-30",
  "pre_training_readiness": {
    "score": 52.24,
    "components": {"hrv": 38.35, "resting_hr": 63.08, "sleep": 59.92},
    "insufficient_history": false,
    "validated": false
  },
  "completed_training_load": {
    "score": 32.07,
    "internal_load": 27.43,
    "external_load": 37.74,
    "insufficient_history": false,
    "validated": false
  },
  "history_count": 29,
  "data_quality": "ok",
  "data_source": "synthetic",
  "validated": false
}
```

Use the live endpoint for exact component values from the current generated CSV.

### `GET /athletes/{athlete_id}/readiness/latest`

Returns only the latest pre-training readiness object.

```bash
curl http://127.0.0.1:8000/athletes/ATH-001/readiness/latest
```

### `GET /athletes/{athlete_id}/training-load/latest`

Returns only the latest completed Training Load Score and internal/external decomposition.

```bash
curl http://127.0.0.1:8000/athletes/ATH-001/training-load/latest
```

### `POST /athletes/{athlete_id}/decision-support`

Combines readiness, the latest seven completed load scores, and a planned session supplied by the caller.

```bash
curl -X POST http://127.0.0.1:8000/athletes/ATH-001/decision-support \
  -H "Content-Type: application/json" \
  -d '{"planned_internal_load": 70, "planned_external_load": 60}'
```

Both planned components must be between 0 and 100. The response contains a planned score, recent-load mean, attention codes, and a non-prescriptive message.

### `GET /athletes/{athlete_id}/history`

Returns chronological scored sessions for one athlete.

```bash
curl "http://127.0.0.1:8000/athletes/ATH-001/history?limit=10"
```

The optional `limit` accepts values from 1 to 365. When omitted, every available record is returned.

Response envelope:

```json
{
  "athlete_id": "ATH-001",
  "count": 10,
  "records": [
    {
      "athlete_id": "ATH-001",
      "date": "2026-01-21",
      "pre_training_readiness": {
        "score": 52.1,
        "components": {"hrv": 54.0, "resting_hr": 49.0, "sleep": 52.57},
        "validated": false
      },
      "completed_training_load": {
        "score": 43.87,
        "internal_load": 42.01,
        "external_load": 46.15,
        "validated": false
      },
      "history_count": 20,
      "data_quality": "ok",
      "data_source": "synthetic",
      "validated": false
    }
  ]
}
```

The history values above illustrate the schema. Use the live endpoint for authoritative values from the current CSV.

### Errors

An unknown athlete returns HTTP `404`:

```json
{
  "detail": "Unknown athlete_id: NOT-REAL"
}
```

An invalid history limit returns FastAPI's HTTP `422` validation response.

## Response field reference

| Field | Meaning |
|---|---|
| `athlete_id` | Fictional athlete identifier |
| `date` | Session date |
| `pre_training_readiness.score` | Provisional pre-session context score, 0-100 |
| `pre_training_readiness.components` | HRV, resting-HR, and sleep components |
| `completed_training_load.internal_load` | Personalized completed internal demand, 0-100 |
| `completed_training_load.external_load` | Personalized completed external demand, 0-100 |
| `completed_training_load.score` | Weighted completed Training Load Score, 0-100 |
| `baseline_internal_load_raw` | Mean raw internal load from eligible prior sessions |
| `baseline_external_load_raw` | Mean raw external load from eligible prior sessions |
| `baseline_resting_hr` | Mean resting HR from eligible prior sessions |
| `baseline_hrv` | Mean HRV from eligible prior sessions |
| `history_count` | Number of earlier athlete sessions |
| `insufficient_history` | Whether fewer than three earlier sessions are available |
| `data_quality` | Current simple quality status |
| `data_source` | Always `synthetic` in this PoC |
| `validated` | Always `false` in this PoC |

## Testing

Run the suite from the project directory:

```bash
pytest -q
```

Current expected result:

```text
16 passed
```

Coverage includes:

- Fixed-seed data reproducibility
- Expected record and athlete counts
- Synthetic provenance tagging
- Chronological sorting
- Duplicate handling
- Missing columns and invalid values
- Internal and external load formulas
- Prior-only baseline correctness
- Future-data leakage prevention
- Load Score weighting and boundaries
- Readiness weighting, boundaries, and prior-only behavior
- Separation between readiness and completed load inputs
- Decision Support validation and attention flags
- Warm-up history flags
- API response structure
- History limiting and unknown athletes

### Manual API smoke test

With Uvicorn running:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/athletes/ATH-001/latest
curl --fail http://127.0.0.1:8000/athletes/ATH-001/readiness/latest
curl --fail http://127.0.0.1:8000/athletes/ATH-001/training-load/latest
curl --fail http://127.0.0.1:8000/openapi.json
```

## Design decisions

### Why use expanding means?

The PoC needs transparent behavior with very little history. A prior-only expanding mean is easy to inspect, explain, and test. EWMA, robust rolling statistics, hierarchical models, or learned representations may become appropriate after real data characteristics are understood.

### Why keep readiness and training load separate?

Pre-training context and completed-session demand answer different questions. A high-demand session can occur alongside either high or low readiness. Combining them into one score would obscure the distinction and create timing leakage. Decision Support may compare them while preserving both original values.

### Why reject invalid values instead of imputing?

Silent imputation could make fabricated values appear measured. The PoC fails visibly. Production should distinguish device dropout, delayed sync, user non-response, invalid readings, and genuinely absent measurements.

### Why is there no trained model?

The records are synthetic and contain no validated outcome target. A trained model would demonstrate execution but not real performance prediction. The deterministic foundation is completed first.

### Why load the CSV into memory?

The default dataset contains only 600 rows. In-memory loading keeps the PoC simple. Production requires persistent storage, ingestion jobs, versioned calculations, and controlled recomputation.

## Known limitations

- All data is generated and represents no real athlete.
- Internal load is an illustrative proxy.
- Distance alone is an incomplete external-load measure.
- Expanding means are sensitive to outliers and long-term change.
- Normalization and warm-up behavior are engineering choices.
- Rest days and multiple sessions per day are not modeled.
- Duplicate athlete/date rows are collapsed, so double-session days are unsupported.
- There are no sport, position, session type, competition, or schedule features.
- There is no database, authentication, authorization, audit log, consent workflow, or encryption design.
- There is no sensor metadata, confidence, missingness reason, or device-quality assessment.
- There is no trained ML model, registry, experiment tracking, drift monitoring, or model versioning.
- Readiness and Decision Support are provisional deterministic rules, not validated trained models.
- There is no validated recovery, fatigue, injury, cycle, biomarker, or clinical model.
- API state is loaded at startup; changing the CSV does not automatically reload the running process.
- The API contract is not versioned.

## Production readiness

### Scientific and performance validation

- Define outputs with sports-science and clinical stakeholders.
- Compare calculations against accepted reference equipment and methods.
- Predefine evaluation metrics and acceptable error ranges.
- Validate across sports, roles, session types, and athlete subgroups.
- Establish minimum-history and abstention criteria.
- Test sensor failure, outliers, travel, illness, and schedule changes.

### Data and privacy

- Establish lawful basis, consent, retention, deletion, and access policies.
- Treat health and cycle information as sensitive data.
- Implement role-based access and audit logging.
- Encrypt data in transit and at rest.
- Separate identity from analytical features where appropriate.
- Document lineage, device provenance, units, time zones, and corrections.

### Engineering

- Add persistent storage and schema migrations.
- Add authenticated ingestion and API access.
- Add structured logging, metrics, alerting, and tracing.
- Version datasets, features, algorithms, and responses.
- Add integration, security, load, and recovery tests.
- Define deployment and rollback procedures.

### Model governance

- Distinguish deterministic algorithms from trained models in UI and documentation.
- Record training-data provenance and licensing.
- Use chronological and athlete-level evaluation splits.
- Monitor error, calibration, missingness, drift, and subgroup performance.
- Report uncertainty and abstain when data quality is insufficient.
- Keep clinicians and qualified staff responsible for medical and return-to-play decisions.

## Suggested next phases

### Phase 1A: frontend integration

- Connect the coach dashboard to the current API.
- Display component decomposition and insufficient-history states.
- Keep recovery and readiness visually separate from demand.
- Add athlete and date-range filters.

### Phase 1B: realistic session schema

- Support multiple daily sessions with a `session_id`.
- Add sport, position, session type, planned load, and competition context.
- Add high-speed running, sprint distance, acceleration, and deceleration fields.
- Introduce explicit missingness and sensor-quality fields.

### Phase 1C: optional ML workflow demonstration

- Create a clearly synthetic next-session target.
- Build lagged features without target leakage.
- Split train and test data chronologically and, where appropriate, by athlete.
- Train a small scikit-learn baseline model.
- Report MAE and RMSE against a naive baseline.
- Expose predictions through a separately labeled experimental endpoint.
- State that synthetic performance is not evidence of real-world accuracy.

### Phase 2: validated real-data pilot

- Replace synthetic input with consented pilot data.
- Validate measurement and calculation stability.
- Refine individual baseline and abstention logic.
- Add confidence, explanations, and human-review workflows.
- Evaluate prospective utility before making product claims.

## License and usage

No standalone software license is declared in this PoC. Before external distribution or commercial reuse, the project owner should select a license and review dependency, privacy, intellectual-property, and data-governance obligations.

## Final safety statement

This repository demonstrates software architecture and deterministic scoring with simulated data. It does not establish that AVAIL can predict performance, recovery, injury, menstrual-cycle state, physiological stress, or medical outcomes. Any future real-world capability requires measurement validation, athlete consent, prospective evaluation, and qualified human oversight.
