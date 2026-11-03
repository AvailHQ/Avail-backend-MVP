"""FastAPI interface for the AVAIL ML backend proof of concept."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ml.generate_data import DEFAULT_OUTPUT, generate_synthetic_data
from ml.scoring import build_scored_sessions

app = FastAPI(
    title="AVAIL ML Backend PoC",
    version="1.1.0",
    description=(
        "Synthetic-data technical demonstration separating pre-training readiness, "
        "completed training load, and coach-reviewed decision support."
    ),
)


class PlannedTraining(BaseModel):
    """Coach-provided planned demand components on the same provisional 0-100 scale."""

    planned_internal_load: float = Field(ge=0, le=100)
    planned_external_load: float = Field(ge=0, le=100)


def load_records(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        generate_synthetic_data(path)
    return build_scored_sessions(pd.read_csv(path))


RECORDS = load_records()


def _clean_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def _serialize_record(row: pd.Series) -> dict[str, Any]:
    return {
        "athlete_id": row["athlete_id"],
        "date": row["date"],
        "pre_training_readiness": {
            "score": row["readiness_score"],
            "components": {
                "hrv": row["readiness_hrv"],
                "resting_hr": row["readiness_resting_hr"],
                "sleep": row["readiness_sleep"],
            },
            "baselines": {
                "hrv": _clean_value(row["baseline_hrv"]),
                "resting_hr": _clean_value(row["baseline_resting_hr"]),
                "sleep_hours": _clean_value(row["baseline_sleep_hours"]),
            },
            "insufficient_history": bool(row["insufficient_history"]),
            "validated": False,
        },
        "completed_training_load": {
            "score": row["training_load_score"],
            "internal_load": row["internal_load"],
            "external_load": row["external_load"],
            "baselines": {
                "internal_load_raw": _clean_value(row["baseline_internal_load_raw"]),
                "external_load_raw": _clean_value(row["baseline_external_load_raw"]),
            },
            "insufficient_history": bool(row["insufficient_history"]),
            "validated": False,
        },
        "history_count": int(row["history_count"]),
        "data_quality": row["data_quality"],
        "data_source": row["data_source"],
        "validated": False,
    }


def _serialize(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_serialize_record(row) for _, row in frame.iterrows()]


def _athlete_records(athlete_id: str) -> pd.DataFrame:
    selected = RECORDS[RECORDS["athlete_id"] == athlete_id]
    if selected.empty:
        raise HTTPException(status_code=404, detail=f"Unknown athlete_id: {athlete_id}")
    return selected


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "data_source": "synthetic", "validated": False}


@app.get("/athletes")
def athletes() -> dict[str, object]:
    ids = RECORDS["athlete_id"].drop_duplicates().tolist()
    return {"athletes": ids, "count": len(ids), "data_source": "synthetic", "validated": False}


@app.get("/athletes/{athlete_id}/latest")
def latest(athlete_id: str) -> dict[str, Any]:
    return _serialize(_athlete_records(athlete_id).tail(1))[0]


@app.get("/athletes/{athlete_id}/readiness/latest")
def latest_readiness(athlete_id: str) -> dict[str, Any]:
    row = _athlete_records(athlete_id).iloc[-1]
    record = _serialize_record(row)
    return {
        "athlete_id": record["athlete_id"],
        "date": record["date"],
        "pre_training_readiness": record["pre_training_readiness"],
        "data_source": "synthetic",
    }


@app.get("/athletes/{athlete_id}/training-load/latest")
def latest_training_load(athlete_id: str) -> dict[str, Any]:
    row = _athlete_records(athlete_id).iloc[-1]
    record = _serialize_record(row)
    return {
        "athlete_id": record["athlete_id"],
        "date": record["date"],
        "completed_training_load": record["completed_training_load"],
        "data_source": "synthetic",
    }


@app.post("/athletes/{athlete_id}/decision-support")
def decision_support(athlete_id: str, plan: PlannedTraining) -> dict[str, Any]:
    athlete = _athlete_records(athlete_id)
    latest_row = athlete.iloc[-1]
    readiness = float(latest_row["readiness_score"])
    recent_load = round(float(athlete.tail(7)["training_load_score"].mean()), 2)
    planned_load = round(0.55 * plan.planned_internal_load + 0.45 * plan.planned_external_load, 2)

    attention_points: list[str] = []
    if bool(latest_row["insufficient_history"]):
        attention_points.append("insufficient_individual_history")
    if readiness < 40 and planned_load >= recent_load:
        attention_points.append("low_readiness_with_non_reduced_planned_demand")
    if planned_load > recent_load + 20:
        attention_points.append("planned_demand_materially_above_recent_load")
    if readiness < 40:
        attention_points.append("review_pre_training_readiness_signals")

    return {
        "athlete_id": athlete_id,
        "date": latest_row["date"],
        "decision_support": {
            "status": "review" if attention_points else "no_automatic_flag",
            "readiness_score": readiness,
            "recent_completed_load_7_session_mean": recent_load,
            "planned_internal_load": plan.planned_internal_load,
            "planned_external_load": plan.planned_external_load,
            "planned_training_load_score": planned_load,
            "attention_points": attention_points,
            "message": (
                "Review the contributing signals and planned session context. "
                "This output does not prescribe a training change."
            ),
            "validated": False,
        },
        "data_source": "synthetic",
    }


@app.get("/athletes/{athlete_id}/history")
def history(
    athlete_id: str,
    limit: int | None = Query(default=None, ge=1, le=365),
) -> dict[str, object]:
    selected = _athlete_records(athlete_id)
    if limit is not None:
        selected = selected.tail(limit)
    records = _serialize(selected)
    return {"athlete_id": athlete_id, "count": len(records), "records": records}
