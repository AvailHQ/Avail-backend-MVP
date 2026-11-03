"""Validation and quality checks for the AVAIL synthetic session dataset."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "athlete_id",
    "date",
    "duration_min",
    "avg_hr",
    "resting_hr",
    "hrv",
    "distance_km",
    "sleep_hours",
    "perceived_exertion",
}
NUMERIC_COLUMNS = sorted(REQUIRED_COLUMNS - {"athlete_id", "date"})
VALID_RANGES = {
    "duration_min": (1, 360),
    "avg_hr": (40, 240),
    "resting_hr": (25, 140),
    "hrv": (1, 300),
    "distance_km": (0, 100),
    "sleep_hours": (0, 24),
    "perceived_exertion": (1, 10),
}


def preprocess_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, de-duplicate and chronologically order session observations."""
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    cleaned = frame.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid_date = cleaned["date"].isna()
    missing_value = cleaned[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    invalid_range = pd.Series(False, index=cleaned.index)
    for column, (low, high) in VALID_RANGES.items():
        invalid_range |= ~cleaned[column].between(low, high, inclusive="both")

    if invalid_date.any() or missing_value.any() or invalid_range.any():
        problems = cleaned.index[invalid_date | missing_value | invalid_range].tolist()
        raise ValueError(f"Invalid or missing values in rows: {problems[:10]}")
    if cleaned["athlete_id"].astype(str).str.strip().eq("").any():
        raise ValueError("athlete_id cannot be empty")

    duplicate_mask = cleaned.duplicated(["athlete_id", "date"], keep="first")
    cleaned = cleaned.loc[~duplicate_mask].copy()
    cleaned["data_quality"] = "ok"
    cleaned["data_source"] = "synthetic"
    return cleaned.sort_values(["athlete_id", "date"], kind="stable").reset_index(drop=True)

