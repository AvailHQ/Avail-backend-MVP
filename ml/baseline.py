"""Prior-only individual baselines and component normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 3


def prior_expanding_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    """Mean of strictly earlier observations per athlete; current/future rows are excluded."""
    return frame.groupby("athlete_id", sort=False)[column].transform(
        lambda values: values.expanding(min_periods=MIN_HISTORY).mean().shift(1)
    )


def normalize_to_individual_baseline(values: pd.Series, baselines: pd.Series) -> pd.Series:
    """Map baseline to 50, twice baseline to 100; warm-up rows receive neutral 50."""
    safe_baseline = baselines.where(baselines > 0)
    normalized = 50.0 * values / safe_baseline
    return normalized.replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0, 100).round(2)


def add_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("internal_load_raw", "external_load_raw", "resting_hr", "hrv", "sleep_hours"):
        result[f"baseline_{column}"] = prior_expanding_mean(result, column).round(3)
    result["history_count"] = result.groupby("athlete_id", sort=False).cumcount()
    result["insufficient_history"] = result["history_count"] < MIN_HISTORY
    return result
