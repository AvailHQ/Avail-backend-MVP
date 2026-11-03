"""Provisional pre-training readiness context, kept separate from training load."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _relative_component(
    values: pd.Series,
    baselines: pd.Series,
    tolerance: float,
    higher_is_better: bool,
) -> pd.Series:
    """Map a personal baseline to 50 and a tolerance-sized deviation to 0 or 100."""
    safe_baseline = baselines.where(baselines > 0)
    deviation = (values / safe_baseline) - 1.0
    direction = 1.0 if higher_is_better else -1.0
    component = 50.0 + direction * 50.0 * deviation / tolerance
    return component.replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0, 100).round(2)


def add_readiness_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an unvalidated readiness proxy using only pre-session signals."""
    result = frame.copy()
    result["readiness_hrv"] = _relative_component(
        result["hrv"], result["baseline_hrv"], tolerance=0.20, higher_is_better=True
    )
    result["readiness_resting_hr"] = _relative_component(
        result["resting_hr"], result["baseline_resting_hr"], tolerance=0.15, higher_is_better=False
    )
    result["readiness_sleep"] = _relative_component(
        result["sleep_hours"], result["baseline_sleep_hours"], tolerance=0.25, higher_is_better=True
    )
    result["readiness_score"] = (
        0.40 * result["readiness_hrv"]
        + 0.30 * result["readiness_resting_hr"]
        + 0.30 * result["readiness_sleep"]
    ).round(2)
    result["readiness_validated"] = False
    return result
