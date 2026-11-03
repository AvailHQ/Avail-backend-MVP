"""AVAIL Load Score V1 orchestration."""

from __future__ import annotations

import pandas as pd

from ml.baseline import add_baselines, normalize_to_individual_baseline
from ml.load import calculate_external_load, calculate_internal_load
from ml.preprocessing import preprocess_sessions
from ml.readiness import add_readiness_scores


def build_scored_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic, prior-only Load Score V1 records."""
    scored = preprocess_sessions(frame)
    scored["internal_load_raw"] = calculate_internal_load(scored)
    scored["external_load_raw"] = calculate_external_load(scored)
    scored = add_baselines(scored)
    scored = add_readiness_scores(scored)
    scored["internal_load"] = normalize_to_individual_baseline(
        scored["internal_load_raw"], scored["baseline_internal_load_raw"]
    )
    scored["external_load"] = normalize_to_individual_baseline(
        scored["external_load_raw"], scored["baseline_external_load_raw"]
    )
    scored["training_load_score"] = (
        0.55 * scored["internal_load"] + 0.45 * scored["external_load"]
    ).round(2)
    # Backward-compatible alias; new integrations should use training_load_score.
    scored["load_score"] = scored["training_load_score"]
    scored["validated"] = False
    scored["date"] = scored["date"].dt.date.astype(str)
    return scored
