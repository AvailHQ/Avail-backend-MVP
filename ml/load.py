"""Illustrative training-demand calculations, not validated physiology measures."""

from __future__ import annotations

import pandas as pd


def calculate_internal_load(frame: pd.DataFrame) -> pd.Series:
    """Return a duration x relative-heart-rate proxy in arbitrary units."""
    heart_rate_reserve_proxy = ((frame["avg_hr"] - frame["resting_hr"]) / 100).clip(lower=0)
    return (frame["duration_min"] * heart_rate_reserve_proxy).round(3)


def calculate_external_load(frame: pd.DataFrame) -> pd.Series:
    """Return distance in kilometres as the simple external-demand proxy."""
    return frame["distance_km"].astype(float).round(3)

