import pandas as pd
import pytest

from ml.baseline import MIN_HISTORY
from ml.generate_data import generate_synthetic_data
from ml.load import calculate_external_load, calculate_internal_load
from ml.preprocessing import preprocess_sessions
from ml.scoring import build_scored_sessions


def test_generation_is_reproducible_and_complete(tmp_path):
    first = generate_synthetic_data(tmp_path / "one.csv")
    second = generate_synthetic_data(tmp_path / "two.csv")
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 600
    assert first["athlete_id"].nunique() == 20
    assert first["data_source"].eq("synthetic").all()


def test_preprocessing_sorts_and_removes_duplicates(tmp_path):
    frame = generate_synthetic_data(tmp_path / "data.csv", athlete_count=1, day_count=5)
    dirty = pd.concat([frame.iloc[::-1], frame.iloc[[0]]], ignore_index=True)
    clean = preprocess_sessions(dirty)
    assert len(clean) == 5
    assert clean["date"].is_monotonic_increasing


@pytest.mark.parametrize("column,value", [("duration_min", -1), ("sleep_hours", 25), ("avg_hr", None)])
def test_invalid_values_rejected(tmp_path, column, value):
    frame = generate_synthetic_data(tmp_path / "data.csv", athlete_count=1, day_count=5)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="Invalid or missing"):
        preprocess_sessions(frame)


def test_missing_column_rejected(tmp_path):
    frame = generate_synthetic_data(tmp_path / "data.csv", athlete_count=1, day_count=5)
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_sessions(frame.drop(columns=["hrv"]))


def test_load_calculations_are_deterministic():
    frame = pd.DataFrame({"duration_min": [60], "avg_hr": [150], "resting_hr": [50], "distance_km": [8.2]})
    assert calculate_internal_load(frame).iloc[0] == 60.0
    assert calculate_external_load(frame).iloc[0] == 8.2


def test_baseline_uses_only_prior_observations(tmp_path):
    frame = generate_synthetic_data(tmp_path / "data.csv", athlete_count=1, day_count=8)
    original = build_scored_sessions(frame)
    changed = frame.copy()
    changed.loc[7, ["duration_min", "avg_hr", "distance_km"]] = [300, 220, 90]
    rescored = build_scored_sessions(changed)
    columns = ["baseline_internal_load_raw", "baseline_external_load_raw", "load_score"]
    pd.testing.assert_frame_equal(original.loc[:6, columns], rescored.loc[:6, columns])
    assert original.loc[MIN_HISTORY, "baseline_internal_load_raw"] == pytest.approx(
        original.loc[: MIN_HISTORY - 1, "internal_load_raw"].mean()
    )


def test_score_formula_boundaries_and_warmup(tmp_path):
    scored = build_scored_sessions(generate_synthetic_data(tmp_path / "data.csv", athlete_count=2, day_count=8))
    expected = (0.55 * scored["internal_load"] + 0.45 * scored["external_load"]).round(2)
    pd.testing.assert_series_equal(scored["load_score"], expected, check_names=False)
    assert scored["load_score"].between(0, 100).all()
    assert scored.groupby("athlete_id").head(MIN_HISTORY)["insufficient_history"].all()


def test_readiness_is_separate_bounded_and_prior_only(tmp_path):
    frame = generate_synthetic_data(tmp_path / "data.csv", athlete_count=1, day_count=8)
    original = build_scored_sessions(frame)
    changed = frame.copy()
    changed.loc[7, ["hrv", "resting_hr", "sleep_hours"]] = [20, 85, 4.5]
    rescored = build_scored_sessions(changed)
    assert original["readiness_score"].between(0, 100).all()
    pd.testing.assert_series_equal(
        original.loc[:6, "readiness_score"], rescored.loc[:6, "readiness_score"]
    )
    assert rescored.loc[7, "readiness_score"] < original.loc[7, "readiness_score"]


def test_training_load_does_not_use_readiness_signals_beyond_resting_hr_proxy(tmp_path):
    frame = generate_synthetic_data(tmp_path / "data.csv", athlete_count=1, day_count=8)
    original = build_scored_sessions(frame)
    changed = frame.copy()
    changed.loc[7, ["hrv", "sleep_hours"]] = [20, 4.5]
    rescored = build_scored_sessions(changed)
    assert rescored.loc[7, "training_load_score"] == original.loc[7, "training_load_score"]
