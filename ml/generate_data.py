"""Generate a reproducible, explicitly synthetic athlete training dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ATHLETE_COUNT = 20
DAY_COUNT = 30
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_athletes.csv"


def generate_synthetic_data(
    output_path: Path | str = DEFAULT_OUTPUT,
    athlete_count: int = ATHLETE_COUNT,
    day_count: int = DAY_COUNT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate and save fictional training records; no row represents a real person."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=day_count, freq="D")
    rows: list[dict[str, object]] = []

    for number in range(1, athlete_count + 1):
        athlete_id = f"ATH-{number:03d}"
        fitness = rng.normal(0, 1)
        resting_base = np.clip(rng.normal(58, 5), 46, 72)
        hrv_base = np.clip(rng.normal(62, 12), 35, 95)
        for date in dates:
            recovery_noise = rng.normal(0, 1)
            duration = int(np.clip(rng.normal(65 + 5 * fitness, 20), 20, 120))
            exertion = int(np.clip(round(rng.normal(6.0, 1.8)), 1, 10))
            avg_hr = int(np.clip(105 + duration * 0.35 + exertion * 4 + rng.normal(0, 8), 105, 190))
            distance = np.clip(duration * rng.normal(0.115 + 0.008 * fitness, 0.018), 1.0, 18.0)
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "date": date.date().isoformat(),
                    "duration_min": duration,
                    "avg_hr": avg_hr,
                    "resting_hr": round(float(np.clip(resting_base + recovery_noise * 2.2, 40, 85)), 1),
                    "hrv": round(float(np.clip(hrv_base - recovery_noise * 4 + rng.normal(0, 5), 20, 130)), 1),
                    "distance_km": round(float(distance), 2),
                    "sleep_hours": round(float(np.clip(rng.normal(7.4 - recovery_noise * 0.15, 0.8), 4.5, 10)), 1),
                    "perceived_exertion": exertion,
                    "data_source": "synthetic",
                }
            )

    frame = pd.DataFrame(rows).sort_values(["athlete_id", "date"], kind="stable")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame.reset_index(drop=True)


if __name__ == "__main__":
    data = generate_synthetic_data()
    print(f"Wrote {len(data)} synthetic records to {DEFAULT_OUTPUT}")

