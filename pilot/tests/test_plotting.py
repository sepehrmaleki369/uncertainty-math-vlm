import pandas as pd
import pytest

from pilot.plotting import REQUIRED_COLUMNS, load_results


def test_load_results_happy_path(tmp_path):
    df = pd.DataFrame(
        {
            "perception_entropy": [0.0, 1.6],
            "reasoning_entropy": [0.0, 0.7],
            "transcription_correct": [True, False],
            "grading_correct": [True, False],
            "temp0_entropy_transcription": [0.0, 0.0],
            "temp0_entropy_grading": [0.0, 0.0],
        }
    )
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)

    result = load_results(csv_path)
    assert set(REQUIRED_COLUMNS).issubset(result.columns)
    assert len(result) == 2


def test_load_results_missing_column_raises(tmp_path):
    df = pd.DataFrame({"perception_entropy": [0.0, 1.6]})
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_results(csv_path)
