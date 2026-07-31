import pandas as pd
import pytest

from pilot.plotting import (
    REQUIRED_COLUMNS,
    check_parse_failure_rate,
    check_temperature_zero_anchor,
    load_results,
    summarize_results,
)


def _make_df(**overrides):
    data = {
        "perception_entropy": [0.0, 1.6, 0.2, 1.2],
        "reasoning_entropy": [0.0, 0.7, 0.1, 0.9],
        "transcription_correct": [True, False, True, False],
        "grading_correct": [True, False, True, False],
        "temp0_entropy_transcription": [0.0, 0.0, 0.0, 0.0],
        "temp0_entropy_grading": [0.0, 0.0, 0.0, 0.0],
        "n_transcription_parse_failures": [0, 0, 0, 0],
        "n_grading_parse_failures": [0, 0, 0, 0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_load_results_happy_path(tmp_path):
    csv_path = tmp_path / "results.csv"
    _make_df().to_csv(csv_path, index=False)
    result = load_results(csv_path)
    assert set(REQUIRED_COLUMNS).issubset(result.columns)
    assert len(result) == 4


def test_load_results_missing_column_raises(tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"perception_entropy": [0.0, 1.6]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_results(csv_path)


def test_temp0_anchor_passes_when_all_zero():
    result = check_temperature_zero_anchor(_make_df(), "temp0_entropy_transcription")
    assert result["passed"] is True
    assert result["n_nonzero"] == 0


def test_temp0_anchor_flags_nonzero():
    df = _make_df(temp0_entropy_transcription=[0.0, 0.69, 0.0, 0.0])
    result = check_temperature_zero_anchor(df, "temp0_entropy_transcription")
    assert result["passed"] is False
    assert result["n_nonzero"] == 1


def test_parse_failure_rate_counts_aggregate():
    df = _make_df(n_transcription_parse_failures=[5, 0, 1, 0])
    result = check_parse_failure_rate(df, k=5)
    assert result["transcription"]["n_failed_samples"] == 6
    assert result["transcription"]["n_total_samples"] == 20
    assert result["transcription"]["frac_failed"] == pytest.approx(0.3)
    # An item where every sample failed collapses to one confident cluster.
    assert result["transcription"]["n_items_all_failed"] == 1


def test_parse_failure_rate_tolerates_missing_columns():
    df = _make_df().drop(columns=["n_transcription_parse_failures"])
    assert check_parse_failure_rate(df)["transcription"] is None


def test_summarize_handles_string_booleans_from_csv_roundtrip(tmp_path):
    """bool('False') is True, so a naive cast would mark every row correct."""
    csv_path = tmp_path / "results.csv"
    _make_df().to_csv(csv_path, index=False)
    summary = summarize_results(load_results(csv_path))
    assert summary["perception"]["n_correct"] == 2
    assert summary["perception"]["n_incorrect"] == 2


def test_summarize_handles_empty_group():
    df = _make_df(grading_correct=[True, True, True, True])
    summary = summarize_results(df)
    assert summary["reasoning"]["n_incorrect"] == 0
