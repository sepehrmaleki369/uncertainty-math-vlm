import math

import pandas as pd
import pytest

from pilot.plotting import (
    REQUIRED_COLUMNS,
    check_parse_failure_rate,
    check_temperature_zero_anchor,
    classify_k_resample_result,
    compute_auroc,
    join_k_resample,
    load_results,
    summarize_k_comparison,
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


# --- compute_auroc ---


def test_compute_auroc_perfect_separation():
    df = pd.DataFrame(
        {"entropy": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0], "correct": [True, True, True, False, False, False]}
    )
    assert compute_auroc(df, "entropy", "correct") == pytest.approx(1.0)


def test_compute_auroc_no_signal_identical_distributions():
    df = pd.DataFrame({"entropy": [0.5, 0.5, 0.5, 0.5], "correct": [True, True, False, False]})
    assert compute_auroc(df, "entropy", "correct") == pytest.approx(0.5)


def test_compute_auroc_hand_computed_small_example():
    # incorrect (positive class): [0.9, 0.3]; correct (negative): [0.1, 0.2, 0.4]
    # Mann-Whitney U: for each (pos, neg) pair, 1 if pos > neg, 0.5 if tied.
    # 0.9 beats all 3 negatives (3), 0.3 beats only 0.1,0.2 (2) -> U=5, n_pos*n_neg=6
    df = pd.DataFrame(
        {"entropy": [0.9, 0.3, 0.1, 0.2, 0.4], "correct": [False, False, True, True, True]}
    )
    assert compute_auroc(df, "entropy", "correct") == pytest.approx(5 / 6)


def test_compute_auroc_reproduces_real_k5_baseline_distribution():
    """Regression test: reconstructs the exact reasoning_entropy value-count
    breakdown observed on the real 2026-07-31 100-item CSV (75 correct / 25
    incorrect, 5 distinct entropy values from K=5 grading samples) as a
    literal fixture -- avoids depending on the untracked results/ CSV file
    while still locking in the real AUROC=0.6184 this session reported,
    cross-checked against scipy.stats.mannwhitneyu independently."""
    values = [0.0, 0.5004024235381879, 0.6730116670092565, 0.9502705392332348, 1.0549201679861442]
    correct_counts = [31, 29, 12, 1, 2]  # sums to 75, matches real n_correct
    incorrect_counts = [7, 7, 10, 1, 0]  # sums to 25, matches real n_incorrect
    entropy, correct = [], []
    for v, c in zip(values, correct_counts):
        entropy += [v] * c
        correct += [True] * c
    for v, c in zip(values, incorrect_counts):
        entropy += [v] * c
        correct += [False] * c
    df = pd.DataFrame({"reasoning_entropy": entropy, "grading_correct": correct})
    assert len(df) == 100
    assert compute_auroc(df, "reasoning_entropy", "grading_correct") == pytest.approx(0.6184, abs=1e-4)


def test_compute_auroc_handles_ties():
    # K=5-style: several items share the exact same entropy value, split
    # across both classes -- must not raise or silently mis-rank.
    df = pd.DataFrame(
        {
            "entropy": [0.5, 0.5, 0.5, 0.5, 0.0, 1.0],
            "correct": [True, True, False, False, True, False],
        }
    )
    result = compute_auroc(df, "entropy", "correct")
    assert 0.0 <= result <= 1.0


def test_compute_auroc_empty_class_returns_nan():
    df = pd.DataFrame({"entropy": [0.1, 0.2, 0.3], "correct": [True, True, True]})
    assert math.isnan(compute_auroc(df, "entropy", "correct"))


# --- join_k_resample ---


def test_join_k_resample_composite_key_required():
    """Regression test: orig_q alone repeats across perturbed variants of the
    same question on the real data (91/100 unique) -- proves the join
    correctly pairs rows by the FULL (orig_q, pert_a) key rather than
    matching any row that merely shares orig_q."""
    baseline = pd.DataFrame(
        {"orig_q": ["q1", "q1"], "pert_a": ["variant_a", "variant_b"], "reasoning_entropy": [0.1, 0.9]}
    )
    resample = pd.DataFrame(
        {
            "orig_q": ["q1", "q1"],
            "pert_a": ["variant_b", "variant_a"],  # deliberately reordered
            "reasoning_entropy_k25": [0.99, 0.11],
        }
    )
    joined = join_k_resample(baseline, resample)
    row_a = joined[joined.pert_a == "variant_a"].iloc[0]
    row_b = joined[joined.pert_a == "variant_b"].iloc[0]
    assert row_a["reasoning_entropy"] == pytest.approx(0.1)
    assert row_a["reasoning_entropy_k25"] == pytest.approx(0.11)
    assert row_b["reasoning_entropy"] == pytest.approx(0.9)
    assert row_b["reasoning_entropy_k25"] == pytest.approx(0.99)


def test_join_k_resample_raises_on_duplicate_key():
    baseline = pd.DataFrame({"orig_q": ["q1", "q1"], "pert_a": ["a", "a"], "reasoning_entropy": [0.1, 0.2]})
    resample = pd.DataFrame({"orig_q": ["q1"], "pert_a": ["a"], "reasoning_entropy_k25": [0.1]})
    with pytest.raises(ValueError, match="duplicate"):
        join_k_resample(baseline, resample)


def test_join_k_resample_raises_on_incomplete_overlap():
    baseline = pd.DataFrame({"orig_q": ["q1", "q2"], "pert_a": ["a", "b"], "reasoning_entropy": [0.1, 0.2]})
    resample = pd.DataFrame({"orig_q": ["q1"], "pert_a": ["a"], "reasoning_entropy_k25": [0.1]})
    with pytest.raises(ValueError, match="overlap"):
        join_k_resample(baseline, resample)


# --- summarize_k_comparison ---


def _make_joined_df():
    return pd.DataFrame(
        {
            "orig_q": [f"q{i}" for i in range(20)],
            "pert_a": [f"a{i}" for i in range(20)],
            "reasoning_entropy": [0.5] * 20,
            "grading_correct": [True] * 15 + [False] * 5,
            "reasoning_entropy_k25": [0.1] * 15 + [1.5] * 5,
            "grading_correct_k25": [True] * 15 + [False] * 5,
        }
    )


def test_summarize_k_comparison_fields_present_and_typed():
    summary = summarize_k_comparison(_make_joined_df(), k_base=5, k_new=25)
    expected_keys = {
        "k_base", "k_new", "auroc_k_base", "auroc_k_new",
        "n_distinct_entropy_k_base", "n_distinct_entropy_k_new",
        "frac_at_max_entropy_k_base", "frac_at_max_entropy_k_new",
        "median_by_correctness_k_base", "median_by_correctness_k_new",
        "majority_vote_flip_rate", "n_incorrect_k_new",
    }
    assert expected_keys.issubset(summary.keys())
    assert isinstance(summary["n_incorrect_k_new"], int)
    assert isinstance(summary["median_by_correctness_k_new"], dict)
    assert summary["n_incorrect_k_new"] == 5


def test_summarize_k_comparison_flip_rate():
    df = _make_joined_df()
    # Flip 2 items' correctness between the base and new columns.
    df.loc[0, "grading_correct_k25"] = False
    df.loc[15, "grading_correct_k25"] = True
    summary = summarize_k_comparison(df, k_base=5, k_new=25)
    assert summary["majority_vote_flip_rate"] == pytest.approx(2 / 20)


# --- classify_k_resample_result ---


def test_classify_confirmed_sharper():
    summary = {
        "n_incorrect_k_new": 20,
        "auroc_k_new": 0.72,
        "median_by_correctness_k_new": {"correct": 0.3, "incorrect": 0.6},
    }
    assert classify_k_resample_result(summary) == "confirmed_sharper"


def test_classify_confirmed_sharper_requires_median_separation():
    """AUROC alone clearing the bar isn't enough -- the median tie observed
    at K=5 must actually break, or it's the same mean/tail-driven signal."""
    summary = {
        "n_incorrect_k_new": 20,
        "auroc_k_new": 0.80,
        "median_by_correctness_k_new": {"correct": 0.5, "incorrect": 0.5},
    }
    assert classify_k_resample_result(summary) == "confirmed_modest"


def test_classify_confirmed_modest():
    summary = {
        "n_incorrect_k_new": 20,
        "auroc_k_new": 0.55,
        "median_by_correctness_k_new": {"correct": 0.3, "incorrect": 0.6},
    }
    assert classify_k_resample_result(summary) == "confirmed_modest"


def test_classify_null_at_k5_was_noise():
    summary = {
        "n_incorrect_k_new": 20,
        "auroc_k_new": 0.549,
        "median_by_correctness_k_new": {"correct": 0.3, "incorrect": 0.6},
    }
    assert classify_k_resample_result(summary) == "null_at_k5_was_noise"


def test_classify_inconclusive_underpowered_at_boundary():
    summary_14 = {
        "n_incorrect_k_new": 14,
        "auroc_k_new": 0.90,
        "median_by_correctness_k_new": {"correct": 0.1, "incorrect": 0.9},
    }
    summary_15 = {
        "n_incorrect_k_new": 15,
        "auroc_k_new": 0.90,
        "median_by_correctness_k_new": {"correct": 0.1, "incorrect": 0.9},
    }
    assert classify_k_resample_result(summary_14) == "inconclusive_underpowered"
    assert classify_k_resample_result(summary_15) == "confirmed_sharper"
