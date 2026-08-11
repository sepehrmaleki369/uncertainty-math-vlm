import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.plotting import (
    REQUIRED_COLUMNS,
    SCALEUP_PREREGISTRATION,
    classify_scaleup_result,
    aurc,
    auroc_sensitivity,
    coverage_at_risk,
    oracle_aurc,
    plot_scoring_categories,
    bootstrap_auroc_ci,
    bootstrap_auroc_difference_ci,
    check_parse_failure_rate,
    classify_capability_check,
    check_temperature_zero_anchor,
    classify_k_resample_result,
    compute_auroc,
    conformal_abstention_threshold,
    evaluate_conformal_abstention,
    join_k_resample,
    load_results,
    majority_class_baseline,
    plot_risk_coverage,
    risk_coverage_curve,
    stratified_auroc,
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


# --- bootstrap_auroc_ci ---


def _k5_baseline_frame():
    """The real 2026-07-31 reasoning-arm distribution as a literal fixture --
    same construction as test_compute_auroc_reproduces_real_k5_baseline_distribution."""
    values = [0.0, 0.5004024235381879, 0.6730116670092565, 0.9502705392332348, 1.0549201679861442]
    correct_counts = [31, 29, 12, 1, 2]
    incorrect_counts = [7, 7, 10, 1, 0]
    entropy, correct = [], []
    for v, c in zip(values, correct_counts):
        entropy += [v] * c
        correct += [True] * c
    for v, c in zip(values, incorrect_counts):
        entropy += [v] * c
        correct += [False] * c
    return pd.DataFrame({"reasoning_entropy": entropy, "grading_correct": correct})


def test_bootstrap_auroc_ci_point_estimate_matches_compute_auroc():
    df = _k5_baseline_frame()
    result = bootstrap_auroc_ci(df, "reasoning_entropy", "grading_correct", n_boot=200)
    assert result["auroc"] == pytest.approx(
        compute_auroc(df, "reasoning_entropy", "grading_correct")
    )


def test_bootstrap_auroc_ci_brackets_the_point_estimate():
    df = _k5_baseline_frame()
    r = bootstrap_auroc_ci(df, "reasoning_entropy", "grading_correct", n_boot=2000)
    assert r["ci_low"] < r["auroc"] < r["ci_high"]
    assert r["n_items"] == 100
    assert r["n_error"] == 25 and r["n_correct"] == 75


def test_bootstrap_auroc_ci_real_reasoning_arm_does_not_exclude_chance():
    """The finding that motivated adding CIs at all (2026-08-02): the reasoning
    arm's headline AUROC 0.6184 was written up as a real-but-weak signal, but its
    95% CI includes 0.5 at n=100 -- it is not distinguishable from chance. This
    locks in that conclusion so a future change can't quietly restore the
    stronger claim."""
    df = _k5_baseline_frame()
    r = bootstrap_auroc_ci(df, "reasoning_entropy", "grading_correct", n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.6184, abs=1e-4)
    assert r["ci_low"] < 0.5
    assert r["excludes_chance"] is False


def test_bootstrap_auroc_ci_strong_separation_excludes_chance():
    df = pd.DataFrame(
        {"entropy": [0.0, 0.05, 0.1, 0.15, 0.2] * 6 + [0.8, 0.85, 0.9, 0.95, 1.0] * 6,
         "correct": [True] * 30 + [False] * 30}
    )
    r = bootstrap_auroc_ci(df, "entropy", "correct", n_boot=2000)
    assert r["excludes_chance"] is True
    assert r["ci_low"] > 0.5


def test_bootstrap_auroc_ci_is_deterministic_for_a_seed():
    df = _k5_baseline_frame()
    kwargs = dict(n_boot=500, seed=7)
    a = bootstrap_auroc_ci(df, "reasoning_entropy", "grading_correct", **kwargs)
    b = bootstrap_auroc_ci(df, "reasoning_entropy", "grading_correct", **kwargs)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_bootstrap_auroc_ci_single_class_returns_nan_ci():
    df = pd.DataFrame({"entropy": [0.1, 0.2, 0.3], "correct": [True, True, True]})
    r = bootstrap_auroc_ci(df, "entropy", "correct", n_boot=50)
    assert math.isnan(r["auroc"]) and math.isnan(r["ci_low"])
    assert r["excludes_chance"] is False


# --- bootstrap_auroc_difference_ci ---


def test_bootstrap_auroc_difference_paired_resample_keeps_item_pairing():
    """A paired difference CI must be narrower than what you'd conclude from two
    overlapping marginal CIs when the two metrics are strongly correlated across
    items -- that correlation is exactly what pairing preserves and what
    comparing two separate CIs throws away."""
    # entropy_b is entropy_a plus a small constant shift: perfectly correlated,
    # so the *difference* in AUROC is near-zero with a tight interval, even
    # though each marginal AUROC has a wide interval at this n.
    entropy = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 5
    correct = [True, True, False, True, False, False] * 5
    df = pd.DataFrame({"ent_a": entropy, "ent_b": entropy, "correct": correct})
    r = bootstrap_auroc_difference_ci(df, "ent_a", "correct", "ent_b", "correct", n_boot=1000)
    assert r["difference"] == pytest.approx(0.0)
    assert r["ci_low"] == pytest.approx(0.0) and r["ci_high"] == pytest.approx(0.0)
    assert r["difference_excludes_zero"] is False


def test_bootstrap_auroc_difference_detects_a_real_gap():
    n = 40
    df = pd.DataFrame(
        {
            "good": list(range(n // 2)) + list(range(100, 100 + n // 2)),  # separates
            "useless": [0.5] * n,  # no signal
            "correct": [True] * (n // 2) + [False] * (n // 2),
        }
    )
    r = bootstrap_auroc_difference_ci(df, "good", "correct", "useless", "correct", n_boot=2000)
    assert r["difference"] == pytest.approx(0.5)
    assert r["difference_excludes_zero"] is True


def test_bootstrap_auroc_difference_reports_differing_error_counts():
    """The K=5 vs K=15 comparison scores the two AUROCs against *different*
    correctness labels (grading_correct flips as K changes). The interval can't
    see that confound, so the counts have to be surfaced for the caller."""
    df = pd.DataFrame(
        {
            "ent_a": [0.1, 0.2, 0.3, 0.4],
            "ent_b": [0.1, 0.2, 0.3, 0.4],
            "correct_a": [True, True, False, False],
            "correct_b": [True, True, True, False],
        }
    )
    r = bootstrap_auroc_difference_ci(df, "ent_a", "correct_a", "ent_b", "correct_b", n_boot=100)
    assert r["n_error_a"] == 2
    assert r["n_error_b"] == 1


def test_bootstrap_auroc_difference_length_mismatch_raises():
    df = pd.DataFrame(
        {
            "ent_a": [0.1, 0.2, 0.3, 0.4],
            "ent_b": [0.1, None, 0.3, 0.4],
            "correct": [True, True, False, False],
        }
    )
    with pytest.raises(ValueError, match="equal-length"):
        bootstrap_auroc_difference_ci(df, "ent_a", "correct", "ent_b", "correct", n_boot=10)


# --- auroc_sensitivity ---


def test_auroc_sensitivity_reports_all_cuts():
    df = pd.DataFrame(
        {
            "entropy": [0.0, 0.5, math.log(5), 0.2, math.log(5), 0.9],
            "correct": [True, True, False, True, False, False],
            "n_transcription_parse_failures": [0, 1, 0, 0, 2, 0],
        }
    )
    out = auroc_sensitivity(df, "entropy", "correct", "n_transcription_parse_failures", k=5, n_boot=100)
    assert set(out) == {"full", "excl_parse_failures", "excl_max_entropy", "excl_both", "robust"}
    assert out["full"]["n_items"] == 6
    assert out["excl_parse_failures"]["n_items"] == 4  # drops the two failure rows
    assert out["excl_max_entropy"]["n_items"] == 4  # drops the two ln(5) rows
    assert out["excl_both"]["n_items"] == 3  # rows 1 and 4 overlap on both cuts


def test_auroc_sensitivity_flags_a_pure_max_entropy_artifact():
    """The artifact this function exists to catch: entropy is flat and
    uninformative everywhere except a max-entropy cell that happens to be
    all-incorrect. AUROC looks respectable on the full set but there is nothing
    left once that cell is removed."""
    df = pd.DataFrame(
        {
            "entropy": [0.5] * 20 + [math.log(5)] * 8,
            "correct": [True] * 10 + [False] * 10 + [False] * 8,
            "fails": [0] * 28,
        }
    )
    out = auroc_sensitivity(df, "entropy", "correct", "fails", k=5, n_boot=1000)
    assert out["full"]["auroc"] > 0.6  # looks like signal
    assert out["excl_max_entropy"]["auroc"] == pytest.approx(0.5)  # nothing underneath
    assert out["robust"] is False


def test_auroc_sensitivity_real_perception_arm_is_robust():
    """Regression test for the 2026-08-02 sensitivity check on the real
    perception arm: the headline 0.788 was challenged as a possible
    parse-failure / max-entropy artifact and survived every cut, holding at
    0.707 with the CI still excluding 0.5 under the harshest simultaneous cut.
    Fixture reproduces the real perception_entropy x correctness crosstab."""
    values = [0.0, 0.5004024235381879, 0.6730116670092565, 0.9502705392332348,
              1.0549201679861442, 1.3321790402101223, math.log(5)]
    correct_counts = [11, 15, 2, 5, 4, 5, 0]  # sums to 42, matches real n_correct
    incorrect_counts = [3, 7, 5, 9, 3, 17, 14]  # sums to 58, matches real n_incorrect
    entropy, correct = [], []
    for v, c in zip(values, correct_counts):
        entropy += [v] * c
        correct += [True] * c
    for v, c in zip(values, incorrect_counts):
        entropy += [v] * c
        correct += [False] * c
    df = pd.DataFrame(
        {"perception_entropy": entropy, "transcription_correct": correct,
         "n_transcription_parse_failures": [0] * 100}
    )
    out = auroc_sensitivity(
        df, "perception_entropy", "transcription_correct",
        "n_transcription_parse_failures", k=5, n_boot=10000, seed=0,
    )
    assert out["full"]["auroc"] == pytest.approx(0.788, abs=0.005)
    assert out["full"]["excludes_chance"] is True
    # 14 all-incorrect max-entropy items removed -- signal degrades but survives.
    assert out["excl_max_entropy"]["auroc"] == pytest.approx(0.721, abs=0.005)
    assert out["excl_max_entropy"]["excludes_chance"] is True
    assert out["robust"] is True


# --- stratified_auroc / majority_class_baseline ---


def test_stratified_auroc_detects_sign_reversal():
    """The grading-arm shape: entropy predicts errors in the large stratum and
    predicts them backwards in the small one, so pooling cancels the signal."""
    # Majority stratum: entropy ranks errors high (good).
    maj = pd.DataFrame({
        "entropy": [0.1] * 20 + [0.9] * 10,
        "correct": [True] * 20 + [False] * 10,
        "gt": [1] * 30,
    })
    # Minority stratum: model is confidently wrong -- low entropy, all wrong.
    minor = pd.DataFrame({
        "entropy": [0.1] * 9 + [0.9],
        "correct": [False] * 9 + [True],
        "gt": [0] * 10,
    })
    df = pd.concat([maj, minor], ignore_index=True)
    out = stratified_auroc(df, "entropy", "correct", "gt", n_boot=500)
    assert out["strata"][1]["auroc"] > 0.9
    assert out["strata"][0]["auroc"] < 0.1
    assert out["sign_reversal"] is True
    assert out["pooled_understates"] is True
    assert out["pooled"]["auroc"] < out["strata"][1]["auroc"]


def test_stratified_auroc_no_reversal_when_strata_agree():
    df = pd.DataFrame({
        "entropy": [0.1, 0.2, 0.8, 0.9] * 4,
        "correct": [True, True, False, False] * 4,
        "gt": [1] * 8 + [0] * 8,
    })
    out = stratified_auroc(df, "entropy", "correct", "gt", n_boot=200)
    assert out["sign_reversal"] is False


def test_stratified_auroc_reports_single_class_stratum_as_nan():
    """An all-wrong stratum is the finding, not a row to silently drop."""
    df = pd.DataFrame({
        "entropy": [0.1, 0.9, 0.2, 0.8],
        "correct": [True, False, False, False],
        "gt": [1, 1, 0, 0],
    })
    out = stratified_auroc(df, "entropy", "correct", "gt", n_boot=100)
    assert 0 in out["strata"]
    assert math.isnan(out["strata"][0]["auroc"])
    assert out["strata"][0]["n_error"] == 2


def test_majority_class_baseline_beats_a_weak_model():
    """Regression test for the 2026-08-02 grading-arm finding: the FERMAT
    sample is 87/13 on has_error, so a constant 'there is an error' predictor
    scores 0.87 while the model's K=5 majority vote scores 0.75."""
    df = pd.DataFrame({"has_error": [True] * 87 + [False] * 13})
    base = majority_class_baseline(df, "has_error")
    assert base["baseline_accuracy"] == pytest.approx(0.87)
    assert base["majority_label"] is True
    assert base["baseline_accuracy"] > 0.75  # the model's actual grading accuracy


def test_majority_class_baseline_handles_minority_majority():
    df = pd.DataFrame({"label": [False] * 9 + [True]})
    base = majority_class_baseline(df, "label")
    assert base["baseline_accuracy"] == pytest.approx(0.9)
    assert base["majority_label"] is False


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


# --- classify_scaleup_result (pre-registered thresholds) ---


def _ci(auroc, ci_low, n_error=50, n_correct=50):
    return {
        "auroc": auroc, "ci_low": ci_low, "ci_high": min(1.0, ci_low + 0.2),
        "n_error": n_error, "n_correct": n_correct,
        "excludes_chance": ci_low > 0.5,
    }


def test_prereg_block_is_dated_and_not_silently_retuned():
    """The pre-registration's whole value is that it predates the data. Lock
    the date and thresholds so a later edit has to be deliberate."""
    assert SCALEUP_PREREGISTRATION["registered"] == "2026-08-02"
    assert SCALEUP_PREREGISTRATION["perception_ci_low_min"] == 0.65
    assert SCALEUP_PREREGISTRATION["perception_error_stratum_ci_low_min"] == 0.65
    assert SCALEUP_PREREGISTRATION["reasoning_stratum_auroc_min"] == 0.70
    assert SCALEUP_PREREGISTRATION["clean_stratum_auroc_max"] == 0.50
    assert SCALEUP_PREREGISTRATION["min_minority_class"] == 30


def test_classify_scaleup_all_hypotheses_confirmed():
    out = classify_scaleup_result({
        "perception": _ci(0.79, 0.70),
        "perception_error_stratum": _ci(0.77, 0.68),
        "reasoning_pooled": _ci(0.70, 0.60),
        "reasoning_error_stratum": _ci(0.75, 0.62),
        "reasoning_clean_stratum": _ci(0.15, 0.02),
    })
    assert out["perception"] == "replicated"
    assert out["reasoning_pooled"] == "signal_confirmed"
    assert out["reasoning_stratified"] == "confirmed"
    assert out["clean_stratum_inversion"] == "confirmed"


def test_classify_scaleup_perception_weaker_but_still_real():
    """Beats chance but not the pilot's effect size -- a distinct outcome from
    both 'replicated' and 'failed', and the most likely one if n=100 was lucky."""
    out = classify_scaleup_result({"perception_error_stratum": _ci(0.62, 0.55)})
    assert out["perception"] == "weaker_than_pilot"


def test_classify_scaleup_perception_failure():
    out = classify_scaleup_result({"perception_error_stratum": _ci(0.53, 0.44)})
    assert out["perception"] == "failed_to_replicate"


def test_classify_scaleup_perception_verdict_uses_the_stratum_not_the_pool():
    """Amended pre-registration (2026-08-02, after the census, before the run):
    the rebalanced sample is 50% clean vs the pilot's 13%, and clean items are
    easier to transcribe (54% vs 40% accuracy at n=100). Pooled perception is
    therefore expected to drift up on composition alone, so the replication
    verdict must read the has_error=1 stratum. Here the pool looks like a
    replication and the stratum does not -- the verdict must follow the stratum."""
    out = classify_scaleup_result({
        "perception": _ci(0.85, 0.78),                  # flattered by composition
        "perception_error_stratum": _ci(0.60, 0.52),    # the honest comparison
    })
    assert out["perception"] == "weaker_than_pilot"
    assert out["perception_pooled_not_comparable"] == "replicated"


def test_classify_scaleup_perception_not_measured_without_the_stratum():
    out = classify_scaleup_result({"perception": _ci(0.79, 0.70)})
    assert out["perception"] == "not_measured"


def test_classify_scaleup_stratified_prediction_can_fail():
    """The point of pre-registering: this must be able to come out 'not
    confirmed'. 0.68 is below the registered 0.70 even though it beats chance."""
    out = classify_scaleup_result({"reasoning_error_stratum": _ci(0.68, 0.56)})
    assert out["reasoning_stratified"] == "not_confirmed"


def test_classify_scaleup_inversion_judged_independently():
    """Signal real but inversion absent -- would mean the bias explanation for
    the pooled collapse is wrong, so these must not be collapsed into one verdict."""
    out = classify_scaleup_result({
        "reasoning_error_stratum": _ci(0.75, 0.62),
        "reasoning_clean_stratum": _ci(0.72, 0.60),
    })
    assert out["reasoning_stratified"] == "confirmed"
    assert out["clean_stratum_inversion"] == "not_confirmed"


def test_classify_scaleup_underpowered_guard():
    out = classify_scaleup_result({
        "perception_error_stratum": _ci(0.85, 0.72, n_error=12, n_correct=200),
        "reasoning_error_stratum": _ci(0.90, 0.80, n_error=8, n_correct=200),
    })
    assert out["perception"] == "inconclusive_underpowered"
    assert out["reasoning_stratified"] == "inconclusive_underpowered"


def test_classify_scaleup_missing_keys_are_not_measured():
    out = classify_scaleup_result({})
    assert out["perception"] == "not_measured"
    assert out["reasoning_pooled"] == "not_measured"
    assert out["reasoning_stratified"] == "not_measured"
    assert out["clean_stratum_inversion"] == "not_measured"


def test_classify_scaleup_nan_clean_stratum_is_not_measured():
    """A stratum with a single correctness class yields nan, which must not be
    read as 'inversion confirmed' by a naive < comparison."""
    out = classify_scaleup_result({
        "reasoning_clean_stratum": {"auroc": float("nan"), "ci_low": float("nan"),
                                    "n_error": 40, "n_correct": 0,
                                    "excludes_chance": False},
    })
    assert out["clean_stratum_inversion"] == "not_measured"


def test_classify_scaleup_boundary_values_land_on_the_confirming_side():
    out = classify_scaleup_result({
        "perception_error_stratum": _ci(0.80, 0.65),  # exactly the threshold
        "reasoning_error_stratum": _ci(0.70, 0.58),   # exactly the threshold
    })
    assert out["perception"] == "replicated"
    assert out["reasoning_stratified"] == "confirmed"


def test_classify_scaleup_inversion_respects_the_power_guard():
    """Regression test for a consistency bug found on the 2026-08-02 n=300 run:
    the inversion branch skipped the min_minority_class guard that every other
    branch applies. The real clean stratum had 137 misgraded items but only 13
    correct ones, so a strongly-inverted AUROC was being graded as a passed
    prediction on a minority class smaller than the one that had just been
    called inconclusive in the error stratum."""
    out = classify_scaleup_result({
        "reasoning_clean_stratum": _ci(0.239, 0.128, n_error=137, n_correct=13),
    })
    assert out["clean_stratum_inversion"] == "inconclusive_underpowered"


def test_classify_scaleup_inversion_confirms_when_adequately_powered():
    out = classify_scaleup_result({
        "reasoning_clean_stratum": _ci(0.24, 0.13, n_error=120, n_correct=60),
    })
    assert out["clean_stratum_inversion"] == "confirmed"


# --- risk-coverage / AURC / conformal abstention ---


def _selective_frame():
    """Entropy ranks errors well but imperfectly: 20 confident-correct,
    20 uncertain-mostly-wrong, plus one confident error to keep it honest."""
    return pd.DataFrame({
        "entropy": [0.0] * 20 + [0.0] + [1.0] * 20,
        "correct": [True] * 20 + [False] + [False] * 16 + [True] * 4,
    })


def test_risk_coverage_curve_shape():
    curve = risk_coverage_curve(_selective_frame(), "entropy", "correct")
    assert list(curve["threshold"]) == [0.0, 1.0]
    # Most-confident bucket: 21 items, 1 wrong.
    assert curve.iloc[0]["n_kept"] == 21
    assert curve.iloc[0]["risk_kept"] == pytest.approx(1 / 21)
    # Full coverage is the last row and must match the overall error rate.
    assert curve.iloc[-1]["coverage"] == pytest.approx(1.0)
    assert curve.iloc[-1]["risk_kept"] == pytest.approx(17 / 41)


def test_risk_coverage_accuracy_and_risk_are_complementary():
    curve = risk_coverage_curve(_selective_frame(), "entropy", "correct")
    assert np.allclose(curve["accuracy_kept"] + curve["risk_kept"], 1.0)


def test_aurc_beats_baseline_when_entropy_ranks_errors():
    out = aurc(_selective_frame(), "entropy", "correct")
    assert out["aurc"] < out["baseline_aurc"]
    assert out["improvement"] > 0


def test_aurc_equals_baseline_when_entropy_is_uninformative():
    """Flat entropy carries no ordering, so deferring by it is deferring at
    random and AURC must collapse onto the base error rate."""
    df = pd.DataFrame({"entropy": [0.5] * 40, "correct": [True] * 20 + [False] * 20})
    out = aurc(df, "entropy", "correct")
    assert out["aurc"] == pytest.approx(out["baseline_aurc"], abs=0.02)


def test_conformal_threshold_picks_an_achievable_operating_point():
    fit = conformal_abstention_threshold(
        _selective_frame(), "entropy", "correct", target_risk=0.30,
        conservative=False,
    )
    assert fit["achievable"] is True
    assert fit["cal_risk"] <= 0.30


def test_conformal_threshold_reports_unachievable_targets():
    """A target below what any operating point can reach is a real outcome, not
    an error -- at a high base error rate most tight targets are unreachable."""
    df = pd.DataFrame({"entropy": [0.5] * 40, "correct": [False] * 40})
    fit = conformal_abstention_threshold(df, "entropy", "correct", target_risk=0.05)
    assert fit["achievable"] is False
    assert math.isnan(fit["threshold"])


def test_conformal_conservative_mode_costs_coverage():
    """The Hoeffding bound must be strictly more cautious than the point
    estimate, otherwise the 1-delta guarantee is not being bought."""
    df = _selective_frame()
    loose = conformal_abstention_threshold(df, "entropy", "correct", 0.42,
                                           conservative=False)
    tight = conformal_abstention_threshold(df, "entropy", "correct", 0.42,
                                           conservative=True)
    assert loose["cal_coverage"] >= tight["cal_coverage"]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
def test_conformal_threshold_rejects_invalid_target(bad):
    with pytest.raises(ValueError, match="target_risk"):
        conformal_abstention_threshold(_selective_frame(), "entropy", "correct", bad)


def test_evaluate_conformal_holds_the_guarantee_on_held_out_data():
    """The whole point: a threshold fitted on calibration data must control
    risk on data it has never seen. Violations should be rare."""
    rng = np.random.default_rng(0)
    n = 400
    entropy = rng.uniform(0, 1, n)
    # Error probability rises with entropy, so the ordering is informative.
    correct = rng.uniform(0, 1, n) > entropy * 0.8
    df = pd.DataFrame({"entropy": entropy, "correct": correct})

    out = evaluate_conformal_abstention(df, "entropy", "correct",
                                        target_risk=0.20, n_splits=200, seed=1)
    assert out["n_valid"] > 0
    assert out["violation_rate"] <= 0.10
    assert out["mean_test_risk"] <= 0.20


def test_evaluate_conformal_counts_unachievable_splits():
    df = pd.DataFrame({"entropy": [0.5] * 60, "correct": [False] * 60})
    out = evaluate_conformal_abstention(df, "entropy", "correct",
                                        target_risk=0.05, n_splits=50)
    assert out["n_unachievable"] == 50
    assert out["n_valid"] == 0


def test_aurc_is_invariant_to_input_order_under_ties():
    """Regression test for a real bug found while building this: the first
    implementation swept per item, so within a group of tied entropy values the
    result depended on the CSV's row order. Ties dominate this project -- K=5
    entropy takes 7 distinct values across 300 items -- so the metric was
    reporting row order as if it were signal."""
    base = pd.DataFrame({"entropy": [0.5] * 20, "correct": [True] * 10 + [False] * 10})
    shuffled = base.iloc[np.random.default_rng(3).permutation(20)].reset_index(drop=True)
    assert aurc(base, "entropy", "correct")["aurc"] == pytest.approx(
        aurc(shuffled, "entropy", "correct")["aurc"]
    )


# --- plot_risk_coverage ---


def test_plot_risk_coverage_returns_axes_with_correct_step_count():
    df = pd.DataFrame({
        "entropy": [0.0] * 10 + [1.0] * 10,
        "correct": [True] * 9 + [False] * 11,
    })
    ax = plot_risk_coverage(df, "entropy", "correct")
    assert ax is not None
    # Exactly one solid step line (the curve) plus the dashed baseline
    # reference line drawn separately -- filter to the solid one.
    solid_lines = [ln for ln in ax.get_lines() if ln.get_linestyle() not in ("--", "dashed")]
    assert len(solid_lines) == 1
    xdata = solid_lines[0].get_xdata()
    assert xdata[0] == pytest.approx(0.5)  # coverage of the first (lowest-entropy) group
    assert xdata[-1] == pytest.approx(1.0)


def test_plot_risk_coverage_baseline_defaults_to_overall_error_rate():
    df = pd.DataFrame({
        "entropy": [0.0, 0.5, 1.0, 1.0],
        "correct": [True, True, False, True],
    })
    ax = plot_risk_coverage(df, "entropy", "correct")
    # The dashed reference line should sit at the overall error rate (1/4).
    hlines = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(hlines) == 1
    assert hlines[0].get_ydata()[0] == pytest.approx(0.25)


# --- classify_capability_check ---


def test_capability_check_at_chance():
    out = classify_capability_check(accuracy=0.517, baseline_accuracy=0.500, n_items=300)
    assert out["verdict"] == "at_chance"
    assert out["entropy_result_meaningful"] is False


def test_capability_check_real_3b_result_is_at_chance():
    """Regression: the actual 3B n=300 grading accuracy (0.517) must classify
    as at_chance, not marginal -- this is the number the whole gate exists to
    catch, so the boundary must not accidentally let it through."""
    out = classify_capability_check(accuracy=0.517, baseline_accuracy=0.500, n_items=300)
    assert out["verdict"] == "at_chance"


def test_capability_check_capable():
    out = classify_capability_check(accuracy=0.70, baseline_accuracy=0.500, n_items=300)
    assert out["verdict"] == "capable"
    assert out["entropy_result_meaningful"] is True


def test_capability_check_marginal_is_not_meaningful():
    """The middle band exists precisely so a so-so accuracy cannot be quietly
    read as validating reasoning entropy -- only 'capable' does that."""
    out = classify_capability_check(accuracy=0.60, baseline_accuracy=0.500, n_items=300)
    assert out["verdict"] == "marginal"
    assert out["entropy_result_meaningful"] is False


def test_capability_check_boundary_values():
    at_capable = classify_capability_check(accuracy=0.65, baseline_accuracy=0.5, n_items=300)
    assert at_capable["verdict"] == "capable"
    just_below = classify_capability_check(accuracy=0.649, baseline_accuracy=0.5, n_items=300)
    assert just_below["verdict"] == "marginal"
    at_chance_edge = classify_capability_check(accuracy=0.55, baseline_accuracy=0.5, n_items=300)
    assert at_chance_edge["verdict"] == "marginal"
    just_below_chance = classify_capability_check(accuracy=0.549, baseline_accuracy=0.5, n_items=300)
    assert just_below_chance["verdict"] == "at_chance"


def test_capability_check_uses_the_passed_baseline_not_a_hardcoded_half():
    """If a future run is not perfectly 50/50, the baseline must come from
    majority_class_baseline, not an assumed 0.5."""
    out = classify_capability_check(accuracy=0.60, baseline_accuracy=0.55, n_items=300)
    assert out["margin_over_baseline"] == pytest.approx(0.05)


# --- classify_stratum_result ----------------------------------------------

def _stratum_ci(auroc, lo, hi, n_error, n_items=150):
    return {
        "auroc": auroc, "ci_low": lo, "ci_high": hi,
        "n_items": n_items, "n_error": n_error, "n_correct": n_items - n_error,
        # Mirrors bootstrap_auroc_ci's ONE-SIDED definition exactly.
        "excludes_chance": lo > 0.5,
    }


def test_classify_stratum_absent_is_not_measured_not_underpowered():
    """A stratum that does not exist (ScratchMath has zero clean items) is a
    different statement from one with too few, and must not collapse into it."""
    from pilot.plotting import classify_stratum_result

    assert classify_stratum_result(None) == "not_measured"
    assert classify_stratum_result({"n_items": 0}) == "not_measured"


def test_classify_stratum_underpowered_beats_a_high_point_estimate():
    """ScratchMath's error stratum reads 0.852 on 4 misgraded items. The
    power bar has to win, or that number gets quoted."""
    from pilot.plotting import classify_stratum_result

    assert classify_stratum_result(_stratum_ci(0.852, 0.766, 0.943, n_error=4)) == \
        "inconclusive_underpowered"


def test_classify_stratum_inverted_is_not_reported_as_no_signal():
    """Regression for a real bug (2026-08-08). bootstrap_auroc_ci defines
    excludes_chance as `ci_low > 0.5`, a one-sided above-chance test, so a
    resolvably INVERTED stratum carries excludes_chance=False. An earlier
    ordering checked that gate first and sent Qwen-7B's CONFIRMED clean-stratum
    inversion -- 0.280 [0.200, 0.366], n_wrong=106 -- to "no_signal",
    mislabelling a confirmed finding as an absence of one. Caught by the
    snapshot of that run disagreeing with the report."""
    from pilot.plotting import classify_stratum_result

    qwen7b_clean = _stratum_ci(0.280, 0.200, 0.366, n_error=106)
    assert qwen7b_clean["excludes_chance"] is False  # the trap
    assert classify_stratum_result(qwen7b_clean) == "confirmed_inverted"

    internvl3_clean = _stratum_ci(0.369, 0.290, 0.454, n_error=97)
    assert classify_stratum_result(internvl3_clean) == "confirmed_inverted"


def test_classify_stratum_confirmed_and_below_threshold_are_distinguished():
    """The distinction that did not exist before InternVL3: powered and
    resolved above chance, but short of the 0.70 confirmation bar."""
    from pilot.plotting import classify_stratum_result

    # Qwen-7B matched n=648, and Qwen-3B n=650 -- both confirmed.
    assert classify_stratum_result(_stratum_ci(0.801, 0.751, 0.846, 73, 648)) == "confirmed"
    assert classify_stratum_result(_stratum_ci(0.854, 0.796, 0.902, 42, 650)) == "confirmed"
    # LLaVA-NeXT n=400 -- confirmed, first cross-family replication.
    assert classify_stratum_result(_stratum_ci(0.775, 0.694, 0.848, 34, 400)) == "confirmed"
    # InternVL3 -- powered, excludes chance, does NOT clear 0.70.
    assert classify_stratum_result(_stratum_ci(0.628, 0.548, 0.706, 45)) == \
        "resolved_below_threshold"


def test_classify_stratum_no_signal_when_ci_spans_chance():
    from pilot.plotting import classify_stratum_result

    assert classify_stratum_result(_stratum_ci(0.522, 0.462, 0.582, 100, 300)) == "no_signal"


def test_classify_stratum_threshold_boundary_is_exact():
    """Lock the cutoff so a future edit cannot quietly move it."""
    from pilot.plotting import classify_stratum_result

    assert classify_stratum_result(_stratum_ci(0.700, 0.651, 0.780, 40)) == "confirmed"
    assert classify_stratum_result(_stratum_ci(0.699, 0.651, 0.780, 40)) == \
        "resolved_below_threshold"
    # Minority exactly at the registered minimum passes; one below does not.
    assert classify_stratum_result(_stratum_ci(0.80, 0.72, 0.88, 30)) == "confirmed"
    assert classify_stratum_result(_stratum_ci(0.80, 0.72, 0.88, 29)) == \
        "inconclusive_underpowered"


# --- oracle_aurc / E-AURC (Geifman & El-Yaniv) ---------------------------
#
# Raw AURC is dominated by the base error rate, so it cannot compare two
# models with different accuracies -- a model that is wrong 60% of the time
# cannot have a good AURC however well it ranks. E-AURC subtracts what a
# perfect ranking would score at that same error rate, which is what makes
# the deferral numbers comparable across models and across datasets.


@pytest.mark.parametrize("rate,expected", [
    (0.0, 0.0),          # never wrong: perfect deferral is free
    (1.0, 1.0),          # always wrong: no ordering can help
    (0.5, 0.5 + 0.5 * math.log(0.5)),
])
def test_oracle_aurc_closed_form_endpoints(rate, expected):
    assert oracle_aurc(rate) == pytest.approx(expected, abs=1e-12)


def test_oracle_aurc_is_below_the_random_baseline_and_rises_with_error():
    """Sanity on the shape: a perfect ranker always beats random deferral
    (whose AURC is just the error rate), and both get worse as the model does."""
    prev = -1.0
    for r in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        o = oracle_aurc(r)
        assert 0.0 < o < r, f"oracle {o} should sit strictly between 0 and {r}"
        assert o > prev
        prev = o


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_oracle_aurc_rejects_impossible_error_rates(rate):
    with pytest.raises(ValueError, match="error_rate"):
        oracle_aurc(rate)


@pytest.mark.parametrize("n,n_err", [(300, 150), (100, 40), (1000, 10), (50, 1)])
def test_a_perfect_ranker_scores_exactly_zero_e_aurc_on_its_own_grid(n, n_err):
    """The defining invariant. A score that puts every correct item ahead of
    every wrong one has nothing left to improve, so e_aurc_on_grid must be 0
    exactly. (e_aurc against the CONTINUOUS oracle is only approximately 0 --
    the closed form integrates a continuum while the data is n discrete
    steps -- which is precisely why both are reported.)"""
    df = pd.DataFrame({"s": range(n), "ok": [True] * (n - n_err) + [False] * n_err})
    out = aurc(df, "s", "ok")
    assert out["e_aurc_on_grid"] == pytest.approx(0.0, abs=1e-12)
    assert out["e_aurc"] == pytest.approx(0.0, abs=0.005)
    assert out["aurc"] < out["baseline_aurc"]


def test_e_aurc_on_grid_does_not_charge_a_coarse_score_for_its_ties():
    """Why the on-grid variant exists. Cluster entropy at K=5 offers 7
    operating points; the continuous oracle assumes one at every coverage.
    Subtracting the continuous oracle blames the score for its GRID as well as
    its RANKING. Here the ranking is perfect and only the grid is coarse, so
    the on-grid gap is zero while the continuous gap is not."""
    df = pd.DataFrame({"entropy": [0.0] * 50 + [1.0] * 50,
                       "correct": [True] * 50 + [False] * 50})
    out = aurc(df, "entropy", "correct")
    assert out["n_operating_points"] == 2
    assert out["e_aurc_on_grid"] == pytest.approx(0.0, abs=1e-12)
    assert out["e_aurc"] > 0.05, "the continuous oracle should penalise the grid"


def test_e_aurc_is_worse_for_a_signal_free_score():
    uninformative = pd.DataFrame({"entropy": [0.5] * 40,
                                  "correct": [True, False] * 20})
    ranked = pd.DataFrame({"entropy": list(range(40)),
                           "correct": [True] * 20 + [False] * 20})
    assert aurc(uninformative, "entropy", "correct")["e_aurc"] > \
        aurc(ranked, "entropy", "correct")["e_aurc"]


# --- coverage_at_risk ----------------------------------------------------


def test_coverage_at_risk_finds_the_largest_qualifying_operating_point():
    df = pd.DataFrame({"entropy": [0.0] * 10 + [1.0] * 10 + [2.0] * 10,
                       "correct": [True] * 10 + [True] * 8 + [False] * 2
                                  + [False] * 10})
    at_10 = coverage_at_risk(df, "entropy", "correct", 0.10)
    assert at_10["achievable"] is True
    assert at_10["n_kept"] == 20 and at_10["risk_kept"] == pytest.approx(0.10)

    at_0 = coverage_at_risk(df, "entropy", "correct", 0.0)
    assert at_0["n_kept"] == 10 and at_0["risk_kept"] == 0.0


def test_coverage_at_risk_reports_unachievable_rather_than_guessing():
    """A common, real outcome at K=5, not an error condition: every operating
    point can sit above the target. Returning the closest one anyway would
    silently violate the risk guarantee the caller asked for."""
    df = pd.DataFrame({"entropy": [0.0] * 10, "correct": [False] * 10})
    out = coverage_at_risk(df, "entropy", "correct", 0.10)
    assert out["achievable"] is False
    assert out["coverage"] == 0.0 and out["n_kept"] == 0


def test_coverage_at_risk_scans_all_points_not_just_the_first_crossing():
    """risk_kept is not monotone in coverage. Here the 2-item point is above
    target and the 4-item point is below it, so an implementation that stopped
    at the first threshold to exceed the target would report unachievable."""
    df = pd.DataFrame({"entropy": [0.0, 0.0, 1.0, 1.0],
                       "correct": [True, False, True, True]})
    out = coverage_at_risk(df, "entropy", "correct", 0.30)
    assert out["achievable"] is True
    assert out["n_kept"] == 4 and out["risk_kept"] == pytest.approx(0.25)


def test_coverage_at_risk_is_monotone_in_the_target():
    df = pd.DataFrame({"entropy": list(range(30)),
                       "correct": [True] * 20 + [False] * 10})
    coverages = [coverage_at_risk(df, "entropy", "correct", t)["coverage"]
                 for t in (0.0, 0.1, 0.2, 0.3, 0.5)]
    assert coverages == sorted(coverages)


@pytest.mark.parametrize("target", [-0.1, 1.5])
def test_coverage_at_risk_rejects_an_impossible_target(target):
    df = pd.DataFrame({"entropy": [0.0], "correct": [True]})
    with pytest.raises(ValueError, match="target_risk"):
        coverage_at_risk(df, "entropy", "correct", target)


# --- plot_scoring_categories ---------------------------------------------


def _category_summary(models=("A",)):
    from pilot.rescore import CATEGORIES
    rows = []
    for m in models:
        for i, c in enumerate(CATEGORIES):
            rows.append({"model": m, "category": c, "n": 10 * (i + 1)})
    return pd.DataFrame(rows)


def test_plot_scoring_categories_draws_one_bar_per_category_in_order():
    summary = _category_summary()
    ax = plot_scoring_categories(summary.drop(columns="model"))
    from pilot.rescore import CATEGORIES
    assert len(ax.patches) == len(CATEGORIES)
    # Order is CATEGORIES, not sorted by size, so the same category sits in
    # the same row across models and the eye can compare.
    assert [t.get_text() for t in ax.get_yticklabels()] == list(CATEGORIES)


def test_plot_scoring_categories_groups_multiple_models():
    from pilot.rescore import CATEGORIES
    ax = plot_scoring_categories(_category_summary(models=("A", "B")))
    assert len(ax.patches) == 2 * len(CATEGORIES)
    assert ax.get_legend() is not None
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:2] == ["A", "B"]
    # Third key explains the hatch. Model swatches must use a real bar colour:
    # a neutral grey matched neither encoding channel and made readers stop to
    # work out that the key referred to opacity alone.
    handles = ax.get_legend().legend_handles
    assert handles[0].get_facecolor()[:3] == handles[1].get_facecolor()[:3]
    assert handles[0].get_alpha() != handles[1].get_alpha()


def test_scoring_regression_categories_are_hatched_apart_from_model_failures():
    """false_pass_removed items were scored CORRECT by the frozen rule; they
    are scoring failures, not model failures. Sharing the plain orange of
    genuinely_wrong reads as "the model got this wrong", the opposite of what
    they show, so they carry a hatch and a legend key of their own."""
    from pilot.rescore import CATEGORIES
    ax = plot_scoring_categories(_category_summary())

    hatched = {cat for cat, patch in zip(CATEGORIES, ax.patches)
               if patch.get_hatch()}
    assert hatched == {"false_pass_removed", "broken_by_relaxation"}

    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("not a model failure" in t for t in labels)


def test_plot_scoring_categories_rejects_an_unknown_category():
    """A typo'd category would otherwise be silently dropped from the plot,
    so the bars would not sum to the population they claim to describe."""
    bad = pd.DataFrame({"category": ["correct_robust", "typo_here"], "n": [1, 2]})
    with pytest.raises(ValueError, match="unknown categories"):
        plot_scoring_categories(bad)


def test_plot_scoring_categories_omits_categories_absent_from_the_data():
    ax = plot_scoring_categories(
        pd.DataFrame({"category": ["correct_robust", "genuinely_wrong"], "n": [3, 4]}))
    assert [t.get_text() for t in ax.get_yticklabels()] == \
        ["correct_robust", "genuinely_wrong"]


# --- contact_sheet -------------------------------------------------------
#
# Built because inline rendering does not survive the Colab -> local sync:
# the 2026-08-10 notebook-17 run has zero image/png outputs despite calling
# plt.show() and erroring nowhere. Evidence has to be written to a file.


def _stub_images(n, size=(60, 40)):
    from PIL import Image
    return [Image.new("RGB", size, "white") for _ in range(n)]


def test_contact_sheet_pages_and_pads_the_last_page():
    from pilot.plotting import contact_sheet
    figs = contact_sheet(_stub_images(7), [f"i{i}" for i in range(7)],
                         ncols=3, per_page=4)
    assert len(figs) == 2
    # Cell size must stay constant across pages, so the short final page is
    # padded with blank axes rather than laid out smaller.
    assert len(figs[0].axes) == len(figs[1].axes)
    # get_title() reads the CENTER title; captions are set with loc="left".
    titled = [ax for ax in figs[1].axes if ax.get_title(loc="left")]
    assert len(titled) == 3          # 7 items = 4 + 3


def test_contact_sheet_rejects_mismatched_captions():
    """A caption drifting off its image would mislabel the evidence, which is
    worse than no sheet at all."""
    from pilot.plotting import contact_sheet
    with pytest.raises(ValueError, match="captions"):
        contact_sheet(_stub_images(3), ["only", "two"])


def test_contact_sheet_of_nothing_is_empty_not_a_blank_page():
    from pilot.plotting import contact_sheet
    assert contact_sheet([], []) == []


def test_contact_sheet_captions_land_on_their_own_cells():
    from pilot.plotting import contact_sheet
    captions = ["alpha", "beta", "gamma"]
    fig = contact_sheet(_stub_images(3), captions, ncols=3, per_page=3)[0]
    assert [ax.get_title(loc="left") for ax in fig.axes
            if ax.get_title(loc="left")] == captions


# --- accuracy by distinct answers ----------------------------------------
#
# The plainest statement of the perception result and the one a reader
# understands without knowing what an AUROC is. Locked because it is headed
# for a figure, and because deriving it the obvious way gets it wrong.


def _distinct_table(csv_name):
    from pilot.plotting import accuracy_by_distinct_answers, distinct_from_entropy
    path = (Path(__file__).resolve().parents[2] / "results" / csv_name)
    if not path.exists():
        pytest.skip(f"{csv_name} not present (download from Drive)")
    df = pd.read_csv(path)
    df["n_distinct"] = distinct_from_entropy(df["perception_entropy"])
    return accuracy_by_distinct_answers(df).set_index("n_distinct")


def test_distinct_from_entropy_inverts_the_k5_values_exactly():
    """For K=5 the seven reachable entropies map one-to-one onto the
    partitions of 5, so the inversion is exact rather than approximate."""
    from pilot.plotting import distinct_from_entropy
    values = [0.0, 0.5004024235, 0.6730116670, 0.9502705392,
              1.0549201680, 1.3321790402, 1.6094379124]
    assert distinct_from_entropy(values) == [1, 2, 2, 3, 3, 4, 5]


def test_distinct_from_entropy_rejects_an_unreachable_value():
    """A value off the K=5 grid means the run was not K=5, and silently
    rounding it to the nearest would fabricate a distinct count."""
    from pilot.plotting import distinct_from_entropy
    with pytest.raises(ValueError, match="reachable"):
        distinct_from_entropy([0.30])


@pytest.mark.parametrize("csv_name,expected", [
    ("scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv",
     {1: (38, 0.921), 2: (62, 0.629), 3: (79, 0.380), 4: (60, 0.167), 5: (61, 0.066)}),
    ("pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv",
     {1: (56, 0.875), 2: (57, 0.614), 3: (76, 0.329), 4: (60, 0.267), 5: (51, 0.000)}),
])
def test_accuracy_falls_with_the_number_of_distinct_answers(csv_name, expected):
    table = _distinct_table(csv_name)
    for n_distinct, (n, acc) in expected.items():
        assert int(table.loc[n_distinct, "n"]) == n
        assert table.loc[n_distinct, "accuracy"] == pytest.approx(acc, abs=0.005)
    assert int(table["n"].sum()) == 300


def test_the_trend_is_monotone_at_the_ends_on_both_models():
    """The claim is the trend, not any single cell. Pixtral's 4-distinct cell
    sits slightly above its 3-distinct one (n=60, noise), so monotonicity is
    asserted where it is actually claimed -- across the full range."""
    for csv_name in ("scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv",
                     "pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv"):
        t = _distinct_table(csv_name)["accuracy"]
        assert t.loc[1] > 0.85
        assert t.loc[5] < 0.10
        assert t.loc[1] - t.loc[5] > 0.80          # a 13x range on Qwen
        assert t.loc[1] > t.loc[2] > t.loc[3]      # monotone where it is dense


def test_recomputing_distinct_counts_would_contradict_the_reported_numbers():
    """The trap this nearly fell into. Recomputing labels from the raw samples
    gives a DIFFERENT table for the 2026-08-02 Qwen run -- 80% rather than 92%
    in the one-distinct cell -- because that run was scored in a Colab session
    without SymPy's LaTeX parser and 43/300 items labelled differently.

    Every locked figure (AUROC 0.835, accuracy 39.3%) comes from the stored
    column, so a figure built from a recompute would contradict the paper's
    own headline. Pixtral is unaffected: its recompute matches the stored
    column exactly, which is why the discrepancy is easy to miss."""
    import ast

    import pilot.canonicalize
    import pilot.entropy
    import pilot.parsing
    from pilot.plotting import accuracy_by_distinct_answers

    path = (Path(__file__).resolve().parents[2] / "results"
            / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")
    if not path.exists():
        pytest.skip("Qwen CSV not present")
    df = pd.read_csv(path)
    df["n_distinct"] = [
        pilot.entropy.distinct_count(
            [pilot.canonicalize.canonical_answer_label(
                pilot.parsing.parse_transcription(s))
             for s in ast.literal_eval(r["all_transcription_samples_raw"])])
        for _, r in df.iterrows()]
    recomputed = accuracy_by_distinct_answers(df).set_index("n_distinct")

    stored = _distinct_table(path.name)
    assert recomputed.loc[1, "accuracy"] == pytest.approx(0.80, abs=0.02)
    assert stored.loc[1, "accuracy"] == pytest.approx(0.921, abs=0.005)
    assert abs(recomputed.loc[1, "accuracy"] - stored.loc[1, "accuracy"]) > 0.10
