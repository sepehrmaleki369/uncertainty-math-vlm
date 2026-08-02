import math

import pandas as pd
import pytest

from pilot.plotting import (
    REQUIRED_COLUMNS,
    SCALEUP_PREREGISTRATION,
    classify_scaleup_result,
    auroc_sensitivity,
    bootstrap_auroc_ci,
    bootstrap_auroc_difference_ci,
    check_parse_failure_rate,
    check_temperature_zero_anchor,
    classify_k_resample_result,
    compute_auroc,
    join_k_resample,
    load_results,
    majority_class_baseline,
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
