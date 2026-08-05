"""Locks the 2026-08-05 7B grading capability check against the saved CSV.

The notebook (pilot/05_7b_capability_check.ipynb) gated on accuracy and
printed pooled/stratified AUROC as diagnostic-only, before any formal
registration. This file records the post-hoc follow-up: applying the
project's existing power standard (SCALEUP_PREREGISTRATION's
min_minority_class=30, established for the 3B n=300 run) to the 7B result,
consistently rather than inventing a new or looser bar.

The CSV is untracked (Drive-only); tests skip cleanly when absent.
"""

from pathlib import Path

import pandas as pd
import pytest

from pilot.plotting import (
    bootstrap_auroc_ci,
    classify_capability_check,
    majority_class_baseline,
    stratified_auroc,
    SCALEUP_PREREGISTRATION,
)

RESULTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "grading_7b_n300_qwen2.5-vl-7b-instruct_20260805T115859Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_is_the_full_balanced_300_unquantized(run):
    assert len(run) == 300
    assert int(run["has_error"].astype(bool).sum()) == 150
    assert (run["quantized"] == False).all()  # noqa: E712 -- explicit bool check
    assert run["model_id"].unique().tolist() == ["Qwen/Qwen2.5-VL-7B-Instruct"]


def test_capability_gate_is_marginal(run):
    """59.0% accuracy against a 50.0% baseline: better than 3B's 51.7%, but
    below the 0.65 bar for a clearly resolved capability. The gate exists
    precisely so this result is not read as either a clean positive or a
    clean negative."""
    baseline = majority_class_baseline(run, "has_error")
    accuracy = float(run["grading_correct"].mean())
    gate = classify_capability_check(
        accuracy=accuracy, baseline_accuracy=baseline["baseline_accuracy"],
        n_items=len(run),
    )
    assert accuracy == pytest.approx(0.590, abs=0.001)
    assert baseline["baseline_accuracy"] == pytest.approx(0.500, abs=0.001)
    assert gate["verdict"] == "marginal"
    assert gate["entropy_result_meaningful"] is False


def test_pooled_auroc_shows_no_signal(run):
    r = bootstrap_auroc_ci(run, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.520, abs=0.001)
    assert r["excludes_chance"] is False


def test_stratified_result_replicates_the_3b_shape(run):
    """The core finding: pooling cancels a real effect, same as 3B. Both
    sign_reversal and pooled_understates must hold, and the has_error=1
    point estimate (0.761) must sit close to 3B's 0.756 -- not claiming
    either is individually confirmed (both are underpowered), only that the
    pattern is stable across model size."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    assert out["sign_reversal"] is True
    assert out["pooled_understates"] is True

    error_stratum = out["strata"][True]
    assert error_stratum["auroc"] == pytest.approx(0.761, abs=0.001)
    assert error_stratum["n_error"] == 17  # matches 3B's 8 in being underpowered
    assert abs(error_stratum["auroc"] - 0.756) < 0.02  # 3B's has_error=1 AUROC


def test_has_error_stratum_is_underpowered_by_the_registered_standard(run):
    """Applying SCALEUP_PREREGISTRATION's min_minority_class=30 (the 3B
    standard) consistently, not a new bar invented for 7B. n_wrong=17 fails
    it, same as 3B's n_wrong=8 -- the point estimate (0.761) cannot be
    reported as a confirmed result."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=2000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    error_stratum = out["strata"][True]
    minority = min(error_stratum["n_error"], error_stratum["n_correct"])
    assert minority == 17
    assert minority < min_n


def test_clean_stratum_is_adequately_powered_and_confirms_inversion(run):
    """Unlike 3B (13 clean items total, hopelessly underpowered), 7B's
    balanced sample gives the clean stratum 150 items / 44 misgraded --
    clearing the registered minimum of 30 for the first time. The inversion
    (entropy predicts backwards on clean items) is therefore a genuinely
    resolved finding here, not merely a suggestive direction."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    clean_stratum = out["strata"][False]
    minority = min(clean_stratum["n_error"], clean_stratum["n_correct"])

    assert minority == 44
    assert minority >= min_n  # the power bar 3B's clean stratum (13) never cleared
    assert clean_stratum["auroc"] == pytest.approx(0.280, abs=0.001)
    assert clean_stratum["ci_high"] < 0.5  # resolvably below chance, not just a trend


def test_parse_failures_and_response_bias_improve_but_do_not_disappear(run):
    """7B follows the output format more reliably than 3B (0.1% vs 0.9% parse
    failures) and is less error-biased (80% vs 93% 'there is a mistake'), but
    the same directional bias that broke the 3B pooled result is still
    present, which is consistent with the sign-reversal pattern replicating."""
    assert int(run["n_grading_parse_failures"].sum()) == 2
    assert run["said_error"].mean() == pytest.approx(0.797, abs=0.005)
    assert run["said_error"].mean() < 0.93  # improved over 3B, not eliminated
