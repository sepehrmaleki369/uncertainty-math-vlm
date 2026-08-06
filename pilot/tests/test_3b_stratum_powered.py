"""Locks the 2026-08-06 3B has_error=1 stratum-powering run against the saved CSV.

Direct follow-on to test_7b_stratum_powered.py, and the experiment that
answers the question that run's confirmation raised: is the has_error=1
stratified effect (entropy predicts grading correctness within items that
actually contain an error) specific to 7B, or does it hold at 3B too? 3B's
own n=300 run had the same shape but was never powered -- only 8 misgraded
items (results/scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv,
report S7.2). pilot/08_3b_error_stratum_power.ipynb drew 500 more
has_error=1 items (sized larger than 7B's 200, since 3B's error rate on
this stratum is roughly half 7B's) and merged them in.

The CSV is untracked (Drive-only); tests skip cleanly when absent.
"""

from pathlib import Path

import pandas as pd
import pytest

from pilot.plotting import (
    bootstrap_auroc_ci,
    stratified_auroc,
    SCALEUP_PREREGISTRATION,
)

RESULTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "grading_3b_stratum_powered_n800_qwen2.5-vl-3b-instruct_20260806T150044Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_is_300_reference_plus_500_extra_error_items(run):
    """650/150 on has_error -- the 3B extension needed more than double 7B's
    200 extra items, because 3B's error rate on this stratum is roughly
    half 7B's (it says "error" even more often, 93% vs 80%)."""
    assert len(run) == 800
    assert int(run["has_error"].astype(bool).sum()) == 650
    assert int((~run["has_error"].astype(bool)).sum()) == 150
    assert run["model_id"].unique().tolist() == ["Qwen/Qwen2.5-VL-3B-Instruct"]


def test_pooled_auroc_is_not_the_number_to_report(run):
    """Same trap as the 7B n=500 CSV: pooled reads above chance (0.610) only
    because the sample is no longer 50/50 (650 vs 150) -- an artifact of the
    stratum-targeted extension, not evidence the sign-reversal problem
    resolved. stratified_auroc is still the only correct read."""
    r = bootstrap_auroc_ci(run, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.610, abs=0.001)


def test_has_error_stratum_is_confirmed_and_close_to_7b(run):
    """The headline result: 3B's has_error=1 stratum, once powered, lands at
    0.854 -- confirmed (clears the registered 0.70 threshold, CI excludes
    chance), and close to 7B's own confirmed 0.834
    (test_7b_stratum_powered.py). Two independent model sizes now confirm
    the same stratified effect at a similar magnitude: this is evidence the
    effect is model-size-independent, not a 7B-specific behavior."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    threshold = SCALEUP_PREREGISTRATION["reasoning_stratum_auroc_min"]

    error_stratum = out["strata"][True]
    minority = min(error_stratum["n_error"], error_stratum["n_correct"])
    assert error_stratum["n_error"] == 42
    assert minority >= min_n
    assert error_stratum["auroc"] == pytest.approx(0.854, abs=0.001)
    assert error_stratum["ci_low"] > threshold - 0.01
    assert error_stratum["excludes_chance"] is True
    # Close to 7B's confirmed 0.834 -- not identical, but well within a
    # magnitude that supports "model-size-independent," not "coincidence."
    assert abs(error_stratum["auroc"] - 0.834) < 0.05


def test_clean_stratum_still_underpowered_as_expected(run):
    """No new clean items were drawn (the extension targets only
    has_error=1), so this must reproduce the original 3B n=300 clean-stratum
    result exactly and remain underpowered -- unlike 7B, 3B's clean stratum
    has never been powered (only 13 correct items, since 3B is wrong on
    clean items 91% of the time)."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    clean_stratum = out["strata"][False]
    minority = min(clean_stratum["n_error"], clean_stratum["n_correct"])

    assert clean_stratum["n_items"] == 150
    assert minority == 13
    assert minority < min_n
    assert clean_stratum["auroc"] == pytest.approx(0.239, abs=0.001)


def test_sign_reversal_holds(run):
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    assert out["sign_reversal"] is True
    assert out["pooled_understates"] is True


def test_response_bias_and_parse_failures_match_3b_pattern(run):
    """says_error stays at 3B's established ~93% level (the 500 new items
    are all has_error=1, and 3B's existing bias scores well on them, same
    mechanical effect as the 7B extension's said_error shift). Parse
    failures (32/4000 = 0.8%) are consistent with 3B's known higher parse-
    failure rate relative to 7B."""
    assert run["said_error"].mean() == pytest.approx(0.931, abs=0.005)
    assert int(run["n_grading_parse_failures"].sum()) == 32
