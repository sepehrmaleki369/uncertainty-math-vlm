"""Locks the 2026-08-06 7B-matched-to-3B (n=650) run against the saved CSV.

Direct follow-on to test_3b_stratum_powered.py: 7B's has_error=1 stratum was
already confirmed at n=350 (0.834 [0.768, 0.891]), but 3B needed 650 items
to reach the same power, leaving the two models compared at different total
n. This run extends 7B by 300 more has_error=1 items (skip=350) purely so
both models are reported on (almost) the identical item set -- not a power
fix, since 7B was already confirmed at n=350.

Two items were dropped during the merge: FERMAT's Hub copy appears to have
drifted slightly since notebook 06's round-1 extra ran (same seed=42 +
skip=150 no longer reproduces an ordering perfectly disjoint from a fresh
skip=350 draw), so 2 of the freshly-drawn round-2 items happened to already
be in round-1. Diagnosed by printing the overlap before dropping it -- both
were ordinary questions, consistent with upstream dataset drift rather than
a sampling bug. Final n is 648 has_error=1 items (798 total), not 650.

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
    / "grading_7b_matched_n650_qwen2.5-vl-7b-instruct_20260806T172004Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_is_300_reference_plus_200_plus_298_extra(run):
    """798 total, not 800 -- 2 of the round-2 draw's 300 items were dropped
    as duplicates of round-1 items (see module docstring)."""
    assert len(run) == 798
    assert int(run["has_error"].astype(bool).sum()) == 648
    assert int((~run["has_error"].astype(bool)).sum()) == 150
    assert (run["quantized"] == False).all()  # noqa: E712
    assert run["model_id"].unique().tolist() == ["Qwen/Qwen2.5-VL-7B-Instruct"]


def test_pooled_auroc_is_not_the_number_to_report(run):
    """Same trap as every other stratum-targeted extension in this project:
    pooled reads 0.641 (above chance) purely because the sample is no
    longer 50/50 (648 vs 150) -- stratified_auroc is still the only correct
    read."""
    r = bootstrap_auroc_ci(run, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.641, abs=0.001)


def test_has_error_stratum_confirmed_refines_but_does_not_overturn_n350(run):
    """The headline number moves from 0.834 (n=350) to 0.801 (n=648) --
    still clears the registered 0.70 threshold and the power minimum, and
    0.801 sits comfortably inside the n=350 result's own CI [0.768, 0.891].
    This is a refinement from more data, not a contradictory finding."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    threshold = SCALEUP_PREREGISTRATION["reasoning_stratum_auroc_min"]

    error_stratum = out["strata"][True]
    minority = min(error_stratum["n_error"], error_stratum["n_correct"])
    assert error_stratum["n_error"] == 73
    assert minority >= min_n
    assert error_stratum["auroc"] == pytest.approx(0.801, abs=0.001)
    assert error_stratum["ci_low"] > threshold - 0.01
    assert error_stratum["excludes_chance"] is True
    # Consistent with (not identical to) the earlier n=350 point estimate.
    assert 0.768 <= error_stratum["auroc"] <= 0.891


def test_still_overlaps_3b_confirmed_ci(run):
    """3B's matched-n confirmed result is 0.854 [0.796, 0.902]
    (test_3b_stratum_powered.py). The two models' CIs still overlap here
    (7B's upper bound 0.846 > 3B's lower bound 0.796), so the two remain
    statistically indistinguishable -- just not as numerically close as the
    earlier n=350-vs-n=650 comparison (0.834 vs 0.854) made them look."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    error_stratum = out["strata"][True]
    assert error_stratum["ci_high"] > 0.796  # overlaps 3B's ci_low


def test_clean_stratum_unchanged(run):
    """No new clean items were drawn; must reproduce the reference run's
    already-confirmed clean-stratum result exactly."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    clean_stratum = out["strata"][False]
    assert clean_stratum["n_items"] == 150
    assert clean_stratum["auroc"] == pytest.approx(0.280, abs=0.001)
    assert clean_stratum["ci_high"] < 0.5


def test_sign_reversal_holds(run):
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    assert out["sign_reversal"] is True
    assert out["pooled_understates"] is True


def test_parse_failures_stay_low(run):
    """Parse failures stay at 2 total (both from the reference run -- the
    548 new items across both extension rounds contributed zero new
    failures), consistent with 7B's established low parse-failure rate."""
    assert int(run["n_grading_parse_failures"].sum()) == 2
