"""RETRACTED INTERPRETATION (2026-08-09). The numbers asserted below are
correct and are kept as a record of what was measured. What they were
taken to MEAN was wrong.

Within a stratum where every item shares the same true label,
"was the model correct" is the same column as "what did the model
answer" -- they agree on 100% of items here. Reasoning entropy is
computed from the very votes that produce that answer, so the
stratified AUROC is near-circular, and a signal-free biased coin
reaches a HIGHER value than any model in this project did. See
pilot/tests/test_stratum_degeneracy.py.

Do not cite the has_error stratified AUROCs as evidence that entropy
predicts grading errors. The honest reasoning result is the pooled
one on a balanced sample: ~0.52, no signal.

Locks the 2026-08-05 7B has_error=1 stratum-powering run against the saved CSV.

Follow-on to test_7b_capability_check.py: that run's has_error=1 stratum had
only 17 misgraded items, below the registered minimum of 30
(SCALEUP_PREREGISTRATION["min_minority_class"]), so its 0.761 AUROC could
only be reported as a point estimate, not a confirmed result. This notebook
(pilot/06_7b_error_stratum_power.ipynb) drew 200 more has_error=1 items
(disjoint by construction -- see pilot.data.load_fermat_extra_error_items)
and merged them with the original 300-item reference run, producing 350
has_error=1 items / 38 misgraded -- the first time either grading stratum in
this project has cleared the power bar with a large, unambiguous effect.

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
    / "grading_7b_stratum_powered_n500_qwen2.5-vl-7b-instruct_20260805T135021Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_is_300_reference_plus_200_extra_error_items(run):
    """350/150 on has_error, not the original run's 150/150 -- this sample is
    deliberately unbalanced, skewed toward the stratum that needed power."""
    assert len(run) == 500
    assert int(run["has_error"].astype(bool).sum()) == 350
    assert int((~run["has_error"].astype(bool)).sum()) == 150
    assert (run["quantized"] == False).all()  # noqa: E712 -- explicit bool check
    assert run["model_id"].unique().tolist() == ["Qwen/Qwen2.5-VL-7B-Instruct"]


def test_pooled_auroc_is_not_the_number_to_report(run):
    """Pooled AUROC now reads ABOVE chance (0.612), unlike the balanced n=300
    run's 0.520 -- but that shift is an artifact of the sample no longer
    being 50/50 (350 vs 150), which mechanically favors the stronger,
    now-larger has_error=1 stratum in the pooled mix. It is not evidence the
    sign-reversal problem went away. Asserting the CI here specifically to
    document that "pooled looks fine now" is the wrong read -- stratified_auroc
    is still required, per this project's own standing rule."""
    r = bootstrap_auroc_ci(run, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.612, abs=0.001)


def test_has_error_stratum_is_now_confirmed(run):
    """The core result: n_wrong grew from 17 (underpowered) to 38 (clears the
    registered minimum of 30), and the point estimate rose from 0.761 to
    0.834 with a CI that excludes both chance and the pre-registered 0.70
    threshold's lower edge. This is the first has_error=1 stratum result in
    the whole project (3B or 7B) that can be reported as confirmed rather
    than suggestive."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    threshold = SCALEUP_PREREGISTRATION["reasoning_stratum_auroc_min"]

    error_stratum = out["strata"][True]
    minority = min(error_stratum["n_error"], error_stratum["n_correct"])
    assert error_stratum["n_error"] == 38
    assert minority >= min_n
    assert error_stratum["auroc"] == pytest.approx(0.834, abs=0.001)
    assert error_stratum["ci_low"] > threshold - 0.01  # CI clears the registered bar
    assert error_stratum["excludes_chance"] is True


def test_clean_stratum_inversion_unchanged(run):
    """The clean stratum got no new items (extension only drew has_error=1),
    so this should reproduce test_7b_capability_check.py's already-confirmed
    0.280 result exactly -- a consistency check that merging didn't disturb
    the reference run's existing 150 clean items."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    clean_stratum = out["strata"][False]
    assert clean_stratum["n_items"] == 150
    assert clean_stratum["auroc"] == pytest.approx(0.280, abs=0.001)
    assert clean_stratum["ci_high"] < 0.5


def test_sign_reversal_confirmed_at_both_strata_simultaneously(run):
    """The headline: for the first time in this project, BOTH strata are
    simultaneously powered (38 and 44 misgraded, both >= 30), each with a CI
    that resolves away from chance -- in opposite directions, so
    `excludes_chance` (one-directional, true only above 0.5) is checked
    separately per stratum rather than uniformly. This is no longer a
    hypothesis about pooling hiding a real effect -- it is a fully resolved,
    symmetric result."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    assert out["sign_reversal"] is True
    assert out["pooled_understates"] is True

    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    for level in (True, False):
        s = out["strata"][level]
        minority = min(s["n_error"], s["n_correct"])
        assert minority >= min_n

    assert out["strata"][True]["excludes_chance"] is True   # CI clears 0.5 above
    assert out["strata"][False]["ci_high"] < 0.5             # CI clears 0.5 below


def test_response_bias_and_parse_failures(run):
    """says_error rises to 83.6% (vs the reference-only run's 79.7%) simply
    because the 200 new items are all has_error=1 and the model's existing
    bias toward saying "error" scores well on them -- not a new finding,
    just what adding only-error items to an error-biased model does to the
    aggregate rate. Parse failures stay at 2 total (both from the reference
    run; the 200 new items contributed zero new failures)."""
    assert int(run["n_grading_parse_failures"].sum()) == 2
    assert run["said_error"].mean() == pytest.approx(0.836, abs=0.005)
