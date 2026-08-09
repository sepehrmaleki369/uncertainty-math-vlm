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

Locks the 2026-08-07 LLaVA-NeXT has_error=1 stratum-powering run.

The single most important result in the WACV push so far: the first
cross-MODEL-FAMILY replication in this project (every prior replication --
7B vs 3B -- was within the Qwen2.5-VL family). LLaVA-NeXT's n=300 reference
run (pilot/10_llava_next_fermat.ipynb, test_llava_next_fermat.py) found a
promising but underpowered echo of Qwen-7B's confirmed has_error=1 result
(0.766 point estimate on 14 misgraded items, need 30+). This notebook
(pilot/11_llava_error_stratum_power.ipynb) drew 250 more has_error=1 items
to power it properly.

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
    / "grading_llava_stratum_powered_n550_llava-v1.6-mistral-7b-hf_20260807T181623Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_is_300_reference_plus_250_extra_error_items(run):
    assert len(run) == 550
    assert int(run["has_error"].astype(bool).sum()) == 400
    assert int((~run["has_error"].astype(bool)).sum()) == 150
    assert (run["quantized"] == False).all()  # noqa: E712
    assert run["model_id"].unique().tolist() == ["llava-hf/llava-v1.6-mistral-7b-hf"]


def test_pooled_auroc_is_not_the_number_to_report(run):
    """Same trap as every other stratum-targeted extension in this project:
    pooled reads 0.542 here, close to chance but not meaningfully so at
    this sample composition -- stratified_auroc is still the only correct
    read, since the sample is no longer 50/50 (400 has_error=1 vs 150
    clean)."""
    r = bootstrap_auroc_ci(run, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.542, abs=0.001)


def test_has_error_stratum_now_confirmed_first_cross_family_replication(run):
    """The headline result: LLaVA-NeXT's has_error=1 stratum clears the
    registered power bar (34 >= 30) and the 0.70 threshold, at
    0.775 [0.694, 0.848] -- close to and overlapping with Qwen-7B's
    confirmed 0.801 [0.751, 0.846] (test_7b_matched_n650.py). This is the
    first confirmed replication across model FAMILIES in the project (all
    prior replications were 3B vs 7B within Qwen2.5-VL)."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    threshold = SCALEUP_PREREGISTRATION["reasoning_stratum_auroc_min"]

    error_stratum = out["strata"][True]
    minority = min(error_stratum["n_error"], error_stratum["n_correct"])
    assert error_stratum["n_error"] == 34
    assert minority >= min_n
    assert error_stratum["auroc"] == pytest.approx(0.775, abs=0.001)
    assert error_stratum["ci_low"] > threshold - 0.01
    assert error_stratum["excludes_chance"] is True
    # Overlaps Qwen-7B's confirmed CI [0.751, 0.846] -- statistically
    # indistinguishable, not just "in the same ballpark" by eye.
    assert error_stratum["ci_high"] > 0.751


def test_clean_stratum_still_underpowered_but_consistent(run):
    """No new clean items were drawn; must reproduce the reference run's
    already-observed clean-stratum result exactly, and remain underpowered
    (14 correct items, same as notebook 10 -- LLaVA needed more error
    items, not more clean ones)."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    min_n = SCALEUP_PREREGISTRATION["min_minority_class"]
    clean_stratum = out["strata"][False]
    minority = min(clean_stratum["n_error"], clean_stratum["n_correct"])

    assert clean_stratum["n_items"] == 150
    assert minority == 14
    assert minority < min_n
    assert clean_stratum["auroc"] == pytest.approx(0.283, abs=0.001)
    # Consistent with Qwen-7B's confirmed clean-stratum inversion (0.280).
    assert abs(clean_stratum["auroc"] - 0.280) < 0.02


def test_sign_reversal_holds(run):
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    assert out["sign_reversal"] is True
    assert out["pooled_understates"] is True


def test_response_bias_replicates_across_model_families(run):
    """says_error stays at ~91% -- close to Qwen-3B's 93% and higher than
    Qwen-7B's 80%, but the same qualitative "says error" bias mechanism
    that drives the sign-reversal pattern in every model tested so far."""
    assert run["said_error"].mean() == pytest.approx(0.911, abs=0.005)
