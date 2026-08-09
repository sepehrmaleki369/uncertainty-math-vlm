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

Locks the 2026-08-06 LLaVA-NeXT n=300 FERMAT run against the saved CSV.

pilot/10_llava_next_fermat.ipynb ran clean (adapter worked on the first
try, no fallback needed), but the perception result is diagnostic-only,
not a confirmed AUROC -- this file records why, the same way
test_7b_capability_check.py records a "marginal" gate rather than treating
a middling number as either a clean positive or negative.

The CSV is untracked (Drive-only); tests skip cleanly when absent.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from pilot.plotting import bootstrap_auroc_ci, stratified_auroc

RESULTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scaleup_n300_bal50_llava-v16-mistral-7b-hf_20260806T231143Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_is_the_full_balanced_300_unquantized(run):
    assert len(run) == 300
    assert int(run["has_error"].astype(bool).sum()) == 150
    assert (run["quantized"] == False).all()  # noqa: E712
    assert run["model_id"].unique().tolist() == ["llava-hf/llava-v1.6-mistral-7b-hf"]


def test_perception_is_gated_by_a_real_capability_gap_not_a_pipeline_bug(run):
    """Only 9/300 transcriptions are correct (3.0%, vs Qwen's ~42%). Verified
    this is not a scoring bug -- independently recomputing transcription_correct
    from the raw text via pilot.parsing/pilot.canonicalize matches the stored
    column on all 300 rows. Verified it is not primarily a format-following
    failure either -- parse failures are only 9.6% (144/1500), so the model
    mostly produces well-formed **Question:**/**Answer:** output; it just gets
    the content wrong. The diagnosis: LLaVA can read a multiple-choice letter
    but cannot transcribe free-response handwritten derivations."""
    assert int(run["transcription_correct"].sum()) == 9
    assert int(run["n_transcription_parse_failures"].sum()) == 144

    is_mcq = (
        run["orig_q"].str.contains(r"\\item\[|Option", regex=True, na=False)
        | run["pert_a"].str.contains("Option", na=False)
    )
    mcq_accuracy = run.loc[is_mcq, "transcription_correct"].mean()
    free_response_accuracy = run.loc[~is_mcq, "transcription_correct"].mean()

    assert int(is_mcq.sum()) == 57
    assert mcq_accuracy == pytest.approx(0.123, abs=0.005)
    assert free_response_accuracy == pytest.approx(0.008, abs=0.005)
    # The gap is the whole story: MCQ items are ~15x more likely to be
    # correct than free-response ones, and free-response accuracy is
    # indistinguishable from a floor effect (2/243).
    assert mcq_accuracy > free_response_accuracy * 10


def test_perception_auroc_is_not_reportable(run):
    """9 correct items is an order of magnitude below the registered minimum
    of 30 used everywhere else in this project. The point estimate (0.710)
    is not wrong, exactly -- it is uninterpretable at this n, with a CI that
    barely excludes chance."""
    r = bootstrap_auroc_ci(run, "perception_entropy", "transcription_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.710, abs=0.005)
    assert r["ci_low"] < 0.55  # barely clears chance -- not a result to report


def test_reasoning_pooled_at_chance_same_as_qwen(run):
    assert run["grading_correct"].mean() == pytest.approx(0.500, abs=0.001)


def test_reasoning_stratified_replicates_qwen7b_shape_but_underpowered(run):
    """The promising part: LLaVA's clean-stratum point estimate (0.283) is
    within 0.01 of Qwen-7B's CONFIRMED 0.280 (test_7b_matched_n650.py) --
    same sign-reversal mechanism, same rough magnitude, different model
    family. Not confirmed here: both strata have only 14 misgraded items,
    same underpowered starting point Qwen-7B was in before notebook 06."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    assert out["sign_reversal"] is True
    assert out["pooled_understates"] is True

    error_stratum = out["strata"][True]
    clean_stratum = out["strata"][False]
    assert error_stratum["n_error"] == 14
    assert clean_stratum["n_error"] == 136  # i.e. 14 correct on clean items
    assert error_stratum["auroc"] == pytest.approx(0.766, abs=0.005)
    assert clean_stratum["auroc"] == pytest.approx(0.283, abs=0.005)
    # Close to Qwen-7B's confirmed clean-stratum result (0.280).
    assert abs(clean_stratum["auroc"] - 0.280) < 0.02
