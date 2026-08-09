"""The single-label-stratum trap, and the retraction it forced.

This project reported, across three model families and two model sizes,
that reasoning entropy predicts grading errors within the has_error=1
stratum at AUROC 0.775-0.854. That claim was wrong, and these tests lock
the evidence that retired it so it cannot be reintroduced by someone
reading the old numbers.

Two independent demonstrations:

1. IDENTITY. Within a stratum where every item has the same true label,
   "was the model correct" and "what did the model answer" are the same
   column. On the 7B run they agree on 100% of items, and the AUROC
   against correctness is numerically identical to the AUROC against the
   model's own verdict. Entropy is computed from the very votes that
   produce that verdict, so the measurement is near-circular.

2. NULL. A model with no item-level signal at all -- one that answers
   "error" with a fixed probability per sample -- reaches a HIGHER AUROC
   than any of the three real models did. On an all-error stratum the
   majority is wrong only when most samples dissent, and dissent is
   exactly what raises entropy, so the association is arithmetic.

The perception arm is not affected, and one test here pins the structural
reason: there, correctness depends on which label wins rather than on how
concentrated the votes are, so a model can agree with itself and still be
wrong -- which happens on real items.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import pilot.parsing
from pilot.plotting import (
    bias_only_null_auroc,
    compute_auroc,
    correctness_collapses_onto_prediction,
)

ROOT = Path(__file__).resolve().parents[2]
GRADING_7B = (ROOT / "results"
              / "grading_7b_matched_n650_qwen2.5-vl-7b-instruct_20260806T172004Z.csv")
PERCEPTION = (ROOT / "results"
              / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")


def _majority_said_error(raw):
    digits = [pilot.parsing.parse_grading(s) for s in ast.literal_eval(raw)]
    votes = [d for d in digits if d is not None]
    return bool(votes) and sum(votes) > len(votes) / 2


@pytest.fixture(scope="module")
def grading():
    if not GRADING_7B.exists():
        pytest.skip(f"{GRADING_7B} not present (download from Drive)")
    df = pd.read_csv(GRADING_7B)
    df["said_error"] = df["all_grading_samples_raw"].apply(_majority_said_error)
    return df


# --- 1. the identity ------------------------------------------------------

def test_correctness_is_the_models_own_verdict_within_a_stratum(grading):
    """The retraction in one assertion: inside has_error=1 the two columns
    are the same data, so an AUROC against correctness cannot be evidence
    about error prediction."""
    err = grading[grading["has_error"].astype(bool)]
    d = correctness_collapses_onto_prediction(err, "grading_correct", "said_error")

    assert d["agreement"] == pytest.approx(1.0, abs=0.001)
    assert d["degenerate"] is True

    a_correct = compute_auroc(err, "reasoning_entropy", "grading_correct")
    a_verdict = compute_auroc(err, "reasoning_entropy", "said_error")
    assert a_correct == pytest.approx(a_verdict, abs=0.001)
    assert a_correct == pytest.approx(0.801, abs=0.005)


def test_the_clean_stratum_is_the_same_identity_with_the_sign_flipped(grading):
    """The 'sign reversal' this project treated as its key finding. In the
    clean stratum correctness is the exact COMPLEMENT of the verdict, so
    anything correlating with the verdict must invert between strata. The
    two AUROCs sum to 1 -- that is arithmetic, not a discovery."""
    clean = grading[~grading["has_error"].astype(bool)]
    d = correctness_collapses_onto_prediction(clean, "grading_correct", "said_error")

    assert d["agreement"] == pytest.approx(0.0, abs=0.001)   # exact complement
    assert d["degenerate"] is True

    a_correct = compute_auroc(clean, "reasoning_entropy", "grading_correct")
    a_verdict = compute_auroc(clean, "reasoning_entropy", "said_error")
    assert a_correct + a_verdict == pytest.approx(1.0, abs=0.002)
    assert a_correct == pytest.approx(0.280, abs=0.005)


# --- 2. the null ----------------------------------------------------------

@pytest.mark.parametrize("label,n,rate,observed", [
    ("Qwen-3B",    650, 0.843, 0.854),
    ("Qwen-7B",    648, 0.813, 0.801),
    ("LLaVA-NeXT", 400, 0.780, 0.775),
])
def test_a_signalless_biased_coin_matches_or_beats_every_reported_result(
        label, n, rate, observed):
    """No item-level information whatsoever, yet the null median exceeds
    what each real model achieved. A result that cannot beat this floor is
    not evidence of an uncertainty signal."""
    null = bias_only_null_auroc(n, rate, k=5, n_sims=600, seed=0)
    assert null["n_valid"] > 500
    assert null["median"] > observed, (
        f"{label}: null median {null['median']:.3f} should exceed the "
        f"observed {observed:.3f}; if this ever fails the retraction needs "
        f"revisiting")
    assert null["ci_low"] > 0.70, (
        f"{label}: even the null's 2.5th percentile clears the "
        f"pre-registered 0.70 'confirmation' threshold, which is why that "
        f"threshold could not distinguish signal from arithmetic here")


# --- 3. why perception is unaffected --------------------------------------

def test_perception_does_not_collapse_because_the_label_is_not_binary():
    """The structural difference. Grading correctness is a function of the
    vote COUNT, which is also what entropy measures. Perception correctness
    depends on WHICH label wins, so unanimity does not imply correctness --
    and on the real run, 3 of 38 unanimous items are wrong. That decoupling
    is what makes the perception AUROC a real measurement."""
    if not PERCEPTION.exists():
        pytest.skip(f"{PERCEPTION} not present (download from Drive)")
    df = pd.read_csv(PERCEPTION)

    unanimous = df[df["perception_entropy"] < 1e-9]
    n_wrong = int((~unanimous["transcription_correct"].astype(bool)).sum())
    assert len(unanimous) == 38
    assert n_wrong == 3, "confidently-wrong items must exist for the metric to mean anything"

    # Correctness is genuinely not the model's own agreement level: the
    # confidently-wrong cell is non-empty at both ends of the entropy range.
    at_max = df[df["perception_entropy"] > 1.6]
    assert int(at_max["transcription_correct"].astype(bool).sum()) == 4, \
        "and confidently-uncertain-but-right items must exist too"
