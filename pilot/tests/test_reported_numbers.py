"""Locks the exact figures reported in report/summary.pdf against the saved run.

Every number we sent out is asserted here. If a code change, a dependency
change, or an environment difference would move any of them, this file fails
and names the number that moved -- instead of the discrepancy being found by
someone reading the PDF.

The CSV lives in results/, which is untracked (it comes from Drive by hand), so
these tests skip cleanly when it is absent rather than failing on a fresh clone.
Run them before sending any updated figures.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.canonicalize import latex_parser_available
from pilot.plotting import bootstrap_auroc_ci, majority_class_baseline

RESULTS = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS.exists():
        pytest.skip(f"scale-up CSV not present at {RESULTS} (download it from Drive)")
    return pd.read_csv(RESULTS)


def test_sample_is_the_balanced_300(run):
    """Guards against pointing these assertions at a different run entirely."""
    assert len(run) == 300
    assert int(run["has_error"].astype(bool).sum()) == 150
    assert majority_class_baseline(run, "has_error")["baseline_accuracy"] == pytest.approx(0.5)


def test_perception_auroc_as_reported(run):
    """summary.pdf: AUROC 0.835 [0.787, 0.879]."""
    r = bootstrap_auroc_ci(run, "perception_entropy", "transcription_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.835, abs=0.001)
    assert r["ci_low"] == pytest.approx(0.787, abs=0.001)
    assert r["ci_high"] == pytest.approx(0.879, abs=0.001)
    assert r["excludes_chance"] is True


def test_reasoning_auroc_as_reported(run):
    """summary.pdf: AUROC 0.522 [0.462, 0.582], no signal."""
    r = bootstrap_auroc_ci(run, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.522, abs=0.001)
    assert r["ci_low"] == pytest.approx(0.462, abs=0.001)
    assert r["ci_high"] == pytest.approx(0.582, abs=0.001)
    assert r["excludes_chance"] is False  # the CI contains chance -- the headline claim


def test_grading_accuracy_is_at_chance(run):
    """summary.pdf: 0.517 against a 0.500 baseline, 0.947 on error items,
    0.087 on clean ones, answers 'there is an error' on 93% of items."""
    gt = run["has_error"].astype(bool)
    assert run["grading_correct"].mean() == pytest.approx(0.517, abs=0.001)
    assert run.loc[gt, "grading_correct"].mean() == pytest.approx(0.947, abs=0.001)
    assert run.loc[~gt, "grading_correct"].mean() == pytest.approx(0.087, abs=0.001)

    says_error = np.where(gt, run["grading_correct"], ~run["grading_correct"])
    assert says_error.mean() == pytest.approx(0.93, abs=0.005)


def test_clean_stratum_inversion_as_reported(run):
    """summary.pdf: 0.239, below 0.5 -- entropy runs backwards on clean items."""
    gt = run["has_error"].astype(bool)
    r = bootstrap_auroc_ci(run[~gt], "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.239, abs=0.001)
    assert r["ci_high"] < 0.5


def test_abstention_rule_as_reported(run):
    """summary.pdf: when all five transcriptions disagree, wrong 57 of 61
    (93.4% precision), catching 31% of transcription errors."""
    at_max = np.isclose(run["perception_entropy"], math.log(5))
    flagged = run[at_max]
    n_wrong = int((~flagged["transcription_correct"]).sum())

    assert len(flagged) == 61
    assert n_wrong == 57
    assert n_wrong / len(flagged) == pytest.approx(0.934, abs=0.001)

    total_errors = int((~run["transcription_correct"]).sum())
    assert n_wrong / total_errors == pytest.approx(0.313, abs=0.005)


def test_transcription_accuracy_is_the_reported_lower_bound(run):
    """summary.pdf reports 0.393 explicitly as a strict lower bound, not an
    estimate -- see test_scoring_is_environment_dependent below for why."""
    assert int(run["transcription_correct"].sum()) == 118
    assert run["transcription_correct"].mean() == pytest.approx(0.393, abs=0.001)


def test_confidently_wrong_is_rare(run):
    """summary.pdf: only 3 of 300 items were both fully consistent and scored
    wrong. The claim that the model is almost never confidently wrong."""
    steady_and_wrong = (run["perception_entropy"] < 1e-9) & (~run["transcription_correct"])
    assert int(steady_and_wrong.sum()) == 3


@pytest.mark.skipif(not latex_parser_available(), reason="needs SymPy's LaTeX parser")
def test_scoring_is_environment_dependent_and_this_run_predates_the_pin(run):
    """Documents a known, deliberate gap rather than hiding it.

    This CSV was scored in a Colab environment whose SymPy could not parse as
    much LaTeX as a correctly-pinned one. Re-scoring the identical raw samples
    locally yields 141/300 transcription-correct instead of 118/300, because
    more mathematically-identical answers collapse into one cluster.

    The reported figures are the Colab ones -- that is the pre-registered run,
    and re-scoring after seeing results is exactly what the pre-registration
    exists to prevent. This test pins the size of the gap so nobody has to
    rediscover it, and fails if the relationship changes.
    """
    import ast

    import pilot.canonicalize as C
    import pilot.entropy as E
    import pilot.parsing as P

    rescored = []
    for _, r in run.iterrows():
        raw = ast.literal_eval(r["all_transcription_samples_raw"])
        labels = [C.canonical_answer_label(P.parse_transcription(t)) for t in raw]
        majority, _ = E.majority_cluster(labels)
        rescored.append(majority == C.canonical_answer_label(r["pert_a"]))

    # Better parsing merges more answers, so accuracy rises above the reported
    # figure. The reported 118 is therefore conservative, never flattering.
    assert sum(rescored) == 141
    assert sum(rescored) > int(run["transcription_correct"].sum())
