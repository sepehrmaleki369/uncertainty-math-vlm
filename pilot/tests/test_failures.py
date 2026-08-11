"""The genuinely_wrong pre-sorter, and the two ways it over-triggered.

classify_scoring_outcome says why an item SCORED as it did; nothing subdivided
its largest failure bucket. This proposes a reason so the human pass is
confirm-or-correct rather than a hundred items read cold.

The counts moved 34% -> 23% -> 15% for extraction_issue as two false-positive
rules were tightened, and BOTH are pinned below, because a pre-sorter that
confidently mislabels is worse than none -- its counts get quoted.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import pilot.failures as failures

RESULTS = Path(__file__).resolve().parents[2] / "results"
QWEN = RESULTS / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv"
PIXTRAL = RESULTS / "pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv"


@pytest.fixture(scope="module")
def qwen():
    if not QWEN.exists():
        pytest.skip("Qwen n=300 CSV not present (download from Drive)")
    return pd.read_csv(QWEN)


@pytest.fixture(scope="module")
def presorted(qwen):
    return failures.presort(qwen)


# --- the two over-triggers, both found on real data ----------------------

def test_a_short_numeric_ground_truth_is_not_an_extraction_issue():
    """First over-trigger. Any label of <=2 characters counted as degenerate,
    so ground truths of '4', '25' and '91' were called scoring failures --
    but a short NUMBER is very plausibly the real answer. Only a stray single
    LETTER is the bug-3 signature (SymPy pulling one symbol out of prose)."""
    samples = ["**Answer:** the result is 7"]
    res = failures.classify_failure(samples, "the result is 4")
    assert res["label"] != "extraction_issue", res

    bare_symbol = failures.classify_failure(
        ["**Answer:** 42"], "Therefore the required number is $p$")
    assert bare_symbol["label"] == "extraction_issue"
    assert "bare symbol" in bare_symbol["why"]


def test_sympy_function_names_are_not_prose(qwen):
    """Second over-trigger, and the costlier one. SymPy prints eq, tan, log,
    integral as bare words, so the prose detector fired on its own output and
    labelled item 55 -- the CONFIRMED auto-correction case, a real model
    error -- as a scoring failure. Mislabelling a genuine model error as a
    parser artifact is the exact direction that would flatter the method."""
    row = qwen.iloc[55]
    res = failures.classify_failure(
        ast.literal_eval(row["all_transcription_samples_raw"]), row["pert_a"])
    assert res["label"] == "notation_misread", res
    assert "sympy:eq(tan" in res["gt_label"]

    # ...while still catching genuine SymPy-parsed prose. Note the synthetic
    # case that first came to mind -- "Hence, the required number of words is
    # 24" -- does NOT work as a control: strict_parse (bug 3's fix) already
    # stops that becoming a sympy: label at all, so the rule correctly never
    # sees it. Only residual cases survive, e.g. item 202's sympy:mathbb*r
    # from a \mathbb{R} that lost its backslash.
    from pilot import rescore
    assert rescore.answer_label(
        "Hence, the required number of words is 24",
        "final_term_v4").startswith("text:")

    row202 = qwen.iloc[202]
    res202 = failures.classify_failure(
        ast.literal_eval(row202["all_transcription_samples_raw"]), row202["pert_a"])
    assert res202["label"] == "extraction_issue", res202
    assert res202["gt_label"] == "sympy:mathbb*r"


# --- shape -----------------------------------------------------------------

def test_every_genuinely_wrong_item_gets_exactly_one_label(presorted, qwen):
    assert len(presorted) == 104
    assert set(presorted["label"]) <= set(failures.LABELS)
    assert presorted["label"].notna().all()
    assert presorted["item"].is_unique


def test_needs_visual_is_the_default_not_a_guess(presorted):
    """More than half the bucket genuinely needs the page. That is the honest
    outcome, not a shortfall -- a classifier that assigned a confident text
    label to everything would be inventing the counts it reports."""
    share = (presorted["label"] == "needs_visual").mean()
    assert 0.45 < share < 0.70
    assert (presorted["label"] == "needs_visual").sum() == 59


def test_the_counts_replicate_across_model_families():
    """Whatever the pre-sorter is measuring, it is not a Qwen quirk:
    extraction_issue 15.4% vs 14.6%, and zero hallucinations on both."""
    if not PIXTRAL.exists():
        pytest.skip("Pixtral CSV not present")
    q = failures.presort(pd.read_csv(QWEN))
    p = failures.presort(pd.read_csv(PIXTRAL))
    for frame, n_extract in ((q, 16), (p, 19)):
        assert (frame["label"] == "extraction_issue").sum() == n_extract
        assert (frame["label"] == "hallucination").sum() == 0
    assert abs((q["label"] == "extraction_issue").mean()
               - (p["label"] == "extraction_issue").mean()) < 0.03


def test_every_proposal_carries_its_evidence(presorted):
    """`why` is what makes the label auditable rather than an assertion."""
    assert presorted["why"].str.len().min() > 10
    assert presorted["why"].notna().all()


# --- the audit sample ------------------------------------------------------

def test_audit_sample_is_reproducible_and_spans_the_strata(presorted):
    """Selected by rule, so it cannot be tuned: the same call reproduces it.
    High- AND low-entropy wrong items both appear, because a confidently
    wrong page and a maximally uncertain one fail for different reasons."""
    a = failures.select_audit_sample(presorted, n_per_stratum=8)
    b = failures.select_audit_sample(presorted, n_per_stratum=8)
    assert a["item"].tolist() == b["item"].tolist()
    assert 25 <= len(a) <= 50
    assert a["item"].is_unique
    assert {"unanimous_wrong", "max_entropy_wrong", "low_entropy_wrong",
            "likely_scoring_artifact", "needs_visual"} <= set(a["stratum"])


def test_coding_sheet_keeps_the_proposal_separate_from_the_verdict(presorted):
    """If the human edited proposed_label in place there would be no record of
    how often the pre-sorter was wrong -- which is itself worth reporting."""
    sheet = failures.coding_sheet(
        failures.select_audit_sample(presorted, n_per_stratum=8))
    assert (sheet["final_label"] == "").all()
    assert sheet["proposed_label"].notna().all()
    for col in ("item", "stratum", "proposed_label", "proposed_because",
                "final_label", "notes", "entropy", "model_label", "gt_label"):
        assert col in sheet.columns, col
