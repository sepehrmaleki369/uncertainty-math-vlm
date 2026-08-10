"""Math-Verify as a triage tool, and the reason it is not the scorer.

Two things are locked here. First, that Math-Verify gets the isolated cases
right -- including REJECTING both of FERMAT's confirmed injected errors, which
is the property that makes it usable for triage at all. Second, the $-wrapping
gotcha, because it is silent, it inverts results, and it produced every false
positive in the first three runs of this analysis.

The decision it supports: the frozen string rule stays the reported metric.
Driving the score with Math-Verify raises accuracy (63.3% -> 71.0% on Qwen)
but roughly half the newly-accepted items are false positives concentrated on
has_error=1 rows, because FERMAT injects units, coefficients and notation --
exactly what an equivalence checker normalises away.

math-verify is an optional dependency; every test here skips without it.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import pilot.mathverify as mathverify
import pilot.rescore as rescore

pytestmark = pytest.mark.skipif(
    not mathverify.math_verify_available(),
    reason="math-verify not installed (optional audit dependency)")

RESULTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


# --- 1. the $-wrapping gotcha --------------------------------------------

def test_bare_input_silently_degrades_to_last_number_wins():
    """The defect mv_parse exists to prevent. On an unwrapped span
    Math-Verify's parse() falls back to plain-number extraction and throws the
    structure away, so `x = 5` and `z = 5` compare EQUAL. Wrapped, they do
    not. This inverted three runs of results before it was found."""
    from math_verify import parse, verify

    assert verify(parse("x = 5"), parse("z = 5")) is True      # WRONG, and silent
    assert verify(parse("$x = 5$"), parse("$z = 5$")) is False  # correct

    assert str(parse("x = 5")[0]) == "5"                       # variable discarded
    assert str(parse("$x = 5$")[0]) == "Eq(x, 5)"


def test_mv_parse_always_wraps():
    """The wrapper is the whole point of the function; a caller must not be
    able to reach the degraded path by passing a bare span."""
    assert mathverify.mv_equivalent(
        mathverify.mv_parse("x = 5"), mathverify.mv_parse("z = 5")) is False
    assert mathverify.mv_equivalent(
        mathverify.mv_parse("6^n + 5^n"), mathverify.mv_parse("6^n + 5n")) is False


@pytest.mark.parametrize("span", [None, "", "   "])
def test_mv_parse_of_nothing_is_none(span):
    assert mathverify.mv_parse(span) is None


def test_mv_equivalent_is_false_when_either_side_is_unparseable():
    """A failed parse must never read as agreement."""
    assert mathverify.mv_equivalent(None, mathverify.mv_parse("5")) is False
    assert mathverify.mv_equivalent(mathverify.mv_parse("5"), None) is False
    assert mathverify.mv_equivalent(None, None) is False


# --- 2. the sanity suite --------------------------------------------------

@pytest.mark.parametrize("label,a,b,expected", mathverify.SANITY_CASES)
def test_sanity_cases(label, a, b, expected):
    assert mathverify.mv_equivalent(
        mathverify.mv_parse(a), mathverify.mv_parse(b)) is expected, label


def test_the_injected_errors_are_rejected():
    """The property that makes Math-Verify usable for triage at all. FERMAT's
    errors on items 55 and 273 are a sign flip and a wrong numerator; merging
    either would destroy the has_error signal outright, so this is the test to
    re-run before trusting any future version of the dependency."""
    sign_flip = (r"\frac{\tan x+\tan y}{1-\tan x\tan y}",
                 r"\frac{\tan x+\tan y}{1+\tan x\tan y}")
    numerator = (r"\frac{3}{4}", r"\frac{2}{4}")
    for a, b in (sign_flip, numerator):
        assert mathverify.mv_equivalent(
            mathverify.mv_parse(a), mathverify.mv_parse(b)) is False


# --- 3. why it is not the scorer -----------------------------------------

def test_majority_semantics_not_any_of_k(run):
    """mv_score_item must mirror majority_cluster. An early version asked
    whether ANY of the K samples matched, which is a far weaker criterion and
    inflated the result from 80.3% to 86.7%."""
    row = run.iloc[0]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    scored = mathverify.mv_score_item(samples, row["pert_a"])
    assert 1 <= scored["majority_size"] <= len(samples)
    assert 1 <= scored["n_clusters"] <= len(samples)
    assert scored["majority_size"] >= len(samples) / scored["n_clusters"] - 1e-9


@pytest.mark.parametrize("idx,note", [
    (108, "y^2/(9/4) vs y^2/(11/4): different coefficients, matched on the chain"),
    (71, "the ground truth's answer is dy/dx = 1, not the pi mentioned in an aside"),
])
def test_math_verify_accepts_items_it_should_not(run, idx, note):
    """The evidence for keeping it out of the scoring path. These are real
    has_error-bearing rows where the injected difference is exactly what an
    equivalence checker erases. If a future version fixes these, revisit the
    decision -- but do not assume it."""
    row = run.iloc[idx]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    assert mathverify.mv_score_item(samples, row["pert_a"])["correct"] is True, note

    string_rule = rescore.score_item(samples, row["pert_a"], "final_term_v4")
    assert string_rule["transcription_correct"] is False


def test_disagreement_queue_is_a_review_list_not_a_correction(run):
    """The supported use. Both directions are reported, with the two spans, so
    a human can decide -- nothing is applied."""
    subset = run.head(40)
    string_correct = rescore.rescore_run(subset, "final_term_v4")[
        "transcription_correct"]
    queue = mathverify.disagreement_queue(subset, string_correct)

    assert set(queue.columns) >= {"item", "direction", "has_error",
                                  "mv_majority_span", "gt_span"}
    assert set(queue["direction"]) <= {"string_wrong_mv_right",
                                       "string_right_mv_wrong"}
    # Every queued row genuinely disagrees with the string rule.
    for _, r in queue.iterrows():
        assert bool(string_correct[r["item"]]) is (
            r["direction"] == "string_right_mv_wrong")
