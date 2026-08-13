"""Locks the cross-audit scorer-reliability diagnostics (2026-08-12).

Three human passes exist, and the temptation is to average their agreement
into one number. This file exists to make that impossible to do by accident:

  * two of the three sets agree with `strict_v1` **by construction**, and the
    tests assert those agreements as TAUTOLOGIES rather than results;
  * the sets overlap by 78 items, so no count may be summed across them;
  * four items carry opposite determinate labels between passes, and the
    sheet that records them must keep BOTH notes or it cannot be adjudicated.

Requires the Qwen n=300 CSV for the span views; the label-only assertions run
without it.
"""

from pathlib import Path

import pandas as pd
import pytest

import pilot.audit_diagnostics as AD
import pilot.rescore as rescore
import pilot.strict_v2 as S

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv"


@pytest.fixture(scope="module")
def audits():
    return AD.load_audit_sets(str(ROOT / "reference" / "audit"))


@pytest.fixture(scope="module")
def run():
    if not RUN.exists():
        pytest.skip("Qwen n=300 CSV not present (Drive-only)")
    return pd.read_csv(RUN)


@pytest.fixture(scope="module")
def diag(run, audits):
    v1 = rescore.rescore_run(run, "strict_v1")["transcription_correct"].astype(bool)
    v2 = S.rescore_v2(run)["correct_strict_v2_display_primary"].astype(bool)
    return AD.per_set_diagnostics(v1, v2, audits).set_index("audit_set")


def test_every_label_maps_to_a_correctness_verdict(audits):
    """A vocabulary drifting out of TRUTH_MAP would silently drop items from
    the determinate set and move every agreement figure."""
    assert set(audits) == {"genuinely_wrong_census", "v1_correct_spotcheck",
                           "v2_high_priority"}
    assert [len(d) for d in audits.values()] == [104, 100, 108]
    for d in audits.values():
        assert d["truth"].isin(["correct", "wrong", "indeterminate"]).all()


def test_extraction_issue_is_indeterminate_not_wrong(audits):
    """The distinction the whole audit rests on: an unearned verdict leaves
    the model's answer UNDECIDED. Mapping it to "wrong" would manufacture a
    false-fail count out of nothing."""
    assert AD.TRUTH_MAP["extraction_issue"] == "indeterminate"
    assert AD.TRUTH_MAP["needs_visual"] == "indeterminate"
    assert AD.TRUTH_MAP["notation_misread"] == "wrong"
    assert AD.TRUTH_MAP["true_correct"] == "correct"


def test_the_spotcheck_agrees_with_v1_by_construction_not_by_measurement(diag):
    """THE REASON THERE IS NO POOLED NUMBER.

    Set 2 is items strict_v1 called CORRECT and its determinate labels all
    mean "model correct", so its 100% agreement is definitional: the
    vocabulary in that sheet cannot express disagreement. Averaging it with
    the others manufactures a number that is mostly tautology.
    """
    name = "v1_correct_spotcheck"
    assert bool(diag.loc[name, "truth_one_directional"]) is True
    assert diag.loc[name, "v1_agreement"] == 1.0
    assert bool(diag.loc[name, "v1_tautological"]) is True
    assert diag.loc[name, "v1_detectable_error"] == "none"
    assert diag.loc[name, "n_determinate"] == 79


def test_the_census_stopped_being_one_directional_when_item_95_was_recoded(diag):
    """UPDATED 2026-08-13, and the flag flipping is the point.

    The census was DESIGNED one-directionally: it drew only items strict_v1
    called wrong, and the coding vocabulary in use at the time had no label
    meaning "the model was actually right". On 2026-08-13 the coder re-read
    item 95 and recoded it `needs_visual` -> `true_correct`, because the model
    span `(2x - y + z)(2x - y + z)` and the truth `(2x - y + z)^2` are
    equivalent. That is one genuine disagreement, so the set now technically
    CAN express one and `v1_tautological` is False.

    **The caveat it supported has not gone away.** One late recode does not
    make a one-directional draw into a two-directional audit: 103 of 104 items
    still carry labels that could only ever agree with the rule. The pooled
    agreement figure remains unquotable for the same reason as before, which
    is why that is asserted separately rather than inferred from this flag.
    """
    d = diag.loc["genuinely_wrong_census"]
    assert bool(d["truth_one_directional"]) is False
    assert bool(d["v1_tautological"]) is False
    assert d["n_determinate"] == 24
    assert d["v1_agreement"] == pytest.approx(23 / 24, abs=0.005)


def test_only_the_high_priority_set_can_disagree_in_both_directions(diag):
    d = diag.loc["v2_high_priority"]
    assert bool(d["truth_one_directional"]) is False
    assert bool(d["v1_tautological"]) is False and bool(d["v2_tautological"]) is False
    assert d["v1_detectable_error"] == "both"
    assert d["n_determinate"] == 68
    assert d["v1_agreement"] == pytest.approx(0.5147, abs=0.005)
    assert d["v2_agreement"] == pytest.approx(0.4853, abs=0.005)


def test_degeneracy_is_per_rule_not_per_set(diag):
    """Set 2 is tautological for v1 but NOT for v2: v2 was not used to select
    the set, so it can disagree there, and its 92.4% is a real one-directional
    measurement of v2's false-fail rate. One flag per set would have discarded
    that."""
    d = diag.loc["v1_correct_spotcheck"]
    assert bool(d["v1_tautological"]) is True
    assert bool(d["v2_tautological"]) is False
    assert d["v2_detectable_error"] == "false_fail_only"
    assert d["v2_agreement"] == pytest.approx(0.924, abs=0.005)
    assert d["v2_false_fail"] == 6
    assert d["v2_false_pass"] == 0


def test_neither_rule_produces_a_false_pass_on_any_audited_set(diag):
    """Consistent across all three: where a human could decide, neither rule
    ever called correct something the human called wrong. Every disagreement
    is a false FAIL."""
    for name in diag.index:
        assert diag.loc[name, "v1_false_pass"] == 0
        assert diag.loc[name, "v2_false_pass"] == 0


def test_the_sets_overlap_so_counts_cannot_be_summed(audits):
    rep = AD.overlap_report(audits)
    assert rep["n_union"] == 234
    assert rep["n_sum_of_parts"] == 312
    assert rep["n_double_counted"] == 78
    assert rep["pairwise_overlap"]["genuinely_wrong_census|v2_high_priority"] == 47
    assert rep["pairwise_overlap"]["v1_correct_spotcheck|v2_high_priority"] == 31
    assert rep["pairwise_overlap"]["genuinely_wrong_census|v1_correct_spotcheck"] == 0


def test_the_four_hard_contradictions(audits):
    """The first intra-rater reliability measurement in this project. Items
    coded `notation_misread` (model wrong) in the census and `true_correct` in
    the high-priority pass -- both determinate, opposite directions."""
    rep = AD.overlap_report(audits)
    hard = {c["item"] for c in rep["hard_contradictions"]}
    assert hard == {108, 149, 222, 230}
    for c in rep["hard_contradictions"]:
        assert c["label_a"] == "notation_misread"
        assert c["label_b"] == "true_correct"
    assert len(rep["soft_differences"]) == 27


def test_the_contradiction_sheet_keeps_both_notes(run, audits):
    """Its only job. A row without both readings cannot be adjudicated later,
    which is the entire reason the sheet exists rather than a list of ids."""
    sheet = AD.contradiction_sheet(run, audits)
    assert len(sheet) == 4
    assert sheet["item"].tolist() == [108, 149, 222, 230]
    for _, r in sheet.iterrows():
        assert isinstance(r["note_a"], str) and len(r["note_a"]) > 20
        assert isinstance(r["note_b"], str) and len(r["note_b"]) > 20
        assert r["label_a"] != r["label_b"]
        assert r["model_span"] and r["truth_span"]
    assert (sheet["adjudicated_label"] == "").all(), "nothing is resolved here"
    assert (sheet["adjudication_note"] == "").all()


def test_deduplicated_counts_are_over_the_union(audits):
    dd = AD.deduplicated_counts(audits)
    assert dd["n_union"] == 234
    assert sum(dd["label_counts"].values()) == 234
    assert sum(dd["truth_counts"].values()) == 234
    assert dd["n_hard_contradictions_masked"] == 4, (
        "precedence silently picks one side for these; the count must be "
        "surfaced so they are not forgotten")
    assert dd["precedence"][0] == "v2_high_priority"


def test_no_public_function_returns_a_pooled_accuracy(run, audits):
    """The module's whole point. Nothing here may hand back a single number
    that reads as corrected model accuracy."""
    v1 = pd.Series(True, index=run.index)
    v2 = pd.Series(True, index=run.index)
    for fn, args in ((AD.per_set_diagnostics, (v1, v2, audits)),
                     (AD.overlap_report, (audits,)),
                     (AD.deduplicated_counts, (audits,))):
        got = fn(*args)
        assert not isinstance(got, float)
        if isinstance(got, dict):
            assert not any("accuracy" in k for k in got)
        else:
            assert not any("accuracy" in c for c in got.columns)


def test_the_frozen_pipeline_is_untouched():
    assert rescore.RULES == ("strict_v1", "fixed_v2", "relaxed_v3",
                             "final_term_v4")
