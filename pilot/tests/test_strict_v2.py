"""Locks `strict_v2_display_primary` -- span-primary scoring, SymPy demoted.

The rule exists because the false-pass audit found the SPAN, not the
normalizer, is what goes wrong: 14 of 16 first-pass false passes were the
extractor choosing a wrong or partial span, only 2 were SymPy collapse.

Two properties matter more than the accuracy number and are asserted hardest:

  * **v2 does NOT fix false passes** where both sides extract the same wrong
    span. Span-matching agrees exactly as SymPy did. What v2 buys is FLAGS.
  * **The "any risk flag" queue is not selective** -- 71% of the run, catching
    what a random 71% would catch. Only the high-priority tier enriches.

Both are the kind of thing that would otherwise get quietly oversold.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import pilot.rescore as rescore
import pilot.strict_v2 as S

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv"
SPOT = [ROOT / "reference" / "audit" / "spotcheck_40_qwen_strict_v1_correct_20260811.csv",
        ROOT / "reference" / "audit" / "spotcheck_extra60_qwen_strict_v1_correct_20260812.csv"]


@pytest.fixture(scope="module")
def run():
    if not RUN.exists():
        pytest.skip("Qwen n=300 CSV not present (Drive-only)")
    return pd.read_csv(RUN)


@pytest.fixture(scope="module")
def scored(run):
    return S.rescore_v2(run)


@pytest.fixture(scope="module")
def false_passes():
    both = pd.concat([pd.read_csv(p) for p in SPOT])
    return set(both.loc[both["final_label"] == "extraction_issue",
                        "item"].astype(int))


def test_the_frozen_pipeline_is_untouched():
    """The whole freeze discipline rests on this: adding a rule must not add
    itself to RULES, or every locked snapshot silently changes meaning."""
    assert rescore.RULES == ("strict_v1", "fixed_v2", "relaxed_v3",
                             "final_term_v4")
    assert S.RULE_NAME not in rescore.RULES


def test_normalize_span_is_cosmetic_only():
    """It must NOT do algebra. Reordering or evaluating is exactly how two
    different answers land on one label, which is the bug being fixed."""
    assert S.normalize_span(r"x = 3, \, y = 1") == S.normalize_span("x = 3, y=1")
    assert S.normalize_span(r"\textcolor{red}{2 \text{ cm}}") == "2 cm"
    # different answers must stay different
    assert S.normalize_span("x = 3") != S.normalize_span("x = 4")
    assert S.normalize_span("a + b") != S.normalize_span("b + a")
    assert S.normalize_span(None) == ""


def test_sympy_never_decides_correctness(run):
    """The core inversion. `sympy_match` is carried as metadata and must not
    be what `correct_strict_v2_display_primary` is computed from."""
    r = S.score_item_v2(
        ast.literal_eval(run.loc[132, "all_transcription_samples_raw"]),
        run.loc[132, "pert_a"], run.loc[132, "orig_q"])
    assert r["correct_strict_v2_display_primary"] == (
        r["model_span_norm"] == r["truth_span_norm"] != "")
    assert "sympy_match" in r


def test_the_worked_examples_flag_as_the_audit_described(run):
    """The four cases the human pass reasoned about explicitly."""
    def sc(i):
        return S.score_item_v2(
            ast.literal_eval(run.loc[i, "all_transcription_samples_raw"]),
            run.loc[i, "pert_a"], run.loc[i, "orig_q"])

    # x = 3, y = 1 -> SymPy kept only eq(x,3): correct, but flagged.
    r = sc(132)
    assert r["correct_strict_v2_display_primary"] is True
    assert r["sympy_partial_parse_risk"] and r["system_answer"]

    # 12, 35, 37 -> SymPy kept only 12: correct, but flagged.
    r = sc(147)
    assert r["correct_strict_v2_display_primary"] is True
    assert r["multi_answer_collapse"] and r["multi_value_answer"]

    # span "zx" against a long polynomial: v2 still says correct (both sides
    # extracted the same wrong span) but must FLAG it.
    r = sc(180)
    assert r["correct_strict_v2_display_primary"] is True
    assert r["tiny_suspicious_non_mcq"]

    # one-letter B on a NON-mcq item must not read as an option answer.
    r = sc(289)
    assert r["is_mcq"] is False
    assert r["mcq_option"] is False
    assert r["tiny_suspicious_non_mcq"] is True


def test_a_one_letter_span_is_fine_on_a_real_mcq(run):
    """The distinction `tiny_valid_mcq` exists for: items 25 and 93 have
    `option d` as the key, which is a legitimate short answer."""
    for i in (25, 93):
        r = S.score_item_v2(
            ast.literal_eval(run.loc[i, "all_transcription_samples_raw"]),
            run.loc[i, "pert_a"], run.loc[i, "orig_q"])
        assert r["is_mcq"] is True
        assert r["mcq_option"] is True
        assert r["tiny_suspicious_non_mcq"] is False


def test_v2_does_not_rescue_accuracy(scored):
    """It is a re-basing of the comparison, not a fix. 47.0% -> 46.0%, with
    disagreement in BOTH directions. Anyone hoping v2 raises accuracy should
    read this test instead."""
    s = S.accuracy_summary(scored)
    assert s["strict_v1_correct"] == 141
    assert s["strict_v1_accuracy"] == pytest.approx(0.470, abs=0.001)
    assert s["strict_v2_correct"] == 138
    assert s["strict_v2_accuracy"] == pytest.approx(0.460, abs=0.001)
    assert s["n_disagree"] == 31
    assert s["v1_only_correct"] == 17 and s["v2_only_correct"] == 14


def test_v2_does_not_fix_the_known_false_passes(scored, false_passes):
    """THE HONEST LIMIT. Where both sides extract the same wrong span,
    span-matching agrees exactly as SymPy did, so v2 still scores it correct.
    It flips only a quarter of them. The value is the flags, not the verdict."""
    flipped = sum(1 for i in false_passes
                  if not scored.loc[i, "correct_strict_v2_display_primary"])
    assert flipped == 5, "v2 flips 5 of the 20; it does not solve the problem"
    assert flipped < len(false_passes) / 2


def test_the_any_flag_queue_is_not_selective(scored, false_passes):
    """Measured, and it is why `review_priority` exists. Selecting on ANY risk
    flag takes 71% of the run and catches 15 of 20 false passes -- which is
    what a random 71% would catch. Reporting that queue as a triage win would
    be wrong."""
    sheet = S.disagreement_and_risk_sheet(scored)
    assert len(sheet) == 212
    caught = len(false_passes & set(sheet["item"]))
    expected_by_chance = len(false_passes) * len(sheet) / len(scored)
    assert caught == 15
    assert abs(caught - expected_by_chance) < 2, "no meaningful enrichment"


def test_the_high_priority_tier_is_the_one_that_enriches(scored, false_passes):
    """36% of the run, the same 15 false passes: 2.1x over chance, and a 48%
    hit rate against a 20% base. This is the tier a human pass should read."""
    pri = S.review_priority(scored)
    high = set(scored.index[pri == "high"])
    assert len(high) == 108
    caught = len(false_passes & high)
    assert caught == 15
    enrichment = caught / (len(false_passes) * len(high) / len(scored))
    assert enrichment > 2.0


def test_set_and_system_flags_are_not_over_triggering(scored):
    """Both were fixed after firing on 45% and 31% of items -- `\\frac{a}{b}`
    counted as a set, and the chain `a=b=c` as a system. An over-triggering
    flag is worse than none because it drowns the queue."""
    assert int(scored["set_answer"].sum()) == 14
    assert int(scored["system_answer"].sum()) == 21
    assert S.answer_flags(r"\frac{a}{b}", "sympy:a/b", False)["set_answer"] is False
    assert S.answer_flags(r"\{1, 2, 3\}", "sympy:1", False)["set_answer"] is True
    assert S.answer_flags("a = b = c", "sympy:eq(a,b)", False)["system_answer"] is False
    assert S.answer_flags("x = 3, y = 1", "sympy:eq(x,3)", False)["system_answer"] is True


AUDIT_CSV = (ROOT / "reference" / "audit"
             / "strict_v2_high_priority_human_audit_20260812.csv")
AUDIT_HTML = AUDIT_CSV.with_suffix(".html")


@pytest.fixture(scope="module")
def audit_sheet():
    if not AUDIT_CSV.exists():
        pytest.skip("audit sheet not built")
    return pd.read_csv(AUDIT_CSV)


def test_the_audit_sheet_has_the_requested_shape(audit_sheet):
    required = ["item_id", "image_path", "priority", "span_m_disp",
                "span_t_disp", "label_m", "label_t", "strict_v1_correct",
                "strict_v2_correct", "rules_disagree", "final_label",
                "confidence", "note"]
    for c in required:
        assert c in audit_sheet.columns, c
    for f in S.RISK_FLAGS + ("sympy_partial_parse_risk",
                             "sympy_malformed_derivative",
                             "multi_answer_collapse"):
        assert f in audit_sheet.columns, f
    assert len(audit_sheet) == 108
    assert (audit_sheet["priority"] == "high").all()
    assert audit_sheet["item_id"].is_unique


def test_the_human_columns_are_now_fully_coded(audit_sheet):
    """This sheet SHIPPED empty and was coded on 2026-08-12 -- the assertion
    was flipped when the labels landed, not deleted, so the automatic columns
    and the human verdict stay visibly separate and the coding stays pinned.

    `true_wrong` is in the vocabulary because 66 of the 108 are items the
    scorer called WRONG, and the false-pass vocabulary the sheet originally
    shipped with had no way to say that verdict was earned.
    """
    for c in ("final_label", "confidence", "note"):
        assert audit_sheet[c].fillna("").ne("").all(), f"{c} has uncoded rows"
    assert set(audit_sheet["final_label"]) <= {
        "true_correct", "true_wrong", "extraction_issue", "needs_visual"}
    counts = audit_sheet["final_label"].value_counts().to_dict()
    assert counts["true_correct"] == 65
    assert counts["extraction_issue"] == 39
    assert counts["true_wrong"] == 3
    assert counts["needs_visual"] == 1
    # extraction_status answers a DIFFERENT question from final_label and must
    # not be collapsed into it.
    assert "extraction_status" in audit_sheet.columns
    assert set(audit_sheet["extraction_status"]) == {
        "extraction_ok", "extraction_issue", "needs_visual"}


def test_the_sort_is_exactly_as_specified(audit_sheet):
    """rules_disagree, then tiny_suspicious_non_mcq, then
    sympy_partial_parse_risk, then multi_answer_collapse, all descending,
    then item_id ascending."""
    key = audit_sheet[list(S.AUDIT_SORT)].astype(int).apply(
        lambda r: tuple(-v for v in r), axis=1)
    rows = list(zip(key, audit_sheet["item_id"]))
    assert rows == sorted(rows), "sheet is not in the requested order"
    assert bool(audit_sheet.iloc[0]["rules_disagree"]) is True


def test_image_paths_are_relative_and_per_item(audit_sheet):
    """The HTML sits next to an images/ folder; absolute paths would break the
    moment the folder is downloaded from Drive."""
    for _, r in audit_sheet.iterrows():
        assert r["image_path"] == f"images/item{int(r['item_id']):03d}.png"
        assert not r["image_path"].startswith("/")


def test_the_html_renders_every_row_with_blank_fields():
    if not AUDIT_HTML.exists():
        pytest.skip("audit html not built")
    html = AUDIT_HTML.read_text()
    sheet = pd.read_csv(AUDIT_CSV)
    assert html.count('class="card') == len(sheet)
    for f in ("final_label", "confidence", "note"):
        assert html.count(f'data-f="{f}"') == len(sheet)
    for i in sheet["item_id"].astype(int):
        assert f"images/item{i:03d}.png" in html
    # the instruction that outranks everything on the page
    assert "overrides every automatic label" in html
    assert 'value=""' in html, "the blank option must be selectable"
