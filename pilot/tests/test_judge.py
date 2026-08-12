"""Locks the LiveMath-Judge adapter. Stub backend only -- the model never runs here.

`jnanliu/LiveMath-Judge` is a math-EQUIVALENCE judge being repurposed as a
transcription-FIDELITY judge, which is the whole risk. These tests pin the
three things that decide whether the experiment is valid:

  * the prompt keeps the model's own template verbatim and only APPENDS, since
    deviating from the format a 3B fine-tune was trained on degrades it;
  * the verdict parser takes the LAST boxed answer, because the prompt itself
    contains a literal `\\boxed{yes}` and taking the first would score
    almost every item correct;
  * the gate on items 55 and 273 fails closed.

No GPU, no network, no model download.
"""

import pandas as pd
import pytest

import pilot.judge as J
import pilot.rescore as rescore
import pilot.strict_v2 as strict_v2


def stub(reply):
    """A backend that always answers the same thing."""
    return lambda prompt: reply


def scripted(replies):
    """A backend answering from a list, in call order."""
    seq = list(replies)

    def backend(prompt):
        return seq.pop(0) if seq else ""
    return backend


@pytest.fixture
def tiny_run():
    """Two rows standing in for items 55 and 273, at those exact labels."""
    rows = {
        55: {"all_transcription_samples_raw": repr(
                [r"**Answer:** \tan(x+y) = \frac{\tan x + \tan y}{1 - \tan x \tan y}"] * 5),
             "pert_a": r"\tan (x + y) = \frac{\tan x + \tan y}{1 + \tan x \tan y}",
             "orig_q": "Prove the tangent addition formula.", "has_error": True},
        273: {"all_transcription_samples_raw": repr([r"**Answer:** P(E) = \frac{3}{4}"] * 5),
              "pert_a": r"P(E) = \frac{2}{4}",
              "orig_q": "Find the probability.", "has_error": True},
    }
    return pd.DataFrame.from_dict(rows, orient="index")


# --- prompt construction --------------------------------------------------

def test_the_native_template_is_verbatim_and_fidelity_only_appends():
    native = J.build_prompt("Q", "GOLD", "ANS", fidelity=False)
    fid = J.build_prompt("Q", "GOLD", "ANS", fidelity=True)
    for marker in ("Please act as an expert in grading mathematics exam papers",
                   "3. You do not need to recalculate the problem answers",
                   "Original Question: Q", "Standard Answer: GOLD",
                   "Examinee's Answer: ANS", "Analysis:"):
        assert marker in native, marker
        assert marker in fid, marker
    assert "IMPORTANT" not in native
    assert len(fid) > len(native)


def test_the_fidelity_clause_overrides_the_equivalence_criterion():
    """Criterion 2 says equivalent formulas count as correct -- exactly what
    must NOT apply when the standard answer is deliberately wrong. The clause
    has to sit inside the criteria list and before the instruction, or the
    model reads it as an afterthought."""
    fid = J.build_prompt("Q", "GOLD", "ANS", fidelity=True)
    assert "4. IMPORTANT" in fid
    assert "overrides criterion 2" in fid
    assert "silently corrected" in fid
    assert fid.index("4. IMPORTANT") > fid.index("2. Some answers may be")
    assert fid.index("4. IMPORTANT") < fid.index("Please judge whether")


def test_the_prompt_carries_all_three_fields():
    p = J.build_prompt("WHATQ", "WHATGOLD", "WHATANS")
    assert "WHATQ" in p and "WHATGOLD" in p and "WHATANS" in p
    p2 = J.build_prompt(None, None, None)
    assert "Original Question:" in p2      # empties must not break formatting


# --- verdict parsing ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    (r"Analysis: they match. \boxed{yes}", "correct"),
    (r"Analysis: they differ. \boxed{no}", "incorrect"),
    (r"\boxed{ YES }", "correct"),
    (r"\boxed{No}", "incorrect"),
    ("", "unclear"),
    (None, "unclear"),
    ("Analysis: I am thinking about it", "unclear"),
])
def test_parse_verdict(text, expected):
    assert J.parse_verdict(text) == expected


def test_the_last_boxed_verdict_wins_not_the_first():
    """THE TRAP. The prompt itself says 'If they match, output \\boxed{yes},
    otherwise output \\boxed{no}'. Any echo of it puts a `yes` before the real
    answer, and a first-match parser would score nearly every item correct --
    a bug that looks like an excellent judge."""
    echo = (r"If they match, output \boxed{yes}, otherwise output \boxed{no}."
            "\nAnalysis: the examinee corrected the error."
            "\n" + r"\boxed{no}")
    assert J.parse_verdict(echo) == "incorrect"
    echo_yes = (r"output \boxed{yes} otherwise \boxed{no}" "\n" r"\boxed{yes}")
    assert J.parse_verdict(echo_yes) == "correct"


def test_a_bare_yes_no_is_only_read_from_the_final_line():
    assert J.parse_verdict("no boxed here\nyes") == "correct"
    # "no" appearing mid-prose must not be harvested as a verdict
    assert J.parse_verdict("there is no boxed answer at all\nAnalysis") == "unclear"


# --- per-item output shape ------------------------------------------------

def test_judge_item_emits_every_requested_column(tiny_run):
    import ast
    r = J.judge_item(stub(r"\boxed{no}"),
                     ast.literal_eval(tiny_run.loc[55, "all_transcription_samples_raw"]),
                     tiny_run.loc[55, "pert_a"], tiny_run.loc[55, "orig_q"])
    for c in ("question", "ground_truth_answer", "model_answer",
              "livemath_raw_output", "livemath_label", "parse_failed",
              "verdict", "fidelity_prompt"):
        assert c in r, c
    assert r["livemath_label"] == "no"
    assert r["parse_failed"] is False
    assert r["verdict"] == "incorrect"


def test_parse_failure_is_flagged_and_labelled_empty(tiny_run):
    """`unclear` means OUR parse failed. The model has no abstain verdict --
    its own prompt maps 'difficult to judge' onto `no` -- so this must never
    be reported as judge uncertainty."""
    import ast
    r = J.judge_item(stub("Analysis: hmm, hard to say"),
                     ast.literal_eval(tiny_run.loc[55, "all_transcription_samples_raw"]),
                     tiny_run.loc[55, "pert_a"])
    assert r["verdict"] == "unclear"
    assert r["parse_failed"] is True
    assert r["livemath_label"] == ""


# --- the gate -------------------------------------------------------------

def test_the_gate_items_are_the_two_confirmed_silent_corrections():
    assert J.GATE_ITEMS == (55, 273)


def test_gate_passes_only_when_both_items_are_rejected(tiny_run):
    g = J.run_gate(stub(r"\boxed{no}"), tiny_run)
    assert g["passed"] is True
    assert g["verdict_label"] == "gate_passed"
    assert set(g["verdicts"]) == set(J.GATE_ITEMS)
    assert all(v == "incorrect" for v in g["verdicts"].values())
    assert g["note"] == ""


def test_gate_fails_when_the_judge_accepts_a_corrected_error(tiny_run):
    """The failure this exists to catch: the judge says the model's
    mathematically-better answer matches, i.e. it recalculated."""
    g = J.run_gate(stub(r"\boxed{yes}"), tiny_run)
    assert g["passed"] is False
    assert g["verdict_label"] == "gated_judges_mathematics"
    assert "grading mathematics" in g["note"]


def test_gate_fails_closed_on_one_bad_item(tiny_run):
    """Both must be rejected. One out of two is a fail, not a pass."""
    g = J.run_gate(scripted([r"\boxed{no}", r"\boxed{yes}"]), tiny_run)
    assert g["passed"] is False


def test_gate_fails_closed_on_unparseable_output(tiny_run):
    """An unreadable judge is not a passing judge."""
    g = J.run_gate(stub("no verdict at all"), tiny_run)
    assert g["passed"] is False


def test_the_two_prompts_are_gated_independently(tiny_run):
    """The native prompt is recorded for comparison and must not be able to
    authorise the 300-item run."""
    fid = J.run_gate(stub(r"\boxed{no}"), tiny_run, fidelity=True)
    nat = J.run_gate(stub(r"\boxed{yes}"), tiny_run, fidelity=False)
    assert fid["fidelity_prompt"] is True and fid["passed"] is True
    assert nat["fidelity_prompt"] is False and nat["passed"] is False


# --- diagnostics ----------------------------------------------------------

def test_diagnostics_report_both_accuracy_conventions_and_per_class():
    judged = pd.DataFrame({
        "verdict": ["correct", "incorrect", "unclear", "correct"],
        "has_error": [True, True, False, False],
    }, index=[1, 2, 3, 4])
    v1 = pd.Series([True, True, False, False], index=[1, 2, 3, 4])
    v2 = pd.Series([True, False, False, True], index=[1, 2, 3, 4])
    human = pd.Series({1: "correct", 2: "wrong", 4: "wrong"})
    d = J.judge_diagnostics(judged, v1, v2, human=human)
    assert d["counts"] == {"correct": 2, "incorrect": 1, "unclear": 1}
    assert d["accuracy_excluding_unclear"] == pytest.approx(2 / 3)
    assert d["accuracy_unclear_as_wrong"] == pytest.approx(2 / 4)
    assert d["human_per_class"]["wrong"]["n"] == 2
    assert d["human_false_pass"] == 1          # item 4: human wrong, judge yes
    assert d["human_false_fail"] == 0
    assert "human_overall_agreement_DO_NOT_QUOTE_ALONE" in d, (
        "the overall figure must be named so it cannot be quoted bare")


# --- isolation ------------------------------------------------------------

def test_no_scorer_rule_is_modified():
    """This is a separate pilot. strict_v1 and strict_v2 must be untouched."""
    assert rescore.RULES == ("strict_v1", "fixed_v2", "relaxed_v3",
                             "final_term_v4")
    assert strict_v2.RULE_NAME == "strict_v2_display_primary"
    assert J.MODEL_ID == "jnanliu/LiveMath-Judge"


# --- exploratory diagnostic (notebook 25) ---------------------------------

def test_the_sample_is_exactly_20_20_with_both_probes_counted_in():
    """55 and 273 are both `has_error=1`. Adding them ON TOP of the stratum
    would make it 22 while the code claimed 20, so they are forced in and
    counted."""
    run = pd.DataFrame({
        "has_error": [i % 2 == 0 for i in range(300)],
    }, index=range(300))
    run.loc[55, "has_error"] = True
    run.loc[273, "has_error"] = True
    v2 = pd.DataFrame({f: [False] * 300 for f in
                       ("mcq_option", "tiny_valid_mcq", "derivative_equation",
                        "set_answer", "system_answer", "multi_value_answer",
                        "text_conclusion")}, index=range(300))
    for i in range(0, 300, 7):
        v2.loc[i, "set_answer"] = True
    for i in range(1, 300, 5):
        v2.loc[i, "mcq_option"] = True
    s = J.diagnostic_sample(run, v2, human=None)
    assert len(s) == 40
    assert int(s["has_error"].sum()) == 20
    assert int((~s["has_error"]).sum()) == 20
    assert set(J.GATE_ITEMS) <= set(s.index)
    assert s.loc[list(J.GATE_ITEMS), "forced"].all()


def test_the_sample_is_seeded_and_spreads_answer_types():
    run = pd.DataFrame({"has_error": [i % 2 == 0 for i in range(300)]},
                       index=range(300))
    run.loc[55, "has_error"] = True
    run.loc[273, "has_error"] = True
    v2 = pd.DataFrame({f: [False] * 300 for f in
                       ("mcq_option", "tiny_valid_mcq", "derivative_equation",
                        "set_answer", "system_answer", "multi_value_answer",
                        "text_conclusion")}, index=range(300))
    for n, f in enumerate(("mcq_option", "derivative_equation", "set_answer",
                           "system_answer", "multi_value_answer",
                           "text_conclusion")):
        for i in range(n, 300, 6):
            v2.loc[i, f] = True
    a = J.diagnostic_sample(run, v2, human=None)
    b = J.diagnostic_sample(run, v2, human=None)
    assert a.index.tolist() == b.index.tolist(), "must be reproducible"
    assert a["answer_type"].nunique() >= 5


def test_human_labelled_items_are_preferred():
    run = pd.DataFrame({"has_error": [True] * 100 + [False] * 200},
                       index=range(300))
    v2 = pd.DataFrame({f: [False] * 300 for f in
                       ("mcq_option", "tiny_valid_mcq", "derivative_equation",
                        "set_answer", "system_answer", "multi_value_answer",
                        "text_conclusion")}, index=range(300))
    human = pd.Series({i: "correct" for i in range(0, 300, 3)})
    s = J.diagnostic_sample(run, v2, human)
    picked = [i for i in s.index if i not in J.GATE_ITEMS]
    labelled = sum(1 for i in picked if i in human.index)
    assert labelled / len(picked) > 0.5, (
        "the sampler should lean toward items the judge can be scored against")


@pytest.mark.parametrize("text,expected", [
    ("Analysis: the answers match exactly. \\boxed{yes}", False),
    ("The examinee wrote a different denominator. \\boxed{no}", False),
    ("There are 3 favourable outcomes so the correct answer is 3/4. \\boxed{yes}", True),
    ("We compute 2/4 = 0.5, so it should be 3/4. \\boxed{yes}", True),
    ("The standard answer is wrong. \\boxed{yes}", True),
])
def test_looks_like_solving(text, expected):
    """Catches the failure the gate found on item 273: the judge recalculating
    instead of comparing. A review flag, never a verdict."""
    assert J.looks_like_solving(text) is expected


def test_answer_type_priority_is_stable():
    assert J.answer_type({"mcq_option": True, "set_answer": True}) == "mcq"
    assert J.answer_type({"set_answer": True, "system_answer": True}) == "set"
    assert J.answer_type({"text_conclusion": True}) == "text_conclusion"
    assert J.answer_type({}) == "numeric_or_algebra"


def test_the_diagnostic_summary_reports_no_accuracy(tmp_path):
    """The whole point of separating this from the scoring path. A gated
    judge's numbers must not be presentable as accuracy."""
    both = pd.DataFrame({
        "has_error": [True, False], "answer_type": ["mcq", "set"],
        "human_label": ["correct", "wrong"],
        "ground_truth_answer": ["a", "b"], "model_answer": ["a", "c"],
        "verdict_native": ["correct", "correct"],
        "verdict_fidelity": ["correct", "incorrect"],
        "raw_native": ["x", "y"], "raw_fidelity": ["x", "y"],
        "solving_native": [False, False], "solving_fidelity": [True, False],
    }, index=[55, 273])
    sample = pd.DataFrame({"has_error": [True, False],
                           "answer_type": ["mcq", "set"],
                           "human_label": ["correct", "wrong"],
                           "forced": [True, True]}, index=[55, 273])
    p = tmp_path / "s.md"
    text = J.diagnostic_summary_md(str(p), both, sample)
    low = text.lower()
    assert "not a scoring run" in low and "gate failed" in low
    import re
    assert not re.search(r"\baccuracy\b\s*[:=]", text, re.I)
    for section in ("## Sample composition", "## Native vs fidelity prompt",
                    "## has_error=1 behaviour", "## Judge appears to solve"):
        assert section in text, section


# --- Omni-Judge, second candidate on the SAME gate -------------------------

@pytest.mark.parametrize("field,expected", [
    ("TRUE", "correct"), ("FALSE", "incorrect"),
    ("  true  ", "correct"), ("false.", "incorrect"),
    (None, "unclear"), ("maybe", "unclear"), ("", "unclear"),
])
def test_parse_omni_judgement(field, expected):
    """Omni-Judge answers TRUE/FALSE where LiveMath-Judge answers yes/no.
    Anything unrecognised is `unclear` -- OUR parse failed -- so the two
    adapters stay comparable."""
    assert J.parse_omni_judgement(field) == expected


def test_run_gate_with_uses_the_identical_bar(tiny_run):
    """Comparing two judges against two different gates would prove nothing,
    so the generic runner reuses GATE_ITEMS and the same all-must-be-incorrect
    rule."""
    calls = []

    def rejecting(question, gold, answer):
        calls.append((question, gold, answer))
        return "incorrect", "justification: does not match"

    g = J.run_gate_with(rejecting, tiny_run, label="Omni-Judge")
    assert g["judge"] == "Omni-Judge"
    assert g["passed"] is True
    assert set(g["verdicts"]) == set(J.GATE_ITEMS)
    assert len(calls) == len(J.GATE_ITEMS)
    # it must be handed the MAJORITY answer, not raw samples
    assert all(isinstance(c[2], str) for c in calls)


def test_run_gate_with_fails_closed(tiny_run):
    accepting = lambda q, g, a: ("correct", "justification: equivalent")
    g = J.run_gate_with(accepting, tiny_run, label="Omni-Judge")
    assert g["passed"] is False
    assert g["verdict_label"] == "gated_unsafe_as_scorer"
    assert "do not run the 300" in g["note"]

    half = iter([("incorrect", "a"), ("correct", "b")])
    g2 = J.run_gate_with(lambda q, gg, a: next(half), tiny_run)
    assert g2["passed"] is False, "one of two is a fail"

    g3 = J.run_gate_with(lambda q, gg, a: ("unclear", "?"), tiny_run)
    assert g3["passed"] is False, "unparseable is not passing"


def test_the_generic_gate_records_the_solving_flag(tiny_run):
    """Omni-Judge emits a justification, unlike LiveMath-Judge's bare boxed
    verdict, so for the first time the solving heuristic has a transcript to
    read."""
    def solving(question, gold, answer):
        return "correct", "the correct answer is 3/4 so the student is right"

    g = J.run_gate_with(solving, tiny_run, label="Omni-Judge")
    assert g["passed"] is False
    assert all(g["solving"].values()), "justification must be scanned"


def test_the_livemath_gate_is_unchanged_by_the_refactor(tiny_run):
    """Adding a second candidate must not move the first one's bar."""
    g = J.run_gate(stub(r"\boxed{no}"), tiny_run)
    assert g["passed"] is True and g["verdict_label"] == "gate_passed"
    assert J.GATE_ITEMS == (55, 273)


def test_bpe_mangling_is_detected_not_mistaken_for_a_judge_failure():
    """The Omni-Judge run on 2026-08-12 returned all-None from
    `parse_response`, which the adapter mapped to `unclear` and the gate read
    as a failure. The cause was the custom tokenizer's DECODE returning
    space-separated BPE tokens -- `F AL ĠS ĠE` for `FALSE`. A verdict read off
    a mangled transcript is not a verdict."""
    mangled = "E quiv ale Ġn ce Jud gment F AL ĠS ĠE # Ġ# J us Ġt ification"
    assert J.looks_bpe_mangled(mangled) is True
    assert J.looks_bpe_mangled("## Equivalence Judgment FALSE") is False
    assert J.looks_bpe_mangled("") is False
    assert J.looks_bpe_mangled(None) is False


def test_demangle_recovers_the_judgement_but_not_perfectly():
    """A fallback for reading a transcript that would otherwise be lost. It
    cannot restore spaces the marker never recorded, so it is not a substitute
    for decoding correctly -- hence the notebook decodes with a plain
    tokenizer and only falls back to this."""
    mangled = ("F AL ĠS ĠE # Ġ# J us Ġt ification The Ġstudent 's Ġanswer "
               "Ġis Ġincorrect")
    out = J.demangle_bpe(mangled)
    assert "The student's answer is incorrect" in out
    assert "FAL S E" in out or "FALSE" in out
    assert J.demangle_bpe(None) == ""
