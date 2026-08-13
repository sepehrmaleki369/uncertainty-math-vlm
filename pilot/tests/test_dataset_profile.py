"""Locks `pilot.dataset_profile`. Offline, synthetic fixtures, no model.

The risks here are not arithmetic. They are the three ways a distribution
table can look informative and mean nothing:

  * grouping by a variable that the model's own behaviour helped define, so
    the group's accuracy is partly the grouping;
  * printing an AUROC inside a group where every item carries the same label,
    which is the degeneracy behind this project's one retracted result;
  * reading `\\textcolor{red}` as injected-error metadata when it fires on
    most CLEAN pages too.
"""

import pandas as pd
import pytest

import pilot.dataset_profile as dp
import pilot.strict_v2 as strict_v2


def _profile(n=120, correct_frac=0.5, types=None, has_error=None):
    """A synthetic profile frame with the columns group_metrics consumes.

    Types CYCLE rather than sitting in contiguous blocks, so every type spans
    both correctness classes -- a blocked fixture makes each type single-class
    and every AUROC vacuously unavailable, which would let a broken guard pass.
    Human labels are a SUPERSET of determinate ones, as they are in the real
    audit: `extraction_issue` carries a raw label and no determinate truth.
    """
    types = types or ["numeric", "text_conclusion"]
    rows = []
    for i in range(n):
        correct = i < int(n * correct_frac)
        labelled = i % 5 == 0
        indeterminate = labelled and (i % 10 == 0)
        rows.append({
            "item_id": i,
            "has_error": (i % 2 == 0) if has_error is None else (i in has_error),
            "strict_v1_correct": correct,
            "strict_v2_correct": correct,
            "answer_type": types[i % len(types)],
            "question_type_if_available": "free_response",
            # Entropy tracks correctness, so a real AUROC is recoverable where
            # one is licensed.
            "entropy": 0.2 if correct else 1.4,
            "parse_failed": i % 10 == 0,
            "max_entropy": i % 7 == 0,
            "truth_span_len": 5 + (i % 40),
            "human_label_if_available": (
                ("extraction_issue" if indeterminate else "true_correct")
                if labelled else ""),
            "human_truth_if_available": (
                "" if not labelled else
                ("indeterminate" if indeterminate else "correct")),
        })
    return pd.DataFrame(rows).set_index("item_id", drop=False)


# --- answer types come from the truth side only ----------------------------

def test_answer_types_ignore_the_model_span():
    """THE GROUPING GUARD. strict_v2 ORs each risk flag across model and truth
    because a risk either side is a risk for the comparison. Grouping that way
    would make a group's accuracy partly a statement about which items the
    model answered badly."""
    truth = "x = 3"
    model_prose = "the required number is twenty four apples"
    gt_flags = strict_v2.answer_flags(truth, "sympy:eq(x,3)", False)
    model_flags = strict_v2.answer_flags(model_prose, "text:...", False)
    assert model_flags["text_conclusion"], "fixture must trip the model side"
    assert not gt_flags["text_conclusion"]
    assert dp.truth_answer_type(gt_flags, truth) == "single_expression", (
        "a prose MODEL span must not move the item into the prose group")


@pytest.mark.parametrize("span,label,expected", [
    ("42", "sympy:42", "numeric"),
    ("11130 cm", "text:11130 cm", "numeric"),
    ("x^2 + 1", "sympy:x**2+1", "single_expression"),
    ("", "", "unknown"),
])
def test_truth_answer_type_basic_shapes(span, label, expected):
    flags = strict_v2.answer_flags(span, label, False)
    assert dp.truth_answer_type(flags, span) == expected


def test_uninferable_types_are_declared_not_silently_dropped():
    """The brief asked for table/cell and proof categories. Neither is
    derivable from the stored fields, and saying so is the deliverable."""
    assert "table_cell_answer" in dp.UNINFERABLE_TYPES
    assert "proof_conclusion" in dp.UNINFERABLE_TYPES
    assert "table_cell_answer" not in dp.ANSWER_TYPE_ORDER


# --- the AUROC must be refused, not approximated ---------------------------

def test_no_auroc_inside_a_group_defined_by_the_label():
    """Splitting on a rule's verdict makes correctness constant within each
    group. That is the retracted stratified result's exact failure shape, so
    it is detected structurally rather than left to a reader."""
    g = dp.group_metrics(_profile(), "strict_v1_correct").set_index("group")
    for level in g.index:
        assert g.loc[level, "auroc"] is None
        assert g.loc[level, "auroc_status"] == "degenerate_split_defined_by_the_label"


def test_no_auroc_below_the_registered_minority_minimum():
    prof = _profile(n=100, correct_frac=0.9)     # minority = 10
    g = dp.group_metrics(prof, "question_type_if_available").set_index("group")
    assert g.loc["free_response", "auroc"] is None
    assert g.loc["free_response", "auroc_status"] == "underpowered_minority_10"


def test_an_auroc_is_reported_when_the_split_licenses_one():
    g = dp.group_metrics(_profile(n=200), "answer_type").set_index("group")
    row = g.loc["numeric"]
    assert row["auroc"] is not None and row["auroc_ci"]
    assert row["auroc_status"] in ("excludes_chance", "includes_chance")


def test_group_accuracies_and_shares_are_real():
    prof = _profile(n=100, correct_frac=0.3)
    g = dp.group_metrics(prof, "question_type_if_available").set_index("group")
    assert g.loc["free_response", "n"] == 100
    assert g.loc["free_response", "strict_v1_accuracy"] == pytest.approx(0.30)
    assert g.loc["free_response", "share"] == pytest.approx(1.0)


def test_extraction_issue_is_not_folded_into_wrong():
    """`extraction_issue` means the verdict was UNEARNED, which leaves the
    model's answer undecided. Counting it as wrong is the easiest available
    way to manufacture a result here, so it can never reach false_pass."""
    prof = _profile(n=100)
    g = dp.group_metrics(prof, "question_type_if_available").set_index("group")
    assert g.loc["free_response", "extraction_issue_rate"] > 0
    assert g.loc["free_response", "false_pass_v1"] == 0
    assert g.loc["free_response", "n_human_determinate"] < \
        g.loc["free_response", "n_human_labelled"]


# --- the red markup is not error metadata ----------------------------------

def test_red_spans_are_brace_balanced():
    """`[^{}]*` truncates at the first inner brace, which is extractor bug 1."""
    assert dp.red_spans(r"\textcolor{red}{\hat{b}}") == [r"\hat{b}"]
    assert dp.red_spans(r"a \textcolor{red}{1} b \textcolor{red}{2}") == ["1", "2"]
    assert dp.red_spans(None) == []


def test_red_markup_is_rejected_as_error_location_when_it_fires_on_clean_pages():
    """THE DECISIVE CHECK. If red marked the injected error it would be rare on
    clean pages. On the real sample it appears on 73% of them."""
    run = pd.DataFrame({
        "has_error": [True] * 4 + [False] * 4,
        "pert_a": [r"\textcolor{red}{7}"] * 4 + [r"\textcolor{red}{also note}"] * 3
        + ["plain"],
    })
    r = dp.red_markup_report(run)
    assert r["clean_items_with_red"] == 3
    assert r["clean_with_red_share"] == pytest.approx(0.75)
    assert r["usable_as_error_location"] is False


def test_an_error_item_without_red_is_reported():
    """Absence of red does not mean absence of an error, so the exceptions are
    listed rather than rounded away."""
    run = pd.DataFrame({"has_error": [True, True],
                        "pert_a": [r"\textcolor{red}{7}", "no markup"]})
    assert dp.red_markup_report(run)["error_items_without_red"] == [1]


# --- metadata availability is checked, not assumed -------------------------

def test_absent_metadata_is_reported_absent():
    run = pd.DataFrame({"has_error": [True], "pert_a": ["x"], "orig_q": ["q"],
                        "handwriting_style": [True], "image_quality": [True]})
    fields = {f["field"]: f for f in dp.metadata_availability(run)["fields"]}
    for name in ("injected error location", "injected error type/category",
                 "original clean answer", "page / image id",
                 "problem type / subject / topic"):
        assert fields[name]["available"] is False, name
    assert fields["has_error"]["available"] is True
    assert fields["image quality"]["available"] is True


def test_orig_q_is_not_mistaken_for_the_clean_answer():
    """`orig_q` is the QUESTION. Its name invites reading it as the original
    answer, which would turn 'no clean answer stored' into a false positive."""
    run = pd.DataFrame({"has_error": [True], "pert_a": ["x"], "orig_q": ["q"]})
    fields = {f["field"]: f for f in dp.metadata_availability(run)["fields"]}
    assert fields["original clean answer"]["available"] is False
    assert "orig_q" not in fields["original clean answer"]["columns_checked"]


# --- example selection ------------------------------------------------------

def test_example_selection_is_balanced_and_seeded():
    prof = _profile(n=200, correct_frac=0.5,
                    types=["numeric", "text_conclusion", "set_answer",
                           "single_expression"])
    a = dp.example_selection(prof)
    b = dp.example_selection(prof)
    assert list(a.index) == list(b.index), "selection must be reproducible"
    assert len(a) == 20
    assert int(a["has_error"].sum()) == 10
    assert int(a["strict_v1_correct"].sum()) == 10


def test_the_dominant_answer_type_can_reach_the_sheet():
    """A fixed ANSWER_TYPE_ORDER round robin put `numeric` and
    `single_expression` LAST, so with five slots per cell the two types
    covering 54% of the corpus could never appear. Shuffling made the
    exclusion unbiased instead of systematic."""
    common = "single_expression"
    prof = _profile(n=200, types=[common] * 18 + ["set_answer",
                                                 "text_conclusion"])
    picked = dp.example_selection(prof)
    assert common in set(picked["answer_type"]), (
        "the most common answer type must be reachable by the sampler")


def test_a_vacuous_match_is_the_note_that_wins():
    """A long span collapsing to a one-symbol label on BOTH sides is the
    mechanism behind most audited false passes, and it is invisible from the
    verdict. It must outrank the cheaper notes (parse failure, rule
    disagreement) or the tile says nothing useful about item 117."""
    row = pd.Series({
        "label_m": "sympy:p", "label_t": "sympy:p",
        "span_t_disp": r"P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{4}{9}.",
        "span_m_disp": "P(A|B) = ...", "answer_type": "set_answer",
        "strict_v1_correct": True, "strict_v2_correct": False,
        "max_entropy": False, "parse_failed": True, "truth_span_len": 60,
    })
    assert "vacuous" in dp._tile_note(row)


def test_a_short_truth_span_is_not_called_vacuous():
    """`sympy:4` against a genuinely short numeric truth is a real match, not
    a collapse. Flagging it would make the note meaningless on MCQ items."""
    row = pd.Series({
        "label_m": "sympy:4", "label_t": "sympy:4", "span_t_disp": "4",
        "span_m_disp": "4", "answer_type": "numeric",
        "strict_v1_correct": True, "strict_v2_correct": True,
        "max_entropy": False, "parse_failed": False, "truth_span_len": 1,
    })
    assert "vacuous" not in dp._tile_note(row)


# --- the MCQ review set -----------------------------------------------------
#
# The detector is a heuristic. These pin the two things that decide whether an
# audit built on it can produce an unbiased corrected count: that the unsound
# trigger is visible to the reviewer, and that the review set is TWO-SIDED.

def _v2row(**kw):
    base = {"is_mcq": True, "truth_span": "option a", "model_span": "option a",
            "truth_label": "text:option a", "model_label": "text:option a",
            "model_tier": "option", "truth_tier": "option",
            "correct_strict_v2_display_primary": True}
    base.update(kw)
    for f in strict_v2.RISK_FLAGS:
        base.setdefault(f, False)
    return base


def _mcq_inputs(questions, answers, is_mcq, spans=None):
    n = len(questions)
    run = pd.DataFrame({
        "orig_q": questions, "pert_a": answers,
        "has_error": [False] * n, "perception_entropy": [0.5] * n,
    })
    v2s = pd.DataFrame([
        _v2row(is_mcq=is_mcq[i],
               truth_span=(spans or answers)[i]) for i in range(n)])
    return run, pd.Series([True] * n), v2s


@pytest.mark.parametrize("blob,expected", [
    (r"If \(P(A) = \frac{7}{13}\) and \(P(B) = \frac{9}{13}\)", "(A) = \\frac{7}{13}\\) and \\(P(B)"),
    (r"a relation where $f(a) = f(b)$", "(a) = f(b)"),
])
def test_the_weak_trigger_returns_what_it_matched(blob, expected):
    """A `paren_letters_only` hit is only recognisable as a false positive once
    you can SEE that what matched was probability or function notation. Item
    117 is `P(A)...P(B)`, item 195 is `f(a) = f(b)`; neither is multiple
    choice."""
    t = dp.mcq_trigger(blob, "", "")
    assert t["presort"] == "paren_letters_only"
    assert expected in t["paren_letters_match"]
    assert t["flagged_by_detector"]


def test_the_option_word_outranks_the_weak_trigger():
    t = dp.mcq_trigger(r"Which is correct? \begin{enumerate}\item x\end{enumerate}",
                       "option d", "Answer: option d")
    assert t["presort"] == "option_word_and_choice_list"
    assert "truth_span" in t["option_word_in"]


def test_a_choice_list_alone_is_a_candidate_not_a_verdict():
    """LaTeX `enumerate` is equally how a multi-part question writes
    '(i) ... (ii) ...', so a list is a reason to LOOK, never a second
    automatic verdict."""
    t = dp.mcq_trigger(r"\begin{enumerate}\item Find x\item Find y\end{enumerate}",
                       "x = 3", "x = 3")
    assert t["presort"] == "choice_list_only"
    assert not t["flagged_by_detector"]


def test_the_review_set_is_two_sided():
    """THE BIAS GUARD. Reviewing only flagged items can find false positives
    and nothing else, so a corrected count would shrink by construction. That
    is the one-sided mistake the `genuinely_wrong` census already made once."""
    run, v1, v2s = _mcq_inputs(
        questions=["pick one: option a or b",
                   r"\begin{enumerate}\item Find x\end{enumerate}",
                   "plain question"],
        answers=["option a", "x = 3", "7"],
        is_mcq=[True, False, False])
    rev = dp.mcq_review_set(run, v1, v2s)
    assert set(rev["mcq_group"]) == {"flagged_mcq_like", "candidate_missed"}
    assert list(rev.loc[rev["mcq_group"] == "candidate_missed", "item_id"]) == [1]
    assert 2 not in set(rev["item_id"]), "a plain item must not enter the set"


def test_weak_trigger_is_flagged_even_when_a_choice_list_demotes_the_presort():
    """Items 170 and 237 are MATCHING exercises: they carry an enumerate, so
    `presort` becomes `choice_list_only`, yet the detector fired on the
    unsound regex. Keying suspicion off `presort` alone would hide them."""
    run, v1, v2s = _mcq_inputs(
        questions=[r"Match: $(a)$ pairs with $(b)$. \begin{enumerate}\item p\end{enumerate}"],
        answers=["(b)"], is_mcq=[True], spans=["(b)"])
    rev = dp.mcq_review_set(run, v1, v2s)
    assert rev.loc[0, "presort"] == "choice_list_only"
    assert bool(rev.loc[0, "flagged_by_weak_trigger_only"]), (
        "no 'option' word anywhere means the detector fired on the regex")
    assert "WEAK trigger matched" in dp.mcq_caption(rev.loc[0])


def test_manifest_paging_matches_the_render_order():
    """The notebook renders each group in the manifest's order, so page/cell
    must be derived from that same order or a reviewer follows the CSV to the
    wrong page."""
    n = 25
    run, v1, v2s = _mcq_inputs(["pick an option"] * n, ["option a"] * n,
                               [True] * n)
    rev = dp.mcq_review_set(run, v1, v2s, per_page=9)
    for pos, (_, r) in enumerate(rev.iterrows()):
        assert r["contact_sheet_file"] == f"mcq_flagged_p{pos // 9 + 1}.png"
        assert r["contact_sheet_cell"] == pos % 9 + 1


def test_the_human_columns_ship_empty():
    run, v1, v2s = _mcq_inputs(["pick an option"], ["option a"], [True])
    rev = dp.mcq_review_set(run, v1, v2s)
    assert rev.loc[0, "confirmed_mcq"] == ""
    assert rev.loc[0, "reviewer_note"] == ""


def test_the_caption_never_claims_confirmed_mcq():
    """Every sheet is labelled MCQ-*like*. A caption asserting MCQ would be
    the automatic verdict the human pass exists to replace."""
    run, v1, v2s = _mcq_inputs(["pick an option"], ["option a"], [True])
    cap = dp.mcq_caption(dp.mcq_review_set(run, v1, v2s).loc[0])
    assert "MCQ?" in cap
    assert "confirmed" not in cap.lower()


def test_the_sensitivity_refuses_to_call_the_number_quotable():
    """The count moves with a detector judgement no human has ruled on, so the
    magnitude is not quotable even though the direction is stable."""
    run, v1, v2s = _mcq_inputs(
        questions=["pick one: option a", r"$P(A) = P(B)$",
                   r"\begin{enumerate}\item Find x\end{enumerate}"],
        answers=["option a", "0.5", "x = 3"],
        is_mcq=[True, True, False])
    rev = dp.mcq_review_set(run, v1, v2s)
    prof = pd.DataFrame({
        "item_id": [0, 1, 2], "strict_v1_correct": [True, False, False],
    }).set_index("item_id", drop=False)
    s = dp.mcq_accuracy_sensitivity(prof, rev)
    assert s["n_flagged"] == 2 and s["n_weak_trigger"] == 1
    assert s["n_candidate_missed"] == 1
    assert s["quotable"] is False
    assert s["range"][0] <= s["as_reported"]["accuracy"] <= s["range"][1]


# --- scorer-vs-human confusion examples ------------------------------------
#
# The risk here is presenting a clean 2x2 to a reader who will take it at face
# value, when the audit's largest group does not belong in any of the four
# cells.

@pytest.mark.parametrize("v1_correct,label,expected", [
    (True,  "true_correct",      "TP"),
    (False, "true_correct",      "FN"),
    (False, "notation_misread",  "TN"),
    (False, "copied_wrong_line", "TN"),
    (False, "true_wrong",        "TN"),
    (True,  "notation_misread",  "FP"),
    # THE ASYMMETRY that decides the whole design.
    (True,  "extraction_issue",  "FP"),
    (False, "extraction_issue",  "INDETERMINATE"),
    (False, "needs_visual",      "INDETERMINATE"),
])
def test_confusion_category(v1_correct, label, expected):
    assert dp.confusion_category(v1_correct, label) == expected


def test_an_unearned_FAIL_is_never_a_true_negative():
    """`extraction_issue` says the verdict was not earned, NOT that the model
    was wrong. On a fail it leaves the model's answer undecided, so folding it
    into TN would inflate the scorer's apparent accuracy on the largest group
    in the audit (71 of 234 items)."""
    assert dp.confusion_category(False, "extraction_issue") != "TN"
    assert dp.confusion_category(False, "extraction_issue") != "FN"


def _conf_inputs(labels, v1_flags):
    n = len(labels)
    run = pd.DataFrame({"has_error": [i % 2 == 0 for i in range(n)],
                        "perception_entropy": [0.5] * n}, index=range(n))
    v2s = pd.DataFrame([{
        "correct_strict_v2_display_primary": v1_flags[i],
        "model_span": f"m{i}", "truth_span": f"t{i}",
        "model_label": f"text:m{i}", "truth_label": f"text:t{i}",
    } for i in range(n)], index=range(n))
    audit = pd.DataFrame({"final_label": labels,
                          "note": [f"note {i}" for i in range(n)]},
                         index=range(n))
    return run, pd.Series(v1_flags, index=range(n)), v2s, audit


def test_requested_items_are_forced_in_and_come_first():
    labels = ["extraction_issue"] * 10
    run, v1, v2s, audit = _conf_inputs(labels, [True] * 10)
    ex = dp.confusion_examples(run, v1, v2s, audit, n_per_category=4,
                               prefer={"FP": [7, 3]})
    got = ex.loc[ex["category"] == "FP", "item_id"].tolist()
    assert got[:2] == [7, 3], got


def test_selection_spreads_across_mechanisms():
    """A uniform draw on TN returns six `notation_misread` items and hides
    `copied_wrong_line` and `true_wrong` entirely. The rarest label goes
    first so a one-item mechanism is never crowded out."""
    labels = ["notation_misread"] * 20 + ["copied_wrong_line"] + ["true_wrong"] * 3
    run, v1, v2s, audit = _conf_inputs(labels, [False] * len(labels))
    ex = dp.confusion_examples(run, v1, v2s, audit, n_per_category=6)
    assert len(set(ex["human_label"])) == 3, sorted(set(ex["human_label"]))
    assert "copied_wrong_line" in set(ex["human_label"])


def test_the_caption_carries_every_requested_field():
    run, v1, v2s, audit = _conf_inputs(["true_correct"] * 3, [True] * 3)
    cap = dp.confusion_caption(dp.confusion_examples(
        run, v1, v2s, audit, n_per_category=3).iloc[0])
    for token in ("item ", "err=", "H=", "scorer v1=", "v2=", "human=",
                  "span M:", "span T:", "label M:", "label T:", "why:"):
        assert token in cap, f"{token!r} missing:\n{cap}"


def test_the_why_line_wraps_instead_of_truncating():
    """It was cut mid-word ('the match is vacu...') and only a rendered page
    showed it. That line is what a reader outside this project needs."""
    run, v1, v2s, audit = _conf_inputs(["true_correct"], [False])
    row = dp.confusion_examples(run, v1, v2s, audit, n_per_category=1).iloc[0]
    row = row.copy()
    row["note"] = ("a deliberately long explanation of why this item sits in "
                   "this cell, easily longer than one caption line")
    lines = dp.confusion_caption(row).splitlines()
    why = [x for x in lines[6:]]
    assert len(why) == 2, why
    assert not why[-1].endswith("..."), why


def test_the_readme_defines_the_groups_before_it_counts_them(tmp_path):
    run, v1, v2s, audit = _conf_inputs(
        ["true_correct", "extraction_issue", "notation_misread"],
        [True, False, False])
    ex = dp.confusion_examples(run, v1, v2s, audit, n_per_category=2)
    pop = pd.DataFrame({"item_id": [0, 1, 2],
                        "category": ["TP", "INDETERMINATE", "TN"]})
    text = dp.write_confusion_readme(str(tmp_path / "R.md"), ex, population=pop)
    assert "describe the SCORER, not the model" in text
    assert text.index("| **TP** |") < text.index("items audited")
    assert "not a population rate" in text
    for cat in dp.CONFUSION_ORDER:
        assert cat in text


def test_needs_visual_splits_by_verdict_so_titles_never_contradict_captions():
    """SHIPPED ONCE. `needs_visual` was routed to INDETERMINATE regardless of
    verdict, so item 5 -- scorer PASSED, coder could not read the page --
    rendered under a sheet titled "scorer said WRONG" while its own caption
    read v1=CORRECT. A reader without context sees a contradiction in the
    data, not a routing bug."""
    assert dp.confusion_category(False, "needs_visual") == "INDETERMINATE"
    assert dp.confusion_category(True, "needs_visual") == "NEEDS_VISUAL"


def test_a_passed_undecidable_item_is_not_called_a_false_pass():
    """`needs_visual` on a pass is weaker than `extraction_issue` on a pass:
    nobody found the verdict unearned, the coder could not tell. Calling it FP
    would assert more than the audit does."""
    assert dp.confusion_category(True, "needs_visual") != "FP"
    assert dp.confusion_category(True, "extraction_issue") == "FP"


def test_every_group_declares_the_verdict_its_title_states():
    assert set(dp.CONFUSION_VERDICT) == set(dp.CONFUSION_ORDER)
    assert dp.CONFUSION_VERDICT["INDETERMINATE"] is False
    assert dp.CONFUSION_VERDICT["NEEDS_VISUAL"] is True


def test_the_title_invariant_raises_and_names_the_offender():
    run, v1, v2s, audit = _conf_inputs(["true_correct"], [True])
    ex = dp.confusion_examples(run, v1, v2s, audit, n_per_category=1)
    assert dp.assert_confusion_groups_match_titles(ex)["checked"] == 1
    broken = ex.copy()
    broken.loc[0, "strict_v1_correct"] = False      # TP with v1=WRONG
    with pytest.raises(AssertionError, match="contradicts their sheet title"):
        dp.assert_confusion_groups_match_titles(broken)


def test_the_legend_defines_every_field_the_captions_use():
    for frag in ("scorer v1", "scorer v2", "span M", "span T", "label M",
                 "label T", "sympy:", "text:", "human ="):
        assert frag in dp.CONFUSION_LEGEND, frag


def test_the_legend_is_not_repeated_inside_every_tile():
    """Once per sheet. The fields are identical on every cell, so repeating
    them would triple the text before a reader reaches the item."""
    run, v1, v2s, audit = _conf_inputs(["true_correct"] * 2, [True] * 2)
    ex = dp.confusion_examples(run, v1, v2s, audit, n_per_category=2)
    for _, r in ex.iterrows():
        assert "LEGEND" not in dp.confusion_caption(r)


def test_contact_sheet_footer_is_optional_and_off_by_default():
    """Existing callers (notebooks 17, 23, 28) must render unchanged."""
    import inspect

    import pilot.plotting as P
    assert inspect.signature(P.contact_sheet).parameters["footer"].default == ""


def test_the_needs_visual_note_does_not_claim_the_model_was_correct():
    """The generic fallback asserted 'human confirms the model was correct',
    which contradicts a sheet titled 'correctness UNKNOWN'. Caught by
    rendering the single-tile sheet and reading it."""
    run, v1, v2s, audit = _conf_inputs(["needs_visual"], [True])
    ex = dp.confusion_examples(run, v1, v2s, audit, n_per_category=1)
    assert ex.loc[0, "category"] == "NEEDS_VISUAL"
    note = ex.loc[0, "note"]
    assert "could not decide" in note
    assert "confirms the model was correct" not in note
