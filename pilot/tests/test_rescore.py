"""Versioned scoring rules, and the extractor defects that motivated them.

The perception AUROC is computed against `transcription_correct`, which is an
exact match between two canonicalized strings. Asked how much of the "wrong"
pile is a misread page versus a pedantic comparison, this project had no
answer on record. Inspecting the real n=300 run found four separate things,
three of them genuine bugs:

1. A genuine bug. canonicalize.structural_clean unwraps \\textcolor{}{} with
   a [^{}]* body, so every NESTED case survives into the label. That is
   85/300 ground truths, 59 of them has_error=1 -- FERMAT marks the injected
   error in red, so the defect concentrates on exactly the items the error
   label depends on.
2. A second genuine bug. extract_final_answer's last-line tier splits on "."
   as a sentence terminator, which also splits DECIMAL NUMBERS: "the area is
   75.46 cm." extracts as "46 cm".
3. A third genuine bug. parse_latex does not fail loudly on input it only
   partly understands -- it parses a PREFIX and silently returns it. "Hence,
   the required number of words is 24" becomes h*(e*(n*(c*e))), the answer
   discarded; "40^\\circ 20' = \\frac{121\\pi}{540}" becomes 40**circ*20, the
   "=" and the whole answer discarded. 37/300 ground truths, 44/1500 samples.
4. Cosmetic mismatches. "2^3 = 8" vs "2^{3} = 8"; "0 = 9" vs "0 = 9,";
   "11130 cm" vs "11130 \\, cm"; "(x,z)" vs "(x, z)".
5. Scope mismatches. The model reports "= 75.46 cm^2" while the ground truth
   spells out "= \\pi r^2 = ... = 75.46 cm^2" -- same answer, scored wrong.

Together those move accuracy from 47.0% to 63.3%. The reason this is a
sensitivity analysis and not a correction to the headline: the AUROC survives
all four rules, so the signal belongs to the entropy, not to the comparison.
strict_v1 stays frozen because it produced every locked result and all of
reference/*.json.

Relaxing is not uniformly generous, which is why bugs 2 and 3 had to be fixed
rather than noted. Bug 2 truncated BOTH sides of item 101 to "5 square
meters", and a looser rule then scored that agreement as correct. Bug 3 is
worse in kind: it is COLLAPSING, mapping \\frac{1210}{540} and
\\frac{121\\pi}{540} -- different answers -- onto one label, which deflates
entropy as well as manufacturing matches. Fixing them LOWERS accuracy (144 ->
141 at fixed_v2), which is the honest direction. A relaxation is only
trustworthy once the extractor feeding it is not silently mangling its inputs.
"""

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

import pilot.canonicalize as canonicalize
import pilot.entropy as entropy
import pilot.parsing as parsing
import pilot.rescore as rescore

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


# --- 1. the frozen rule stays frozen --------------------------------------

def test_strict_v1_is_bit_identical_to_the_frozen_pipeline(run):
    """The invariant that lets the other three rules exist at all. If this
    ever fails, every reference/*.json snapshot is describing a rule the code
    no longer implements."""
    scored = rescore.rescore_run(run, "strict_v1")
    for i, row in run.iterrows():
        samples = ast.literal_eval(row["all_transcription_samples_raw"])
        labels = [canonicalize.canonical_answer_label(parsing.parse_transcription(s))
                  for s in samples]
        majority, _ = entropy.majority_cluster(labels)
        assert scored.loc[i, "perception_entropy"] == pytest.approx(
            entropy.cluster_entropy(labels), abs=1e-12)
        assert bool(scored.loc[i, "transcription_correct"]) == (
            majority == canonicalize.canonical_answer_label(row["pert_a"]))


def test_rescore_run_does_not_mutate_its_input(run):
    before = run.copy(deep=True)
    rescore.rescore_run(run, "final_term_v4")
    pd.testing.assert_frame_equal(run, before)


def test_unknown_rule_is_rejected_rather_than_silently_scored():
    with pytest.raises(ValueError, match="unknown rule"):
        rescore.answer_label("x", rule="lenient")


# --- 2. the nested \textcolor defect --------------------------------------

def test_structural_clean_still_has_the_nested_textcolor_defect():
    """Pins the defect deliberately left in place. \\textcolor{red}{x} is
    handled; \\textcolor{red}{\\hat{b}} is not, because the regex body is
    [^{}]*. Documented, not fixed, because structural_clean defines the
    frozen scoring rule."""
    assert "textcolor" not in canonicalize.structural_clean(r"\textcolor{red}{x}")
    assert "textcolor" in canonicalize.structural_clean(r"\textcolor{red}{\hat{b}}")


@pytest.mark.parametrize("text,macro,expected", [
    (r"\textcolor{red}{\hat{b}}", "textcolor", r"\hat{b}"),      # the defect case
    (r"\textcolor{red}{x}", "textcolor", "x"),                    # the easy case
    (r"\text{cm}^2", "text", "cm^2"),                             # one-arg form
    (r"\text{a}\text{b}", "text", "ab"),                          # repeated
    (r"\textcolor{red}{\frac{a}{b}}+1", "textcolor", r"\frac{a}{b}+1"),
    (r"1+\textcolor{blue}{2}=3", "textcolor", "1+2=3"),           # mid-expression
    (r"\textcolor", "textcolor", r"\textcolor"),                  # no args: verbatim
    (r"\textcolor{red}{x", "textcolor", r"\textcolor{red}{x"),    # unbalanced: verbatim
])
def test_unwrap_latex_macro(text, macro, expected):
    assert canonicalize.unwrap_latex_macro(text, macro) == expected


def test_unwrap_does_not_eat_a_macro_that_merely_starts_with_the_name():
    """macro="text" must not consume \\textcolor, or the colour argument
    ("red") becomes the answer -- a silent, plausible-looking wrong label."""
    assert canonicalize.unwrap_latex_macro(r"\textcolor{red}{z}", "text") == \
        r"\textcolor{red}{z}"
    assert canonicalize.unwrap_latex_macro(r"\textbf{q}", "text") == r"\textbf{q}"


def test_the_last_line_tier_splits_decimal_numbers():
    """Bug 2, pinned in both directions: the frozen default truncates at the
    decimal point, fix_decimal_split=True does not. A trailing sentence
    period must still split, or the fix would swallow the whole paragraph."""
    text = "The area is 75.46 cm."
    assert canonicalize.extract_final_answer(text) == "46 cm"
    assert canonicalize.extract_final_answer(text, fix_decimal_split=True) == \
        "The area is 75.46 cm"

    two_sentences = "First we simplify. The answer is 3.5"
    assert canonicalize.extract_final_answer(
        two_sentences, fix_decimal_split=True) == "The answer is 3.5"


def test_item_101_is_a_false_pass_that_the_decimal_fix_removes(run):
    """The reason bug 2 could not just be documented. FERMAT's injected error
    on this item is a fabricated unit -- a population reported in "square
    meters" -- so the ground truth ends "...is 23152.5 square meters.". The
    decimal split truncated that to "5 square meters" on the ground-truth
    side AND on two model samples, and relaxing then scored the shared
    mangling as agreement. Correct under no rule once the extractor is fixed."""
    row = run.iloc[101]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    assert canonicalize.extract_final_answer(row["pert_a"]) == "5 square meters}"
    for rule in rescore.RULES:
        assert rescore.score_item(samples, row["pert_a"], rule)[
            "transcription_correct"] is False, f"false pass under {rule}"


def test_the_defect_lands_on_has_error_items(run):
    """Why it is worth fixing rather than noting: FERMAT marks the injected
    error in red, so the ground truths the defect corrupts are mostly the
    ones whose label the reasoning arm depends on."""
    nested = run["pert_a"].astype(str).str.contains(
        r"\\textcolor\{[^{}]*\}\{[^{}]*\{", regex=True)
    assert int(nested.sum()) == 85
    assert int((nested & run["has_error"].astype(bool)).sum()) == 59


# --- 2c. parse_latex silently parsing only a prefix -----------------------

def test_sympy_silently_parses_a_prefix_and_discards_the_rest():
    """Bug 3 at source, both variants. The frozen path trusts these; the
    strict path does not. Note what is LOST, not merely mangled: the number
    24 in the first, the entire "= <answer>" in the second."""
    prose = "Hence, the required number of words is 24"
    assert canonicalize.canonicalize_math(prose) == "sympy:h*(e*(n*(c*e)))"
    assert canonicalize.canonicalize_math(prose, strict_parse=True).startswith("text:")

    truncated = r"40^\circ 20' = \frac{121\pi}{540} \text{ radians}."
    assert canonicalize.canonicalize_math(truncated) == "sympy:40**circ*20"
    assert canonicalize.canonicalize_math(
        truncated, strict_parse=True).startswith("text:")


def test_the_existing_symbol_guard_misses_four_letter_words():
    """Why a new check was needed rather than a tweak. canonicalize_math
    already rejects a parse with more than four DISTINCT single-character
    symbols, aimed at exactly this failure. "hence" yields h, e, n, c -- four,
    because the e repeats -- so it passes underneath the guard."""
    from sympy.parsing.latex import parse_latex
    expr = parse_latex(canonicalize.structural_clean("hence"))
    assert len([sym for sym in expr.free_symbols if len(str(sym)) == 1]) == 4


def test_the_collapse_is_what_makes_bug_3_worse_than_bug_1():
    """Two DIFFERENT answers reduced to one identical label. That does not
    just mis-score an item, it merges clusters -- understating entropy and
    creating a match out of nothing. Both must survive as distinct labels."""
    a = r"40^\circ 20' = \frac{1210}{540} \text{ radians}."
    b = r"40^\circ 20' = \frac{121\pi}{540} \text{ radians}."
    assert canonicalize.canonicalize_math(a) == canonicalize.canonicalize_math(b)
    assert canonicalize.canonicalize_math(a, strict_parse=True) != \
        canonicalize.canonicalize_math(b, strict_parse=True)


@pytest.mark.parametrize("text", [
    r"x = 1",                    # a real equation must still parse
    r"\frac{1}{256}j",
    r"\sin^{-1}(x)",             # backslash commands are not bare words
    r"xy",                       # 2-letter implicit multiplication is allowed
    r"4 \times 3 = 12",
])
def test_strict_parse_does_not_reject_legitimate_mathematics(text):
    """The check must not be so eager that it sends real expressions to the
    text tier -- that would trade a silent wrong answer for a silent strict
    one."""
    strict = canonicalize.canonicalize_math(text, strict_parse=True)
    assert strict == canonicalize.canonicalize_math(text)


@pytest.mark.parametrize("cleaned,expr,expected", [
    ("hence", None, False),                 # bare word >= 3 letters
    ("radians", None, False),
    ("xy", None, True),                     # 2 letters: implicit multiplication
    (r"\frac{1}{2}", None, True),           # backslash command, not a word
])
def test_sympy_parse_is_trustworthy_word_check(cleaned, expr, expected):
    import sympy
    assert canonicalize.sympy_parse_is_trustworthy(
        cleaned, expr if expr is not None else sympy.Symbol("q")) is expected


def test_sympy_parse_is_trustworthy_rejects_a_dropped_equals():
    """The second criterion, isolated: input asserting an equation whose parse
    is not a relation means the "=" and everything after it was discarded."""
    import sympy
    x = sympy.Symbol("x")
    assert canonicalize.sympy_parse_is_trustworthy("x = 1", sympy.Eq(x, 1)) is True
    assert canonicalize.sympy_parse_is_trustworthy("x = 1", x) is False
    assert canonicalize.sympy_parse_is_trustworthy("x", x) is True


def test_bug_3_affects_a_meaningful_share_of_the_real_run(run):
    """Prevalence, so the fix is not justified by two cherry-picked strings."""
    import re
    bare = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{3,}")

    def is_prose_sympy(label, final):
        if not label.startswith("sympy:") or final is None:
            return False
        return bool(bare.search(re.sub(r"\\[A-Za-z]+", "", final)))

    n_gt = n_samples = 0
    for _, row in run.iterrows():
        tr = rescore.trace_item(
            ast.literal_eval(row["all_transcription_samples_raw"]),
            row["pert_a"], "strict_v1")
        n_gt += is_prose_sympy(tr["ground_truth"]["label"],
                               tr["ground_truth"]["final_answer"])
        n_samples += sum(is_prose_sympy(s["label"], s["final_answer"])
                         for s in tr["samples"])
    assert n_gt == 37
    assert n_samples == 44


# --- 3. final-term extraction ---------------------------------------------

@pytest.mark.parametrize("label,expected", [
    (r"= \pi r^2 = \frac{22}{7} \times (4.9)^2 = 75.46 cm^2", "75.46 cm^2"),
    ("2^3 = 8", "8"),
    ("75.46", "75.46"),                          # no '=' at all: unchanged
    (r"\frac{a=b}{c}", r"\frac{a=b}{c}"),        # '=' inside braces is not top level
    ("f(x=1) = 3", "3"),                         # '=' inside parens is not top level
    ("x <= 5", "x <= 5"),                        # inequality never split
    ("x >= 5", "x >= 5"),
    ("a == b", "a == b"),
    ("x =", "x"),                                # trailing '=' falls back
])
def test_final_term(label, expected):
    assert rescore.final_term(label) == expected


def test_final_term_never_empties_a_non_empty_label():
    """A non-empty label reducing to "" would collapse every such item into
    one cluster, manufacturing agreement out of malformed answers. (An
    already-empty label stays empty -- that is the identity, not a collapse,
    and cannot arise anyway since parse failures map to the sentinel.)"""
    for label in ("=", "==", "= =", "= = =", "x =", "= x"):
        assert rescore.final_term(label) != ""


def test_parse_failures_are_never_relaxed_into_a_real_answer():
    """The sentinel must survive every rule, or failed samples would cluster
    with each other AND with genuine answers."""
    sentinel = entropy.normalize_string(entropy.PARSE_FAILURE_SENTINEL)
    for rule in rescore.RULES:
        assert rescore.answer_label(None, rule) == sentinel


# --- 4. the real items this was built to explain --------------------------

@pytest.mark.parametrize("idx,note", [
    (9,   "(x,z) vs (x, z) -- one space"),
    (34,  "0 = 9 vs 0 = 9, -- trailing comma"),
    (190, "2^{3} = 8 vs 2^3 = 8 -- braces on the exponent"),
    (280, "11130 cm vs 11130 \\, cm -- LaTeX thin space"),
    (73,  "22.50 vs Rs 22.50 -- currency symbol"),
])
def test_cosmetic_near_misses_are_wrong_under_strict_and_right_when_relaxed(
        run, idx, note):
    """Real items from the n=300 run where the model read the page correctly
    and the comparison said otherwise."""
    row = run.iloc[idx]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    assert rescore.score_item(samples, row["pert_a"], "strict_v1")[
        "transcription_correct"] is False, note
    assert rescore.score_item(samples, row["pert_a"], "relaxed_v3")[
        "transcription_correct"] is True, note


def test_answer_only_vs_full_chain_needs_the_final_term_rule(run):
    """Item 49: the model reports '= 75.46 cm^2', the ground truth spells out
    '= \\pi r^2 = ... = 75.46 cm^2'. Cosmetic normalization cannot fix this
    one -- it is a scope difference, which is what final_term_v4 is for."""
    row = run.iloc[49]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    assert rescore.score_item(samples, row["pert_a"], "relaxed_v3")[
        "transcription_correct"] is False
    assert rescore.score_item(samples, row["pert_a"], "final_term_v4")[
        "transcription_correct"] is True


# --- 5. extractor tier instability ----------------------------------------

@pytest.mark.parametrize("field,tier", [
    ("The answer is Option: C", "option"),
    (r"so \boxed{42}", "boxed"),
    (r"therefore \[ x = 3 \]", "display_math"),
    ("the pair has no solution", "last_line"),
    (None, "parse_fail"),
])
def test_extraction_tier(field, tier):
    assert rescore.extraction_tier(field) == tier


def test_the_extractor_changes_branch_across_samples_on_half_the_items(run):
    """The confound worth reporting on its own: some of what the perception
    arm scores as model uncertainty is the extractor picking a different line
    of an otherwise unchanged derivation. Mean entropy rises monotonically
    with the number of branches used."""
    scored = rescore.rescore_run(run, "strict_v1")
    n_tiers = [rescore.tier_instability(
        ast.literal_eval(r["all_transcription_samples_raw"]))["n_distinct_tiers"]
        for _, r in run.iterrows()]
    scored["n_tiers"] = n_tiers

    assert int(sum(n > 1 for n in n_tiers)) == 153

    means = scored.groupby("n_tiers")["perception_entropy"].mean()
    assert means[1] == pytest.approx(0.685, abs=0.01)
    assert means[2] == pytest.approx(1.031, abs=0.01)
    assert means[3] == pytest.approx(1.243, abs=0.01)
    assert means[1] < means[2] < means[3]


# --- 6. the claim the sensitivity table supports --------------------------

def test_the_auroc_survives_every_scoring_rule(run):
    """The point of the whole module. Accuracy moves a lot (47.0% -> 63.3%);
    the AUROC decays gracefully and never touches chance. A signal that
    existed only because of a strict string comparison would not do that."""
    table = rescore.scoring_sensitivity(run, n_boot=4000).set_index("rule")

    assert table.loc["strict_v1", "n_correct"] == 141
    assert table.loc["fixed_v2", "n_correct"] == 141
    assert table.loc["relaxed_v3", "n_correct"] == 162
    assert table.loc["final_term_v4", "n_correct"] == 190
    assert table.loc["strict_v1", "accuracy"] == pytest.approx(0.470, abs=0.005)
    assert table.loc["final_term_v4", "accuracy"] == pytest.approx(0.633, abs=0.005)

    assert table.loc["strict_v1", "auroc"] == pytest.approx(0.850, abs=0.01)
    assert table.loc["fixed_v2", "auroc"] == pytest.approx(0.838, abs=0.01)
    assert table.loc["relaxed_v3", "auroc"] == pytest.approx(0.802, abs=0.01)
    assert table.loc["final_term_v4", "auroc"] == pytest.approx(0.817, abs=0.01)

    assert table["excludes_chance"].all()
    assert (table["ci_low"] > 0.70).all()

    # Relaxing also removes spurious max-entropy items: 5 samples that agree
    # on the value but differ in formatting stop looking like disagreement.
    assert table.loc["strict_v1", "n_at_max_entropy"] == 50
    assert table.loc["final_term_v4", "n_at_max_entropy"] == 21


def test_accuracy_is_monotone_except_where_the_bug_fix_moves_items_both_ways(run):
    """fixed_v2 is a correction, not a loosening, so it is the one step that
    is NOT guaranteed monotone -- and it is not: 141 -> 141 net, with items
    moving in both directions. The nested-\\textcolor and decimal fixes each
    recover real matches, while the strict-parse fix removes matches that only
    existed because two different answers collapsed to one truncated label.
    Only the two genuine loosenings below are guaranteed to be monotone."""
    table = rescore.scoring_sensitivity(run, n_boot=200).set_index("rule")
    assert table.loc["relaxed_v3", "n_correct"] >= table.loc["fixed_v2", "n_correct"]
    assert table.loc["final_term_v4", "n_correct"] >= table.loc["relaxed_v3", "n_correct"]


def test_final_term_v4s_gain_is_partly_short_label_coincidence(run):
    """The caveat that keeps 63.3% from being quotable as an accuracy. v4
    reduces an answer to the term after the last '=', and some of those are a
    single character -- a wrong answer can land on "1" or "5" by coincidence.
    9 of the 30 items v4 newly scores correct rest on a <=2-character match,
    so the defensible statement is a RANGE (54-63%), not v4's point value."""
    v3 = rescore.rescore_run(run, "relaxed_v3")
    v4 = rescore.rescore_run(run, "final_term_v4")
    gained = v4[v4.transcription_correct & ~v3.transcription_correct]

    assert len(gained) == 30
    short = gained["gt_label"].astype(str).str.len() <= 2
    assert int(short.sum()) == 9
    # The majority of the gain is still substantive, which is why v4 is
    # reported at all rather than dropped.
    assert int((~short).sum()) > int(short.sum())


def test_trace_item_stages_line_up_with_the_score(run):
    """notebook 17 renders trace_item; if it could disagree with score_item
    the inspection view would be showing a different pipeline than the one
    being reported."""
    for idx in (9, 34, 49, 190):
        row = run.iloc[idx]
        samples = ast.literal_eval(row["all_transcription_samples_raw"])
        for rule in ("strict_v1", "final_term_v4"):
            tr = rescore.trace_item(samples, row["pert_a"], rule)
            sc = rescore.score_item(samples, row["pert_a"], rule)
            assert tr["correct"] == sc["transcription_correct"]
            assert tr["majority_label"] == sc["majority_label"]
            assert tr["perception_entropy"] == pytest.approx(
                sc["perception_entropy"], abs=1e-12)
            assert [s["label"] for s in tr["samples"]] == sc["labels"]
            assert math.isfinite(tr["perception_entropy"])


# --- 7. classifying the whole population ----------------------------------
#
# The four buckets notebook 17 originally used to draw examples from
# OVERLAPPED -- an item could be both a cosmetic mismatch and
# extractor-tier-unstable -- so they could show a case of X but could not say
# what the 300 items are made of. These categories are mutually exclusive and
# total, which is what makes them countable, plottable and comparable across
# models.

PIXTRAL_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv"
)

EXPECTED_CATEGORIES = {
    "qwen": {"correct_robust": 134, "bug_fix_recovered": 6,
             "cosmetic_mismatch": 22, "scope_mismatch": 27,
             "false_pass_removed": 6, "broken_by_relaxation": 1,
             "genuinely_wrong": 104},
    "pixtral": {"correct_robust": 118, "bug_fix_recovered": 9,
                "cosmetic_mismatch": 11, "scope_mismatch": 25,
                "false_pass_removed": 6, "broken_by_relaxation": 1,
                "genuinely_wrong": 130},
}


@pytest.fixture(scope="module")
def pixtral():
    if not PIXTRAL_CSV.exists():
        pytest.skip(f"{PIXTRAL_CSV} not present (download from Drive)")
    return pd.read_csv(PIXTRAL_CSV)


@pytest.fixture(scope="module")
def classified(run):
    return rescore.classify_scoring_outcome(run)


@pytest.fixture(scope="module")
def classified_pixtral(pixtral):
    return rescore.classify_scoring_outcome(pixtral)


def test_every_item_gets_exactly_one_known_category(classified,
                                                    classified_pixtral):
    """Total and mutually exclusive. If this fails the bar plot is lying about
    the population, because some items are in no bar or in two."""
    for name, c in (("qwen", classified), ("pixtral", classified_pixtral)):
        assert len(c) == 300, name
        assert c["category"].notna().all(), name
        assert set(c["category"]) <= set(rescore.CATEGORIES), name
        assert c["category"].value_counts().sum() == 300, name


@pytest.mark.parametrize("which", ["qwen", "pixtral"])
def test_category_counts(which, classified, classified_pixtral):
    c = classified if which == "qwen" else classified_pixtral
    assert c["category"].value_counts().to_dict() == EXPECTED_CATEGORIES[which]


@pytest.mark.parametrize("which", ["qwen", "pixtral"])
def test_categories_reconcile_with_the_sensitivity_table(
        which, run, pixtral, classified, classified_pixtral):
    """The taxonomy and the table beside it must describe the same run. Every
    item correct under the frozen rule is in exactly one of three categories,
    so they have to sum to that rule's n_correct."""
    df = run if which == "qwen" else pixtral
    c = classified if which == "qwen" else classified_pixtral
    n_v1 = int(rescore.scoring_sensitivity(df, rules=("strict_v1",),
                                           n_boot=50)["n_correct"].iloc[0])
    counts = c["category"].value_counts()
    assert (counts["correct_robust"] + counts["false_pass_removed"]
            + counts["broken_by_relaxation"]) == n_v1


def test_the_non_monotone_categories_are_not_empty(classified,
                                                   classified_pixtral):
    """Both exist because the rules are NOT monotone. A cumulative scheme
    would silently fold them into 'correct' and hide the two cases where
    scoring goes backwards -- which are the interesting ones."""
    for c in (classified, classified_pixtral):
        assert (c["category"] == "false_pass_removed").sum() == 6
        assert (c["category"] == "broken_by_relaxation").sum() == 1


def test_false_pass_items_are_the_bug_3_collapse(run, classified):
    """What a false pass looks like: bug 3 mapped two unrelated strings onto
    one label, so the frozen rule called them equal. Item 31's model text is
    about the rectangle's LENGTH and the truth about its PERIMETER, and both
    canonicalize to sympy:2*(c*m)."""
    idx = classified.index[classified["category"] == "false_pass_removed"].tolist()
    assert idx == [2, 31, 37, 117, 239, 294]

    row = run.iloc[31]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    strict = rescore.score_item(samples, row["pert_a"], "strict_v1")
    assert strict["transcription_correct"] is True
    assert strict["majority_label"] == strict["gt_label"] == "sympy:2*(c*m)"

    fixed = rescore.score_item(samples, row["pert_a"], "fixed_v2")
    assert fixed["transcription_correct"] is False
    assert fixed["majority_label"] != fixed["gt_label"]


def test_item_101_is_genuinely_wrong_not_a_false_pass(classified):
    """Guards a plausible misreading. Item 101 was the motivating false pass,
    but only under a relaxed rule applied BEFORE the decimal fix existed. With
    the fix in place it is wrong under every rule, so it belongs in
    genuinely_wrong -- the false_pass_removed category is for items the FROZEN
    rule scores correct, which item 101 never did."""
    assert classified.loc[101, "category"] == "genuinely_wrong"
    assert not classified.loc[101, "correct_strict_v1"]


def test_later_regression_is_separate_from_the_category(classified):
    """One label cannot honestly carry both 'first fixed at v3' and 'broken
    again at v4'. Two Qwen items are exactly that, so the fact is recorded in
    its own column and the category still names the rule that fixed them."""
    regressed = classified.index[classified["later_regression"]].tolist()
    assert regressed == [129, 206]
    assert classified.loc[129, "category"] == "cosmetic_mismatch"
    assert classified.loc[206, "category"] == "bug_fix_recovered"
    for i in regressed:
        assert not classified.loc[i, "correct_final_term_v4"]


def test_tier_instability_cross_cuts_the_taxonomy_so_it_is_a_flag(classified):
    """Why extractor-tier instability is a column and not a bar: it occurs in
    categories that end correct AND categories that end wrong, at materially
    different rates. A bucket defined by it would overlap every other one.

    Both splits are pinned because they are easy to confuse and differ: 42%
    vs 67% grouping by the LOOSEST rule's verdict, 41% vs 60% grouping by the
    frozen rule's. Quoting one figure against the other denominator is the
    mistake this test exists to prevent."""
    multi = classified["n_distinct_tiers"] > 1

    ends_correct = classified["category"].isin(
        ["correct_robust", "bug_fix_recovered", "cosmetic_mismatch",
         "scope_mismatch"])
    assert multi[ends_correct].mean() == pytest.approx(0.418, abs=0.01)
    assert multi[~ends_correct].mean() == pytest.approx(0.667, abs=0.01)

    strict = classified["correct_strict_v1"]
    assert multi[strict].mean() == pytest.approx(0.411, abs=0.01)
    assert multi[~strict].mean() == pytest.approx(0.598, abs=0.01)

    # The property that makes it a flag: present at both ends, never total.
    assert 0 < multi[ends_correct].mean() < multi[~ends_correct].mean() < 1


@pytest.mark.parametrize("which,expected", [
    ("qwen", {"textcolor": 6, "sympy_prefix": 5, "decimal": 1}),
    ("pixtral", {"textcolor": 8, "sympy_prefix": 6, "decimal": 1}),
])
def test_single_fix_attribution(which, expected, classified,
                                classified_pixtral):
    """Each flip is attributed by applying one fix alone. Only the two
    bug-related categories get an attribution -- a cosmetic mismatch has no
    bug to blame."""
    c = classified if which == "qwen" else classified_pixtral
    assert c["attributed_bug"].dropna().value_counts().to_dict() == expected
    attributed = c["attributed_bug"].notna()
    assert set(c.loc[attributed, "category"]) == {
        "bug_fix_recovered", "false_pass_removed"}


def test_scoring_category_summary_is_complete_and_ordered(classified):
    """Every category appears even at zero, in CATEGORIES order, so two models
    plot as comparable rows rather than as differently-shaped tables."""
    s = rescore.scoring_category_summary(classified, label="Qwen")
    assert list(s["category"]) == list(rescore.CATEGORIES)
    assert s["n"].sum() == 300
    assert s["share"].sum() == pytest.approx(1.0)
    assert (s["model"] == "Qwen").all()


def test_classification_does_not_mutate_its_input(run):
    before = run.copy(deep=True)
    rescore.classify_scoring_outcome(run)
    pd.testing.assert_frame_equal(run, before)


# --- 8. the reading view --------------------------------------------------

def test_first_difference_pinpoints_a_cosmetic_divergence():
    """The line the view was missing. Two labels that look identical on screen
    differ by one space, and the reader needs the column to see it."""
    d = rescore.first_difference(r"text:(x,z) \in r", r"text:(x, z) \in r")
    assert d["index"] == 8
    assert d["model"] == repr("z")
    assert d["truth"] == repr(" ")
    assert rescore.first_difference("same", "same") is None


def test_first_difference_handles_one_label_being_a_prefix():
    d = rescore.first_difference("24", "24 cm")
    assert d["index"] == 2
    assert "longer" in d["note"]


def test_format_trace_names_every_stage_and_both_sides(run):
    """The view must show what the parser took from EACH side and the verdict
    between them -- that is the whole question it exists to answer."""
    row = run.iloc[9]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    tr = rescore.trace_item(samples, row["pert_a"], "strict_v1")
    text = rescore.format_trace(tr, question=row["orig_q"],
                                category="cosmetic_mismatch")

    for expected in ("QUESTION", "GROUND TRUTH", "raw answer (pert_a)",
                     "parse_transcription", "extract_final_answer",
                     "COMPARISON LABEL", "MODEL sample 1 of 5",
                     "model majority label", "ground-truth label",
                     "verdict", "first difference at col",
                     "cosmetic_mismatch"):
        assert expected in text, f"{expected!r} missing from format_trace"

    assert "DIFFER" in text and "MATCH" not in text.split("verdict")[1][:20]


def test_format_trace_verdict_agrees_with_score_item(run):
    """The view and the numbers must not be able to disagree.

    Matched by regex rather than by reconstructing the column padding: an
    assertion that hard-codes the layout fails on a cosmetic formatting change
    while saying nothing about whether the verdict is right."""
    import re as _re
    verdict_re = _re.compile(r"^\s*verdict:\s+(MATCH|DIFFER)\s*$", _re.MULTILINE)

    for i in (9, 31, 49, 101, 190):
        row = run.iloc[i]
        samples = ast.literal_eval(row["all_transcription_samples_raw"])
        for rule in ("strict_v1", "final_term_v4"):
            tr = rescore.trace_item(samples, row["pert_a"], rule)
            correct = rescore.score_item(
                samples, row["pert_a"], rule)["transcription_correct"]
            found = verdict_re.findall(
                rescore.format_trace(tr, question=row["orig_q"]))
            assert found == ["MATCH" if correct else "DIFFER"], (
                f"item {i} under {rule}: view says {found}, score says {correct}")


def test_format_trace_marks_truncation_rather_than_cutting_silently(run):
    """A silently cut string reads as the model having stopped there, which is
    exactly the confusion this module exists to remove."""
    row = run.iloc[9]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    tr = rescore.trace_item(samples, row["pert_a"], "strict_v1")
    assert "…[+" in rescore.format_trace(tr, raw_chars=40)


# --- 7. what the unanimous-and-wrong audit actually found -----------------
#
# Read 2026-08-11 from the contact sheets. "Unanimous and wrong" was described
# as the population entropy cannot help with by construction. It is mostly the
# population where the models were RIGHT and the scoring rule was wrong, which
# is a different statement and a better one for the method.


def _unanimous_wrong(df):
    c = rescore.classify_scoring_outcome(df)
    gw = c.index[c["category"] == "genuinely_wrong"]
    return sorted(i for i in gw if c.loc[i, "entropy"] < 1e-9)


def test_the_unanimous_and_wrong_sets(run):
    """Small, stable, and the highest-value pages in the audit. Item 218 is
    unanimous and wrong on BOTH models -- 10/10 samples at zero entropy."""
    assert _unanimous_wrong(run) == [176, 208, 218]


@pytest.mark.parametrize("idx,note", [
    (218, "Option $\\text{A}$ -- _OPTION_RE needs a BARE letter after 'option'"),
    (208, "3x^2 = -4y vs x^2 = -4y/3, the same parabola"),
    (176, "eq(x,4) vs a bare 4"),
])
def test_every_qwen_unanimous_wrong_item_is_a_scoring_failure(run, idx, note):
    """None of the three is a misread. The model's answer is right in each and
    the comparison is what fails, so 'entropy cannot flag these' understates
    the case -- entropy was silent because there was nothing to flag."""
    row = run.iloc[idx]
    samples = ast.literal_eval(row["all_transcription_samples_raw"])
    scored = rescore.score_item(samples, row["pert_a"], "final_term_v4")
    assert scored["transcription_correct"] is False, note
    assert scored["perception_entropy"] == pytest.approx(0.0, abs=1e-9)


def _equation_parts(label):
    """Split a lowercased `eq(a,b)` label. normalize_string lowercases, so the
    label is NOT SymPy's Eq -- an isinstance check silently never fires and a
    scan built on one under-reports. This cost a wrong first count (1 vs 3)."""
    import re
    m = re.match(r"^eq\((.*)\)$", label[len("sympy:"):], re.S)
    if not m:
        return None
    body, depth = m.group(1), 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return body[:i], body[i + 1:]
    return None


def test_normalize_string_lowercases_eq_so_isinstance_checks_never_fire():
    """Pins the gotcha above, because it is invisible and produces a plausible
    wrong number rather than an error."""
    import sympy
    label = rescore.answer_label("x = 1", "fixed_v2")
    assert label == "sympy:eq(x, 1)"
    assert not isinstance(sympy.sympify(label[len("sympy:"):]), sympy.Eq)
    assert _equation_parts(label) is not None


def test_mathematically_equivalent_labels_are_scored_wrong(run):
    """A fifth scoring issue, distinct from the four extractor bugs: SymPy's
    canonical form does not normalise an equation scaled by a constant, so
    3x^2 = -4y and x^2 = -4y/3 compare unequal. Three items on Qwen, and none
    on Pixtral -- small, but it inflates genuinely_wrong."""
    import sympy

    def equivalent(a, b):
        pa, pb = _equation_parts(a), _equation_parts(b)
        if not (pa and pb):
            return False
        da = sympy.simplify(sympy.sympify(pa[0]) - sympy.sympify(pa[1]))
        db = sympy.simplify(sympy.sympify(pb[0]) - sympy.sympify(pb[1]))
        if da == 0 or db == 0:
            return da == db
        ratio = sympy.simplify(da / db)
        return bool(ratio.is_number and ratio != 0)

    scored = rescore.rescore_run(run, "final_term_v4")
    classified = rescore.classify_scoring_outcome(run)
    hits = [
        i for i in classified.index[classified["category"] == "genuinely_wrong"]
        if str(scored.loc[i, "majority_label"]).startswith("sympy:eq(")
        and str(scored.loc[i, "gt_label"]).startswith("sympy:eq(")
        and scored.loc[i, "majority_label"] != scored.loc[i, "gt_label"]
        and equivalent(str(scored.loc[i, "majority_label"]),
                       str(scored.loc[i, "gt_label"]))
    ]
    assert hits == [99, 151, 208]


def test_item_273_is_the_second_auto_correction_case(run):
    """The mechanism worth pre-registering. FERMAT's injected error is the red
    2 in P(E) = 2/4, while the page's own working says three outcomes are
    favourable. Pixtral transcribed 3/4 -- the correct value, not the one on
    the page. Item 55 is the same mechanism on both models.

    NOT a measured effect: the genuinely_wrong rate on has_error=1 vs clean
    items is +6.7% [-4.0%, +17.3%] on both families, spanning zero."""
    assert bool(run.loc[273, "has_error"]) is True
    assert r"\frac{\textcolor{red}{2}}{4}" in str(run.loc[273, "pert_a"])
    assert "favourable to E is 3" in str(run.loc[273, "pert_a"])

    assert bool(run.loc[55, "has_error"]) is True
    assert r"{1 + \tan x \tan y}" in str(run.loc[55, "pert_a"])


# --- 8. the \boxed{} experiment ------------------------------------------
#
# Registered before the run. The perception arm cannot resolve the extractor
# confound by rescoring -- every rule reads the same ambiguous text -- so
# notebook 19 removes it by making extraction deterministic instead.


def test_the_boxed_prompt_is_additive_and_still_says_ocr():
    """The confound the prompt is written to avoid is the model switching from
    transcribing the page to SOLVING it, which would change what is measured
    while still producing plausible numbers."""
    import pilot.prompts as prompts
    boxed = prompts.TRANSCRIPTION_USER_PROMPT_BOXED
    assert boxed.startswith(prompts.TRANSCRIPTION_USER_PROMPT)
    assert r"\boxed{" in boxed
    for phrase in ("Do not solve", "do not correct",
                   "do not add anything that is not written"):
        assert phrase in boxed, phrase
    # An answer with no single final result must not be forced into a box.
    assert "no single final result" in boxed


def test_boxed_compliance_counts_the_answer_field_only():
    """A \\boxed{} in the Question field would not make extraction
    deterministic, so compliance is measured on the parsed Answer."""
    assert rescore.boxed_compliance([r"**Answer:** so \boxed{42}"])["frac_boxed"] == 1.0
    assert rescore.boxed_compliance(["**Answer:** so 42"])["frac_boxed"] == 0.0
    assert rescore.boxed_compliance(
        [r"**Question:** \boxed{q}" + "\n**Answer:** 42"])["frac_boxed"] == 0.0

    mixed = rescore.boxed_compliance([r"**Answer:** \boxed{1}", "**Answer:** 2"])
    assert mixed["frac_boxed"] == 0.5 and mixed["all_boxed"] is False


@pytest.mark.parametrize("compliance,auroc,excl,multi,expected", [
    (0.50, 0.90, True,  0.00, "gated_low_compliance"),
    (0.95, 0.90, True,  0.40, "manipulation_failed"),
    (0.95, 0.80, True,  0.02, "signal_is_not_extractor"),
    (0.95, 0.55, False, 0.02, "signal_was_extractor"),
    (0.95, 0.72, True,  0.02, "inconclusive"),
    # boundaries, so the registered bars are the tested ones
    (0.80, 0.75, True,  0.10, "signal_is_not_extractor"),
    (0.79, 0.75, True,  0.10, "gated_low_compliance"),
])
def test_classify_boxed_result(compliance, auroc, excl, multi, expected):
    ci = {"auroc": auroc, "excludes_chance": excl}
    assert rescore.classify_boxed_result(compliance, ci, multi) == expected


def test_signal_was_extractor_is_reachable():
    """The outcome the experiment exists to be ABLE to report. A registered
    bar that cannot produce its own negative is not a bar."""
    ci = {"auroc": 0.52, "excludes_chance": False}
    assert rescore.classify_boxed_result(0.95, ci, 0.01) == "signal_was_extractor"


def test_the_signal_holds_where_extraction_was_already_deterministic(run):
    """The extractor confound, addressed WITHOUT the \\boxed{} run.

    Notebook 19 tried to make extraction deterministic by prompt. Both
    Qwen-3B (33.5% compliance) and Qwen-7B (2/5 greedy probes) refused the
    instruction, so that manipulation is unavailable on models this size.

    The conditional version is free: restrict to items where the extractor
    fired a SINGLE tier across all five samples, so tier-switching cannot
    contribute to the entropy by construction. If the confound were
    generating the signal, the AUROC there would fall. It does not -- 0.845
    against 0.835 on the full sample, and Pixtral behaves the same way.

    This is a subset analysis, not a manipulation. It cannot rule out
    WITHIN-tier variation (item 9 has all five samples in display_math and
    still disagrees), and the subsets differ in difficulty -- accuracy is
    46.9% here against 32.0% on the switching items. State it as evidence
    against the confound, not as its elimination.
    """
    from pilot.plotting import bootstrap_auroc_ci

    n_tiers = pd.Series(
        [rescore.tier_instability(
            ast.literal_eval(r["all_transcription_samples_raw"]))["n_distinct_tiers"]
         for _, r in run.iterrows()], index=run.index)

    stable = run[n_tiers == 1]
    assert len(stable) == 147

    full = bootstrap_auroc_ci(run, "perception_entropy",
                              "transcription_correct", n_boot=4000, seed=0)
    cond = bootstrap_auroc_ci(stable, "perception_entropy",
                              "transcription_correct", n_boot=4000, seed=0)

    assert full["auroc"] == pytest.approx(0.835, abs=0.01)
    assert cond["auroc"] == pytest.approx(0.845, abs=0.015)
    assert cond["excludes_chance"] is True
    # The claim: not weaker where extraction could not vary.
    assert cond["auroc"] >= full["auroc"] - 0.02
