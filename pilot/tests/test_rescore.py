"""Versioned scoring rules, and the extractor defects that motivated them.

The perception AUROC is computed against `transcription_correct`, which is an
exact match between two canonicalized strings. Asked how much of the "wrong"
pile is a misread page versus a pedantic comparison, this project had no
answer on record. Inspecting the real n=300 run found three separate things:

1. A genuine bug. canonicalize.structural_clean unwraps \\textcolor{}{} with
   a [^{}]* body, so every NESTED case survives into the label. That is
   85/300 ground truths, 59 of them has_error=1 -- FERMAT marks the injected
   error in red, so the defect concentrates on exactly the items the error
   label depends on.
2. A second genuine bug. extract_final_answer's last-line tier splits on "."
   as a sentence terminator, which also splits DECIMAL NUMBERS: "the area is
   75.46 cm." extracts as "46 cm".
3. Cosmetic mismatches. "2^3 = 8" vs "2^{3} = 8"; "0 = 9" vs "0 = 9,";
   "11130 cm" vs "11130 \\, cm"; "(x,z)" vs "(x, z)".
4. Scope mismatches. The model reports "= 75.46 cm^2" while the ground truth
   spells out "= \\pi r^2 = ... = 75.46 cm^2" -- same answer, scored wrong.

Together those move accuracy from 47.0% to 63.3%. The reason this is a
sensitivity analysis and not a correction to the headline: the AUROC survives
all four rules, so the signal belongs to the entropy, not to the comparison.
strict_v1 stays frozen because it produced every locked result and all of
reference/*.json.

Relaxing is not uniformly generous, which is why bug 2 had to be fixed rather
than noted: it truncated BOTH sides of item 101 to "5 square meters", and a
looser rule then scored that agreement as correct -- a false pass, not a
recovered read. A relaxation is only trustworthy once the extractor feeding
it is not silently mangling its inputs.
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
    assert table.loc["fixed_v2", "n_correct"] == 144
    assert table.loc["relaxed_v3", "n_correct"] == 166
    assert table.loc["final_term_v4", "n_correct"] == 190
    assert table.loc["strict_v1", "accuracy"] == pytest.approx(0.470, abs=0.005)
    assert table.loc["final_term_v4", "accuracy"] == pytest.approx(0.633, abs=0.005)

    assert table.loc["strict_v1", "auroc"] == pytest.approx(0.850, abs=0.01)
    assert table.loc["fixed_v2", "auroc"] == pytest.approx(0.839, abs=0.01)
    assert table.loc["relaxed_v3", "auroc"] == pytest.approx(0.798, abs=0.01)
    assert table.loc["final_term_v4", "auroc"] == pytest.approx(0.819, abs=0.01)

    assert table["excludes_chance"].all()
    assert (table["ci_low"] > 0.70).all()

    # Relaxing also removes spurious max-entropy items: 5 samples that agree
    # on the value but differ in formatting stop looking like disagreement.
    assert table.loc["strict_v1", "n_at_max_entropy"] == 50
    assert table.loc["final_term_v4", "n_at_max_entropy"] == 24


def test_accuracy_is_monotone_except_where_the_bug_fix_moves_items_both_ways(run):
    """fixed_v2 is a correction, not a loosening, so it is the one step that
    could legitimately lower accuracy. It does not here (141 -> 146), but the
    test states which steps are guaranteed monotone and which is not."""
    table = rescore.scoring_sensitivity(run, n_boot=200).set_index("rule")
    assert table.loc["relaxed_v3", "n_correct"] >= table.loc["fixed_v2", "n_correct"]
    assert table.loc["final_term_v4", "n_correct"] >= table.loc["relaxed_v3", "n_correct"]


def test_final_term_v4s_gain_is_partly_short_label_coincidence(run):
    """The caveat that keeps 63.3% from being quotable as an accuracy. v4
    reduces an answer to the term after the last '=', and some of those are a
    single character -- a wrong answer can land on "1" or "5" by coincidence.
    9 of the 26 items v4 newly scores correct rest on a <=2-character match,
    so the defensible statement is a RANGE (55-63%), not v4's point value."""
    v3 = rescore.rescore_run(run, "relaxed_v3")
    v4 = rescore.rescore_run(run, "final_term_v4")
    gained = v4[v4.transcription_correct & ~v3.transcription_correct]

    assert len(gained) == 26
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
