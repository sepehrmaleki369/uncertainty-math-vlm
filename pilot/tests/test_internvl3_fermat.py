"""Locks the 2026-08-07 InternVL3-8B n=300 FERMAT run against the saved CSV.

pilot/12_internvl3_fermat.ipynb ran clean on the second attempt (the first
crashed with `AttributeError: 'InternVLChatModel' object has no attribute
'all_tied_weights_keys'` -- a transformers v4-vs-v5 trust_remote_code
break, fixed by switching to the native OpenGVLab/InternVL3-8B-hf
integration; see git log for the fix commit).

The headline perception AUROC (0.915) is HIGHER than Qwen's confirmed
0.835 despite InternVL3's transcription accuracy (21.0%) being about half
Qwen's (~42%) -- exactly the "too good to be true" shape this project has
hit before, so it was checked against the same auroc_sensitivity cut every
other perception result here gets checked against, and it does NOT
survive: 0.915 -> 0.556 (chance) excluding max-entropy items, versus
Qwen's 0.835 -> 0.796 under the identical cut. This file locks that
diagnosis, not the raw 0.915 as a standalone finding -- do not cite 0.915
without the sensitivity result alongside it.

Root cause (confirmed against raw samples, not guessed): 202/300 items
(67%) pin at exact max entropy ln(5), and within that group accuracy is
0.99% (2/202) -- essentially a floor. The initial hypothesis (degenerate
repetition-loop garbage, drawn from 1-2 examples) turned out to explain
only ~30% of that group (61/200 have a 6x-repeated phrase); the dominant
mechanism (~70%, 139/200) is that InternVL3's 5 retries each land on a
different incomplete slice of a multi-step derivation rather than
converging on a stated final answer -- real model instability, not a
parsing bug (every relevant column recomputes from raw samples with 0
mismatches; see test_no_scoring_bug_recomputes_from_raw_samples below).

The reasoning has_error=1 stratum (0.628) introduces this project's first
"powered and resolved-above-chance but below the 0.70 confirmation
threshold" verdict -- distinct from both `confirmed` and
`inconclusive_underpowered` used everywhere else.

The CSV is untracked (Drive-only); tests skip cleanly when absent.
"""

import ast
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

import pilot.canonicalize
import pilot.parsing
from pilot.plotting import auroc_sensitivity, bootstrap_auroc_ci, stratified_auroc

RESULTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scaleup_n300_bal50_internvl3-8b-hf_20260807T205407Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_is_the_full_balanced_300_unquantized(run):
    assert len(run) == 300
    assert int(run["has_error"].astype(bool).sum()) == 150
    assert (run["quantized"] == False).all()  # noqa: E712
    assert run["model_id"].unique().tolist() == ["OpenGVLab/InternVL3-8B-hf"]


def test_k_is_5_for_every_item_both_arms(run):
    k_transcription = run["all_transcription_samples_raw"].apply(
        lambda r: len(ast.literal_eval(r))
    )
    k_grading = run["all_grading_samples_raw"].apply(lambda r: len(ast.literal_eval(r)))
    assert (k_transcription == 5).all()
    assert (k_grading == 5).all()


def test_no_scoring_bug_recomputes_from_raw_samples(run):
    """Independently recompute every derived column from the raw sample text
    using the unmodified pipeline functions (parse_transcription ->
    canonical_answer_label -> cluster_entropy, the same sequence notebook
    12's scoring cell uses). 0 mismatches confirms the surprising AUROC is
    real model behavior, not a pipeline bug -- the check that matters most
    given how this project has been burned by scoring bugs before."""
    import pilot.entropy

    def recompute_transcription(row):
        samples = ast.literal_eval(row["all_transcription_samples_raw"])
        parsed = [pilot.parsing.parse_transcription(s) for s in samples]
        labels = [pilot.canonicalize.canonical_answer_label(p) for p in parsed]
        entropy = pilot.entropy.cluster_entropy(labels)
        majority, _ = pilot.entropy.majority_cluster(labels)
        gt = pilot.canonicalize.canonical_answer_label(row["pert_a"])
        n_fail = sum(1 for p in parsed if p is None)
        return pd.Series({
            "entropy": entropy,
            "correct": majority == gt,
            "n_fail": n_fail,
            "n_distinct": len(set(labels)),
        })

    recomputed = run.apply(recompute_transcription, axis=1)
    assert (recomputed["entropy"].round(6) == run["perception_entropy"].round(6)).all()
    assert (recomputed["correct"] == run["transcription_correct"].astype(bool)).all()
    assert (recomputed["n_fail"] == run["n_transcription_parse_failures"]).all()

    def recompute_grading(row):
        samples = ast.literal_eval(row["all_grading_samples_raw"])
        parsed = [pilot.parsing.parse_grading(s) for s in samples]
        labels = [None if d is None else str(d) for d in parsed]
        entropy = pilot.entropy.cluster_entropy(labels)
        majority, _ = pilot.entropy.majority_cluster(labels)
        correct = majority in {"0", "1"} and int(majority) == int(row["has_error"])
        return pd.Series({"entropy": entropy, "correct": correct})

    recomputed_g = run.apply(recompute_grading, axis=1)
    assert (recomputed_g["entropy"].round(6) == run["reasoning_entropy"].round(6)).all()
    assert (recomputed_g["correct"] == run["grading_correct"].astype(bool)).all()


def test_max_entropy_items_genuinely_have_5_distinct_labels(run):
    """202/300 items pin at exactly ln(5). Confirms each one really does have
    5 distinct canonical transcription labels, not fewer with a tie -- an
    earlier ad hoc check flagged some as suspicious, which turned out to be
    a bug in the check script (double-applying extract_final_answer,
    skipping parse_transcription), not the pipeline. Fixed and reconfirmed
    here with the correct call sequence."""
    def n_distinct(raw):
        samples = ast.literal_eval(raw)
        parsed = [pilot.parsing.parse_transcription(s) for s in samples]
        labels = [pilot.canonicalize.canonical_answer_label(p) for p in parsed]
        return len(set(labels))

    at_max = run["perception_entropy"].apply(lambda e: math.isclose(e, math.log(5)))
    assert int(at_max.sum()) == 202
    distinct_counts = run.loc[at_max, "all_transcription_samples_raw"].apply(n_distinct)
    assert (distinct_counts == 5).all()


def test_parse_failures_cluster_together_not_as_spurious_unique_labels(run):
    """Multiple parse failures on the same item normalize to one shared
    sentinel and cluster together -- they don't each inflate entropy as
    distinct labels. n_transcription_parse_failures recomputes exactly."""
    def n_fail(raw):
        samples = ast.literal_eval(raw)
        return sum(1 for s in samples if pilot.parsing.parse_transcription(s) is None)

    recomputed = run["all_transcription_samples_raw"].apply(n_fail)
    assert (recomputed == run["n_transcription_parse_failures"]).all()


def test_perception_auroc_direction_is_correct(run):
    mean_correct = run.loc[run["transcription_correct"].astype(bool), "perception_entropy"].mean()
    mean_wrong = run.loc[~run["transcription_correct"].astype(bool), "perception_entropy"].mean()
    assert mean_wrong > mean_correct  # higher entropy -> more likely wrong


def test_perception_headline_auroc_does_not_survive_sensitivity_cut(run):
    """The number that must never be quoted alone: 0.915 raw, collapsing to
    0.556 (chance) once max-entropy items are excluded -- unlike Qwen's
    0.835 -> 0.796 under the identical cut, which survives."""
    r = bootstrap_auroc_ci(run, "perception_entropy", "transcription_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.915, abs=0.005)
    assert r["ci_low"] == pytest.approx(0.880, abs=0.01)

    sens = auroc_sensitivity(
        run, "perception_entropy", "transcription_correct",
        "n_transcription_parse_failures", k=5, n_boot=10000, seed=0,
    )
    assert sens["excl_max_entropy"]["auroc"] == pytest.approx(0.556, abs=0.01)
    assert sens["robust"] is False  # does NOT survive -- the opposite of Qwen's result


def test_bimodal_accuracy_split_drives_the_headline(run):
    """The mechanism: accuracy within the max-entropy group is a floor
    (0.99%, 2/202); within the rest it's 62.2% (61/98). This binary
    coherent-vs-incoherent split, not a graded relationship, is what the
    raw 0.915 is actually measuring."""
    at_max = run["perception_entropy"].apply(lambda e: math.isclose(e, math.log(5)))
    acc_max = run.loc[at_max, "transcription_correct"].astype(bool).mean()
    acc_non_max = run.loc[~at_max, "transcription_correct"].astype(bool).mean()
    assert acc_max == pytest.approx(0.0099, abs=0.003)
    assert acc_non_max == pytest.approx(0.622, abs=0.01)


def test_repetition_loops_are_a_minority_mechanism_not_the_dominant_one(run):
    """Correction to an earlier chat claim: reading 1-2 examples suggested
    degenerate repetition-loop garbage was the driver. A full-population
    check shows that's only ~30% of the max-entropy-wrong group; the
    dominant mechanism (~70%) is the model's 5 retries landing on different
    incomplete derivation fragments, not textual repetition specifically."""
    def has_repeat_loop(raw):
        samples = ast.literal_eval(raw)
        for s in samples:
            words = str(s).split()
            for n in (2, 3, 4):
                grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
                counts = Counter(grams)
                if counts and max(counts.values()) >= 6:
                    return True
        return False

    at_max = run["perception_entropy"].apply(lambda e: math.isclose(e, math.log(5)))
    wrong = ~run["transcription_correct"].astype(bool)
    group = run.loc[at_max & wrong, "all_transcription_samples_raw"]
    n_repeat = group.apply(has_repeat_loop).sum()
    assert len(group) == 200
    assert n_repeat == 61  # ~30%, not the dominant mechanism


def test_reasoning_entropy_distribution_is_not_degenerate(run):
    """Unlike perception, the reasoning-entropy distribution shows no
    single-value pinning problem (4 reachable values, no group over ~54%)
    -- so the has_error=1 stratum result below isn't an artifact of the
    same kind."""
    counts = run["reasoning_entropy"].round(3).value_counts()
    assert len(counts) == 4
    assert counts.max() / len(run) < 0.6


def test_reasoning_stratified_has_error_1_powered_but_below_confirm_threshold(run):
    """New verdict type for this project: has_error=1 clears the n=30 power
    minimum (n_wrong=45) and its CI excludes chance (0.548 > 0.5), but does
    NOT clear the pre-registered 0.70 confirmation threshold used for
    Qwen-3B/7B and LLaVA-NeXT. Distinct from `confirmed` and from
    `inconclusive_underpowered` -- call it resolved_but_below_threshold."""
    out = stratified_auroc(run, "reasoning_entropy", "grading_correct",
                           "has_error", n_boot=10000, seed=0)
    error_stratum = out["strata"][True]
    clean_stratum = out["strata"][False]

    assert error_stratum["n_error"] == 45
    assert error_stratum["auroc"] == pytest.approx(0.628, abs=0.005)
    assert error_stratum["ci_low"] > 0.5  # excludes chance
    assert error_stratum["ci_low"] < 0.70  # but does not clear the confirm bar
    minority = min(error_stratum["n_error"], error_stratum["n_correct"])
    assert minority >= 30  # powered by this project's registered standard

    assert clean_stratum["n_error"] == 97
    assert clean_stratum["auroc"] == pytest.approx(0.369, abs=0.005)
    assert clean_stratum["ci_high"] < 0.5  # inverted, as in every other model/family


def test_reasoning_pooled_near_chance_same_story_as_every_other_model(run):
    assert run["grading_correct"].mean() == pytest.approx(0.527, abs=0.005)
