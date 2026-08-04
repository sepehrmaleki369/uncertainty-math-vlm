"""Locks the figures from the 2026-08-04 confidence/K=10/prompt-variant run
against the saved CSVs, the same way test_reported_numbers.py locks the n=300
balanced run. Both CSVs are untracked (Drive-only); tests skip cleanly when
absent.

Closes the biggest remaining gap in the perception story from the n=300 run:
whether entropy beats an independent uncertainty signal, not just a cruder
version of itself (distinct-answer count). It does, decisively.
"""

from pathlib import Path

import pandas as pd
import pytest

from pilot.plotting import bootstrap_auroc_ci, bootstrap_auroc_difference_ci

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CONFIDENCE_CSV = RESULTS_DIR / "confidence_k10_qwen25-vl-3b-instruct_20260804T203938Z.csv"
VARIANTS_CSV = RESULTS_DIR / "grading_variants_qwen25-vl-3b-instruct_20260804T203938Z.csv"


@pytest.fixture(scope="module")
def confidence():
    if not CONFIDENCE_CSV.exists():
        pytest.skip(f"{CONFIDENCE_CSV} not present (download from Drive)")
    return pd.read_csv(CONFIDENCE_CSV)


@pytest.fixture(scope="module")
def variants():
    if not VARIANTS_CSV.exists():
        pytest.skip(f"{VARIANTS_CSV} not present (download from Drive)")
    return pd.read_csv(VARIANTS_CSV)


def test_confidence_run_joins_the_full_reference_sample(confidence):
    assert len(confidence) == 300


# --- A: entropy vs token confidence ---


def test_entropy_beats_token_confidence_decisively(confidence):
    """The headline of this run: entropy AUROC 0.835 vs -mean-logprob 0.537,
    paired +0.297 [+0.214, +0.380], resolved. Token confidence at best barely
    beats chance; entropy is nowhere close to being redundant with it."""
    r_ent = bootstrap_auroc_ci(confidence, "perception_entropy_k5",
                               "transcription_correct_k5", n_boot=10000, seed=0)
    r_conf = bootstrap_auroc_ci(confidence, "neg_mean_logprob",
                                "transcription_correct_k5", n_boot=10000, seed=0)
    assert r_ent["auroc"] == pytest.approx(0.835, abs=0.001)
    assert r_conf["auroc"] == pytest.approx(0.537, abs=0.001)
    assert r_ent["excludes_chance"] is True
    assert r_conf["excludes_chance"] is False  # confidence alone does not beat chance

    d = bootstrap_auroc_difference_ci(
        confidence, "perception_entropy_k5", "transcription_correct_k5",
        "neg_mean_logprob", "transcription_correct_k5", n_boot=10000, seed=0,
    )
    assert d["difference"] == pytest.approx(0.297, abs=0.001)
    assert d["difference_excludes_zero"] is True


def test_min_logprob_is_not_a_reliable_baseline(confidence):
    """-min-logprob's AUROC sits at or below chance (0.460), but do not read
    that as "token confidence is actively misleading" -- 81% of items cluster
    in a narrow band (neg_min_logprob > 20.5) that plausibly reflects batched-
    generation padding rather than genuine per-token content uncertainty (see
    this module's docstring discussion / chat record 2026-08-05). Split by that
    band, both halves sit inside a wide CI overlapping 0.5, which does not
    support a clean artifact story either -- the honest conclusion is only
    that -min-logprob is not usable, not why. -mean-logprob is the trustworthy
    half of this baseline and is asserted on its own above."""
    r = bootstrap_auroc_ci(confidence, "neg_min_logprob", "transcription_correct_k5",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.460, abs=0.001)
    assert r["excludes_chance"] is False


# --- C: does K=10 sharpen the signal? ---


def test_k10_modestly_sharpens_the_signal(confidence):
    """Scored against the K=10-recomputed correctness label (the fair
    comparison -- see this file's sibling discussion of why the K=5 label
    gives a different, less meaningful number). K=5 0.761, K=10 0.806, paired
    +0.045 [+0.005, +0.087], resolved -- confirms the leave-one-out subsample
    prediction from the n=300 analysis that K=5 was not yet saturated."""
    r5 = bootstrap_auroc_ci(confidence, "perception_entropy_k5",
                            "transcription_correct_k10", n_boot=10000, seed=0)
    r10 = bootstrap_auroc_ci(confidence, "perception_entropy_k10",
                             "transcription_correct_k10", n_boot=10000, seed=0)
    assert r5["auroc"] == pytest.approx(0.761, abs=0.001)
    assert r10["auroc"] == pytest.approx(0.806, abs=0.001)

    d = bootstrap_auroc_difference_ci(
        confidence, "perception_entropy_k10", "transcription_correct_k10",
        "perception_entropy_k5", "transcription_correct_k10", n_boot=10000, seed=0,
    )
    assert d["difference"] == pytest.approx(0.045, abs=0.001)
    assert d["difference_excludes_zero"] is True


def test_k10_has_finer_resolution_than_k5(confidence):
    """The structural reason K=10 helps: 7 reachable entropy values at K=5,
    34 at K=10 on this sample."""
    assert confidence["perception_entropy_k5"].round(6).nunique() == 7
    assert confidence["perception_entropy_k10"].round(6).nunique() == 34


def test_k5_and_k10_correctness_labels_disagree_on_some_items(confidence):
    """Even the 'ground truth' majority vote is not fully stable at K=5 --
    recomputing it with 10 samples flips some items. Worth knowing before
    treating transcription_correct as a fixed target."""
    flips = (confidence["transcription_correct_k5"]
             != confidence["transcription_correct_k10"]).sum()
    assert flips > 0


# --- B: grading prompt variants (screening) ---


def test_no_grading_variant_clears_the_screening_bar(variants):
    """Pre-registered screen: a variant only earns a full 300-item run if it
    clears the baseline by a wide margin. None of the three alternatives do,
    at n=100 per variant. This is itself a finding: it argues against the
    93%-says-error behaviour being a simple, fixable prompt artifact."""
    acc_by_variant = variants.groupby("variant")["grading_correct"].mean()
    baseline_acc = acc_by_variant["baseline"]
    for name in ("restate", "balanced", "confidence"):
        assert acc_by_variant[name] - baseline_acc < 0.10, (
            f"{name} cleared the screening bar ({acc_by_variant[name]:.3f} vs "
            f"baseline {baseline_acc:.3f}) -- re-examine whether it deserves a full run"
        )


def test_restate_flips_response_bias_without_fixing_accuracy(variants):
    """The interesting negative: 'restate the result first' collapses the
    says-error rate from 94% to 38% -- a huge behavioural shift -- while
    accuracy barely moves (0.510 -> 0.550). The model picks a different
    near-constant answer rather than actually discriminating, which weighs
    against 'yes-bias is a fixable prompt artifact' and toward a genuine
    capability ceiling."""
    says_error = variants.groupby("variant")["said_error"].mean()
    acc = variants.groupby("variant")["grading_correct"].mean()
    assert says_error["baseline"] == pytest.approx(0.94, abs=0.01)
    assert says_error["restate"] == pytest.approx(0.38, abs=0.01)
    assert says_error["baseline"] - says_error["restate"] > 0.5
    assert acc["restate"] - acc["baseline"] < 0.10
