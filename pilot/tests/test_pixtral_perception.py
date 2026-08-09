"""Locks the 2026-08-09 Pixtral-12B n=300 run: the perception replication.

This is the result the perception claim needed. Before it, AUROC 0.835 rested
on a single model family, and two attempts at a second had failed for
different reasons -- LLaVA-NeXT at a 3.0% transcription floor, InternVL3 with
a 0.915 that collapsed to 0.556 under the artifact control.

Pixtral-12B (Mistral, architecturally independent of Qwen) reaches 0.828
[0.782, 0.871] on the same 300 images, an interval that almost entirely
overlaps Qwen-3B's, and -- the part that matters -- it survives the control
InternVL3 failed. A high raw number is easy; a number that holds after the
max-entropy items are removed is the claim.

Model chosen on evidence rather than reputation: the FERMAT paper benchmarks
nine VLMs on this dataset and finds Pixtral-124B among the strongest at
reading the handwriting. Mistral is genuinely independent, unlike MiniCPM-V
and olmOCR, which are Qwen2-VL fine-tunes.

The CSV is untracked (Drive-only); tests skip cleanly when absent.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import pilot.canonicalize
import pilot.entropy
import pilot.parsing
from pilot.plotting import auroc_sensitivity, bootstrap_auroc_ci

RESULTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv"
)


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def test_run_shape_matches_every_other_reference_run(run):
    """Same balanced n=300 draw and same K as the Qwen run, in bf16 -- a
    4-bit fallback would confound the model comparison with a precision one."""
    assert len(run) == 300
    assert int(run["has_error"].astype(bool).sum()) == 150
    assert (run["quantized"] == False).all()  # noqa: E712
    assert run["model_id"].unique().tolist() == ["mistral-community/pixtral-12b"]
    assert (run["k_transcription"] == 5).all()


def test_no_scoring_bug_recomputes_from_raw_samples(run):
    def recompute(row):
        samples = ast.literal_eval(row["all_transcription_samples_raw"])
        parsed = [pilot.parsing.parse_transcription(s) for s in samples]
        labels = [pilot.canonicalize.canonical_answer_label(p) for p in parsed]
        majority, _ = pilot.entropy.majority_cluster(labels)
        gt = pilot.canonicalize.canonical_answer_label(row["pert_a"])
        return pd.Series({
            "entropy": pilot.entropy.cluster_entropy(labels),
            "correct": majority == gt,
            "n_fail": sum(1 for p in parsed if p is None),
        })

    rec = run.apply(recompute, axis=1)
    assert (rec["entropy"].round(6) == run["perception_entropy"].round(6)).all()
    assert (rec["correct"] == run["transcription_correct"].astype(bool)).all()
    assert (rec["n_fail"] == run["n_transcription_parse_failures"]).all()


def test_perception_replicates_qwen_on_a_second_model_family(run):
    """0.828 against Qwen-3B's 0.835, on the same images. The intervals
    overlap almost entirely, so the two are not resolvably different."""
    r = bootstrap_auroc_ci(run, "perception_entropy", "transcription_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.828, abs=0.005)
    assert r["ci_low"] == pytest.approx(0.782, abs=0.01)
    assert r["ci_high"] == pytest.approx(0.871, abs=0.01)
    assert r["excludes_chance"] is True

    # Qwen-3B's interval is [0.787, 0.879]; these overlap on almost their
    # whole length, which is the actual replication claim.
    assert r["ci_low"] < 0.879 and r["ci_high"] > 0.787

    assert run["transcription_correct"].astype(bool).mean() == pytest.approx(0.417, abs=0.005)


def test_it_survives_the_control_that_killed_internvl3(run):
    """The decisive contrast. InternVL3 scored HIGHER raw (0.915) and
    collapsed to 0.556 once max-entropy items were removed, because 67% of
    its items sat at the ceiling. Pixtral has 17% there and holds."""
    s = auroc_sensitivity(run, "perception_entropy", "transcription_correct",
                          "n_transcription_parse_failures", k=5, n_boot=10000, seed=0)
    assert s["excl_max_entropy"]["auroc"] == pytest.approx(0.758, abs=0.01)
    assert s["excl_both"]["auroc"] == pytest.approx(0.772, abs=0.01)
    assert s["excl_both"]["ci_low"] > 0.5
    assert s["robust"] is True

    import math
    at_max = run["perception_entropy"].apply(lambda e: math.isclose(e, math.log(5)))
    assert at_max.mean() == pytest.approx(0.17, abs=0.02)   # InternVL3: 0.673


def test_the_signal_is_graded_not_a_breakdown_detector(run):
    """The structural check from the retraction: correctness must not be a
    relabelling of vote concentration. Unanimous-but-wrong items exist, so
    agreement does not imply correctness here either."""
    unanimous = run[run["perception_entropy"] < 1e-9]
    n_wrong = int((~unanimous["transcription_correct"].astype(bool)).sum())
    assert len(unanimous) > 20
    assert n_wrong > 0, "a confidently-wrong item must exist or the metric is degenerate"


def test_abstention_precision_is_perfect_here_and_that_is_suspicious(run):
    """51/51 -- every item where all five transcriptions disagreed was wrong.
    Recorded, but NOT to be reported as better than Qwen's 57/61 without a
    caveat: this project has already seen perfect precision at n=100 (14/14)
    decay to 93.4% once the sample grew. Treat 100% as small-sample
    optimism about the tail, not an established property."""
    import math
    at_max = run["perception_entropy"].apply(lambda e: math.isclose(e, math.log(5)))
    flagged = run[at_max]
    n_wrong = int((~flagged["transcription_correct"].astype(bool)).sum())
    assert len(flagged) == 51
    assert n_wrong == 51
