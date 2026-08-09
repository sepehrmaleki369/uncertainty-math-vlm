"""Locks the 2026-08-08 ScratchMath sizing run: a capability/engagement gate.

Second dataset, Qwen2.5-VL-7B held fixed (confirmed 0.801 on FERMAT at
n=648) so only the dataset varies. The run came back GATED, and the reason
matters more than the bare misgrade rate:

Every ScratchMath item contains an error, and the model answers "there is
an error" 90% of the time, so it scores 96% accuracy by construction. The
misgrade rate is 4.0% -- under the pre-registered 5% bar, so the stratum
cannot be powered without ~750 items.

But the decisive evidence is qualitative and quantified over all 500
samples: in 24% of them the model explicitly states it cannot read or use
the image ("does not contain any handwritten work", "impossible to
determine"), and in 82% of those it emits `Error: 1` regardless. On an
all-error dataset that scores as correct. So a large share of the 96% is
the response bias firing through explicit non-engagement, not grading
competence.

Powering it further would not fix this: non-engagement is markedly higher
on misgraded items (0.40 vs 0.23), so a larger run would substantially be
measuring "can the model read this scratchwork" rather than uncertainty
about the mathematics -- a different construct from the FERMAT result, and
therefore not comparable to it. Combined with 70/100 items at exactly zero
entropy, there is nothing for entropy to rank.

Verdict recorded here: gated / not comparable. Decided with the user: no
further ScratchMath GPU work.

The CSV is untracked (Drive-only); tests skip cleanly when absent.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import pilot.entropy
import pilot.parsing

RESULTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scratchmath_sizing_n100_qwen25-vl-7b-instruct_20260808T121016Z.csv"
)

# Phrases in which the model states it cannot read or use the image. Kept as
# a module constant because the 24%/82% figures below are only meaningful
# relative to this exact list -- changing it changes the claim.
CANNOT_READ_PHRASES = [
    "does not contain", "no handwritten", "not contain any", "is unclear",
    "quite unclear", "impossible to determine", "cannot be determined",
    "no indication", "illegible", "not legible", "difficult to interpret",
    "does not appear to contain", "no visible", "unrelated", "does not seem to",
    "not clear", "no mathematical work", "cannot determine", "hard to read",
]

GATE_MIN_MISGRADE_RATE = 0.05  # pre-registered in notebook 14 before the run


@pytest.fixture(scope="module")
def run():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    return pd.read_csv(RESULTS_CSV)


def _samples(row):
    return ast.literal_eval(row["all_grading_samples_raw"])


def test_run_shape_model_fixed_and_full_precision(run):
    """Qwen-7B in bf16, matching the FERMAT run this is compared against --
    a 4-bit fallback would confound dataset with precision."""
    assert len(run) == 100
    assert run["model_id"].unique().tolist() == ["Qwen/Qwen2.5-VL-7B-Instruct"]
    assert (run["quantized"] == False).all()  # noqa: E712
    assert all(len(_samples(r)) == 5 for _, r in run.iterrows())


def test_dataset_is_all_error_so_no_clean_stratum_exists(run):
    """The structural fact that constrains everything: ScratchMath has no
    has_error=0 items, so the clean-stratum inversion confirmed on FERMAT
    (0.280) cannot be tested here and no balanced sample can be built."""
    assert (run["has_error"] == 1).all()
    assert int((run["has_error"] == 0).sum()) == 0


def test_no_scoring_bug_recomputes_from_raw_samples(run):
    def recompute(row):
        parsed = [pilot.parsing.parse_grading(s) for s in _samples(row)]
        labels = [None if d is None else str(d) for d in parsed]
        majority, _ = pilot.entropy.majority_cluster(labels)
        return pd.Series({
            "entropy": pilot.entropy.cluster_entropy(labels),
            "correct": majority == "1",
            "n_fail": sum(1 for d in parsed if d is None),
        })

    rec = run.apply(recompute, axis=1)
    assert (rec["entropy"].round(6) == run["reasoning_entropy"].round(6)).all()
    assert (rec["correct"] == run["grading_correct"].astype(bool)).all()
    assert (rec["n_fail"] == run["n_grading_parse_failures"]).all()


def test_accuracy_is_inflated_by_construction_not_competence(run):
    """96% accuracy against a says-error rate of 90% on an all-error set.
    The accuracy number must never be quoted as grading competence."""
    assert run["grading_correct"].astype(bool).mean() == pytest.approx(0.96, abs=0.005)
    assert run["says_error_frac"].mean() == pytest.approx(0.90, abs=0.01)
    # Parsing is not the problem -- the model follows the format perfectly.
    assert int(run["n_grading_parse_failures"].sum()) == 0


def test_gate_fires_misgrade_rate_below_the_preregistered_bar(run):
    n_misgraded = int((~run["grading_correct"].astype(bool)).sum())
    assert n_misgraded == 4
    assert n_misgraded / len(run) == pytest.approx(0.04, abs=0.001)
    assert n_misgraded / len(run) < GATE_MIN_MISGRADE_RATE  # GATED


def test_the_decisive_evidence_non_engagement_still_answers_error(run):
    """The reason this is a gate and not merely an underpowered run: when
    the model says it cannot evaluate the image, it answers "error" anyway
    82% of the time, which scores as correct on an all-error dataset."""
    total = 0
    cannot_read = 0
    cannot_read_says_error = 0
    for _, row in run.iterrows():
        for s in _samples(row):
            total += 1
            if any(p in str(s).lower() for p in CANNOT_READ_PHRASES):
                cannot_read += 1
                if pilot.parsing.parse_grading(s) == 1:
                    cannot_read_says_error += 1

    assert total == 500
    assert cannot_read == 120
    assert cannot_read / total == pytest.approx(0.24, abs=0.005)
    assert cannot_read_says_error == 98
    assert cannot_read_says_error / cannot_read == pytest.approx(0.82, abs=0.01)


def test_powering_it_further_would_measure_legibility_not_reasoning(run):
    """Non-engagement is concentrated on the misgraded items, so a larger
    run would substantially be measuring whether the model can read the
    scratchwork -- a different construct from the FERMAT result, hence
    'not comparable' rather than merely 'underpowered'."""
    def frac_cannot_read(row):
        samples = _samples(row)
        n = sum(1 for s in samples
                if any(p in str(s).lower() for p in CANNOT_READ_PHRASES))
        return n / len(samples)

    run = run.copy()
    run["_frac"] = run.apply(frac_cannot_read, axis=1)
    correct = run["grading_correct"].astype(bool)
    assert run.loc[~correct, "_frac"].mean() == pytest.approx(0.40, abs=0.02)
    assert run.loc[correct, "_frac"].mean() == pytest.approx(0.233, abs=0.02)
    assert run.loc[~correct, "_frac"].mean() > run.loc[correct, "_frac"].mean()


def test_entropy_is_degenerate_nothing_left_to_rank(run):
    """70/100 items have all five samples agreeing. With 4 misgrades and
    this much unanimity there is no usable ranking signal, independent of
    the power argument."""
    counts = run["reasoning_entropy"].round(3).value_counts()
    assert int(counts.get(0.0, 0)) == 70
    assert set(counts.index) == {0.0, 0.5, 0.673}


def test_filtering_out_non_engagement_makes_the_metric_undefined(run):
    """The check that settles whether non-engagement could just be filtered
    out: it cannot. All four misgraded items sit in the flagged group, so
    discarding flagged items leaves a subset with no minority class and the
    AUROC comes back nan -- undefined rather than cleaner.

    Locked because it is the sharpest single argument for the gate, and
    because a future change to CANNOT_READ_PHRASES or to the scoring path
    could quietly turn "undefined" into a plausible-looking number.
    """
    import math

    from pilot.plotting import bootstrap_auroc_ci, compute_auroc

    run = run.copy()
    run["_n_flagged"] = run["all_grading_samples_raw"].apply(
        lambda raw: sum(1 for s in _samples_of(raw)
                        if any(p in str(s).lower() for p in CANNOT_READ_PHRASES))
    )
    correct = run["grading_correct"].astype(bool)

    # The partition is not clean: most items are a mix, only 3 are fully flagged.
    assert int((run["_n_flagged"] >= 1).sum()) == 60
    assert int((run["_n_flagged"] == 5).sum()) == 3

    kept = run[run["_n_flagged"] == 0]
    assert len(kept) == 40
    assert int((~kept["grading_correct"].astype(bool)).sum()) == 0   # every misgrade discarded
    assert int((~correct).sum()) == 4                                # ...and there were only 4

    assert math.isnan(compute_auroc(kept, "reasoning_entropy", "grading_correct"))
    r = bootstrap_auroc_ci(kept, "reasoning_entropy", "grading_correct",
                           n_boot=200, seed=0)
    assert r["n_error"] == 0
    assert math.isnan(r["auroc"])
    assert r["excludes_chance"] is False


def _samples_of(raw):
    return ast.literal_eval(raw)
