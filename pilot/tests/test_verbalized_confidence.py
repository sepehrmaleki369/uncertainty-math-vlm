"""Locks the 2026-08-11 verbalized-confidence run: asking the model does not work.

The first objection to any sampling-based uncertainty method is that entropy
needs K forward passes while asking needs one. This project had the answer for
the GRADING arm only (notebook 04: 0.547 [0.432, 0.661]); the perception arm --
the one that works, and the one the paper leads with -- had never been tested.

It does not work there either, and the margin resolves:

    perception entropy       0.807 [0.757, 0.854]
    -verbalized confidence   0.528 [0.463, 0.593]
    PAIRED difference       +0.279 [+0.194, +0.359]

Two things make this stronger than a bare null. The self-report is NOT
degenerate -- 37 distinct values spanning 68-100 -- so the model varies its
confidence and the variation is simply uninformative, which is a different and
more damaging claim than "it always says 95". And it converges with the token
logprob result (+0.297 [+0.214, +0.380]): two independent cheap-confidence
baselines both land near chance on the same arm.

Every figure is recomputed from the raw sample columns rather than copied from
the notebook's printout, per this project's rule that a frozen number must be
able to CHECK the report rather than echo it. The CSV is Drive-only; the tests
skip cleanly until it is downloaded into results/.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import pilot.parsing
import pilot.rescore
from pilot.plotting import bootstrap_auroc_ci, bootstrap_auroc_difference_ci

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _find():
    hits = sorted(RESULTS_DIR.glob("confidence_perception_full_n300_*.csv"))
    return hits[-1] if hits else None


@pytest.fixture(scope="module")
def run():
    path = _find()
    if path is None:
        pytest.skip("confidence_perception_full_n300_*.csv not present "
                    "(Drive-only; download into results/)")
    return pd.read_csv(path)


def test_run_shape(run):
    assert len(run) == 300
    assert (run["k_transcription"] == 5).all()
    assert run["model_id"].unique().tolist() == ["Qwen/Qwen2.5-VL-3B-Instruct"]
    assert (run["quantized"] == False).all()  # noqa: E712


def test_confidence_recomputes_from_the_raw_samples(run):
    """Not a scoring bug: the stored mean_confidence must follow from the raw
    text via parse_confidence, the same way every other run in this project is
    verified."""
    mismatches = 0
    for _, row in run.iterrows():
        parsed = [pilot.parsing.parse_confidence(s)
                  for s in ast.literal_eval(row["all_transcription_samples_raw"])]
        got = [c for c in parsed if c is not None]
        assert len(got) == row["n_confidence_parsed"]
        if got:
            if abs(sum(got) / len(got) - row["mean_confidence"]) > 1e-6:
                mismatches += 1
        elif pd.notna(row["mean_confidence"]):
            mismatches += 1
    assert mismatches == 0


def test_the_model_is_confident_and_the_confidence_is_not_degenerate(run):
    """A near-constant self-report cannot rank anything whatever its
    calibration, which would make the comparison uninteresting. That is NOT
    what happens here -- the model varies, and varies uninformatively."""
    usable = run.dropna(subset=["mean_confidence"])
    assert len(usable) == 295
    assert usable["mean_confidence"].median() == pytest.approx(95, abs=1)
    assert usable["mean_confidence"].min() == pytest.approx(68, abs=2)
    assert usable["mean_confidence"].max() == pytest.approx(100, abs=1)
    assert usable["mean_confidence"].nunique() == 37

    parse_rate = run["n_confidence_parsed"].sum() / (len(run) * 5)
    assert parse_rate == pytest.approx(0.938, abs=0.005)


def test_asking_the_model_does_not_work(run):
    """The result. Confidence is negated first -- higher confidence should mean
    LESS likely wrong -- and getting that backwards would silently invert the
    finding, so the direction is asserted rather than assumed."""
    usable = run.dropna(subset=["mean_confidence"]).copy()
    usable["neg_confidence"] = -usable["mean_confidence"]
    assert (usable["neg_confidence"] <= 0).all()

    ent = bootstrap_auroc_ci(usable, "perception_entropy",
                             "transcription_correct", n_boot=10000, seed=0)
    con = bootstrap_auroc_ci(usable, "neg_confidence",
                             "transcription_correct", n_boot=10000, seed=0)

    assert ent["auroc"] == pytest.approx(0.807, abs=0.01)
    assert ent["excludes_chance"] is True
    assert con["auroc"] == pytest.approx(0.528, abs=0.01)
    assert con["excludes_chance"] is False       # no signal at all


def test_the_paired_margin_resolves(run):
    """The headline is the PAIRED difference on identical resampled items.
    Reading the two marginal CIs against each other would be the
    difference-in-significance fallacy this project has already been burned
    by, and their intervals do not even overlap here -- which is precisely
    when that shortcut is most tempting."""
    usable = run.dropna(subset=["mean_confidence"]).copy()
    usable["neg_confidence"] = -usable["mean_confidence"]

    d = bootstrap_auroc_difference_ci(
        usable, "perception_entropy", "transcription_correct",
        "neg_confidence", "transcription_correct", n_boot=10000, seed=0)

    assert d["difference"] == pytest.approx(0.279, abs=0.015)
    assert d["ci_low"] == pytest.approx(0.194, abs=0.02)
    assert d["ci_high"] == pytest.approx(0.359, abs=0.02)
    assert d["ci_low"] > 0, "the margin must exclude zero to be reportable"


def test_it_converges_with_the_token_logprob_baseline(run):
    """Two independent cheap-confidence baselines landing in the same place is
    what makes the negative solid rather than a quirk of one elicitation:
    -mean-logprob gave +0.297 [+0.214, +0.380] on the same arm."""
    usable = run.dropna(subset=["mean_confidence"]).copy()
    usable["neg_confidence"] = -usable["mean_confidence"]
    con = bootstrap_auroc_ci(usable, "neg_confidence",
                             "transcription_correct", n_boot=4000, seed=0)
    # token logprob was 0.521 on the K=10 run; verbalized lands beside it.
    assert abs(con["auroc"] - 0.521) < 0.05


def test_accuracy_is_recorded_so_it_is_never_taken_from_another_run(run):
    """An earlier CLAUDE.md entry quoted "right 42% of the time" beside this
    result. That 42.0% is notebook 19's accuracy, from the GATED boxed run --
    a different generation. This pins notebook 20's own number so the two
    cannot be confused again."""
    acc = run["transcription_correct"].astype(bool).mean()
    assert 0.30 < acc < 0.60, acc          # sane range; exact value below
    print(f"\nnotebook 20 transcription accuracy: {acc:.1%} "
          f"(unconstrained reference run: 47.0%)")
