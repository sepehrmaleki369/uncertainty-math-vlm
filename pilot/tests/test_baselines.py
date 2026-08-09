"""Baselines for the perception signal: unit behaviour, and the real numbers.

Two jobs. The first half tests distinct_count / majority_fraction /
majority_margin as functions. The second locks what they score on the real
n=300 run, because their whole purpose is to answer a question a reader
will ask -- is entropy over K samples doing anything that counting distinct
answers does not? -- and the answer is only usable if the numbers behind it
cannot drift.

All four signals come from the SAME five samples per item. No extra
generation is involved, which is what makes them fair ablations of the
summary rather than alternative experiments.
"""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import pilot.canonicalize
import pilot.parsing
from pilot.entropy import (
    cluster_entropy,
    distinct_count,
    majority_fraction,
    majority_margin,
)
from pilot.plotting import bootstrap_auroc_ci, bootstrap_auroc_difference_ci

ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = (ROOT / "results"
               / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")
SNAPSHOT = ROOT / "reference" / "perception_baselines_20260802.json"

_ID = lambda x: x  # noqa: E731 -- labels are pre-canonicalised in these tests


# --- unit behaviour -------------------------------------------------------

def test_distinct_count_spans_one_to_k():
    assert distinct_count(["a"] * 5, normalize_fn=_ID) == 1
    assert distinct_count(list("abcde"), normalize_fn=_ID) == 5
    assert distinct_count([], normalize_fn=_ID) == 0


def test_majority_fraction_direction_is_confidence_not_uncertainty():
    """Reads the opposite way to entropy. Getting this backwards would
    silently invert an AUROC, so it is asserted rather than assumed."""
    assert majority_fraction(["a"] * 5, normalize_fn=_ID) == 1.0
    assert majority_fraction(list("abcde"), normalize_fn=_ID) == 0.2
    assert cluster_entropy(["a"] * 5, normalize_fn=_ID) == 0.0  # entropy: opposite end


def test_margin_separates_splits_that_fraction_cannot():
    """The reason majority_margin is worth reporting alongside the fraction:
    a 3-2 and a 3-1-1 split have the same majority fraction but different
    margins, and only the margin distinguishes them."""
    three_two = ["a", "a", "a", "b", "b"]
    three_one_one = ["a", "a", "a", "b", "c"]

    assert majority_fraction(three_two, normalize_fn=_ID) == \
        majority_fraction(three_one_one, normalize_fn=_ID) == 0.6
    assert majority_margin(three_two, normalize_fn=_ID) == pytest.approx(0.2)
    assert majority_margin(three_one_one, normalize_fn=_ID) == pytest.approx(0.4)
    # Entropy also separates them, and orders them the same way round.
    assert cluster_entropy(three_one_one, normalize_fn=_ID) > \
        cluster_entropy(three_two, normalize_fn=_ID)


def test_unanimous_and_all_distinct_are_the_extremes():
    assert majority_margin(["a"] * 4, normalize_fn=_ID) == 1.0
    assert majority_margin(list("abcd"), normalize_fn=_ID) == 0.0


def test_empty_input_is_handled_not_crashed():
    assert majority_fraction([], normalize_fn=_ID) == 0.0
    assert majority_margin([], normalize_fn=_ID) == 0.0


# --- the real numbers -----------------------------------------------------

@pytest.fixture(scope="module")
def scored():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} not present (download from Drive)")
    df = pd.read_csv(RESULTS_CSV)
    rows = []
    for raw in df["all_transcription_samples_raw"]:
        labels = [pilot.canonicalize.canonical_answer_label(
                      pilot.parsing.parse_transcription(s))
                  for s in ast.literal_eval(raw)]
        rows.append({
            "distinct_count": distinct_count(labels, normalize_fn=_ID),
            "neg_majority_frac": -majority_fraction(labels, normalize_fn=_ID),
            "neg_majority_margin": -majority_margin(labels, normalize_fn=_ID),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def test_entropy_beats_every_simpler_summary_of_the_same_samples(scored):
    """The result the paper needs: using the full cluster distribution is
    worth a small but resolved amount over counting distinct answers or
    looking only at the majority. Every paired interval excludes zero.

    If this ever fails, the paper's claim that entropy is not merely
    distinct-count-in-disguise no longer holds and must be rewritten.
    """
    expected = {
        "distinct_count": (0.790, 0.045),
        "neg_majority_frac": (0.793, 0.042),
        "neg_majority_margin": (0.779, 0.055),
    }
    ent = bootstrap_auroc_ci(scored, "perception_entropy", "transcription_correct",
                             n_boot=10000, seed=0)
    assert ent["auroc"] == pytest.approx(0.835, abs=0.005)

    for col, (auroc, diff) in expected.items():
        r = bootstrap_auroc_ci(scored, col, "transcription_correct",
                               n_boot=10000, seed=0)
        assert r["auroc"] == pytest.approx(auroc, abs=0.005), col

        d = bootstrap_auroc_difference_ci(
            scored, "perception_entropy", "transcription_correct",
            col, "transcription_correct", n_boot=10000, seed=0)
        assert d["difference"] == pytest.approx(diff, abs=0.005), col
        assert d["difference_excludes_zero"] is True, col
        assert d["ci_low"] > 0, col


def test_the_simple_summaries_are_related_but_not_identical(scored):
    """Spearman 0.83-0.89 against entropy. High enough that the comparison
    above is a fair ablation rather than a straw man, low enough that they
    are not the same measurement."""
    for col, rho in [("distinct_count", 0.893),
                     ("neg_majority_frac", 0.884),
                     ("neg_majority_margin", 0.825)]:
        actual = scored[["perception_entropy", col]].corr(method="spearman").iloc[0, 1]
        assert actual == pytest.approx(rho, abs=0.01), col
        assert actual < 0.95, f"{col} is nearly identical to entropy"


def test_snapshot_matches_the_recomputed_values(scored):
    """The snapshot the paper cites must agree with a fresh recomputation."""
    if not SNAPSHOT.exists():
        pytest.skip(f"{SNAPSHOT} not present")
    snap = json.loads(SNAPSHOT.read_text())
    for name, col in [("sampling_entropy", "perception_entropy"),
                      ("distinct_count", "distinct_count"),
                      ("neg_majority_frac", "neg_majority_frac"),
                      ("neg_majority_margin", "neg_majority_margin")]:
        r = bootstrap_auroc_ci(scored, col, "transcription_correct",
                               n_boot=10000, seed=0)
        assert snap["signals"][name]["auroc_ci"]["auroc"] == pytest.approx(
            r["auroc"], abs=1e-9), name
