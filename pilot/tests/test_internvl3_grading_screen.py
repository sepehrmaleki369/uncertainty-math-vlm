"""Locks the 2026-08-08 InternVL3 grading-prompt screen (pilot/13).

A pre-registered, single-shot screen of three ideas against InternVL3's
has_error=1 reasoning AUROC (baseline 0.628 on the same 150 items). The
point of these tests is to preserve the *honest* reading of the outcome,
which is not the same as the verdict string the notebook printed.

By the letter of the pre-registration, `k10` passes: AUROC 0.7023 >= 0.70,
CI excludes chance, minority class 34 >= 30. That is recorded here as
written -- the bar was not moved after seeing the numbers.

But the paired comparison against the baseline, on the same items, is what
this project requires before calling any difference real, and it does not
resolve for ANY of the three variants:
    commit  +0.069 [-0.028, +0.166]
    k10     +0.074 [-0.024, +0.171]
    restate -0.008 [-0.112, +0.095]
k10 cleared the 0.70 threshold by 0.0023 while its own CI runs
[0.616, 0.781] -- the lower bound sits far below the bar it "passed", and
the 0.005 gap separating k10 (pass) from commit (fail) is noise. The 0.70
bar was borrowed from the confirmation runs (0.775-0.854), where CIs were
tight and unambiguously above it; applied to an n=150 screen with a ~+-0.08
interval it is a much weaker statement. Decided with the user: no
confirmation run, 0.628 stands as InternVL3's reported reasoning result.

The one thing that DID resolve is a secondary, non-pre-registered outcome:
k10's grading accuracy, 0.700 -> 0.773, +0.073 [+0.013, +0.133].

The CSV is untracked (Drive-only); tests skip cleanly when absent.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pilot.entropy
import pilot.parsing
from pilot.plotting import bootstrap_auroc_ci, bootstrap_auroc_difference_ci

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SCREEN_CSV = RESULTS_DIR / "internvl3_grading_screen_n150_20260807T231004Z.csv"
REFERENCE_CSV = RESULTS_DIR / "scaleup_n300_bal50_internvl3-8b-hf_20260807T205407Z.csv"

KEY = ["orig_q", "pert_a"]


@pytest.fixture(scope="module")
def screen():
    if not SCREEN_CSV.exists():
        pytest.skip(f"{SCREEN_CSV} not present (download from Drive)")
    return pd.read_csv(SCREEN_CSV)


@pytest.fixture(scope="module")
def baseline():
    if not REFERENCE_CSV.exists():
        pytest.skip(f"{REFERENCE_CSV} not present (download from Drive)")
    ref = pd.read_csv(REFERENCE_CSV)
    return ref[ref["has_error"] == 1].copy()


def _paired(screen, baseline, variant):
    a = screen[screen["screen_variant"] == variant][
        KEY + ["reasoning_entropy", "grading_correct"]
    ].rename(columns={"reasoning_entropy": "ent_v", "grading_correct": "corr_v"})
    b = baseline[KEY + ["reasoning_entropy", "grading_correct"]].rename(
        columns={"reasoning_entropy": "ent_base", "grading_correct": "corr_base"}
    )
    merged = a.merge(b, on=KEY, how="inner", validate="one_to_one")
    assert len(merged) == 150
    return merged


def test_screen_shape_and_precision_matches_the_baseline_run(screen):
    """The precision guard added to notebook 13 after a 4-bit fallback was
    caught mid-session: a screen run in 4-bit against a bf16 baseline would
    make prompt effects and precision effects indistinguishable."""
    assert len(screen) == 450  # 3 variants x 150 items
    assert set(screen["screen_variant"].unique()) == {"commit", "restate", "k10"}
    assert (screen["quantized"] == False).all()  # noqa: E712
    assert screen["model_id"].unique().tolist() == ["OpenGVLab/InternVL3-8B-hf"]
    for variant, expected_k in [("commit", 5), ("restate", 5), ("k10", 10)]:
        assert (screen.loc[screen["screen_variant"] == variant, "k"] == expected_k).all()


def test_no_scoring_bug_recomputes_from_raw_samples(screen):
    def recompute(row):
        samples = ast.literal_eval(row["all_grading_samples_raw"])
        parsed = [pilot.parsing.parse_grading(s) for s in samples]
        labels = [None if d is None else str(d) for d in parsed]
        entropy = pilot.entropy.cluster_entropy(labels)
        majority, _ = pilot.entropy.majority_cluster(labels)
        correct = majority in {"0", "1"} and int(majority) == int(row["has_error"])
        return pd.Series({"entropy": entropy, "correct": correct, "k": len(samples)})

    rec = screen.apply(recompute, axis=1)
    assert (rec["entropy"].round(6) == screen["reasoning_entropy"].round(6)).all()
    assert (rec["correct"] == screen["grading_correct"].astype(bool)).all()
    assert (rec["k"] == screen["k"]).all()


def test_baseline_on_the_same_150_items(baseline):
    r = bootstrap_auroc_ci(baseline, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)
    assert r["auroc"] == pytest.approx(0.628, abs=0.005)
    assert r["n_error"] == 45
    assert baseline["grading_correct"].mean() == pytest.approx(0.700, abs=0.005)


def test_no_variant_resolvably_beats_the_baseline_auroc(screen, baseline):
    """The finding that matters. Every paired interval spans zero, so the
    screen found no resolvable improvement on its pre-registered target --
    including k10, whose point estimate crossed the 0.70 bar."""
    expected = {
        "commit": (+0.069, False),
        "k10": (+0.074, False),
        "restate": (-0.008, False),
    }
    for variant, (diff, excludes_zero) in expected.items():
        merged = _paired(screen, baseline, variant)
        d = bootstrap_auroc_difference_ci(
            merged, "ent_v", "corr_v", "ent_base", "corr_base", n_boot=10000, seed=0
        )
        assert d["difference"] == pytest.approx(diff, abs=0.005), variant
        assert d["difference_excludes_zero"] is excludes_zero, variant
        assert d["ci_low"] < 0 < d["ci_high"], variant


def test_k10_passes_the_letter_of_the_prereg_but_by_a_hair(screen):
    """Both halves of this are load-bearing. k10 does meet the bar as
    written, and that is reported rather than explained away. It also
    clears it by 0.0023 with a CI whose lower bound (0.616) is far below
    the threshold -- so the 'pass' does not mean what the same bar meant
    on the tight-CI confirmation runs it was borrowed from."""
    k10 = screen[screen["screen_variant"] == "k10"]
    r = bootstrap_auroc_ci(k10, "reasoning_entropy", "grading_correct",
                           n_boot=10000, seed=0)

    # Passes as written.
    assert r["auroc"] >= 0.70
    assert r["auroc"] == pytest.approx(0.7023, abs=0.005)
    assert r["excludes_chance"] is True
    n_wrong = int((~k10["grading_correct"].astype(bool)).sum())
    assert min(n_wrong, len(k10) - n_wrong) >= 30
    assert n_wrong == 34

    # And by a margin far smaller than its own uncertainty.
    assert r["auroc"] - 0.70 < 0.01
    assert r["ci_low"] < 0.70

    # The pass/fail line between k10 and commit is noise.
    commit = screen[screen["screen_variant"] == "commit"]
    rc = bootstrap_auroc_ci(commit, "reasoning_entropy", "grading_correct",
                            n_boot=10000, seed=0)
    assert rc["auroc"] < 0.70  # "fails"
    assert abs(r["auroc"] - rc["auroc"]) < 0.01  # ...by half a point


def test_k10_accuracy_gain_is_the_one_resolved_result(screen, baseline):
    """Secondary and not pre-registered, but real: more samples make the
    majority vote more accurate (16 items fixed, 5 broken). Same direction
    as the confirmed K=5 -> K=10 perception result."""
    a = screen[screen["screen_variant"] == "k10"][KEY + ["grading_correct"]].rename(
        columns={"grading_correct": "corr_v"})
    b = baseline[KEY + ["grading_correct"]].rename(columns={"grading_correct": "corr_base"})
    merged = a.merge(b, on=KEY, validate="one_to_one")

    va = merged["corr_v"].astype(bool).to_numpy()
    vb = merged["corr_base"].astype(bool).to_numpy()
    assert int((va & ~vb).sum()) == 16  # fixed
    assert int((~va & vb).sum()) == 5   # broke
    assert va.mean() == pytest.approx(0.773, abs=0.005)

    rng = np.random.default_rng(0)
    d = va.astype(float) - vb.astype(float)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    assert lo > 0  # resolved


def test_restate_replicates_its_3b_failure_mode(screen, baseline):
    """Cross-model replication of a prior negative: on 3B, restate swung
    the response distribution hard while leaving accuracy flat -- the model
    swapping which near-constant answer it gives rather than
    discriminating. Same signature here: heavy churn, no gain."""
    a = screen[screen["screen_variant"] == "restate"][KEY + ["grading_correct"]].rename(
        columns={"grading_correct": "corr_v"})
    b = baseline[KEY + ["grading_correct"]].rename(columns={"grading_correct": "corr_base"})
    merged = a.merge(b, on=KEY, validate="one_to_one")

    va = merged["corr_v"].astype(bool).to_numpy()
    vb = merged["corr_base"].astype(bool).to_numpy()
    fixed = int((va & ~vb).sum())
    broke = int((~va & vb).sum())
    assert fixed == 31
    assert broke == 36
    # Lots of movement, no net gain -- the 3B signature.
    assert fixed + broke > 60
    assert va.mean() < vb.mean()
