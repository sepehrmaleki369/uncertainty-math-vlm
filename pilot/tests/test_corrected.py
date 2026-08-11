"""Locks the 2026-08-11 human-audit-corrected sensitivity analysis.

The human pass read all 104 of Qwen's `genuinely_wrong` items and found ~74%
of them to be the scoring pipeline. This file pins what that does to the
metrics, and -- more importantly -- pins the three properties that stop the
analysis being quoted as something it is not:

  * accuracy is ONE-SIDED (only false negatives could be found), so it is a
    range and an upper bound, never "true accuracy";
  * the AUROC FALLS, which was predicted before it was computed;
  * `extraction_issue` is not automatically recoverable -- items whose
    majority label is a parse failure have no answer to restore.

Requires the Qwen n=300 CSV in results/; skips cleanly without it.
"""

from pathlib import Path

import pandas as pd
import pytest

import pilot.corrected as C
from pilot.plotting import bootstrap_auroc_ci

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv"
AUDIT = [ROOT / "reference" / "audit" / "coded_31_qwen_only_qwen_20260811.csv",
         ROOT / "reference" / "audit" / "coded_73_wrong_on_both_20260811.csv"]


@pytest.fixture(scope="module")
def run():
    if not RUN.exists():
        pytest.skip("Qwen n=300 CSV not present (Drive-only)")
    return pd.read_csv(RUN)


@pytest.fixture(scope="module")
def audit():
    return pd.concat([pd.read_csv(p) for p in AUDIT]).set_index("item")


def test_the_audit_covers_every_genuinely_wrong_item(audit):
    """104/104 is what makes this a census rather than a sample, and it is the
    single fact the whole framing rests on."""
    assert len(audit) == 104
    assert audit.index.duplicated().sum() == 0
    assert set(audit["final_label"]) <= {
        "extraction_issue", "notation_misread", "needs_visual",
        "copied_wrong_line", "hallucination"}


def test_hallucination_is_zero_under_human_coding(audit):
    assert (audit["final_label"] == "hallucination").sum() == 0


def test_unrecoverable_items_are_found_from_data_not_from_memory(run, audit):
    """A first pass named 7 items from recollection; the objective rule -- the
    MAJORITY cluster label is a parse failure -- names 4. The rule is the
    thing that gets to decide, which is why it is pinned."""
    assert C.unrecoverable_items(run, audit) == [40, 65, 142, 236]


def test_extraction_issue_does_not_mean_the_model_was_right(run, audit):
    """The carve-out that keeps the correction honest: these four are labelled
    `extraction_issue` yet resolve to unknown, not True."""
    hc = C.human_corrected_correct(run, audit)
    for item in (40, 65, 142, 236):
        assert audit.loc[item, "final_label"] == "extraction_issue"
        assert hc[item] is None


def test_the_correction_is_one_sided(run, audit):
    """No audited item can move correct -> wrong, because only items the frozen
    rule called WRONG were ever looked at. This is the property that makes the
    accuracy figure upper-biased, so it is asserted rather than described."""
    for unknown_as in (False, True):
        out = C.apply_correction(run, audit, unknown_as=unknown_as)
        base = out["transcription_correct"].astype(bool)
        assert not (base & ~out["corrected_correct"]).any()
        assert not out.loc[~out["was_audited"], "corrected_correct"].ne(
            base[~out["was_audited"]]).any()


def test_corrected_accuracy_is_a_range_against_the_strict_v1_baseline(run, audit):
    """Baseline MUST be strict_v1's 141/300, not the stored 39.3% -- the audit
    selected its items from a local rescore, and mixing the two labelings is
    how this number would come to contradict the paper."""
    b = C.corrected_accuracy_bounds(run, audit)
    assert b["baseline_correct"] == 141
    assert b["baseline_accuracy"] == pytest.approx(0.470, abs=0.001)
    assert b["corrected_correct_low"] == 214
    assert b["corrected_correct_high"] == 222
    assert b["corrected_accuracy_low"] == pytest.approx(0.713, abs=0.005)
    assert b["corrected_accuracy_high"] == pytest.approx(0.740, abs=0.005)
    assert b["one_sided"] is True


def test_corrected_accuracy_exceeds_even_the_loosest_automated_rule(run, audit):
    """`final_term_v4` -- the most generous automated rule -- reaches 63.3%.
    The human correction goes past it, which is expected (a human sees
    equivalences no rule encodes) and is also exactly why the one-sidedness
    has to be stated alongside it."""
    b = C.corrected_accuracy_bounds(run, audit)
    assert b["corrected_accuracy_low"] > 0.633


def test_the_auroc_falls_and_that_was_predicted(run, audit):
    """Registered before computing: recovered items carry mean entropy ~1.16
    against ~0.66 for the already-correct ones, so the correct class gets
    noisier and the AUROC must drop. It does, to ~0.79-0.81, and stays well
    clear of chance -- entropy still predicts the REMAINING real failures."""
    for unknown_as, lo, hi in ((False, 0.808, 0.808), (True, 0.794, 0.794)):
        out = C.apply_correction(run, audit, unknown_as=unknown_as)
        got = bootstrap_auroc_ci(out, "perception_entropy", "corrected_correct",
                                 n_boot=4000, seed=0)
        assert got["auroc"] == pytest.approx(lo, abs=0.015)
        assert got["auroc"] < 0.850           # below the uncorrected figure
        assert got["ci_low"] > 0.70           # still clears the registered bar


def test_the_auroc_drop_is_modest_and_not_cleanly_resolved(run, audit):
    """Guards against over-reading the drop. Under one resolution of `unknown`
    the paired interval spans zero; under the other it barely excludes it. The
    honest sentence is 'a modest drop, directionally consistent', not 'the
    signal was partly an artifact'."""
    from pilot.plotting import bootstrap_auroc_difference_ci
    lo = C.apply_correction(run, audit, unknown_as=False)
    d = bootstrap_auroc_difference_ci(
        lo, "perception_entropy", "corrected_correct",
        "perception_entropy", "transcription_correct", n_boot=4000, seed=0)
    assert d["difference"] == pytest.approx(-0.041, abs=0.02)
    assert d["ci_low"] < 0 < d["ci_high"], "spans zero under unknown=wrong"


def test_entropy_flags_real_errors_above_their_base_rate(run, audit):
    """The practitioner-facing result: a deferral rule firing on high entropy
    is enriched for genuine model failures relative to the audited base rate.
    Modest, not dramatic -- most of what it flags is still scoring, because
    scoring is most of the bucket."""
    d = C.deferral_precision(run, audit, quantile=0.75)
    assert d["precision_real_error"] > d["base_rate_real_error"]
    assert d["precision_real_error"] == pytest.approx(0.367, abs=0.03)
    assert d["base_rate_real_error"] == pytest.approx(0.221, abs=0.02)


def test_k5_ties_make_the_top_decile_unreachable(run, audit):
    """Not a bug: 20.3% of items sit at exactly ln(5), so the 90th and 75th
    percentile thresholds collide and both select the same 32 items. Same
    tie-driven resolution limit as the K=5 vs K=10 deferral work."""
    a = C.deferral_precision(run, audit, quantile=0.75)
    b = C.deferral_precision(run, audit, quantile=0.90)
    assert a["threshold"] == b["threshold"]
    assert a["n_flagged"] == b["n_flagged"] == 32


def test_spot_check_sample_is_random_seeded_and_flags_known_false_passes(run):
    """Selection by seed, never by hand -- otherwise the false-pass rate it
    produces is a collection of interesting cases rather than an estimate."""
    s1 = C.correct_item_spot_check_sample(run, n=40)
    s2 = C.correct_item_spot_check_sample(run, n=40)
    assert s1["item"].tolist() == s2["item"].tolist()
    assert len(s1) == 40
    assert s1["item"].duplicated().sum() == 0
    scored_correct = C.rescore.rescore_run(run, rule="strict_v1")
    ok = scored_correct.index[scored_correct["transcription_correct"].astype(bool)]
    assert set(s1["item"]) <= set(ok), "must sample only from CORRECT items"
    assert s1["final_label"].eq("").all(), "ships uncoded"
