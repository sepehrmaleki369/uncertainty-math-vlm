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


# ---------------------------------------------------------------------------
# The false-pass spot check (2026-08-12) -- the other side of the correction
# ---------------------------------------------------------------------------

SPOT = ROOT / "reference" / "audit" / "spotcheck_40_qwen_strict_v1_correct_20260811.csv"


@pytest.fixture(scope="module")
def spot():
    return pd.read_csv(SPOT)


def test_the_spot_check_is_coded_with_its_own_vocabulary(spot):
    """`true_correct` is not in pilot.failures.LABELS on purpose: that
    vocabulary says why a WRONG item failed, this one says whether a CORRECT
    verdict was earned."""
    assert len(spot) == 40
    assert spot["item"].is_unique
    assert spot.isna().sum().sum() == 0
    assert set(spot["final_label"]) <= set(C.SPOTCHECK_LABELS)


def test_all_four_calibration_items_were_caught(spot):
    """31, 117, 239 and 294 are known `false_pass_removed` items seeded into
    the sample. Missing any of them would mean the human pass cannot detect
    the thing it was built to detect, and the rate it produces could not be
    trusted."""
    cal = spot[spot["known_false_pass"]]
    assert len(cal) == 4
    assert set(cal["item"]) == {31, 117, 239, 294}
    assert (cal["final_label"] == "extraction_issue").all()


def test_two_in_five_correct_verdicts_were_not_earned(spot):
    """The headline of the second pass, and the reason the one-sided figure
    could not stand."""
    fp = C.false_pass_rate(spot)
    assert fp["n_false_pass"] == 16
    assert fp["n_true_correct"] == 23
    assert fp["false_pass_rate"] == pytest.approx(0.40, abs=0.001)
    assert fp["ci_low"] == pytest.approx(0.263, abs=0.01)
    assert fp["ci_high"] == pytest.approx(0.554, abs=0.01)


def test_the_two_corrections_very_nearly_cancel(run, audit, spot):
    """The one-sided pass ran accuracy to 71-74%. Removing the false passes
    pulls it back to ~51%, barely above the 47.0% baseline. Reporting only
    the one-sided number would have overstated accuracy by ~20 points --
    which is exactly why the one-sidedness was flagged before it was fixed."""
    b = C.two_sided_accuracy_bounds(run, audit, spot)
    assert b["baseline_accuracy"] == pytest.approx(0.470, abs=0.001)
    assert b["onesided_corrected_accuracy_low"] > 0.70
    assert b["two_sided_accuracy_point"] == pytest.approx(0.514, abs=0.02)
    assert 0.42 < b["two_sided_accuracy_low"] < 0.47
    assert 0.58 < b["two_sided_accuracy_high"] < 0.63
    assert b["two_sided_accuracy_point"] < b["onesided_corrected_accuracy_low"]


def test_false_passes_are_low_entropy_which_is_the_mechanism(run, spot):
    """Why the corrected AUROC falls so far: a false pass is a CONFIDENTLY
    wrong item, and confidently wrong is precisely what entropy cannot flag.
    Mean entropy 0.818 against 0.530 for verified-correct -- higher, but far
    below the 1.37 of a real misread."""
    fp = spot.loc[spot["final_label"] == "extraction_issue", "item"].astype(int)
    tc = spot.loc[spot["final_label"] == "true_correct", "item"].astype(int)
    h_fp = run.loc[fp, "perception_entropy"].mean()
    h_tc = run.loc[tc, "perception_entropy"].mean()
    assert h_fp == pytest.approx(0.818, abs=0.03)
    assert h_tc == pytest.approx(0.530, abs=0.03)
    assert h_fp < 1.0, "well below the 1.37 of a genuine notation misread"


def test_the_unweighted_subset_auroc_is_a_trap(run, audit, spot):
    """After correction each class MIXES both strata, which were sampled at
    100% and 28%. The naive unweighted estimate over-weights the recovered
    high-entropy items and biases the AUROC low. Weighting is the fix, though
    it moves this particular number less than expected."""
    from pilot.plotting import bootstrap_auroc_ci
    d = C.audited_subset_labels(run, audit, spot, unknown_as=False)
    assert set(d["side"]) == {"wrong_side", "correct_side"}
    assert d.loc[d["side"] == "wrong_side", "weight"].eq(1.0).all()
    assert d.loc[d["side"] == "correct_side", "weight"].iloc[0] == pytest.approx(
        141 / 40, abs=0.01)
    cc = d[d["corrected_correct"]]
    assert (cc["side"] == "wrong_side").mean() > 0.70, "the over-weighting"
    naive = bootstrap_auroc_ci(d, "perception_entropy", "corrected_correct",
                               n_boot=1000, seed=0)["auroc"]
    weighted = C.weighted_auroc(d, n_boot=1000, seed=0)["auroc"]
    assert naive == pytest.approx(0.586, abs=0.02)
    assert weighted == pytest.approx(0.572, abs=0.02)


def test_the_corrected_auroc_is_interpretation_dependent_and_not_resolved(
        run, audit, spot):
    """THE UNCOMFORTABLE RESULT, and it must not be quoted as a single number.

    Whether a false pass means "the model was wrong" or "we cannot tell" is a
    judgement the audit did not settle -- some are unambiguous (item 31: 2 cm
    against a handwritten 19.2 cm) while the collapsed-to-one-symbol cases
    (84, 117, 239, 250, 282) leave the model's actual answer unknown. The
    AUROC swings from chance to resolved across that choice.
    """
    fp = set(spot.loc[spot["final_label"] == "extraction_issue",
                      "item"].astype(int))
    d = C.audited_subset_labels(run, audit, spot, unknown_as=False)

    wrong = C.weighted_auroc(d, n_boot=2000, seed=0)
    assert wrong["auroc"] == pytest.approx(0.572, abs=0.03)
    assert wrong["excludes_chance"] is False

    undecidable = C.weighted_auroc(d[~d.index.isin(fp)], n_boot=2000, seed=0)
    assert undecidable["auroc"] == pytest.approx(0.724, abs=0.04)
    assert undecidable["excludes_chance"] is True

    assert undecidable["auroc"] - wrong["auroc"] > 0.10, (
        "the swing across an unsettled interpretive choice is large; report "
        "the RANGE 0.57-0.73, never one endpoint")


SPOT60 = (ROOT / "reference" / "audit"
          / "spotcheck_extra60_qwen_strict_v1_correct_20260812.csv")


def test_the_extension_draw_is_disjoint_and_pools_with_the_first(run, spot):
    """Drawing 40 then 60 from the remaining 101 leaves the UNION a uniform
    random sample of 100 of the 141 correct items, so the two sheets pool
    directly. An overlap would double-count and break that."""
    ext = pd.read_csv(SPOT60)
    assert len(ext) == 60
    assert ext["item"].is_unique
    assert not (set(ext["item"]) & set(spot["item"])), "draws must be disjoint"
    # Shipped uncoded and was coded on 2026-08-12. The assertion is flipped
    # rather than dropped, so the coding stays pinned: 4 false passes against
    # 16 in the first 40, which is the p=0.00007 pass-to-pass disagreement
    # recorded in CLAUDE.md.
    assert ext["final_label"].fillna("").ne("").all(), "has uncoded rows"
    assert set(ext["final_label"]) <= {"true_correct", "extraction_issue",
                                       "needs_visual"}
    assert int((ext["final_label"] == "extraction_issue").sum()) == 4
    assert int((ext["final_label"] == "true_correct").sum()) == 56

    rebuilt = C.correct_item_spot_check_extension(run, spot, n=60)
    assert rebuilt["item"].tolist() == ext["item"].tolist(), "seeded, reproducible"

    scored = C.rescore.rescore_run(run, rule="strict_v1")
    ok = set(scored.index[scored["transcription_correct"].astype(bool)])
    assert set(ext["item"]) <= ok
    assert len(set(ext["item"]) | set(spot["item"])) == 100


def test_the_extension_carries_no_calibration_items(run, spot):
    """Worth knowing before coding it: all 6 known false_pass_removed items
    fall outside this draw, so unlike the first 40 it has no built-in check
    on the coder. Not a defect of the sampling -- a property to state."""
    ext = pd.read_csv(SPOT60)
    assert int(ext["known_false_pass"].sum()) == 0


def test_n100_would_meaningfully_tighten_the_false_pass_interval(spot):
    """The reason to spend the effort. At the observed 40% rate, n=40 gives a
    29-point Wilson interval and n=100 gives 19 -- the difference that could
    separate the 0.57 and 0.73 readings of the corrected AUROC."""
    lo40, hi40 = C._wilson(16, 40)
    lo100, hi100 = C._wilson(40, 100)
    assert (hi40 - lo40) == pytest.approx(0.291, abs=0.01)
    assert (hi100 - lo100) == pytest.approx(0.189, abs=0.01)
    assert (hi100 - lo100) < (hi40 - lo40)


def test_extension_rejects_a_mismatched_existing_sheet(run):
    """Guards the pooling invariant: if the sheet handed in holds items that
    are not strict_v1-correct, the pool it excludes is wrong and the union
    stops being a random sample of the correct items."""
    bogus = pd.DataFrame({"item": [0, 1, 2]})
    with pytest.raises(ValueError, match="non-correct"):
        C.correct_item_spot_check_extension(run, bogus, n=5)


def test_label_views_shows_both_stages(run):
    """The span and the label describe the SAME generation -- the model side
    is the sample that produced the majority label, not sample 0. Getting that
    wrong would show a span from one sample beside a label from another."""
    v = C.label_views(run, 4)
    assert v["item"] == 4
    for k in ("model_span", "model_label", "truth_span", "truth_label",
              "model_tier", "truth_tier", "entropy", "collapsed"):
        assert k in v
    assert str(v["model_label"]).startswith(("sympy:", "text:", "<"))


def test_most_false_passes_are_span_choice_not_sympy_collapse(run, spot):
    """CORRECTS an earlier claim in CLAUDE.md that blamed SymPy. Only 2 of the
    16 false passes are a long span collapsing to one symbol; 6 are a
    one-character SPAN the extractor chose (SymPy encoded it faithfully) and
    8 are a partial span. The fix belongs in extract_final_answer."""
    fp = spot.loc[spot["final_label"] == "extraction_issue", "item"].astype(int)
    short_span = collapse = 0
    for i in fp:
        v = C.label_views(run, int(i))
        lab = str(v["model_label"]).split(":", 1)[-1]
        span = str(v["model_span"] or "")
        if len(lab) <= 3 and len(span) > 12:
            collapse += 1
        elif len(span) <= 12:
            short_span += 1
    assert collapse == 2, "SymPy collapse is the minority mechanism"
    assert short_span == 6
    assert collapse < short_span

    # item 84 was cited as a SymPy collapse; its span is literally "c".
    v84 = C.label_views(run, 84)
    assert str(v84["model_span"]).strip() == "c"
    assert v84["collapsed"] is False
    # item 239 is a genuine collapse: a long span reduced to one symbol.
    v239 = C.label_views(run, 239)
    assert len(str(v239["model_span"])) > 40
    assert v239["model_label"] == "sympy:l"
    assert v239["collapsed"] is True
