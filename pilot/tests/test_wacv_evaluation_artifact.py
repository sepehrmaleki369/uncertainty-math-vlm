"""Locks the releasable artifact: manifests, hashes, audit long form, rater gate.

Offline, no network, no Hugging Face login, no model. The risks pinned here are
release risks rather than arithmetic ones: leaking gated text or secrets,
claiming a reconstruction that was never performed, silently collapsing the
audit overlaps that carry the only reliability evidence, and manufacturing an
inter-rater number from anything other than a second human.
"""

import os

import pandas as pd
import pytest

import pilot.second_rater as SR
import pilot.wacv_artifact as W

RUN_CSV = "results/scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(RUN_CSV),
                                reason="frozen run CSV not downloaded")


@pytest.fixture(scope="module")
def run():
    d = pd.read_csv(RUN_CSV)
    assert len(d) == 300
    return d


@pytest.fixture(scope="module")
def audit_long():
    return W.audit_labels_long()


@pytest.fixture(scope="module")
def manifests(run, audit_long):
    align = W.verify_reconstruction(run)          # offline -> PENDING
    rev = {"dataset_id": W.DATASET_ID, "revision": "deadbeef", "gated": "auto"}
    return W.build_manifests(run, RUN_CSV, audit_long, rev, align)


# --- fail closed -----------------------------------------------------------

def test_offline_reconstruction_refuses_to_claim_verification(run):
    """FERMAT is gated. Without an authenticated download there is nothing to
    align against, and a manifest that claimed otherwise would be worse than
    no manifest."""
    out = W.verify_reconstruction(run)
    assert out["alignment_status"] == W.PENDING
    assert out["verified"] is False
    assert out["n_aligned"] == 0
    assert "authenticat" in out["reason"].lower()


def test_a_partial_alignment_is_a_failure_not_a_success(run):
    """A drifted revision produces a PARTIAL match, which is precisely the
    outcome that must not be accepted."""
    src = run.iloc[:250][["orig_q", "pert_a", "has_error"]].copy()
    out = W.verify_reconstruction(run, source=src, revision="abc123")
    assert out["alignment_status"] == "FAILED"
    assert out["verified"] is False
    assert out["n_aligned"] == 250 and out["n_expected"] == 300
    assert "guess" in out["reason"]


def test_a_full_alignment_verifies(run):
    src = run[["orig_q", "pert_a", "has_error"]].copy()
    out = W.verify_reconstruction(run, source=src, revision="abc123")
    assert out["verified"] is True and out["n_aligned"] == 300


def test_an_order_insensitive_match_is_rejected(run):
    """Finding all 300 contents somewhere is insufficient: the declared
    sampler must reproduce the exact row order used by the frozen run."""
    src = run[["orig_q", "pert_a", "has_error"]].iloc[::-1].reset_index(drop=True)
    out = W.verify_reconstruction(run, source=src, revision="abc123")
    assert out["verified"] is False
    assert out["alignment_status"] == "FAILED"


def test_reconstruction_exactly_mirrors_the_frozen_balanced_loader():
    """Stage B must reproduce the selection code, not merely find all items.

    The source-row metadata may be added, but the ordered selected contents
    must be byte-for-byte the same as ``load_fermat_balanced``.
    """
    from datasets import Dataset
    import pilot.data as D

    source = Dataset.from_dict({
        "orig_q": [f"question {i}" for i in range(24)],
        "pert_a": [f"answer {i}" for i in range(24)],
        "has_error": [i % 3 != 0 for i in range(24)],
    })
    expected = D.load_fermat_balanced(
        n=12, seed=42, target_error_frac=0.5,
        _loader=lambda *args, **kwargs: source,
    ).to_pandas()
    actual = W.reconstruct_balanced_source(
        source, n=12, seed=42, target_error_frac=0.5)

    cols = ["orig_q", "pert_a", "has_error"]
    pd.testing.assert_frame_equal(actual[cols], expected[cols])
    assert actual["source_row_index"].is_unique
    for row in actual.itertuples(index=False):
        i = int(row.source_row_index)
        assert row.orig_q == source[i]["orig_q"]
        assert row.pert_a == source[i]["pert_a"]


def test_frozen_generation_temperature_is_recorded_exactly():
    assert W.RUN_PROTOCOL["temperature"] == 0.7


def test_public_manifest_records_the_frozen_temperature(manifests):
    public, _ = manifests
    assert public["temperature"].eq(0.7).all()


def test_the_revision_is_pinnable_without_auth_even_though_data_is_not():
    """Metadata and data have different access levels here; the manifest can
    pin the commit offline. Tolerates a network-free environment."""
    rev = W.dataset_revision(offline_ok=True)
    assert rev["dataset_id"] == W.DATASET_ID
    assert rev["revision"] is None or len(rev["revision"]) == 40


# --- release safety --------------------------------------------------------

def test_the_public_manifest_carries_no_gated_text(manifests):
    public, private = manifests
    assert "orig_q" not in public.columns and "pert_a" not in public.columns
    assert "all_transcription_samples_raw" not in public.columns
    assert {"orig_q", "pert_a"} <= set(private.columns)
    assert W.assert_public_manifest_is_safe(public, private)["safe"] is True


def test_public_columns_are_an_allow_list_not_a_deny_list(manifests):
    public, private = manifests
    assert list(public.columns) == list(W.PUBLIC_COLUMNS)
    leaky = public.copy()
    leaky["orig_q"] = "gated text"
    with pytest.raises(AssertionError, match="non-allow-listed|gated content"):
        W.assert_public_manifest_is_safe(leaky, private)


@pytest.mark.parametrize("leak", ["/content/drive/MyDrive/x", "hf_ABCDEFGH",
                                  "/Users/someone/secret"])
def test_secrets_and_local_paths_are_refused(manifests, leak):
    public, private = manifests
    bad = public.copy()
    bad.loc[bad.index[0], "audit_selection_reason"] = leak
    with pytest.raises(AssertionError, match="contains"):
        W.assert_public_manifest_is_safe(bad, private)


def test_the_manifest_hashes_the_content_not_just_the_index(manifests):
    """`shuffle(seed=42)` is NOT stable across dataset revisions -- this
    project already had two items overlap between draws that should have been
    disjoint. An index alone therefore does not reproduce the sample, so the
    public manifest publishes per-item content hashes."""
    public, _ = manifests
    assert public["item_content_sha256"].nunique() == len(public)
    assert public["question_sha256"].str.len().eq(64).all()


def test_prompts_are_hashed_from_the_literal_not_paraphrased():
    import pilot.prompts as P
    h = W.prompt_hashes()
    assert h["transcription_user"]["sha256"] == W.sha256_text(P.TRANSCRIPTION_USER_PROMPT)
    assert h["grading_user"]["n_chars"] == len(P.GRADING_USER_PROMPT)


# --- the audit long form ---------------------------------------------------

def test_the_audit_invariants_hold(audit_long):
    """312 rows, 234 unique items, 78 overlaps. Collapsing the overlaps would
    destroy the only intra-rater evidence this project has."""
    inv = W.audit_invariants(audit_long)
    assert inv["n_rows"] == 312
    assert inv["n_unique_items"] == 234
    assert inv["n_overlap_rows"] == 78


def test_the_four_hard_contradictions_stay_visible_and_unadjudicated(audit_long):
    inv = W.audit_invariants(audit_long)
    assert inv["hard_contradictions"] == [108, 149, 222, 230]
    for item in inv["hard_contradictions"]:
        rows = audit_long[audit_long["item_id"] == item]
        assert len(rows) >= 2, item
        assert {"correct", "wrong"} <= set(rows["truth"]), item


def test_the_artifact_states_it_is_not_a_population_estimate(audit_long):
    inv = W.audit_invariants(audit_long)
    assert inv["is_population_estimate"] is False
    assert "not estimates for all" in inv["coverage_note"]


# --- second-rater packet ---------------------------------------------------

@pytest.fixture(scope="module")
def queue(manifests, audit_long, run):
    public, _ = manifests
    return SR.build_queue(public, audit_long, run)


def test_core_and_challenge_are_disjoint_and_the_queue_covers_the_union(queue, audit_long):
    core = set(queue.loc[queue["queue"] == "core", "item_id"])
    chal = set(queue.loc[queue["queue"] == "challenge", "item_id"])
    assert len(core) == SR.N_CORE
    assert chal == set(SR.CHALLENGE_ITEMS)
    assert not (core & chal), "a known contradiction must never enter the core"
    assert set(queue["item_id"]) == set(audit_long["item_id"])


def test_the_queue_is_deterministic(manifests, audit_long, run):
    public, _ = manifests
    a = SR.build_queue(public, audit_long, run)
    b = SR.build_queue(public, audit_long, run)
    assert list(a["item_id"]) == list(b["item_id"])
    assert list(a["review_id"]) == list(b["review_id"])


def test_every_shipped_answer_field_is_blank(queue):
    t = SR.blank_template(queue)
    assert SR.all_answers_blank(t)
    for f in SR.ANSWER_FIELDS:
        assert (t[f].astype(str) == "").all(), f


def test_the_template_leaks_no_blinded_field(queue):
    """The rater must not see the first label, the automatic verdict, entropy,
    has_error, or whether an item is a challenge case."""
    t = SR.blank_template(queue)
    assert not (set(SR.BLINDED_FIELDS) & set(t.columns))
    assert "item_id" not in t.columns, "the working copy is keyed by review_id"


def test_writing_a_template_with_answers_is_refused(queue, tmp_path):
    t = SR.blank_template(queue)
    t.loc[0, "model_correctness"] = "correct"
    with pytest.raises(AssertionError, match="refusing"):
        SR.write_packet(str(tmp_path), queue, t)


def test_agreement_is_refused_until_a_real_file_exists(tmp_path):
    """THE GATE. No completed file, no agreement, no kappa, no claim."""
    out = SR.agreement_or_pending(str(tmp_path))
    assert out["state"] == "PENDING"
    assert "SECOND-RATER STATUS: PENDING" in out["message"]
    assert "kappa" in out["message"]


def test_a_malformed_return_is_rejected(queue, tmp_path):
    t = SR.blank_template(queue)
    t["model_correctness"] = "definitely_correct"        # not in the vocabulary
    p = tmp_path / "second_rater_completed.csv"
    t.to_csv(p, index=False)
    v = SR.validate_completed(str(p), queue)
    assert v["ok"] is False
    assert any("vocabulary" in s for s in v["problems"])


def test_an_all_blank_return_is_rejected(queue, tmp_path):
    p = tmp_path / "second_rater_completed.csv"
    SR.blank_template(queue).to_csv(p, index=False)
    v = SR.validate_completed(str(p), queue)
    assert v["ok"] is False
    assert any("no rows carry" in s for s in v["problems"])


def test_indeterminate_is_a_first_class_answer():
    """`extraction_issue` and `indeterminate` must never be convertible to
    model-wrong; they are in the vocabulary precisely so a rater can decline."""
    assert "indeterminate" in SR.MODEL_CORRECTNESS
    assert "extraction_issue" in SR.FAILURE_CATEGORY
    assert "not_applicable" in SR.FAILURE_CATEGORY
    text = SR.rater_instructions()
    assert "not" in text.lower() and "indeterminate" in text


def test_the_core_allocation_does_not_depend_on_row_order(manifests, audit_long, run):
    """CROSS-PLATFORM REPRODUCIBILITY. 18 of the 120 core seats are handed out
    by largest remainder, the remainder takes only 13 distinct values, and the
    cut falls inside an eight-way tie. With pandas' default (unstable)
    quicksort the tie order followed the pandas version, so Colab and a local
    machine produced different cores from the same seed and the queue file
    hash differed. Shuffling the input rows simulates a different groupby
    order; the core must not move."""
    public, _ = manifests
    a = SR.build_queue(public, audit_long, run)
    shuffled = audit_long.sample(frac=1.0, random_state=7).reset_index(drop=True)
    b = SR.build_queue(public, shuffled, run)
    core_a = sorted(a.loc[a["queue"] == "core", "item_id"])
    core_b = sorted(b.loc[b["queue"] == "core", "item_id"])
    assert core_a == core_b, (
        f"core moved under a row reorder: {len(set(core_a) ^ set(core_b))} items differ")
    assert len(core_a) == SR.N_CORE
