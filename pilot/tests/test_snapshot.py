"""Tests for the run-snapshot mechanism.

The snapshot exists so reported numbers outlive the untracked CSV they were
computed from, and so a future run can be diffed against this one. These tests
cover the diffing logic itself; the reference values are asserted in
test_reported_numbers.py.
"""

import json

import pandas as pd
import pytest

from pilot.snapshot import compare_snapshots, snapshot_metrics, write_snapshot


def _fake_run(n=40, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    entropy = rng.uniform(0, 1.6, n)
    has_error = np.arange(n) % 2 == 0
    return pd.DataFrame({
        "has_error": has_error,
        "perception_entropy": entropy,
        "reasoning_entropy": rng.uniform(0, 1.6, n),
        "transcription_correct": entropy < 0.8,
        "grading_correct": rng.uniform(0, 1, n) > 0.5,
        "n_transcription_parse_failures": [0] * n,
        "n_grading_parse_failures": [0] * n,
        "model_id": ["fake/model"] * n,
        "k_transcription": [5] * n,
        "k_grading": [5] * n,
        "handwriting_style": [True] * n,
        "image_quality": [True] * n,
    })


def test_snapshot_captures_every_headline_metric():
    snap = snapshot_metrics(_fake_run(), label="fake", n_boot=200)
    for section in ("perception", "reasoning", "grading", "abstention",
                    "risk_coverage", "conformal", "integrity"):
        assert section in snap
    assert snap["perception"]["auroc"] > 0.5
    assert snap["n_items"] == 40


def test_snapshot_round_trips_through_json(tmp_path):
    """It has to survive JSON, since that is the whole point of persisting it."""
    csv = tmp_path / "run.csv"
    _fake_run().to_csv(csv, index=False)
    out = tmp_path / "snap.json"
    write_snapshot(csv, out, label="fake")
    reloaded = json.loads(out.read_text())
    assert reloaded["label"] == "fake"
    assert reloaded["perception"]["auroc"] == pytest.approx(
        snapshot_metrics(_fake_run(), n_boot=10000)["perception"]["auroc"]
    )


def test_compare_reports_no_change_for_identical_snapshots():
    snap = snapshot_metrics(_fake_run(), n_boot=200)
    diff = compare_snapshots(snap, snap)
    assert diff["changed"] == {}
    assert diff["headline_changed"] == {}


def test_compare_flags_a_headline_move():
    """The case this is built for: a code or environment change silently moves
    a published number. The diff must name it and mark it as headline."""
    old = snapshot_metrics(_fake_run(), n_boot=200)
    new = json.loads(json.dumps(old))
    new["perception"]["auroc"] = old["perception"]["auroc"] - 0.05

    diff = compare_snapshots(old, new)
    assert "perception.auroc" in diff["changed"]
    assert "perception.auroc" in diff["headline_changed"]
    assert diff["changed"]["perception.auroc"]["delta"] == pytest.approx(-0.05)


def test_compare_ignores_differences_below_tolerance():
    old = snapshot_metrics(_fake_run(), n_boot=200)
    new = json.loads(json.dumps(old))
    new["perception"]["auroc"] += 1e-9
    assert compare_snapshots(old, new)["changed"] == {}


def test_compare_reports_added_and_removed_fields():
    """A schema change between runs must be visible, not silently skipped."""
    old = snapshot_metrics(_fake_run(), n_boot=200)
    new = json.loads(json.dumps(old))
    new["perception"]["new_metric"] = 1.0
    del new["grading"]["accuracy"]

    diff = compare_snapshots(old, new)
    assert "perception.new_metric" in diff["added"]
    assert "grading.accuracy" in diff["removed"]


# --- snapshot_grading_metrics ---------------------------------------------

def _grading_run(n_error=150, n_clean=150, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = n_error + n_clean
    return pd.DataFrame({
        "has_error": [1] * n_error + [0] * n_clean,
        "reasoning_entropy": rng.choice([0.0, 0.5, 0.673], size=n),
        "grading_correct": rng.uniform(0, 1, n) > 0.4,
        "n_grading_parse_failures": [0] * n,
        "model_id": ["fake/model"] * n,
        "quantized": [False] * n,
        "k_grading": [5] * n,
    })


def test_grading_snapshot_omits_the_perception_arm():
    """Grading-only runs have no perception columns. The snapshot must leave
    the arm out rather than emit NaNs that later read as a measured zero."""
    from pilot.snapshot import snapshot_grading_metrics

    snap = snapshot_grading_metrics(_grading_run(), label="t")
    assert snap["arm"] == "grading_only"
    assert "perception" not in snap
    assert snap["reasoning"]["error_stratum"] is not None


def test_grading_snapshot_handles_a_missing_stratum_without_crashing():
    """ScratchMath has zero clean items. That must yield 'not_measured' --
    a different statement from 'underpowered' -- not a crash or a NaN."""
    from pilot.snapshot import snapshot_grading_metrics

    snap = snapshot_grading_metrics(_grading_run(n_error=100, n_clean=0), label="all_error")
    assert snap["composition"]["n_clean"] == 0
    assert snap["reasoning"]["clean_stratum"] is None
    assert snap["reasoning"]["clean_stratum_verdict"] == "not_measured"
    assert snap["reasoning"]["error_stratum"] is not None


def test_grading_snapshot_flags_an_unbalanced_pooled_auroc():
    """The pooled trap: on the n=500/n=800 CSVs pooling reads ~0.61 while the
    strata run in opposite directions. The snapshot must say so in-band."""
    from pilot.snapshot import snapshot_grading_metrics

    balanced = snapshot_grading_metrics(_grading_run(150, 150))
    assert balanced["composition"]["is_balanced"] is True
    assert balanced["reasoning"]["pooled_is_misleading_unless_balanced"] is False

    skewed = snapshot_grading_metrics(_grading_run(650, 150))
    assert skewed["composition"]["is_balanced"] is False
    assert skewed["reasoning"]["pooled_is_misleading_unless_balanced"] is True


def test_grading_snapshot_detects_a_degenerate_entropy_distribution():
    """Both InternVL3 (67% pinned at max) and ScratchMath (70% at zero) were
    invalidated by their entropy distribution, not by their CI. The snapshot
    records that independently of the AUROC."""
    from pilot.snapshot import snapshot_grading_metrics

    df = _grading_run(100, 0)
    df["reasoning_entropy"] = [0.0] * 70 + [0.5] * 20 + [0.673] * 10
    snap = snapshot_grading_metrics(df)
    assert snap["entropy_distribution"]["largest_single_value_share"] == pytest.approx(0.70)
    assert snap["entropy_distribution"]["is_degenerate"] is True
    assert snap["entropy_distribution"]["n_distinct_values"] == 3

    spread = _grading_run(100, 0)
    spread["reasoning_entropy"] = ([0.0] * 25 + [0.5] * 25 + [0.673] * 25 + [1.0] * 25)
    assert snapshot_grading_metrics(spread)["entropy_distribution"]["is_degenerate"] is False


def test_grading_snapshot_extra_is_merged_verbatim():
    """Run-specific evidence (ScratchMath's non-engagement rate) has to ride
    along without the schema pretending it generalises."""
    from pilot.snapshot import snapshot_grading_metrics

    snap = snapshot_grading_metrics(_grading_run(), extra={"verdict": "gated", "x": 1})
    assert snap["extra"] == {"verdict": "gated", "x": 1}
    assert "extra" not in snapshot_grading_metrics(_grading_run())
