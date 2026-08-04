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
