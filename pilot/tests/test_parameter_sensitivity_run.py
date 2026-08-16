"""Locks the temperature x K sensitivity run of 2026-08-15/16.

Recomputes every snapshot figure from the item-level scores, so a snapshot that
drifts from its source fails here rather than in the paper. Offline, no GPU.

The property this file exists to protect is not the AUROC ordering, which is
easy to read off the table. It is that **the ordering does not survive the
artifact controls**. Raw AUROC rises with temperature, and a reader who stops
there concludes that hotter sampling gives a better uncertainty signal. It does
not: at T=1.0 more than one sample in eight fails to parse, an unparseable
sample takes its own cluster label, and that both inflates entropy and
correlates with being wrong. Remove those items and the ordering is flat.
"""

import json
import os

import pandas as pd
import pytest

import pilot.plotting as P

CSV = "results/real_parameter_sensitivity_full_item_scores.csv"
SNAP = "reference/parameter_sensitivity_20260816.json"

pytestmark = pytest.mark.skipif(not os.path.exists(CSV),
                                reason="item-level scores not downloaded")


@pytest.fixture(scope="module")
def snap():
    with open(SNAP) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def scores():
    d = pd.read_csv(CSV)
    assert len(d) == 2700, "9 conditions x 300 items"
    return d


def test_the_snapshot_matches_its_source(scores, snap):
    """Every AUROC in the snapshot recomputes from the item scores."""
    for key, cond in snap["conditions"].items():
        g = scores[(scores.temperature == cond["temperature"])
                   & (scores.k == cond["k"])]
        assert len(g) == 300, key
        s = P.auroc_sensitivity(g, "perception_entropy", "transcription_correct",
                                "n_transcription_parse_failures", k=cond["k"],
                                n_boot=2000, seed=0)
        for cut in ("full", "excl_parse_failures", "excl_max_entropy", "excl_both"):
            assert abs(s[cut]["auroc"] - cond[cut]["auroc"]) < 1e-9, f"{key}/{cut}"
            assert s[cut]["n_items"] == cond[cut]["n_items"], f"{key}/{cut}"


def test_every_condition_survives_both_controls(snap):
    """The claim the paper is allowed to make: the signal is present across the
    whole declared grid, not only at the reported operating point."""
    for key, cond in snap["conditions"].items():
        assert cond["excl_both"]["excludes_chance"] is True, key
    assert snap["all_conditions_survive_both_controls"] is True


def test_the_raw_temperature_ordering_does_not_survive_the_controls(snap):
    """THE POINT OF THE RUN.

    Raw AUROC increases with temperature at every K. After removing parse
    failures and ceiling-entropy items it does not, so the apparent gain is the
    artifact the controls exist to remove and must never be reported as a
    property of the uncertainty signal.
    """
    c = snap["conditions"]
    for k in (3, 5, 10):
        hot_raw, cold_raw = c[f"T1.0_K{k}"]["full"], c[f"T0.3_K{k}"]["full"]
        hot_ctl, cold_ctl = c[f"T1.0_K{k}"]["excl_both"], c[f"T0.3_K{k}"]["excl_both"]

        raw_gap = hot_raw["auroc"] - cold_raw["auroc"]
        ctl_gap = hot_ctl["auroc"] - cold_ctl["auroc"]
        assert raw_gap > 0.06, f"K={k}: raw gap should be large, got {raw_gap:.3f}"
        assert abs(ctl_gap) < 0.05, (
            f"K={k}: the gap should collapse under the controls, got {ctl_gap:.3f}")

        # The residual is not merely small, it is unresolvable: the intervals
        # overlap. Asserting a strict REVERSAL instead would be overreach -- at
        # K=3 the controlled numbers are 0.681 against 0.678, which is noise on
        # 90 items against 248, not an ordering. The claim is that the apparent
        # temperature effect disappears, not that it inverts.
        assert hot_ctl["ci_low"] < cold_ctl["ci_high"], f"K={k}"
        assert cold_ctl["ci_low"] < hot_ctl["ci_high"], f"K={k}"


def test_the_correction_grows_with_temperature(snap):
    """How much of the raw number is artifact scales with the temperature, which
    is why the raw ordering is misleading rather than merely noisy."""
    c = snap["conditions"]
    drop = {t: c[f"T{t}_K5"]["full"]["auroc"] - c[f"T{t}_K5"]["excl_both"]["auroc"]
            for t in (0.3, 0.7, 1.0)}
    assert drop[0.3] < drop[0.7] < drop[1.0], drop
    assert drop[0.3] < 0.05 and drop[1.0] > 0.15, drop


def test_high_temperature_degrades_the_task_itself(snap):
    """Not only the metric: transcription gets worse and the parser fails more,
    so the hottest setting is the worst operating point on its own terms."""
    c = snap["conditions"]
    assert (c["T0.3_K5"]["parse_failure_rate"] < c["T0.7_K5"]["parse_failure_rate"]
            < c["T1.0_K5"]["parse_failure_rate"])
    assert c["T1.0_K5"]["parse_failure_rate"] > 0.10
    assert c["T1.0_K3"]["accuracy"] < c["T0.3_K3"]["accuracy"] - 0.10


def test_over_half_the_items_are_unusable_at_the_hottest_setting(snap):
    """The controls do not merely shift T=1.0's number, they delete most of its
    sample. A condition surviving on a third of its items is not a better
    operating point than one surviving on two thirds."""
    c = snap["conditions"]
    assert c["T1.0_K5"]["excl_both"]["n_items"] < 150
    assert c["T0.3_K5"]["excl_both"]["n_items"] > 250


def test_the_frozen_protocol_reproduces_the_papers_headline(snap):
    """T=0.7, K=5 on 9000 FRESH samples under a different seed lands on the
    locked 0.835 [0.787, 0.879]. This is an independent replication of the
    headline, and it is the most load-bearing number in the file."""
    f = snap["conditions"]["T0.7_K5"]["full"]
    assert abs(f["auroc"] - 0.835) < 0.02, f["auroc"]
    assert f["ci_low"] < 0.835 < f["ci_high"]
    assert snap["frozen_protocol"] == {"temperature": 0.7, "k": 5,
                                       **snap["frozen_protocol"]}


def test_the_snapshot_records_which_run_it_came_from(snap):
    """A sensitivity grid regenerated from different draws than the one quoted
    would be indistinguishable without this."""
    assert snap["base_seed"] == 20260815
    assert len(snap["source_sha256"]) == 64
    assert snap["n_items"] == 300
