"""Locks the binary-stratification null. Offline, CPU-only, no model.

This is a STRESS TEST, not a measurement: the simulated responder has no
item-level information at all, so every AUROC it attains is voting arithmetic.
The tests exist to stop that being misread as a finding, and to pin the
identity the whole argument rests on.
"""

import numpy as np
import pandas as pd
import pytest

import pilot.plotting as P
import pilot.stratification_null as SN


# --- the identity ----------------------------------------------------------

def test_correctness_is_a_relabelling_of_the_prediction():
    """THE PROPOSITION, checked elementwise rather than asserted in prose.
    y=1 => C = M;  y=0 => C = 1 - M."""
    d = SN.collapse_identity(k=5, p=0.8, n_items=256, seed=3)
    assert d["y1_correct_equals_prediction"] is True
    assert d["y0_correct_equals_complement"] is True


@pytest.mark.parametrize("p,k", [(0.7, 5), (0.843, 5), (0.6, 3), (0.9, 15)])
def test_the_two_strata_sum_to_one(p, k):
    """The sign reversal is the same identity with the label flipped, so the
    y=0 and y=1 AUROCs are exact complements. This reproduces the reported
    'the two stratum AUROCs sum to 1' analytically."""
    a1 = SN.exact_stratum_auroc(p, k, 1)["auroc"]
    a0 = SN.exact_stratum_auroc(p, k, 0)["auroc"]
    assert a1 + a0 == pytest.approx(1.0, abs=1e-12)


def test_p_one_half_is_exactly_chance():
    """With an unbiased responder there is no majority to favour either
    stratum, so the manufactured effect vanishes."""
    for k in SN.DEFAULT_K:
        assert SN.exact_stratum_auroc(0.5, k, 1)["auroc"] == pytest.approx(0.5, abs=1e-12)


def test_the_effect_grows_with_bias():
    """Monotone in p above 0.5: the more one-sided the responder, the larger
    the artifact. This is what makes a strongly biased model most vulnerable."""
    vals = [SN.exact_stratum_auroc(p, 5, 1)["auroc"]
            for p in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


# --- oracle vs Monte Carlo -------------------------------------------------

@pytest.mark.parametrize("p,k", [(0.6, 5), (0.8, 5), (0.7, 3), (0.85, 9)])
def test_simulation_matches_the_closed_form(p, k):
    sim = SN.simulate_stratum(p, k, 1, n_items=650, n_sims=600, seed=11)
    exact = SN.exact_stratum_auroc(p, k, 1)["auroc"]
    assert sim["auroc_median"] == pytest.approx(exact, abs=0.02)


def test_balanced_pooling_returns_to_chance():
    """The sanity condition. Pooling the two strata cancels the artifact, which
    is why the pooled figure is the one to report and the stratified one is
    not."""
    for p in (0.7, 0.843, 0.9):
        s = SN.simulate_stratum(p, 5, "pooled_balanced", n_items=650,
                                n_sims=500, seed=7)
        assert s["auroc_median"] == pytest.approx(0.5, abs=0.02), (p, s)


# --- determinism and seeding ----------------------------------------------

def test_cells_are_deterministic_and_independently_seeded():
    a = SN.simulate_stratum(0.7, 5, 1, n_items=200, n_sims=50, seed=42)
    b = SN.simulate_stratum(0.7, 5, 1, n_items=200, n_sims=50, seed=42)
    assert a["auroc_median"] == b["auroc_median"]
    assert SN._cell_seed(1, "y=1", 0.7, 5) != SN._cell_seed(1, "y=1", 0.7, 7)
    assert SN._cell_seed(1, "y=1", 0.7, 5) != SN._cell_seed(1, "y=0", 0.7, 5)


def test_adding_a_grid_point_does_not_renumber_existing_cells():
    """Seeds are hashed from coordinates, not from an incrementing counter, so
    extending the grid later cannot silently change cells already reported."""
    before = SN._cell_seed(5, "y=1", 0.85, 5)
    assert SN._cell_seed(5, "y=1", 0.85, 5) == before


# --- refusals --------------------------------------------------------------

def test_even_k_is_refused():
    """An even K makes the majority ill-defined on a tie, and the tie-break
    convention would then drive the result."""
    with pytest.raises(ValueError, match="odd"):
        SN.exact_stratum_auroc(0.7, 4, 1)
    with pytest.raises(ValueError, match="odd"):
        SN.simulate_stratum(0.7, 4, 1, n_sims=2)


def test_one_class_replicates_are_counted_not_replaced_by_chance():
    """At extreme bias every item lands in one correctness class and the AUROC
    is UNDEFINED. Substituting 0.5 would quietly drag the reported median
    toward chance and understate the artifact."""
    s = SN.simulate_stratum(0.99, 15, 1, n_items=30, n_sims=200, seed=5)
    assert s["n_invalid"] > 0
    assert s["n_valid"] + s["n_invalid"] == 200
    assert not np.isnan(s["auroc_median"]) or s["n_valid"] == 0


# --- conventions shared with the rest of the project ----------------------

def test_auroc_uses_the_projects_own_function_and_tie_handling():
    """Ties are the norm here: at K=5 the vote entropy takes 4 distinct values.
    Routing through compute_auroc keeps the average-rank convention identical
    to every other AUROC in this repository."""
    ent = np.array([0.0, 0.5, 0.5, 0.7, 0.7, 0.7])
    cor = np.array([True, True, False, False, True, False])
    df = pd.DataFrame({"e": ent, "c": cor})
    assert SN._auroc(ent, cor) == pytest.approx(P.compute_auroc(df, "e", "c"))


def test_entropy_is_in_nats_and_peaks_at_a_half():
    e = SN.binary_entropy(np.array([0.0, 0.5, 1.0]))
    assert e[0] == 0.0 and e[2] == 0.0
    assert e[1] == pytest.approx(np.log(2))


def test_the_existing_null_helper_is_untouched():
    """`plotting.bias_only_null_auroc` produced a frozen, reported number. This
    module must not alter its behaviour."""
    import inspect
    src = inspect.getsource(P.bias_only_null_auroc)
    assert "stratification_null" not in src
    out = P.bias_only_null_auroc(n_items=200, says_error_rate=0.8, k=5,
                                 n_sims=100, seed=0)
    assert 0.0 <= out["median"] <= 1.0


# --- the three locked cases ------------------------------------------------

#: (name, n, says-error rate, observed stratified AUROC), from
#: test_stratum_degeneracy.py's own frozen table.
LOCKED = (("Qwen-3B", 650, 0.843, 0.854),
          ("Qwen-7B", 648, 0.813, 0.801),
          ("LLaVA-NeXT", 400, 0.780, 0.775))


@pytest.mark.parametrize("name,n,p,observed", LOCKED)
def test_the_null_exceeds_every_observed_stratified_auroc(name, n, p, observed):
    """THE RETRACTION, re-derived from scratch. A responder with no item-level
    information scores HIGHER than the model did, so the observed figure
    evidences nothing about the model."""
    null_median = SN.simulate_stratum(p, 5, 1, n_items=n, n_sims=800,
                                      seed=20260814)["auroc_median"]
    assert null_median > observed, (name, null_median, observed)
