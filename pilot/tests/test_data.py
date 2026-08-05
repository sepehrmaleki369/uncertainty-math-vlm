import datasets
import pytest

from pilot.data import (
    FERMAT_FIELDS,
    fermat_census,
    load_fermat_balanced,
    load_fermat_extra_error_items,
    load_fermat_sample,
)


def _make_fake_dataset(n_rows: int) -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "image": [f"image_{i}" for i in range(n_rows)],
            "orig_q": [f"question_{i}" for i in range(n_rows)],
            "pert_a": [f"answer_{i}" for i in range(n_rows)],
            "has_error": [i % 2 for i in range(n_rows)],
            "handwriting_style": ["cursive"] * n_rows,
            "image_quality": ["good"] * n_rows,
            "id": [f"id_{i}" for i in range(n_rows)],
            "source": ["textbook"] * n_rows,
        }
    )


def _fake_loader_factory(n_rows: int):
    def _fake_loader(name: str, split: str = "train"):
        return _make_fake_dataset(n_rows)

    return _fake_loader


def test_fermat_fields_sanity():
    assert FERMAT_FIELDS == [
        "image",
        "orig_q",
        "pert_a",
        "has_error",
        "handwriting_style",
        "image_quality",
    ]


def test_extra_columns_removed():
    result = load_fermat_sample(n=5, seed=1, _loader=_fake_loader_factory(100))
    assert set(result.column_names) == set(FERMAT_FIELDS)


def test_n_respected():
    result = load_fermat_sample(n=25, seed=1, _loader=_fake_loader_factory(100))
    assert len(result) == 25


def test_same_seed_reproducible():
    loader = _fake_loader_factory(100)
    result_a = load_fermat_sample(n=10, seed=7, _loader=loader)
    result_b = load_fermat_sample(n=10, seed=7, _loader=loader)
    assert result_a["orig_q"] == result_b["orig_q"]


def test_different_seeds_differ():
    loader = _fake_loader_factory(100)
    result_a = load_fermat_sample(n=10, seed=1, _loader=loader)
    result_b = load_fermat_sample(n=10, seed=2, _loader=loader)
    assert result_a["orig_q"] != result_b["orig_q"]


def test_n_larger_than_dataset_returns_all_without_crashing():
    result = load_fermat_sample(n=25, seed=1, _loader=_fake_loader_factory(10))
    assert len(result) == 10


# --- fermat_census / load_fermat_balanced ---


def _imbalanced_dataset(n_error: int, n_clean: int) -> datasets.Dataset:
    n = n_error + n_clean
    return datasets.Dataset.from_dict(
        {
            "image": [f"image_{i}" for i in range(n)],
            "orig_q": [f"question_{i}" for i in range(n)],
            "pert_a": [f"answer_{i}" for i in range(n)],
            "has_error": [True] * n_error + [False] * n_clean,
            "handwriting_style": ["cursive"] * n,
            "image_quality": ["good"] * n,
            "id": [f"id_{i}" for i in range(n)],
        }
    )


def _imbalanced_loader(n_error: int, n_clean: int):
    def _loader(name: str, split: str = "train"):
        return _imbalanced_dataset(n_error, n_clean)

    return _loader


def test_fermat_census_reports_composition():
    """Reproduces the real FERMAT shape that made the grading task degenerate:
    87/13 on has_error, where a constant predictor scores 0.87."""
    census = fermat_census(_imbalanced_dataset(87, 13))
    assert census["n_total"] == 100
    assert census["n_error"] == 87 and census["n_clean"] == 13
    assert census["frac_error"] == pytest.approx(0.87)
    assert census["max_balanced_n"] == 26  # the clean pool is the binding limit


def test_load_fermat_balanced_hits_target_when_pools_suffice():
    result = load_fermat_balanced(
        n=100, seed=1, target_error_frac=0.5, _loader=_imbalanced_loader(500, 500)
    )
    assert len(result) == 100
    assert sum(bool(x) for x in result["has_error"]) == 50


def test_load_fermat_balanced_respects_non_half_target():
    result = load_fermat_balanced(
        n=100, seed=1, target_error_frac=0.7, _loader=_imbalanced_loader(500, 500)
    )
    assert sum(bool(x) for x in result["has_error"]) == 70


def test_load_fermat_balanced_preserves_ratio_when_pool_too_small(caplog):
    """The failure that must not happen silently: too few clean items and the
    function returns a differently-balanced sample, reintroducing the exact
    confound it exists to remove. It shrinks n and warns instead."""
    with caplog.at_level("WARNING"):
        result = load_fermat_balanced(
            n=300, seed=1, target_error_frac=0.5, _loader=_imbalanced_loader(500, 40)
        )
    n_error = sum(bool(x) for x in result["has_error"])
    assert len(result) == 80  # capped by the 40-item clean pool, ratio preserved
    assert n_error == 40
    assert "Cannot fill" in caplog.text


def test_load_fermat_balanced_shuffles_so_partial_runs_stay_mixed():
    """Stratified selection concatenates error items then clean items. Without
    a final shuffle, a run interrupted halfway would have processed only error
    items -- exactly the imbalance the balancing is meant to prevent."""
    result = load_fermat_balanced(
        n=100, seed=1, target_error_frac=0.5, _loader=_imbalanced_loader(500, 500)
    )
    first_half = [bool(x) for x in result["has_error"][:50]]
    # A perfectly ordered selection would be all-True here.
    assert 0 < sum(first_half) < 50


def test_load_fermat_balanced_drops_extra_columns():
    result = load_fermat_balanced(
        n=20, seed=1, _loader=_imbalanced_loader(100, 100)
    )
    assert set(result.column_names) == set(FERMAT_FIELDS)


def test_load_fermat_balanced_reproducible_for_a_seed():
    kwargs = dict(n=40, target_error_frac=0.5, _loader=_imbalanced_loader(200, 200))
    a = load_fermat_balanced(seed=7, **kwargs)
    b = load_fermat_balanced(seed=7, **kwargs)
    assert a["orig_q"] == b["orig_q"]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_load_fermat_balanced_rejects_invalid_target(bad):
    with pytest.raises(ValueError, match="target_error_frac"):
        load_fermat_balanced(n=10, target_error_frac=bad, _loader=_imbalanced_loader(50, 50))


# --- load_fermat_extra_error_items ---


def test_load_fermat_extra_error_items_disjoint_from_prior_balanced_run():
    """The whole point: items drawn here must never overlap the has_error=1
    items a prior load_fermat_balanced call already used with the same seed."""
    loader = _imbalanced_loader(500, 500)
    prior = load_fermat_balanced(n=300, seed=7, target_error_frac=0.5, _loader=loader)
    prior_error_qs = {q for q, e in zip(prior["orig_q"], prior["has_error"]) if e}
    assert len(prior_error_qs) == 150

    extra = load_fermat_extra_error_items(n_extra=100, seed=7, skip=150, _loader=loader)
    extra_qs = set(extra["orig_q"])

    assert len(extra) == 100
    assert all(bool(x) for x in extra["has_error"])
    assert extra_qs.isdisjoint(prior_error_qs)


def test_load_fermat_extra_error_items_is_the_immediate_next_slice():
    """Not just disjoint from the prior run -- specifically the NEXT items in
    the same seeded ordering, so a second extension call (skip=350) would
    continue seamlessly with no gap."""
    loader = _imbalanced_loader(500, 500)
    first = load_fermat_extra_error_items(n_extra=50, seed=7, skip=150, _loader=loader)
    second = load_fermat_extra_error_items(n_extra=50, seed=7, skip=200, _loader=loader)
    assert set(first["orig_q"]).isdisjoint(set(second["orig_q"]))
    combined_via_one_call = load_fermat_extra_error_items(
        n_extra=100, seed=7, skip=150, _loader=loader
    )
    assert set(first["orig_q"]) | set(second["orig_q"]) == set(combined_via_one_call["orig_q"])


def test_load_fermat_extra_error_items_reproducible_for_a_seed():
    loader = _imbalanced_loader(500, 500)
    a = load_fermat_extra_error_items(n_extra=40, seed=3, skip=100, _loader=loader)
    b = load_fermat_extra_error_items(n_extra=40, seed=3, skip=100, _loader=loader)
    assert a["orig_q"] == b["orig_q"]


def test_load_fermat_extra_error_items_shrinks_and_warns_when_pool_runs_short(caplog):
    loader = _imbalanced_loader(180, 500)  # only 180 error items exist total
    with caplog.at_level("WARNING"):
        result = load_fermat_extra_error_items(n_extra=100, seed=7, skip=150, _loader=loader)
    assert len(result) == 30  # 180 - 150 remaining
    assert "Returning 30" in caplog.text


def test_load_fermat_extra_error_items_all_positive_class():
    loader = _imbalanced_loader(500, 500)
    result = load_fermat_extra_error_items(n_extra=60, seed=7, skip=150, _loader=loader)
    assert all(bool(x) for x in result["has_error"])


def test_load_fermat_extra_error_items_drops_extra_columns():
    loader = _imbalanced_loader(500, 500)
    result = load_fermat_extra_error_items(n_extra=20, seed=7, skip=150, _loader=loader)
    assert set(result.column_names) == set(FERMAT_FIELDS)
