import datasets
import pytest

from pilot.data import FERMAT_FIELDS, load_fermat_sample


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
