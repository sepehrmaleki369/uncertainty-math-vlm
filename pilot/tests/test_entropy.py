import math

import pytest

from pilot.entropy import cluster_entropy, majority_cluster, normalize_string


def test_normalize_string_whitespace_and_case():
    assert normalize_string("  X=5.  ") == "x=5"


def test_normalize_string_multiple_trailing_periods():
    assert normalize_string("answer...") == "answer"


def test_normalize_string_none_returns_sentinel():
    assert normalize_string(None) == "<PARSE_FAILURE>"


def test_cluster_entropy_empty_list():
    assert cluster_entropy([]) == 0.0


def test_cluster_entropy_all_identical():
    assert cluster_entropy(["x=5", "x=5", "x=5"]) == 0.0


def test_cluster_entropy_normalization_collapses_variants():
    assert cluster_entropy(["X=5", " x=5 ", "x=5.", "x=5"]) == 0.0


def test_cluster_entropy_all_distinct():
    assert cluster_entropy(["a", "b", "c", "d", "e"]) == pytest.approx(math.log(5))


def test_cluster_entropy_mixed_clusters():
    # 3x "a" + 2x "b" out of 5
    expected = 0.6730116670092565
    assert cluster_entropy(["a", "a", "a", "b", "b"]) == pytest.approx(expected)


def test_cluster_entropy_none_samples_form_their_own_cluster():
    # a:2, <PARSE_FAILURE>:2, b:1 out of 5 -- Nones must not be silently dropped
    expected = 1.0549201679861442
    assert cluster_entropy(["a", None, "a", None, "b"]) == pytest.approx(expected)


def test_majority_cluster_basic():
    result = majority_cluster(["a", "a", "b"])
    assert result == ("a", 2)


def test_majority_cluster_tie_break_is_deterministic_first_encountered():
    result = majority_cluster(["b", "a", "b", "a"])
    assert result == ("b", 2)


def test_majority_cluster_empty_raises():
    with pytest.raises(ValueError):
        majority_cluster([])
