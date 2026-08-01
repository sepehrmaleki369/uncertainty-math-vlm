import numpy as np
import pytest

from pilot.entropy import PARSE_FAILURE_SENTINEL
from pilot.semantic import (
    embedding_cluster_labels,
    nli_cluster_labels,
    semantic_cluster_entropy,
)

# Label order matches cross-encoder/nli-deberta-v3-small's actual config:
# 0=contradiction, 1=entailment, 2=neutral.
CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2


class FakeNLI:
    """Deterministic stand-in for CrossEncoder -- looks up (premise, hypothesis)
    in a table built from the test's intent, so no model download is needed."""

    def __init__(self, entailing_pairs: set[tuple[str, str]]):
        self.entailing_pairs = entailing_pairs

    def predict(self, pairs):
        rows = []
        for premise, hypothesis in pairs:
            label = ENTAILMENT if (premise, hypothesis) in self.entailing_pairs else CONTRADICTION
            row = np.zeros(3)
            row[label] = 1.0
            rows.append(row)
        return np.array(rows)


class FakeEmbedder:
    """Maps each text to a hand-picked vector so cosine similarity is exact."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts):
        return np.array([self.vectors[t] for t in texts])


def test_nli_cluster_labels_merges_mutual_entailment():
    a, b, c = "the answer is 36", "we get 36", "the answer is 35"
    samples = [a, b, c]
    entailing = {(a, b), (b, a)}  # only a<->b mutually entail
    labels = nli_cluster_labels(samples, _model=FakeNLI(entailing))
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_nli_cluster_labels_requires_bidirectional_entailment():
    """One-directional entailment (a entails b, but not b entails a) must NOT merge --
    that's the whole point of using mutual entailment over a single-direction check."""
    a, b = "x equals 5", "x is a positive number"  # a -> b true, b -> a false
    entailing = {(a, b)}  # only forward
    labels = nli_cluster_labels([a, b], _model=FakeNLI(entailing))
    assert labels[0] != labels[1]


def test_nli_cluster_labels_transitive_merge():
    """A~B and B~C (but A~C not directly in the entailing set) must still merge
    transitively via union-find -- this is the value union-find adds over a
    raw pairwise check."""
    a, b, c = "phrase one", "phrase two", "phrase three"
    entailing = {(a, b), (b, a), (b, c), (c, b)}
    labels = nli_cluster_labels([a, b, c], _model=FakeNLI(entailing))
    assert labels[0] == labels[1] == labels[2]


def test_nli_cluster_labels_parse_failures_isolated():
    samples = [None, "the answer is 36", None]
    labels = nli_cluster_labels(samples, _model=FakeNLI(set()))
    assert labels[0] == PARSE_FAILURE_SENTINEL == labels[2]
    assert labels[1] != PARSE_FAILURE_SENTINEL


def test_nli_cluster_labels_all_distinct_when_no_entailment():
    samples = ["a", "b", "c"]
    labels = nli_cluster_labels(samples, _model=FakeNLI(set()))
    assert len(set(labels)) == 3


def test_embedding_cluster_labels_merges_above_threshold():
    vectors = {"same one": [1.0, 0.0], "same two": [0.99, 0.01], "different": [0.0, 1.0]}
    labels = embedding_cluster_labels(
        list(vectors.keys()), threshold=0.9, _model=FakeEmbedder(vectors)
    )
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_embedding_cluster_labels_respects_threshold_boundary():
    # cos similarity of these two vectors is exactly 0.6
    vectors = {"a": [1.0, 0.0], "b": [0.6, 0.8]}
    labels_loose = embedding_cluster_labels(["a", "b"], threshold=0.5, _model=FakeEmbedder(vectors))
    labels_strict = embedding_cluster_labels(["a", "b"], threshold=0.7, _model=FakeEmbedder(vectors))
    assert labels_loose[0] == labels_loose[1]
    assert labels_strict[0] != labels_strict[1]


def test_semantic_cluster_entropy_all_same_cluster_is_zero():
    entropy = semantic_cluster_entropy(
        ["a", "a variant", "a again"],
        cluster_fn=lambda s: nli_cluster_labels(
            s, _model=FakeNLI({(x, y) for x in s for y in s if x != y})
        ),
    )
    assert entropy == 0.0


def test_semantic_cluster_entropy_empty_input():
    assert semantic_cluster_entropy([], cluster_fn=lambda s: []) == 0.0
