"""Semantic clustering for perception entropy: NLI entailment and embedding similarity.

pilot.canonicalize fixes syntactic noise (LaTeX spacing, bracket nesting) but
can't merge two samples that paraphrase the same derivation differently ("we
get 36" vs "the result equals thirty six") -- on the real pilot run, that left
83/100 items at maximum entropy even after canonicalization. These two
functions are the actual fix, calibrated against real model output (see
module-level tests): bidirectional NLI entailment (the technique semantic
entropy literature actually uses) and, as a cheaper alternative, embedding
cosine similarity above a threshold.

Both take the whole batch of K samples for one item at once, unlike
pilot.canonicalize's per-sample functions -- clustering by similarity is
inherently a batch operation (every sample compared against every other),
not a normalization you can apply to one string in isolation.
"""

import itertools
from typing import Callable, Optional, Sequence

from pilot.canonicalize import structural_clean
from pilot.entropy import PARSE_FAILURE_SENTINEL, entropy_from_labels

# Loaded lazily (first call only) since these are ~100-500MB downloads and
# slow to import -- tests inject a fake model and never touch this.
_embedding_model = None
_nli_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


class _UnionFind:
    """Minimal disjoint-set for merging samples into connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def _prepare_batch(samples: Sequence[Optional[str]]) -> tuple[list[Optional[int]], list[str]]:
    """Split a batch into (per-position index into real_texts, or None for a
    parse failure) and the list of cleaned real texts to actually cluster.

    Parse failures never enter the NLI/embedding comparison -- they all share
    PARSE_FAILURE_SENTINEL as their label directly, consistent with
    pilot.canonicalize.canonicalize_math's handling of None.
    """
    position_to_real_idx: list[Optional[int]] = []
    real_texts: list[str] = []
    for s in samples:
        if s is None:
            position_to_real_idx.append(None)
            continue
        cleaned = structural_clean(s)
        if not cleaned:
            position_to_real_idx.append(None)
            continue
        position_to_real_idx.append(len(real_texts))
        real_texts.append(cleaned)
    return position_to_real_idx, real_texts


def _labels_from_union_find(
    position_to_real_idx: Sequence[Optional[int]], uf: Optional[_UnionFind]
) -> list[str]:
    labels = []
    for idx in position_to_real_idx:
        if idx is None:
            labels.append(PARSE_FAILURE_SENTINEL)
        else:
            labels.append(f"sem_{uf.find(idx)}")
    return labels


def nli_cluster_labels(
    samples: Sequence[Optional[str]],
    _model=None,
) -> list[str]:
    """Cluster samples by bidirectional NLI entailment (mutual entailment = same cluster).

    Follows the semantic-entropy literature's definition: A and B are the
    same meaning if A entails B AND B entails A, checked with a small
    cross-encoder NLI model. Transitively merged via union-find, so if A~B
    and B~C then A, B, C all share one cluster even if A and C weren't
    directly compared as mutually entailing.
    """
    model = _model if _model is not None else _get_nli_model()
    position_to_real_idx, texts = _prepare_batch(samples)
    uf = _UnionFind(len(texts))

    if len(texts) > 1:
        pairs = list(itertools.combinations(range(len(texts)), 2))
        forward = model.predict([(texts[i], texts[j]) for i, j in pairs])
        backward = model.predict([(texts[j], texts[i]) for i, j in pairs])
        # cross-encoder/nli-deberta-v3-small label order: 0=contradiction,
        # 1=entailment, 2=neutral (verified against the model's own config).
        for (i, j), f_scores, b_scores in zip(pairs, forward, backward):
            if f_scores.argmax() == 1 and b_scores.argmax() == 1:
                uf.union(i, j)

    return _labels_from_union_find(position_to_real_idx, uf)


def embedding_cluster_labels(
    samples: Sequence[Optional[str]],
    threshold: float = 0.85,
    _model=None,
) -> list[str]:
    """Cluster samples by cosine similarity of sentence embeddings above threshold.

    Cheaper and cruder than NLI: a single similarity threshold instead of a
    calibrated entailment judgment, so it's more likely to conflate "close
    but numerically different" answers (see this module's tests for the
    calibration this threshold was picked against).
    """
    model = _model if _model is not None else _get_embedding_model()
    position_to_real_idx, texts = _prepare_batch(samples)
    uf = _UnionFind(len(texts))

    if len(texts) > 1:
        embeddings = model.encode(texts)
        for i, j in itertools.combinations(range(len(texts)), 2):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                uf.union(i, j)

    return _labels_from_union_find(position_to_real_idx, uf)


def _cosine_similarity(a, b) -> float:
    import numpy as np

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_cluster_entropy(
    samples: Sequence[Optional[str]],
    cluster_fn: Callable[[Sequence[Optional[str]]], list[str]] = nli_cluster_labels,
) -> float:
    """Entropy over a semantic clustering of samples (NLI by default)."""
    return entropy_from_labels(cluster_fn(samples))
