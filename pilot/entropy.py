"""Cluster entropy over exact string matches.

This is a simplified stand-in for full semantic entropy, which would cluster
near-duplicate but non-identical strings (e.g. via embedding similarity or NLI
entailment). Here, two samples are only considered the same cluster if they are
identical after light normalization. Good enough to check for a signal in the
pilot; the semantic-clustering upgrade is a later step, not needed now.
"""

import math
from collections import Counter
from typing import Optional, Sequence

_PARSE_FAILURE_SENTINEL = "<PARSE_FAILURE>"


def normalize_string(s: Optional[str]) -> str:
    """Strip whitespace, lowercase, and remove trailing periods.

    None (a parse failure) normalizes to a distinguishable sentinel rather than
    being dropped, since dropping it would understate real instability in the
    K samples.
    """
    if s is None:
        return _PARSE_FAILURE_SENTINEL
    return s.strip().lower().rstrip(".")


def cluster_entropy(samples: Sequence[Optional[str]]) -> float:
    """Shannon entropy (nats) over clusters of exact-match-after-normalization samples.

    Uses natural log to match the semantic-entropy uncertainty-quantification
    literature this is a stand-in for, and to keep units consistent with any
    later extension to token log-probabilities.

    For K=5: H=0 when all samples are identical, H=ln(5)~=1.609 when all 5 are
    distinct. Empty input returns 0.0 by convention.
    """
    if len(samples) == 0:
        return 0.0
    normalized = [normalize_string(s) for s in samples]
    counts = Counter(normalized)
    n = len(normalized)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log(p)
    return entropy


def majority_cluster(samples: Sequence[Optional[str]]) -> tuple[str, int]:
    """Return the (normalized string, count) of the most common cluster.

    Ties are broken by first-encountered order (deterministic in Python's
    Counter/dict). Raises ValueError on empty input, since a majority of
    nothing is not a meaningful pilot-time case.
    """
    if len(samples) == 0:
        raise ValueError("Cannot compute majority cluster of an empty sample list")
    normalized = [normalize_string(s) for s in samples]
    counts = Counter(normalized)
    return counts.most_common(1)[0]
