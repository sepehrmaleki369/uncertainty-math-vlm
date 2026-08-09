"""Cluster entropy over exact string matches.

This is a simplified stand-in for full semantic entropy, which would cluster
near-duplicate but non-identical strings (e.g. via embedding similarity or NLI
entailment). Here, two samples are only considered the same cluster if they are
identical after light normalization. Good enough to check for a signal in the
pilot; the semantic-clustering upgrade is a later step, not needed now.
"""

import math
from collections import Counter
from typing import Callable, Hashable, Optional, Sequence

PARSE_FAILURE_SENTINEL = "<PARSE_FAILURE>"
_PARSE_FAILURE_SENTINEL = PARSE_FAILURE_SENTINEL  # kept for internal call sites below


def entropy_from_labels(labels: Sequence[Hashable]) -> float:
    """Shannon entropy (nats) over a sequence of pre-assigned cluster labels.

    Shared by cluster_entropy (labels = normalized strings) and
    pilot.semantic's NLI/embedding clustering (labels = cluster ids from a
    similarity graph) -- both reduce to "how many distinct groups, how
    lopsided." Empty input returns 0.0 by convention.
    """
    if len(labels) == 0:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log(p)
    return entropy


def normalize_string(s: Optional[str]) -> str:
    """Strip whitespace, lowercase, and remove trailing periods.

    None (a parse failure) normalizes to a distinguishable sentinel rather than
    being dropped, since dropping it would understate real instability in the
    K samples.
    """
    if s is None:
        return _PARSE_FAILURE_SENTINEL
    return s.strip().lower().rstrip(".")


def cluster_entropy(
    samples: Sequence[Optional[str]],
    normalize_fn: Callable[[Optional[str]], str] = normalize_string,
) -> float:
    """Shannon entropy (nats) over clusters of samples grouped by normalize_fn.

    Uses natural log to match the semantic-entropy uncertainty-quantification
    literature this is a stand-in for, and to keep units consistent with any
    later extension to token log-probabilities.

    For K=5: H=0 when all samples are identical, H=ln(5)~=1.609 when all 5 are
    distinct. Empty input returns 0.0 by convention.

    normalize_fn defaults to the light strip/lower/trailing-period normalizer
    above; pass pilot.canonicalize.canonicalize_math for math-aware clustering
    (e.g. LaTeX answers where exact-string matching is too strict -- see that
    module's docstring for why this pilot needed it).
    """
    normalized = [normalize_fn(s) for s in samples]
    return entropy_from_labels(normalized)


def majority_cluster(
    samples: Sequence[Optional[str]],
    normalize_fn: Callable[[Optional[str]], str] = normalize_string,
) -> tuple[str, int]:
    """Return the (normalized string, count) of the most common cluster.

    Ties are broken by first-encountered order (deterministic in Python's
    Counter/dict). Raises ValueError on empty input, since a majority of
    nothing is not a meaningful pilot-time case. See cluster_entropy for
    normalize_fn.
    """
    if len(samples) == 0:
        raise ValueError("Cannot compute majority cluster of an empty sample list")
    normalized = [normalize_fn(s) for s in samples]
    counts = Counter(normalized)
    return counts.most_common(1)[0]


# --- Simpler summaries of the same cluster distribution --------------------
#
# These exist as baselines, and specifically to answer the question a reader
# will ask about cluster_entropy: is entropy over K samples doing anything
# that counting distinct answers does not? On the real n=300 perception data
# it is -- entropy beats all three below by a paired margin that excludes
# zero -- but that only became sayable once the comparison existed.
#
# Direction differs between them and is easy to get backwards. distinct_count
# rises with uncertainty, like entropy; majority_fraction and majority_margin
# rise with CONFIDENCE and must be negated before being fed to an AUROC that
# predicts "is wrong". Each docstring states its direction; the sign is not
# baked in here, matching how the token-confidence baseline is stored
# negated at the point of use rather than at the point of computation.


def distinct_count(
    samples: Sequence[Optional[str]],
    normalize_fn: Callable[[Optional[str]], str] = normalize_string,
) -> int:
    """Number of distinct clusters among the samples. Higher = less certain.

    The crudest possible summary: 1 when the model always says the same
    thing, K when it never repeats itself. Empty input returns 0.
    """
    return len({normalize_fn(s) for s in samples})


def majority_fraction(
    samples: Sequence[Optional[str]],
    normalize_fn: Callable[[Optional[str]], str] = normalize_string,
) -> float:
    """Share of samples falling in the largest cluster. Higher = MORE certain.

    Negate before using as an uncertainty score. Empty input returns 0.0,
    which is the conservative direction (reads as maximally uncertain).
    """
    if len(samples) == 0:
        return 0.0
    counts = Counter(normalize_fn(s) for s in samples)
    return counts.most_common(1)[0][1] / len(samples)


def majority_margin(
    samples: Sequence[Optional[str]],
    normalize_fn: Callable[[Optional[str]], str] = normalize_string,
) -> float:
    """(largest cluster - second largest) / K. Higher = MORE certain.

    Distinguishes a 3-2 split from a 3-1-1 split, which distinct_count
    cannot and majority_fraction cannot: both of those score 3/5 on the
    fraction, but the margin is 0.2 and 0.4 respectively. Negate before
    using as an uncertainty score. Empty input returns 0.0.
    """
    if len(samples) == 0:
        return 0.0
    counts = [c for _, c in Counter(normalize_fn(s) for s in samples).most_common()]
    second = counts[1] if len(counts) > 1 else 0
    return (counts[0] - second) / len(samples)
