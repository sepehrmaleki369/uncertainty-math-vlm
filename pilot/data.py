"""Load and filter a random sample of the FERMAT dataset.

FERMAT is downloaded fresh each session (never cached to Drive) via a plain
(non-streaming) datasets.load_dataset call, then deterministically shuffled and
sliced. True IterableDataset streaming only supports approximate reservoir
shuffling, which would not give a genuinely uniform random sample of n items --
so "not persisted to disk across sessions" is implemented here, not
streaming=True.
"""

import logging
from typing import Callable

import datasets

logger = logging.getLogger(__name__)

FERMAT_FIELDS = ["image", "orig_q", "pert_a", "has_error", "handwriting_style", "image_quality"]


def load_fermat_sample(
    n: int = 25,
    seed: int = 42,
    split: str = "train",
    _loader: Callable[..., datasets.Dataset] = datasets.load_dataset,
) -> datasets.Dataset:
    """Load a deterministic random sample of n FERMAT items with only FERMAT_FIELDS kept.

    _loader is a dependency-injection seam for testing: real access to this
    gated dataset requires HF auth and network, so tests pass a fake loader
    returning a small in-memory Dataset instead.
    """
    dataset = _loader("ai4bharat/FERMAT", split=split)
    dataset = dataset.shuffle(seed=seed)

    actual_n = min(n, len(dataset))
    if actual_n < n:
        logger.warning(
            "Requested n=%d but dataset only has %d items; using %d.", n, len(dataset), actual_n
        )

    sample = dataset.select(range(actual_n))

    columns_to_drop = [c for c in sample.column_names if c not in FERMAT_FIELDS]
    if columns_to_drop:
        sample = sample.remove_columns(columns_to_drop)

    return sample
