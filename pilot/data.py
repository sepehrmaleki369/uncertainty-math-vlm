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


def fermat_census(dataset) -> dict:
    """Size and has_error composition of a dataset, for run planning.

    The seed-42 n=100 sample came out 87/13 on has_error, which turned out to
    be the reasoning arm's dominant problem: a constant "there is an error"
    predictor scores 0.87 and beats the model. Before sizing any further run,
    check what balance the full dataset can actually support.
    """
    labels = [bool(x) for x in dataset["has_error"]]
    n_error = sum(labels)
    n_total = len(labels)
    return {
        "n_total": n_total,
        "n_error": n_error,
        "n_clean": n_total - n_error,
        "frac_error": n_error / n_total if n_total else 0.0,
        # The largest 50/50 sample the clean pool can support.
        "max_balanced_n": 2 * min(n_error, n_total - n_error),
    }


def load_fermat_balanced(
    n: int = 300,
    seed: int = 42,
    target_error_frac: float = 0.5,
    split: str = "train",
    _loader: Callable[..., datasets.Dataset] = datasets.load_dataset,
) -> datasets.Dataset:
    """Load n FERMAT items stratified to target_error_frac on has_error.

    FERMAT is ~87% error-containing, and sampling it naively makes the grading
    task nearly degenerate (see fermat_census). Balancing does two things at
    once: it removes the class-imbalance confound, and it *increases power* --
    because the model over-predicts "error", clean items are the ones it gets
    wrong, so balancing pushes the grading-error rate from ~25% toward ~46%.
    A 50/50 sample of n=290 gives the same AUROC precision as an unbalanced
    n=480.

    If a pool is too small to fill its quota, the target *ratio* is preserved
    and n is reduced, with a warning naming the shortfall -- rather than
    silently returning a differently-balanced sample, which would reintroduce
    the exact confound this function exists to remove. Callers that would
    rather have more items than exact balance should lower target_error_frac
    to what the census supports.

    The returned items are shuffled after stratification, so a run that is
    interrupted partway through still has a representative mix -- selecting
    all error items first would leave a resumed partial run fully imbalanced.
    """
    if not 0.0 < target_error_frac < 1.0:
        raise ValueError(f"target_error_frac must be in (0, 1), got {target_error_frac}")

    dataset = _loader("ai4bharat/FERMAT", split=split)
    dataset = dataset.shuffle(seed=seed)

    error_idx, clean_idx = [], []
    for i, has_error in enumerate(dataset["has_error"]):
        (error_idx if has_error else clean_idx).append(i)

    want_error = round(n * target_error_frac)
    want_clean = n - want_error

    # Preserve the ratio when a pool runs short: find the largest scale factor
    # that both pools can satisfy.
    scale = min(
        1.0,
        len(error_idx) / want_error if want_error else 1.0,
        len(clean_idx) / want_clean if want_clean else 1.0,
    )
    take_error = int(want_error * scale)
    take_clean = int(want_clean * scale)

    if scale < 1.0:
        logger.warning(
            "Cannot fill n=%d at target_error_frac=%.2f: dataset has %d error / "
            "%d clean items. Preserving the ratio and returning %d items "
            "(%d error / %d clean) instead. Lower target_error_frac to trade "
            "balance for size.",
            n, target_error_frac, len(error_idx), len(clean_idx),
            take_error + take_clean, take_error, take_clean,
        )

    selected = error_idx[:take_error] + clean_idx[:take_clean]
    sample = dataset.select(selected).shuffle(seed=seed)

    columns_to_drop = [c for c in sample.column_names if c not in FERMAT_FIELDS]
    if columns_to_drop:
        sample = sample.remove_columns(columns_to_drop)

    return sample
