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


def load_fermat_extra_error_items(
    n_extra: int,
    seed: int = 42,
    skip: int = 150,
    split: str = "train",
    _loader: Callable[..., datasets.Dataset] = datasets.load_dataset,
) -> datasets.Dataset:
    """Load n_extra MORE has_error=1 items, disjoint from a prior balanced run.

    Exists for exactly one situation: a stratified analysis (see
    pilot.plotting.stratified_auroc) found one stratum underpowered and the
    other already adequate, so the fix is more items in ONE stratum only --
    drawing a fresh balanced sample would waste calls re-covering the
    stratum that already has enough. Concretely: the 2026-08-05 7B grading
    run had 150 has_error=1 items / 17 misgraded (below the registered
    minimum of 30), while has_error=0 had 150 items / 44 misgraded (already
    adequate). This draws more has_error=1 items to fix only the deficient
    side.

    Disjointness is structural, not probabilistic: with the same seed,
    ``dataset.shuffle(seed=seed)`` reproduces the exact ordering
    load_fermat_balanced used to build error_idx, so
    ``error_idx[:skip]`` is precisely what a prior
    ``load_fermat_balanced(n=..., seed=seed, target_error_frac=...)`` call
    already took (for skip = round(n * target_error_frac)), and
    ``error_idx[skip:skip+n_extra]`` is guaranteed to be the very next
    items in that same ordering, never overlapping. The caller is
    responsible for passing the ``skip`` that actually matches the prior
    run's error-item count -- this function has no way to verify that from
    here, since it does not see the prior run.
    """
    dataset = _loader("ai4bharat/FERMAT", split=split)
    dataset = dataset.shuffle(seed=seed)

    error_idx = [i for i, has_error in enumerate(dataset["has_error"]) if has_error]

    available = max(0, len(error_idx) - skip)
    actual_n = min(n_extra, available)
    if actual_n < n_extra:
        logger.warning(
            "Requested %d extra has_error=1 items beyond the first %d, but only "
            "%d are available after skipping. Returning %d instead.",
            n_extra, skip, available, actual_n,
        )

    selected = error_idx[skip: skip + actual_n]
    sample = dataset.select(selected)

    columns_to_drop = [c for c in sample.column_names if c not in FERMAT_FIELDS]
    if columns_to_drop:
        sample = sample.remove_columns(columns_to_drop)

    return sample


# --- ScratchMath (second dataset, reasoning arm only) ----------------------
#
# Added 2026-08-08 to test whether the has_error=1 stratified reasoning
# result -- confirmed across Qwen-3B/7B and LLaVA-NeXT on FERMAT -- also
# holds on a genuinely different dataset. Three structural differences from
# FERMAT, all of which have to be stated wherever this is reported:
#
#   1. EVERY ScratchMath item contains an error. There is no has_error=0
#      stratum, so the clean-stratum inversion (a confirmed FERMAT finding)
#      simply cannot be tested here, and no balanced sample can be built.
#      This loader therefore only ever supports the has_error=1 analysis.
#   2. The question is a separate TEXT field; the image holds only the
#      student's rough scratchwork, not a complete Question-Answer page.
#      The grading prompt has to supply the question in text, unlike
#      FERMAT's single-image setup.
#   3. Questions and error annotations are in Chinese, and the scratchwork
#      is sparse stylus/tablet writing that is materially harder to read
#      than FERMAT's pages -- so a capability gate is a live risk and must
#      be checked before any entropy number is interpreted.
#
# `student_answer` is deliberately NOT included in the model's input by the
# notebook that uses this: handing over the student's final answer as text
# would let the model check the arithmetic textually and reduce the image to
# decoration, which would stop testing the vision-grounded claim this
# project actually makes. It is kept here only for offline analysis.

SCRATCHMATH_FIELDS = [
    "student_scratchwork",
    "question",
    "answer",
    "student_answer",
    "solution",
    "error_category",
    "error_explanation",
    "question_id",
    "study_level",
]

# The dataset ships two configs; `middle` is closer to FERMAT's grade range
# but has only 241 items, so both are pooled and the level is recorded as a
# column rather than dropped -- it is a plausible covariate for difficulty.
SCRATCHMATH_CONFIGS = ("primary", "middle")


def load_scratchmath_sample(
    n: int = 100,
    seed: int = 42,
    configs: tuple = SCRATCHMATH_CONFIGS,
    _loader: Callable[..., datasets.Dataset] = datasets.load_dataset,
) -> datasets.Dataset:
    """Load a deterministic random sample of n ScratchMath items.

    Every returned item has an error by construction (see the module note
    above), so callers must treat this as a has_error=1-only sample and must
    not compute a pooled or balanced statistic from it.

    Pools the `primary` and `middle` configs before shuffling so a single
    seed gives one reproducible ordering across both, and so an extension
    run can take the next slice of that same ordering (mirroring how
    load_fermat_extra_error_items achieves structural disjointness).
    """
    parts = []
    for config in configs:
        part = _loader("songdj/ScratchMath", config, split="train")
        if "study_level" not in part.column_names:
            part = part.add_column("study_level", [config] * len(part))
        parts.append(part)

    dataset = datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    dataset = dataset.shuffle(seed=seed)

    actual_n = min(n, len(dataset))
    if actual_n < n:
        logger.warning(
            "Requested n=%d ScratchMath items but only %d are available; using %d.",
            n, len(dataset), actual_n,
        )

    sample = dataset.select(range(actual_n))

    columns_to_drop = [c for c in sample.column_names if c not in SCRATCHMATH_FIELDS]
    if columns_to_drop:
        sample = sample.remove_columns(columns_to_drop)

    return sample


def load_scratchmath_extra(
    n_extra: int,
    seed: int = 42,
    skip: int = 100,
    configs: tuple = SCRATCHMATH_CONFIGS,
    _loader: Callable[..., datasets.Dataset] = datasets.load_dataset,
) -> datasets.Dataset:
    """Load n_extra MORE ScratchMath items, disjoint from a prior sample.

    Same structural-disjointness argument as load_fermat_extra_error_items:
    with the same seed and the same config order, the pooled shuffle
    reproduces one ordering, so ``[:skip]`` is exactly what a prior
    load_scratchmath_sample(n=skip, seed=seed) took and ``[skip:skip+n_extra]``
    is guaranteed to be the next, non-overlapping slice. The caller must pass
    the ``skip`` matching the prior run's n -- this function cannot verify it.
    """
    parts = []
    for config in configs:
        part = _loader("songdj/ScratchMath", config, split="train")
        if "study_level" not in part.column_names:
            part = part.add_column("study_level", [config] * len(part))
        parts.append(part)

    dataset = datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    dataset = dataset.shuffle(seed=seed)

    available = max(0, len(dataset) - skip)
    actual_n = min(n_extra, available)
    if actual_n < n_extra:
        logger.warning(
            "Requested %d extra ScratchMath items beyond the first %d, but only "
            "%d are available. Returning %d instead.",
            n_extra, skip, available, actual_n,
        )

    sample = dataset.select(range(skip, skip + actual_n))

    columns_to_drop = [c for c in sample.column_names if c not in SCRATCHMATH_FIELDS]
    if columns_to_drop:
        sample = sample.remove_columns(columns_to_drop)

    return sample
