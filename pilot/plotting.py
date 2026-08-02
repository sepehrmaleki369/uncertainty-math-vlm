"""Step 4 analysis: plots and sanity checks over a results CSV from a Colab run.

Produces the two box plots the pilot exists to answer -- perception entropy
split by transcription correctness, reasoning entropy split by grading
correctness -- plus the two integrity checks that say whether those plots can
be trusted at all (temperature-0 anchor, aggregate parse-failure rate).
"""

import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "perception_entropy",
    "reasoning_entropy",
    "transcription_correct",
    "grading_correct",
    "temp0_entropy_transcription",
    "temp0_entropy_grading",
]

# Palette: categorical slots 1 and 2, validated for CVD separation against the
# light chart surface (worst-pair CVD dE 24.7, normal-vision 33.6).
COLOR_CORRECT = "#2a78d6"
COLOR_INCORRECT = "#eb6834"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_results(
    csv_path: str | Path, required_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Read a results CSV and validate that the expected columns are present.

    required_columns defaults to REQUIRED_COLUMNS (the full baseline schema).
    A K-resample CSV (see join_k_resample) only has the grading-related
    columns, not perception_entropy etc. -- pass a narrower list rather than
    forcing it to satisfy the full baseline schema.
    """
    df = pd.read_csv(csv_path)
    check_columns = REQUIRED_COLUMNS if required_columns is None else required_columns
    missing = [c for c in check_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Results CSV is missing expected columns: {missing}")
    return df


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a correctness column to bool, tolerating 'True'/'False' strings.

    Booleans survive a CSV round-trip as strings, and bool("False") is True --
    so a naive astype(bool) would silently mark every row correct.
    """
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def plot_entropy_by_correctness(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    title: str,
    ax: Optional[plt.Axes] = None,
    k: int = 5,
) -> plt.Axes:
    """Box plot of entropy split by a correctness label, with the raw points shown.

    Individual points are overlaid because n is small (~25 in the pilot): a bare
    box plot of 5 points implies more distributional detail than the data
    supports, and the reader needs to see how thin each group actually is.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))

    correct = _as_bool(df[correctness_col])
    groups = [
        ("Correct", df.loc[correct, entropy_col].dropna().values, COLOR_CORRECT),
        ("Incorrect", df.loc[~correct, entropy_col].dropna().values, COLOR_INCORRECT),
    ]

    ax.set_facecolor(SURFACE)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor(SURFACE)

    # Reference line: with K samples, entropy is bounded above by ln(K), reached
    # only when all K samples differ. Without it the y-scale is uninterpretable.
    max_entropy = math.log(k)
    ax.axhline(max_entropy, color=GRIDLINE, linestyle="--", linewidth=1, zorder=1)
    ax.text(
        2.62, max_entropy, f"max  ln({k})", va="center", ha="right",
        fontsize=8, color=INK_MUTED,
    )

    rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible
    for pos, (label, values, color) in enumerate(groups, start=1):
        if len(values) == 0:
            continue
        ax.boxplot(
            [values], positions=[pos], widths=0.5, showfliers=False,
            medianprops={"color": color, "linewidth": 2},
            boxprops={"color": BASELINE, "linewidth": 1},
            whiskerprops={"color": BASELINE, "linewidth": 1},
            capprops={"color": BASELINE, "linewidth": 1},
        )
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter, values,
            s=42, color=color, alpha=0.85, zorder=3,
            edgecolors=SURFACE, linewidths=1.4,  # surface ring on overlap
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"{label}\nn={len(values)}" for label, values, _ in groups],
        fontsize=10, color=INK_SECONDARY,
    )
    ax.set_ylabel("cluster entropy (nats)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11.5, color=INK_PRIMARY, pad=12, loc="left")
    ax.set_xlim(0.4, 2.7)
    ax.set_ylim(-0.09, max_entropy + 0.16)

    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    return ax


def check_temperature_zero_anchor(
    df: pd.DataFrame, temp0_entropy_col: str, tol: float = 1e-6
) -> dict:
    """Report how many items had non-zero entropy across the 2 greedy draws.

    Meaningful (not tautological) because the anchor uses 2 temperature-0 draws
    per item, not 1: a single-sample entropy is zero by definition and could
    never fail. With 2 draws, a non-zero value flags genuine low-temperature
    instability -- an accidental sampling flag, or batching nondeterminism.
    """
    values = df[temp0_entropy_col].astype(float)
    nonzero = values.abs() > tol
    return {
        "column": temp0_entropy_col,
        "n_items": int(len(values)),
        "n_nonzero": int(nonzero.sum()),
        "frac_nonzero": float(nonzero.mean()) if len(values) else 0.0,
        "max_value": float(values.max()) if len(values) else 0.0,
        "passed": bool(not nonzero.any()),
    }


def check_parse_failure_rate(
    df: pd.DataFrame,
    transcription_failure_col: str = "n_transcription_parse_failures",
    grading_failure_col: str = "n_grading_parse_failures",
    k: int = 5,
) -> dict:
    """Aggregate fraction of all samples in the run that failed to parse.

    Checked in aggregate, not per item: an item where all K samples fail to
    parse collapses to a single confident <PARSE_FAILURE> cluster with entropy
    0, which reads as a clean, stable signal while actually hiding a systematic
    regex or prompt-format regression. A high rate here is the tell.
    """
    out = {}
    for name, col in (
        ("transcription", transcription_failure_col),
        ("grading", grading_failure_col),
    ):
        if col not in df.columns:
            out[name] = None
            continue
        failures = int(df[col].fillna(0).astype(int).sum())
        total = int(len(df) * k)
        out[name] = {
            "n_failed_samples": failures,
            "n_total_samples": total,
            "frac_failed": failures / total if total else 0.0,
            "n_items_all_failed": int((df[col].fillna(0).astype(int) >= k).sum()),
        }
    return out


def summarize_results(df: pd.DataFrame) -> dict:
    """Group counts, medians and means feeding the written summary."""
    summary = {"n_items": int(len(df))}
    for entropy_col, correctness_col, name in (
        ("perception_entropy", "transcription_correct", "perception"),
        ("reasoning_entropy", "grading_correct", "reasoning"),
    ):
        correct = _as_bool(df[correctness_col])
        vals_c = df.loc[correct, entropy_col].dropna()
        vals_i = df.loc[~correct, entropy_col].dropna()
        summary[name] = {
            "n_correct": int(len(vals_c)),
            "n_incorrect": int(len(vals_i)),
            "median_correct": float(vals_c.median()) if len(vals_c) else float("nan"),
            "median_incorrect": float(vals_i.median()) if len(vals_i) else float("nan"),
            "mean_correct": float(vals_c.mean()) if len(vals_c) else float("nan"),
            "mean_incorrect": float(vals_i.mean()) if len(vals_i) else float("nan"),
        }
    summary["parse_failures"] = check_parse_failure_rate(df)
    summary["temp0_transcription"] = check_temperature_zero_anchor(
        df, "temp0_entropy_transcription"
    )
    summary["temp0_grading"] = check_temperature_zero_anchor(
        df, "temp0_entropy_grading"
    )
    return summary


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average (tie-corrected) ranks, 1-based -- same convention as
    pandas' rank(method="average"), but on a bare array so the bootstrap
    loop doesn't pay for DataFrame construction 10k times."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auroc_from_arrays(entropy: np.ndarray, is_error: np.ndarray) -> float:
    """Mann-Whitney AUROC for entropy ranking the is_error==True items above
    the rest. Returns nan if either class is empty."""
    pos = entropy[is_error]
    neg = entropy[~is_error]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([pos, neg]))
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _error_arrays(
    df: pd.DataFrame, entropy_col: str, correctness_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """(entropy, is_error) arrays with rows missing an entropy value dropped."""
    is_error = ~_as_bool(df[correctness_col])
    entropy = df[entropy_col]
    keep = entropy.notna()
    return entropy[keep].to_numpy(dtype=float), is_error[keep].to_numpy(dtype=bool)


def compute_auroc(df: pd.DataFrame, entropy_col: str, correctness_col: str) -> float:
    """AUROC for entropy_col predicting a correctness_col error, via the
    Mann-Whitney rank-sum identity: AUROC = (sum of ranks of the "incorrect"
    class - n_pos*(n_pos+1)/2) / (n_pos * n_neg). No scikit-learn dependency;
    average-rank tie handling matches Mann-Whitney's, which matters here --
    K=5 entropy only takes 7 distinct values, so ties are the norm, not an
    edge case. Verified against the real 2026-07-31 100-item CSV
    (reasoning_entropy/grading_correct): reproduces 0.6184, matching
    scipy.stats.mannwhitneyu independently.

    Returns nan if either class is empty (AUROC is undefined with one class).
    """
    return _auroc_from_arrays(*_error_arrays(df, entropy_col, correctness_col))


def bootstrap_auroc_ci(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for compute_auroc, resampling items with replacement.

    Exists because every AUROC in this pilot is computed on n=100 items with a
    minority class of 18-58, where the sampling error is large enough to decide
    whether a reported difference means anything. Measured on the real data: the
    reasoning arm's headline 0.618 has a 95% CI of [0.489, 0.742] -- it includes
    0.5, i.e. that result is not distinguishable from chance, which the
    point estimate alone completely hides.

    `excludes_chance` is the field to actually branch on when writing up a
    result, rather than eyeballing whether a point estimate "looks" predictive.
    Bootstrap replicates that degenerate to a single class (possible when the
    minority class is small) are skipped rather than counted as 0.5.
    """
    entropy, is_error = _error_arrays(df, entropy_col, correctness_col)
    observed = _auroc_from_arrays(entropy, is_error)

    rng = np.random.default_rng(seed)
    n = len(entropy)
    replicates = []
    if n > 0:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            value = _auroc_from_arrays(entropy[idx], is_error[idx])
            if not math.isnan(value):
                replicates.append(value)

    if not replicates:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.percentile(replicates, 100 * alpha / 2))
        ci_high = float(np.percentile(replicates, 100 * (1 - alpha / 2)))

    return {
        "auroc": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_items": int(n),
        "n_error": int(is_error.sum()),
        "n_correct": int((~is_error).sum()),
        "n_boot_valid": len(replicates),
        # The claim "this beats chance" reduced to one testable field.
        "excludes_chance": bool(ci_low > 0.5) if replicates else False,
    }


def bootstrap_auroc_difference_ci(
    df: pd.DataFrame,
    entropy_col_a: str,
    correctness_col_a: str,
    entropy_col_b: str,
    correctness_col_b: str,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap CI for (AUROC_a - AUROC_b) over the same items.

    The right test for "does condition A beat condition B" when both are
    measured on one item set: resample items once per replicate and recompute
    *both* AUROCs on that same resample, so the shared-item correlation is kept
    rather than treated as two independent samples. Comparing two separately
    computed CIs for overlap instead is the standard error here -- it is a
    strictly more conservative and less informative test.

    `difference_excludes_zero` is the field to branch on. Note this tests only
    sampling error: if the two conditions also disagree about *which* items are
    errors (e.g. the K=5 and K=15 grading labels differ), the comparison carries
    that confound regardless of what this interval says -- see n_error_a /
    n_error_b, which make it visible.
    """
    entropy_a, is_error_a = _error_arrays(df, entropy_col_a, correctness_col_a)
    entropy_b, is_error_b = _error_arrays(df, entropy_col_b, correctness_col_b)
    if len(entropy_a) != len(entropy_b):
        raise ValueError(
            f"paired comparison needs equal-length columns after dropping NaNs, "
            f"got {len(entropy_a)} for {entropy_col_a} and {len(entropy_b)} for {entropy_col_b}"
        )

    observed = _auroc_from_arrays(entropy_a, is_error_a) - _auroc_from_arrays(
        entropy_b, is_error_b
    )

    rng = np.random.default_rng(seed)
    n = len(entropy_a)
    replicates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # one resample, both metrics -- keeps pairing
        a = _auroc_from_arrays(entropy_a[idx], is_error_a[idx])
        b = _auroc_from_arrays(entropy_b[idx], is_error_b[idx])
        if not (math.isnan(a) or math.isnan(b)):
            replicates.append(a - b)

    if not replicates:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.percentile(replicates, 100 * alpha / 2))
        ci_high = float(np.percentile(replicates, 100 * (1 - alpha / 2)))

    return {
        "difference": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_items": int(n),
        "n_error_a": int(is_error_a.sum()),
        "n_error_b": int(is_error_b.sum()),
        "n_boot_valid": len(replicates),
        "difference_excludes_zero": bool(ci_low > 0 or ci_high < 0) if replicates else False,
    }


def auroc_sensitivity(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    failure_col: str,
    k: int = 5,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """AUROC with CIs under the cuts that would expose a degenerate-signal artifact.

    A cluster-entropy AUROC can be manufactured two ways that have nothing to do
    with the model being usefully uncertain:
      - parse failures get their own cluster (PARSE_FAILURE_SENTINEL), so an
        unparseable sample inflates entropy *and* tends to be scored incorrect --
        the metric would then be measuring the parser, not the model;
      - items pinned at max entropy ln(k) can carry the whole ranking if that
        cell happens to be all-incorrect, leaving nothing in the middle.
    Dropping each, and both together, says how much of the signal is left.

    Run on the real perception arm (2026-08-02): 0.788 full -> 0.773 without
    parse failures -> 0.721 without max-entropy items -> 0.707 without both,
    CI still excluding 0.5 at n=74. So neither cut explains the result.
    """
    at_max = np.isclose(df[entropy_col].astype(float), math.log(k))
    clean = df[failure_col].fillna(0).astype(int) == 0

    subsets = {
        "full": df,
        "excl_parse_failures": df[clean],
        "excl_max_entropy": df[~at_max],
        "excl_both": df[clean & ~at_max],
    }
    out = {}
    for name, subset in subsets.items():
        if len(subset) == 0:
            out[name] = None
            continue
        out[name] = bootstrap_auroc_ci(
            subset, entropy_col, correctness_col, n_boot=n_boot, seed=seed
        )
    # The headline survives only if the harshest simultaneous cut still beats chance.
    out["robust"] = bool(out["excl_both"] and out["excl_both"]["excludes_chance"])
    return out


def stratified_auroc(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    stratum_col: str,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """AUROC pooled and within each level of stratum_col, to expose sign reversal.

    Motivated by a real finding on the grading arm (2026-08-02). The FERMAT
    sample is 87/13 imbalanced on ``has_error``, and the model has a strong
    prior toward answering "there is an error" -- it graded 12 of the 13 clean
    answers as containing an error, 6 of them unanimously. Those items are
    *confidently* wrong, so entropy has no disagreement to detect and ranks
    them backwards. Pooling them with the 87 error-containing items cancels a
    genuine signal: reasoning entropy scores 0.756 within the majority stratum
    but only 0.618 pooled.

    A pooled AUROC over a stratum where the predictor's direction reverses is
    not a weaker version of the within-stratum result -- it is a different and
    largely meaningless quantity. ``sign_reversal`` flags that case.

    Strata with a single correctness class yield nan (AUROC undefined) and are
    reported rather than dropped, since an all-wrong stratum is itself the
    finding.
    """
    out = {"pooled": bootstrap_auroc_ci(df, entropy_col, correctness_col, n_boot, seed)}
    strata = {}
    for level, subset in df.groupby(stratum_col):
        strata[level] = bootstrap_auroc_ci(
            subset, entropy_col, correctness_col, n_boot, seed
        )
    out["strata"] = strata

    valid = [s["auroc"] for s in strata.values() if not math.isnan(s["auroc"])]
    out["sign_reversal"] = bool(
        len(valid) > 1 and max(valid) > 0.5 and min(valid) < 0.5
    )
    # Pooling is misleading when a stratum reverses direction *and* the pooled
    # estimate sits below the best stratum -- the shape of a Simpson's paradox.
    out["pooled_understates"] = bool(
        valid and out["pooled"]["auroc"] < max(valid) - 0.05
    )
    return out


def majority_class_baseline(df: pd.DataFrame, label_col: str) -> dict:
    """Accuracy of always predicting the most common label.

    Worth reporting next to any accuracy on this project: the grading task is
    87/13 imbalanced, so a constant "there is an error" predictor scores 0.87
    while the model's majority vote scores 0.75 at K=5 and 0.82 at K=15. An
    accuracy figure that loses to a constant is not evidence the model can do
    the task, and any uncertainty metric layered on it inherits that problem.
    """
    labels = _as_bool(df[label_col])
    frac_true = float(labels.mean())
    return {
        "n": int(len(labels)),
        "frac_positive": frac_true,
        "majority_label": bool(frac_true >= 0.5),
        "baseline_accuracy": max(frac_true, 1.0 - frac_true),
    }


def join_k_resample(baseline_df: pd.DataFrame, resample_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join a baseline results CSV with a grading K-resample CSV on
    (orig_q, pert_a) -- orig_q alone repeats across perturbed variants of the
    same question (verified 91/100 unique on the real data), so it would
    silently misjoin rows; (orig_q, pert_a) is verified unique 100/100.

    Raises ValueError on a duplicate key in either frame, or if the two
    frames don't overlap 1:1, rather than silently dropping unmatched rows.
    """
    key = ["orig_q", "pert_a"]
    for name, df in (("baseline_df", baseline_df), ("resample_df", resample_df)):
        dupes = df.duplicated(subset=key).sum()
        if dupes:
            raise ValueError(f"{name} has {dupes} duplicate (orig_q, pert_a) key(s)")

    baseline_keys = set(map(tuple, baseline_df[key].values))
    resample_keys = set(map(tuple, resample_df[key].values))
    if baseline_keys != resample_keys:
        only_baseline = baseline_keys - resample_keys
        only_resample = resample_keys - baseline_keys
        raise ValueError(
            f"baseline_df and resample_df do not overlap 1:1 on (orig_q, pert_a): "
            f"{len(only_baseline)} only in baseline, {len(only_resample)} only in resample"
        )

    return baseline_df.merge(resample_df, on=key, how="inner", suffixes=("", "_resample"))


def summarize_k_comparison(
    joined_df: pd.DataFrame,
    k_base: int,
    k_new: int,
    base_entropy_col: str = "reasoning_entropy",
    base_correct_col: str = "grading_correct",
    new_entropy_col: Optional[str] = None,
    new_correct_col: Optional[str] = None,
) -> dict:
    """Before/after comparison of a grading K-resample against the K_base baseline.

    new_entropy_col/new_correct_col default to f"reasoning_entropy_k{k_new}"/
    f"grading_correct_k{k_new}", matching the column names the new save cell
    in pilot.ipynb writes.
    """
    new_entropy_col = new_entropy_col or f"reasoning_entropy_k{k_new}"
    new_correct_col = new_correct_col or f"grading_correct_k{k_new}"

    max_ent_base = math.log(k_base)
    max_ent_new = math.log(k_new)
    frac_max_base = float((joined_df[base_entropy_col].round(6) == round(max_ent_base, 6)).mean())
    frac_max_new = float((joined_df[new_entropy_col].round(6) == round(max_ent_new, 6)).mean())

    correct_base = _as_bool(joined_df[base_correct_col])
    correct_new = _as_bool(joined_df[new_correct_col])
    flip_rate = float((correct_base != correct_new).mean())

    return {
        "k_base": k_base,
        "k_new": k_new,
        "auroc_k_base": compute_auroc(joined_df, base_entropy_col, base_correct_col),
        "auroc_k_new": compute_auroc(joined_df, new_entropy_col, new_correct_col),
        "n_distinct_entropy_k_base": int(joined_df[base_entropy_col].round(6).nunique()),
        "n_distinct_entropy_k_new": int(joined_df[new_entropy_col].round(6).nunique()),
        "frac_at_max_entropy_k_base": frac_max_base,
        "frac_at_max_entropy_k_new": frac_max_new,
        "median_by_correctness_k_base": {
            "correct": float(joined_df.loc[correct_base, base_entropy_col].median()),
            "incorrect": float(joined_df.loc[~correct_base, base_entropy_col].median()),
        },
        "median_by_correctness_k_new": {
            "correct": float(joined_df.loc[correct_new, new_entropy_col].median()),
            "incorrect": float(joined_df.loc[~correct_new, new_entropy_col].median()),
        },
        "majority_vote_flip_rate": flip_rate,
        "n_incorrect_k_new": int((~correct_new).sum()),
    }


def classify_k_resample_result(summary: dict) -> str:
    """Decision rule for whether a higher K sharpened the reasoning-entropy
    signal. A pure function over summarize_k_comparison's output, so the
    number that gets reported is the number that gets tested, not a post-hoc
    eyeball call.

    - inconclusive_underpowered: n_incorrect_k_new < 15, don't conclude
      anything from AUROC alone with too few items in the minority class.
    - confirmed_sharper: AUROC >= 0.72 AND the incorrect-group median finally
      exceeds the correct-group median -- at K=5 those medians are *exactly*
      tied, so "sharper" requires that tie to actually break, not just a
      mean/tail-driven AUROC tick up.
    - confirmed_modest: 0.55 <= AUROC < 0.72 -- real but genuinely modest
      regardless of K.
    - null_at_k5_was_noise: AUROC < 0.55 -- the original signal doesn't
      survive resampling.
    """
    if summary["n_incorrect_k_new"] < 15:
        return "inconclusive_underpowered"

    auroc = summary["auroc_k_new"]
    medians = summary["median_by_correctness_k_new"]
    median_separated = medians["incorrect"] > medians["correct"]

    if auroc >= 0.72 and median_separated:
        return "confirmed_sharper"
    if auroc >= 0.55:
        return "confirmed_modest"
    return "null_at_k5_was_noise"


# Registered 2026-08-02, BEFORE the n=300 scale-up run was executed. The
# reasoning-arm stratified result (0.756 within the has_error stratum, 0.891 at
# K=15) was discovered post-hoc on n=100, which is its main weakness -- these
# thresholds convert it into a falsifiable prediction. Do not tune them after
# seeing the scale-up data; add a new dated block instead and say which one a
# reported number was judged against.
SCALEUP_PREREGISTRATION = {
    "registered": "2026-08-02",
    "basis": "n=100 pilot: perception 0.788 [0.697, 0.869]; reasoning digit "
             "pooled 0.618 [0.489, 0.741]; reasoning digit within has_error "
             "stratum 0.756 [0.618, 0.876]",
    # Perception must not just beat chance, it must reproduce its effect size:
    # the n=100 CI lower bound was 0.697, so a replication should clear 0.65.
    "perception_ci_low_min": 0.65,
    # AMENDED 2026-08-02, after the FERMAT census (2244 items, 84.8% error) but
    # BEFORE the n=300 run generated anything. The census confirmed a 50/50
    # sample is feasible, which exposed a gap in the original block: rebalancing
    # changes the *perception* task too. On the n=100 data, clean items are
    # easier to transcribe (54% vs 40% accuracy) and carry lower entropy (0.753
    # vs 0.902), so moving from 13% clean to 50% clean should inflate pooled
    # perception AUROC for reasons unrelated to the metric. Pooled perception is
    # therefore NOT comparable across the two runs; the like-for-like
    # replication test is the has_error=1 stratum, which scored 0.762
    # [0.662, 0.855] at n=100 and gets 150 items here.
    "perception_error_stratum_ci_low_min": 0.65,
    # The clean stratum had only 13 items at n=100 (AUROC 0.940 [0.762, 1.000],
    # far too noisy to mean anything). At 150 items it becomes measurable for
    # the first time -- registered as an observation, with no threshold, because
    # there is no prior worth predicting against.
    "perception_clean_stratum": "measured_for_the_first_time_no_threshold",
    # Pooled reasoning on a *balanced* sample. The n=100 pooled value (0.618)
    # was depressed by the 87/13 imbalance; if the stratified story is right,
    # balancing should lift the pooled number to near the within-stratum one.
    "reasoning_pooled_ci_low_min": 0.55,
    # The central pre-registered prediction: within has_error=True items,
    # reasoning entropy predicts grading errors at 0.70 or better.
    "reasoning_stratum_auroc_min": 0.70,
    # The boldest, most falsifiable part: the model's error-present bias should
    # make entropy *inverted* on clean items. Predicted before seeing the data.
    "clean_stratum_auroc_max": 0.50,
    # Below this, decline to conclude rather than reading a wide interval.
    "min_minority_class": 30,
}


def classify_scaleup_result(
    summary: dict, prereg: Optional[dict] = None
) -> dict:
    """Judge a scale-up run against SCALEUP_PREREGISTRATION, per hypothesis.

    A pure function over already-computed numbers, in the same spirit as
    classify_k_resample_result: the number that gets reported is the number
    that was tested, against thresholds fixed before the data existed.

    ``summary`` is expected to carry (all optional -- missing keys yield
    "not_measured" rather than raising, so a partial run can still be judged):
      perception:            bootstrap_auroc_ci dict
      reasoning_pooled:      bootstrap_auroc_ci dict
      reasoning_error_stratum / reasoning_clean_stratum: bootstrap_auroc_ci dicts

    Verdicts are deliberately blunt strings, not scores -- the point is that a
    result lands in a bucket chosen in advance.
    """
    prereg = prereg or SCALEUP_PREREGISTRATION
    out = {"prereg_registered": prereg["registered"]}

    def underpowered(ci):
        return min(ci.get("n_error", 0), ci.get("n_correct", 0)) < prereg["min_minority_class"]

    def judge_perception(ci, threshold):
        if ci is None:
            return "not_measured"
        if underpowered(ci):
            return "inconclusive_underpowered"
        if ci["ci_low"] >= threshold:
            return "replicated"
        if ci["excludes_chance"]:
            return "weaker_than_pilot"
        return "failed_to_replicate"

    # Pooled perception is reported but is NOT the replication test: the
    # rebalanced sample is 50% clean vs the pilot's 13%, and clean items are
    # easier to transcribe, so this number is expected to drift upward on
    # composition alone. The has_error=1 stratum is the like-for-like test.
    out["perception_pooled_not_comparable"] = judge_perception(
        summary.get("perception"), prereg["perception_ci_low_min"]
    )
    out["perception"] = judge_perception(
        summary.get("perception_error_stratum"),
        prereg["perception_error_stratum_ci_low_min"],
    )

    pooled = summary.get("reasoning_pooled")
    if pooled is None:
        out["reasoning_pooled"] = "not_measured"
    elif underpowered(pooled):
        out["reasoning_pooled"] = "inconclusive_underpowered"
    elif pooled["ci_low"] >= prereg["reasoning_pooled_ci_low_min"]:
        out["reasoning_pooled"] = "signal_confirmed"
    else:
        out["reasoning_pooled"] = "no_signal"

    err = summary.get("reasoning_error_stratum")
    clean = summary.get("reasoning_clean_stratum")
    if err is None:
        out["reasoning_stratified"] = "not_measured"
    elif underpowered(err):
        out["reasoning_stratified"] = "inconclusive_underpowered"
    elif (
        err["auroc"] >= prereg["reasoning_stratum_auroc_min"]
        and err["excludes_chance"]
    ):
        out["reasoning_stratified"] = "confirmed"
    else:
        out["reasoning_stratified"] = "not_confirmed"

    # Judged separately: the inversion prediction can fail while the main
    # stratified prediction holds, and that combination is informative -- it
    # would mean the signal is real but the bias explanation for the pooled
    # collapse is wrong.
    if clean is None or math.isnan(clean.get("auroc", float("nan"))):
        out["clean_stratum_inversion"] = "not_measured"
    elif underpowered(clean):
        # Consistency fix applied 2026-08-03, after the n=300 run. This branch
        # originally skipped the min_minority_class guard that every other
        # branch applies, which is a bug rather than a registered choice -- the
        # threshold itself is unchanged. It matters: on the n=300 run the clean
        # stratum had 137 misgraded items but only 13 correct ones, so the guard
        # now fires and the verdict moves from "confirmed" to underpowered. The
        # direction is still strongly supported there (AUROC 0.239
        # [0.128, 0.374], excluding 0.5 by a wide margin) -- report that
        # alongside, but do not let it be graded as a passed prediction on a
        # 13-item minority class when the same standard called an 8-item one
        # inconclusive.
        out["clean_stratum_inversion"] = "inconclusive_underpowered"
    elif clean["auroc"] < prereg["clean_stratum_auroc_max"]:
        out["clean_stratum_inversion"] = "confirmed"
    else:
        out["clean_stratum_inversion"] = "not_confirmed"

    return out


def main(results_csv: str, output_dir: str = "figures") -> None:
    """Load results, run the integrity checks, save the two box plots."""
    df = load_results(results_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    plot_entropy_by_correctness(
        df, "perception_entropy", "transcription_correct",
        "Perception entropy by transcription correctness", ax=axes[0],
    )
    plot_entropy_by_correctness(
        df, "reasoning_entropy", "grading_correct",
        "Reasoning entropy by grading correctness", ax=axes[1],
    )
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig_path = out_dir / "entropy_by_correctness.png"
    fig.savefig(fig_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

    summary = summarize_results(df)
    print(f"Saved {fig_path}")
    return summary
