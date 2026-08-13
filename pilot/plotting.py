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


def risk_coverage_curve(
    df: pd.DataFrame, entropy_col: str, correctness_col: str
) -> pd.DataFrame:
    """Accuracy retained as a function of how many items you refuse to answer.

    The operating characteristic a deferral system is actually specified by.
    AUROC answers "does entropy rank errors above non-errors"; this answers
    "if I answer only my most confident X%, how accurate am I", which is the
    question a grading pipeline has to make a decision on.

    Answer every item whose entropy is at or below a threshold, defer the rest.
    One row per distinct entropy value, ascending, plus the full-coverage row.

    Note the resolution limit: with K samples of a discrete answer, entropy
    takes few distinct values (7 at K=5 on the real data, of which only 5 give
    distinct coverage levels). You cannot target an arbitrary risk level -- the
    achievable operating points are whatever the entropy grid provides.
    """
    entropy, is_error = _error_arrays(df, entropy_col, correctness_col)
    rows = []
    for threshold in np.unique(np.round(entropy, 6)):
        keep = entropy <= threshold + 1e-9
        n_kept = int(keep.sum())
        if n_kept == 0:
            continue
        n_wrong = int(is_error[keep].sum())
        rows.append({
            "threshold": float(threshold),
            "coverage": n_kept / len(entropy),
            "n_kept": n_kept,
            "n_deferred": int(len(entropy) - n_kept),
            "accuracy_kept": 1.0 - n_wrong / n_kept,
            "risk_kept": n_wrong / n_kept,
            "n_errors_kept": n_wrong,
        })
    return pd.DataFrame(rows)


def aurc(df: pd.DataFrame, entropy_col: str, correctness_col: str) -> dict:
    """Area under the risk-coverage curve, plus the no-signal baseline.

    Lower is better: it is the average error rate you incur as coverage sweeps
    from answering one item to answering everything. Unlike AUROC it is
    sensitive to the *base error rate*, so it must always be read next to the
    baseline -- a model that is wrong 60% of the time cannot have a good AURC
    no matter how well its uncertainty ranks.

    ``baseline_aurc`` is what deferring at random would give (just the overall
    error rate). ``improvement`` is how much of that the ordering buys back.
    On the real n=300 perception data: 0.342 against a 0.607 baseline.
    """
    entropy, is_error = _error_arrays(df, entropy_col, correctness_col)
    if len(entropy) == 0:
        return {"aurc": float("nan"), "baseline_aurc": float("nan"),
                "improvement": float("nan"), "n_items": 0}

    # Integrated over the *achievable* operating points, not per item. A
    # per-item sweep would break ties by input order, and ties dominate here --
    # K=5 entropy takes 7 distinct values across 300 items, so a per-item
    # version silently reports whatever order the CSV happened to be in, and
    # scores uninformative entropy as if it were informative.
    curve = risk_coverage_curve(df, entropy_col, correctness_col)
    prev_coverage = 0.0
    value = 0.0
    for row in curve.itertuples():
        value += (row.coverage - prev_coverage) * row.risk_kept
        prev_coverage = row.coverage

    base = float(is_error.mean())
    oracle_continuous = oracle_aurc(base)
    oracle_grid = _oracle_aurc_on_grid(curve, len(entropy), int(is_error.sum()))
    return {
        "aurc": float(value),
        "baseline_aurc": base,
        "improvement": base - float(value),
        "n_items": int(len(entropy)),
        "n_operating_points": int(len(curve)),
        # E-AURC (Geifman et al.): AURC minus what a perfect ranking would
        # score. Raw AURC is dominated by the base error rate, so it cannot
        # compare two models with different accuracies -- this can.
        "oracle_aurc": oracle_continuous,
        "e_aurc": float(value) - oracle_continuous,
        # The same subtraction against an oracle restricted to THIS score's
        # coverage grid. See _oracle_aurc_on_grid for why both are reported.
        "oracle_aurc_on_grid": oracle_grid,
        "e_aurc_on_grid": float(value) - oracle_grid,
    }


def oracle_aurc(error_rate: float) -> float:
    """AURC of a perfect ranking at a given error rate: r + (1-r)ln(1-r).

    The closed form of integrating the risk-coverage curve of a score that
    puts every correct item ahead of every incorrect one. A perfect ranker
    still has AURC > 0 whenever the model makes mistakes, because at full
    coverage it must answer the wrong items too -- which is exactly why raw
    AURC cannot be compared across models with different accuracies, and why
    E-AURC exists.
    """
    r = float(error_rate)
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"error_rate must be in [0, 1], got {r}")
    if r == 0.0:
        return 0.0
    if r == 1.0:
        return 1.0
    return r + (1.0 - r) * math.log(1.0 - r)


def _oracle_aurc_on_grid(curve: pd.DataFrame, n_items: int, n_errors: int) -> float:
    """Best AURC achievable at the coverage levels this score actually offers.

    `oracle_aurc` assumes a continuous score with an operating point at every
    coverage. Cluster entropy does not have that: at K=5 it takes 7 distinct
    values across 300 items, so the achievable coverages are a coarse grid.
    Subtracting the continuous oracle therefore charges the score for its
    GRID as well as for its RANKING, and those are different deficiencies --
    the grid is fixed by K and closes as K grows (7 points at K=5, 34 at
    K=10), while the ranking is the thing being evaluated.

    This subtracts an oracle held to the same grid, isolating ranking quality.
    Report both: e_aurc for comparability with the literature, e_aurc_on_grid
    when the question is how well the ordering itself does.
    """
    n_correct = n_items - n_errors
    prev_coverage = 0.0
    value = 0.0
    for row in curve.itertuples():
        # A perfect ranker keeping row.n_kept items keeps correct ones first.
        errors_kept = max(0, row.n_kept - n_correct)
        value += (row.coverage - prev_coverage) * (errors_kept / row.n_kept)
        prev_coverage = row.coverage
    return float(value)


def coverage_at_risk(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    target_risk: float,
) -> dict:
    """Most items answerable while keeping the error rate on them <= target.

    The number a deployment is specified by: "answer as much as you can, but
    stay under 30% error." Complements AUROC (ranking quality) and AURC
    (average over all operating points) with a single actionable point.

    Scans EVERY operating point and takes the maximum qualifying coverage
    rather than stopping at the first threshold that crosses the target --
    risk_kept is not guaranteed monotone in coverage, and on a coarse entropy
    grid a later, larger-coverage point can satisfy a target that an earlier
    one misses. Returns achievable=False when no operating point qualifies,
    which is a real and common outcome at K=5, not an error.
    """
    if not 0.0 <= target_risk <= 1.0:
        raise ValueError(f"target_risk must be in [0, 1], got {target_risk}")
    curve = risk_coverage_curve(df, entropy_col, correctness_col)
    if curve.empty:
        return {"achievable": False, "target_risk": float(target_risk),
                "coverage": 0.0, "n_kept": 0, "threshold": float("nan"),
                "risk_kept": float("nan"), "accuracy_kept": float("nan")}

    ok = curve[curve["risk_kept"] <= target_risk + 1e-12]
    if ok.empty:
        return {"achievable": False, "target_risk": float(target_risk),
                "coverage": 0.0, "n_kept": 0, "threshold": float("nan"),
                "risk_kept": float("nan"), "accuracy_kept": float("nan")}

    best = ok.loc[ok["coverage"].idxmax()]
    return {
        "achievable": True,
        "target_risk": float(target_risk),
        "coverage": float(best["coverage"]),
        "n_kept": int(best["n_kept"]),
        "n_deferred": int(best["n_deferred"]),
        "threshold": float(best["threshold"]),
        "risk_kept": float(best["risk_kept"]),
        "accuracy_kept": float(best["accuracy_kept"]),
    }


def conformal_abstention_threshold(
    cal_df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    target_risk: float,
    delta: float = 0.1,
    conservative: bool = True,
) -> dict:
    """Largest entropy threshold whose retained items hold error <= target_risk.

    Calibrated on held-out data, then applied to unseen items -- the point being
    a threshold picked by eyeballing the same data it is evaluated on carries no
    guarantee at all.

    With ``conservative`` (the default) the threshold must satisfy a Hoeffding
    upper bound on the risk rather than the point estimate, so the guarantee
    holds with probability 1 - delta rather than on average. This costs
    coverage, and at small n it costs a lot: the bound term is
    sqrt(log(1/delta) / (2 * n_kept)), which is ~0.10 at n_kept=100.

    Returns nan for the threshold when no operating point satisfies the target,
    which is a real outcome rather than an error -- at a high base error rate
    some risk levels are simply unreachable at any coverage.
    """
    if not 0.0 < target_risk < 1.0:
        raise ValueError(f"target_risk must be in (0, 1), got {target_risk}")

    curve = risk_coverage_curve(cal_df, entropy_col, correctness_col)
    best = None
    for row in curve.itertuples():
        bound = row.risk_kept
        if conservative:
            bound += math.sqrt(math.log(1.0 / delta) / (2 * row.n_kept))
        if bound <= target_risk:
            best = row  # curve is ascending in threshold, so keep the last pass
    if best is None:
        return {"threshold": float("nan"), "cal_coverage": 0.0,
                "cal_risk": float("nan"), "achievable": False}
    return {
        "threshold": float(best.threshold),
        "cal_coverage": float(best.coverage),
        "cal_risk": float(best.risk_kept),
        "achievable": True,
    }


def evaluate_conformal_abstention(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    target_risk: float,
    n_splits: int = 1000,
    cal_frac: float = 0.5,
    seed: int = 0,
    conservative: bool = True,
) -> dict:
    """Split-calibrate a deferral threshold repeatedly and check it holds out.

    The honest test of a guarantee: calibrate on half the items, apply to the
    other half, and count how often the held-out error actually exceeded the
    target. ``violation_rate`` should sit at or below delta; well below means
    the rule is conservative and is deferring more than it needs to.

    Splits where no threshold achieves the target are counted in
    ``n_unachievable`` rather than silently dropped -- at a high base error rate
    that is the common case for a tight target, and hiding it would overstate
    how usable the rule is.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    n_cal = int(round(n * cal_frac))

    coverages, risks, violations, unachievable = [], [], 0, 0
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, test = df.iloc[perm[:n_cal]], df.iloc[perm[n_cal:]]
        fit = conformal_abstention_threshold(
            cal, entropy_col, correctness_col, target_risk,
            conservative=conservative,
        )
        if not fit["achievable"]:
            unachievable += 1
            continue

        t_entropy, t_error = _error_arrays(test, entropy_col, correctness_col)
        keep = t_entropy <= fit["threshold"] + 1e-9
        if keep.sum() == 0:
            unachievable += 1
            continue
        risk = float(t_error[keep].mean())
        coverages.append(float(keep.mean()))
        risks.append(risk)
        violations += int(risk > target_risk)

    n_valid = len(risks)
    return {
        "target_risk": target_risk,
        "n_splits": n_splits,
        "n_valid": n_valid,
        "n_unachievable": unachievable,
        "mean_coverage": float(np.mean(coverages)) if coverages else float("nan"),
        "mean_test_risk": float(np.mean(risks)) if risks else float("nan"),
        "violation_rate": violations / n_valid if n_valid else float("nan"),
    }


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


def classify_capability_check(
    accuracy: float,
    baseline_accuracy: float,
    n_items: int,
    threshold_capable: float = 0.65,
    threshold_chance: float = 0.55,
) -> dict:
    """Gate for the 7B (or any larger-model) reasoning capability check.

    The Phase 2/3 finding was that reasoning entropy measures nothing because
    Qwen2.5-VL-3B cannot grade above chance (51.7% on a 0.500 baseline). The
    report's recommended next step is explicit that a stronger model must be
    gated on \\emph{accuracy} before any entropy/AUROC number is computed --
    interpreting an uncertainty metric over an incompetent grader is the
    mistake this whole project spent its middle third correcting. This
    function is that gate, made a pure, testable decision rather than an
    eyeballed call at the top of a notebook.

    Three bands, matching the report's own stated criterion ("roughly
    0.65-0.70" to call the question live again):
      - below ``threshold_chance``: "at_chance" -- the negative reasoning-arm
        result generalizes to this model too; do not proceed to an AUROC.
      - between the two thresholds: "marginal" -- some signal above the
        baseline, but not clearly enough to trust an uncertainty analysis
        built on top of it. Report both numbers, proceed with a caveat.
      - at or above ``threshold_capable``: "capable" -- the entropy question
        is live again; the existing AUROC/CI code applies unchanged.

    ``baseline_accuracy`` is passed in (from majority_class_baseline) rather
    than assumed to be 0.5, since a differently-balanced sample -- unlikely
    here, since this check is designed to reuse the same n=300 balanced
    sample, but not guaranteed -- would make a hardcoded 0.5 wrong.
    """
    margin = accuracy - baseline_accuracy
    if accuracy >= threshold_capable:
        verdict = "capable"
    elif accuracy < threshold_chance:
        verdict = "at_chance"
    else:
        verdict = "marginal"
    return {
        "verdict": verdict,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "margin_over_baseline": margin,
        "n_items": n_items,
        "threshold_capable": threshold_capable,
        "threshold_chance": threshold_chance,
        # Whether a resulting entropy AUROC would be a meaningful test of the
        # technique. False for "marginal" too, not just "at_chance": an AUROC
        # is worth *computing* diagnostically at any accuracy level, but only
        # "capable" grading makes it a trustworthy answer to the reasoning-arm
        # question -- exactly the distinction the pooled-vs-stratified n=300
        # result depended on getting right.
        "entropy_result_meaningful": verdict == "capable",
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


def plot_risk_coverage(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    ax: Optional[plt.Axes] = None,
    baseline_risk: Optional[float] = None,
) -> plt.Axes:
    """Step plot of error rate among answered items as coverage increases.

    Reads left to right as "if I answer my most confident X% and defer the
    rest, what's my error rate on the X% I answered". Coverage 0 is the
    single most confident item; coverage 1 is answering everything, where
    risk equals the overall error rate by construction.

    A step plot, not a smooth curve, because the underlying operating points
    are genuinely discrete -- K=5 entropy only takes a handful of distinct
    values (7 on the real n=300 perception data), so the curve is flat
    between them and jumps at each one. Drawing a smooth interpolation would
    imply operating points that do not exist.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.4))

    curve = risk_coverage_curve(df, entropy_col, correctness_col)

    ax.set_facecolor(SURFACE)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor(SURFACE)

    coverage = curve["coverage"].to_numpy()
    risk = curve["risk_kept"].to_numpy()
    ax.step(coverage, risk, where="post", color=COLOR_CORRECT, linewidth=2, zorder=3)
    ax.scatter(coverage, risk, color=COLOR_CORRECT, s=28, zorder=4)

    if baseline_risk is None:
        baseline_risk = float((~_as_bool(df[correctness_col])).mean())
    ax.axhline(baseline_risk, color=GRIDLINE, linestyle="--", linewidth=1, zorder=1)
    ax.text(
        0.02, baseline_risk, "answer everything", va="bottom", ha="left",
        fontsize=8, color=INK_MUTED, transform=ax.get_yaxis_transform(),
    )

    ax.set_xlabel("coverage (fraction answered, most confident first)",
                  fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("risk (error rate among answered items)",
                  fontsize=10, color=INK_SECONDARY)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, max(risk.max(), baseline_risk) * 1.15)

    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    return ax


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


def classify_stratum_result(
    ci: Optional[dict],
    threshold: float = SCALEUP_PREREGISTRATION["reasoning_stratum_auroc_min"],
    min_minority: int = SCALEUP_PREREGISTRATION["min_minority_class"],
) -> str:
    """Bucket one stratum's bootstrap_auroc_ci dict against the registered bars.

    A pure function so the verdict that gets reported is the verdict the
    thresholds actually produce, in the same spirit as
    classify_scaleup_result -- not an eyeball call on a point estimate.

    The four buckets exist because this project needed all four, in this
    order of discovery:

      not_measured                -- the stratum is absent from the data
                                     (ScratchMath has no clean items at all,
                                     which is different from having too few).
      inconclusive_underpowered   -- minority class below the registered
                                     minimum; the point estimate is not
                                     interpretable regardless of its value.
      confirmed                   -- clears the threshold AND excludes chance
                                     (Qwen-3B/7B and LLaVA-NeXT, 0.775-0.854).
      resolved_below_threshold    -- powered, CI excludes chance, but does not
                                     clear the threshold. Added 2026-08-08 for
                                     InternVL3's 0.628 [0.548, 0.706]: real
                                     signal, not confirmable, and previously
                                     there was no honest label for it -- it
                                     would have been forced into "confirmed"
                                     or "no signal", both wrong.
      no_signal                   -- powered but the CI includes chance.

    ``inverted`` is reported as its own suffix rather than folded into
    no_signal: a stratum whose CI sits resolvably BELOW 0.5 is a finding
    (the clean-stratum inversion is a confirmed result here), not an
    absence of one.
    """
    if ci is None or ci.get("n_items", 0) == 0:
        return "not_measured"

    n_error = ci.get("n_error", 0)
    n_correct = ci.get("n_correct", ci.get("n_items", 0) - n_error)
    if min(n_error, n_correct) < min_minority:
        return "inconclusive_underpowered"

    # The inversion check MUST come before the excludes_chance gate.
    # bootstrap_auroc_ci defines excludes_chance as `ci_low > 0.5` -- a
    # one-sided, above-chance-only test -- so a resolvably INVERTED stratum
    # reports excludes_chance=False. Checking that first sent Qwen-7B's
    # confirmed clean-stratum inversion (0.280 [0.200, 0.366]) to
    # "no_signal", mislabelling one of this project's confirmed findings as
    # an absence of one. Caught 2026-08-08 by the snapshot of that run
    # disagreeing with the report.
    if ci["ci_high"] < 0.5:
        return "confirmed_inverted"

    if not ci.get("excludes_chance", False):
        return "no_signal"

    if ci["auroc"] >= threshold:
        return "confirmed"
    return "resolved_below_threshold"


# --- The single-label-stratum trap ----------------------------------------
#
# Added 2026-08-09, after it invalidated this project's own headline
# reasoning result. Recorded here as reusable code rather than as prose so
# the check is cheap to run before trusting any future stratified AUROC.
#
# The trap: stratifying a binary-decision task by its ground-truth label
# leaves a subset on which every item has the SAME true answer. Within it,
# "was the model correct?" is no longer a separate fact from "what did the
# model answer?" -- the two columns are identical (or exact complements).
# Any uncertainty measure computed from the model's own samples therefore
# predicts correctness almost by construction, because the majority vote it
# is being scored against is a function of those same samples.
#
# On this project's grading data the effect is total: correctness and the
# model's own verdict agreed on 100% of items within the has_error=1
# stratum, and the stratified AUROC (0.801) was numerically identical to
# the AUROC for predicting the model's own answer. The apparent
# "sign reversal" between strata was likewise an identity, not a finding:
# correct = said-error in one stratum and its complement in the other, so
# anything correlating with the verdict must flip sign between them.


def correctness_collapses_onto_prediction(
    df: pd.DataFrame,
    correctness_col: str,
    prediction_col: str,
) -> dict:
    """How far 'was it correct' is just a relabelling of 'what did it say'.

    Returns ``agreement`` (fraction of items where the two columns match)
    and ``degenerate``, true when they agree -- or disagree -- almost
    always, since an exact complement is just as degenerate as an exact
    match. On a stratum flagged degenerate, an AUROC against correctness
    is measuring the model's self-consistency, not its error rate, and
    should not be reported as error prediction.
    """
    a = df[correctness_col].astype(bool).to_numpy()
    b = df[prediction_col].astype(bool).to_numpy()
    if len(a) == 0:
        return {"agreement": float("nan"), "degenerate": False, "n_items": 0}
    agreement = float((a == b).mean())
    return {
        "agreement": agreement,
        # Either extreme is fatal; 0.0 means "always the complement".
        "degenerate": bool(agreement >= 0.95 or agreement <= 0.05),
        "n_items": int(len(a)),
    }


def bias_only_null_auroc(
    n_items: int,
    says_error_rate: float,
    k: int = 5,
    n_sims: int = 2000,
    seed: int = 0,
) -> dict:
    """AUROC a model with NO item-level signal still attains on a single-label stratum.

    Simulates a model that answers "error" with a fixed per-sample
    probability, identical for every item, so nothing whatsoever
    distinguishes a hard item from an easy one. Entropy over its K votes
    is then pure sampling noise. Any AUROC it attains is manufactured by
    the voting arithmetic: on an all-error stratum the majority is wrong
    only when most samples dissent, and dissent is exactly what raises
    entropy.

    Use it as the floor a real stratified AUROC has to clear. On this
    project's data none of the three models cleared it -- the observed
    values sat *below* the null median, which is what settled the
    question.
    """
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n_sims):
        votes = rng.random((n_items, k)) < says_error_rate
        n_error_votes = votes.sum(axis=1)
        # Entropy of a two-cluster split, as cluster_entropy would compute it.
        p1 = n_error_votes / k
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(np.where(p1 > 0, p1 * np.log(p1), 0.0)
                    + np.where(p1 < 1, (1 - p1) * np.log(1 - p1), 0.0))
        correct = n_error_votes > k / 2          # truth is "error" throughout
        if correct.all() or (~correct).all():
            continue
        aurocs.append(compute_auroc(
            pd.DataFrame({"_e": ent, "_c": correct}), "_e", "_c"))
    arr = np.array([a for a in aurocs if not math.isnan(a)])
    if arr.size == 0:
        return {"median": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_valid": 0}
    lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
    return {"median": float(med), "ci_low": float(lo), "ci_high": float(hi),
            "n_valid": int(arr.size)}


def plot_scoring_categories(
    summary: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    category_col: str = "category",
    count_col: str = "n",
    model_col: str = "model",
) -> plt.Axes:
    """Horizontal bars: what the scored population is actually made of.

    Takes pilot.rescore.scoring_category_summary's output -- one frame, or
    several concatenated with a `model` column, in which case the bars are
    grouped for a side-by-side comparison.

    Bars follow pilot.rescore.CATEGORIES order rather than sorting by size,
    so the same category sits in the same place across models and the eye can
    compare rows. The order also runs from "scored right" through the two
    non-monotone categories to "wrong under every rule", which is the order a
    reader needs to reason about the taxonomy.

    Colour carries one bit only: whether the item ends up correct under the
    loosest rule. Six shades for seven categories would encode nothing.

    Hatching carries a second, orthogonal bit: whether the category is a
    SCORING failure rather than a model failure. `false_pass_removed` items
    were scored *correct* by the frozen rule and only became wrong once the
    extractor bugs were fixed, and `broken_by_relaxation` items go the other
    way. Both are places the rules are non-monotone. Without the hatch they
    sit in the same orange as `genuinely_wrong` and read as "the model got
    this wrong", which is the opposite of what they show.
    """
    from pilot.rescore import CATEGORIES

    present = [c for c in CATEGORIES if c in set(summary[category_col])]
    missing = set(summary[category_col]) - set(CATEGORIES)
    if missing:
        raise ValueError(f"unknown categories, not in rescore.CATEGORIES: {missing}")

    # Categories whose items are correct under the loosest rule.
    ENDS_CORRECT = {"correct_robust", "bug_fix_recovered",
                    "cosmetic_mismatch", "scope_mismatch"}
    # Categories where the SCORING RULE changed its mind, not the model.
    SCORING_REGRESSION = {"false_pass_removed", "broken_by_relaxation"}
    HATCH = "////"

    models = ([None] if model_col not in summary.columns
              else list(dict.fromkeys(summary[model_col])))
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 0.52 * len(present) * max(1, len(models)) + 1.6))

    ax.set_facecolor(SURFACE)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor(SURFACE)

    y = np.arange(len(present))
    height = 0.8 / len(models)
    for m_i, model in enumerate(models):
        sub = summary if model is None else summary[summary[model_col] == model]
        counts = [int(sub.loc[sub[category_col] == c, count_col].sum()) for c in present]
        total = sum(counts) or 1
        offset = (m_i - (len(models) - 1) / 2) * height
        colors = [COLOR_CORRECT if c in ENDS_CORRECT else COLOR_INCORRECT
                  for c in present]
        # Second and later models are drawn lighter so the grouping reads.
        alpha = 1.0 if m_i == 0 else 0.55
        bars = ax.barh(y + offset, counts, height=height * 0.92, color=colors,
                       alpha=alpha, zorder=3)
        # Hatch per bar: barh takes a single hatch for the whole container,
        # so the per-category pattern has to be set on the patches.
        for patch, cat in zip(bars.patches, present):
            if cat in SCORING_REGRESSION:
                patch.set_hatch(HATCH)
                patch.set_edgecolor(SURFACE)
                patch.set_linewidth(0.0)
        for yi, count in zip(y + offset, counts):
            if count:
                ax.text(count + total * 0.008, yi, f"{count}  ({count / total:.0%})",
                        va="center", ha="left", fontsize=8, color=INK_SECONDARY,
                        zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(present, fontsize=9, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlabel("items", fontsize=10, color=INK_SECONDARY)
    ax.set_xlim(0, max(summary[count_col]) * 1.34)
    from matplotlib.patches import Patch
    handles = []
    if len(models) > 1 and models[0] is not None:
        # Swatches use a REAL bar colour, not a neutral grey. Grey matched
        # neither channel and forced the reader to work out that the key
        # referred to opacity alone -- the one place the chart made someone
        # stop.
        handles += [Patch(facecolor=COLOR_CORRECT, alpha=1.0 if i == 0 else 0.55,
                          label=str(m)) for i, m in enumerate(models)]
    if any(c in SCORING_REGRESSION for c in present):
        handles.append(Patch(facecolor=COLOR_INCORRECT, hatch=HATCH,
                             edgecolor=SURFACE, linewidth=0.0,
                             label="scoring-rule failure,\nnot a model failure"))
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=9,
                  loc="center left", bbox_to_anchor=(0.55, 0.52),
                  title="shade = model", title_fontsize=8,
                  labelspacing=0.9, handlelength=1.9)

    ax.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    return ax


def contact_sheet(
    images,
    captions,
    ncols: int = 3,
    per_page: int = 12,
    title: str = "",
    cell_height: float = 3.6,
    caption_fontsize: float = 7.0,
    footer: str = "",
):
    """A grid of handwritten pages with captions, paged. Returns a list of figures.

    Exists because inline rendering cannot be trusted to carry evidence out of
    Colab. On the 2026-08-10 notebook-17 run every `plt.show()` produced
    nothing in the synced copy -- zero `image/png` outputs, no cell error, and
    no repo-side cause (no .gitattributes, no git filters, and no commit of
    that notebook has ever held an image). Whatever drops them, a saved PNG
    file is immune, and one sheet holding a dozen pages is far easier to
    review later than a dozen separate files.

    `images` are PIL images (FERMAT is gated, so they only exist in Colab);
    `captions` is a same-length sequence of short strings drawn under each
    cell. Pages are filled row-major and the final page is padded with blank
    axes so cell sizes stay constant across pages.
    """
    if len(images) != len(captions):
        raise ValueError(
            f"{len(images)} images vs {len(captions)} captions -- a caption "
            "drifting off its image would mislabel the evidence")
    if not images:
        return []

    figures = []
    n_pages = math.ceil(len(images) / per_page)
    for page in range(n_pages):
        chunk = list(zip(images, captions))[page * per_page:(page + 1) * per_page]
        # Rows come from per_page, NOT from this chunk's length: sizing the
        # final short page to its own contents would enlarge its cells and
        # make the last few pages look like a different figure.
        nrows = math.ceil(per_page / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 3.2, nrows * cell_height))
        fig.patch.set_facecolor(SURFACE)
        flat = np.atleast_1d(axes).ravel()

        for ax, (img, caption) in zip(flat, chunk):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_color(GRIDLINE)
            ax.set_title(caption, fontsize=caption_fontsize, color=INK_SECONDARY,
                         loc="left", wrap=True)
        # Pad the last page rather than resizing its cells.
        for ax in flat[len(chunk):]:
            ax.axis("off")

        if title:
            suffix = f"  ({page + 1}/{n_pages})" if n_pages > 1 else ""
            fig.suptitle(title + suffix, fontsize=11, color=INK_SECONDARY,
                         x=0.01, ha="left")
        # Reserve space for the footer BEFORE tight_layout, or the legend is
        # drawn over the bottom row of captions. Optional and empty by
        # default, so every existing caller renders byte-identically.
        bottom = 0.0
        if footer:
            n_lines = footer.count("\n") + 1
            bottom = min(0.22, 0.016 * n_lines + 0.02)
            fig.text(0.01, bottom * 0.55, footer, fontsize=caption_fontsize,
                     color=INK_SECONDARY, ha="left", va="center", linespacing=1.5)
        fig.tight_layout(rect=(0, bottom, 1, 0.97 if title else 1))
        figures.append(fig)
    return figures


def accuracy_by_distinct_answers(df, distinct_col="n_distinct", correctness_col="transcription_correct"):
    """Accuracy at each level of "how many different answers did it give".

    The plainest statement of the perception result, and the one a reader
    understands without knowing what an AUROC is: with K=5 the model gives
    between 1 and 5 distinct answers, and accuracy falls monotonically across
    that range -- 92% at one answer to 7% at five on Qwen-3B.

    Returned rather than plotted so the numbers can be asserted; the figure is
    plot_accuracy_by_distinct_answers.
    """
    correct = df[correctness_col].astype(str).str.lower().isin(["true", "1"]) \
        if df[correctness_col].dtype == object else df[correctness_col].astype(bool)
    out = (pd.DataFrame({"n_distinct": df[distinct_col].astype(int),
                         "correct": correct})
           .groupby("n_distinct")["correct"].agg(["size", "mean"])
           .rename(columns={"size": "n", "mean": "accuracy"}))
    return out.reset_index()


def plot_accuracy_by_distinct_answers(
    tables: dict,
    ax: Optional[plt.Axes] = None,
    k: int = 5,
    annotate_n: bool = True,
) -> plt.Axes:
    """One line per model of accuracy vs. distinct answers, with counts.

    `tables` maps a model name to accuracy_by_distinct_answers' output.

    Drawn as lines rather than bars because the claim is the TREND -- that
    accuracy falls monotonically as self-disagreement rises -- and bars invite
    reading each cell on its own. The n is annotated at every point because
    the end cells are the interesting ones and are also the smallest, so a
    reader must be able to see how much weight each carries.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_facecolor(SURFACE)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor(SURFACE)

    palette = [COLOR_CORRECT, COLOR_INCORRECT, INK_SECONDARY]
    for i, (name, table) in enumerate(tables.items()):
        colour = palette[i % len(palette)]
        ax.plot(table["n_distinct"], table["accuracy"] * 100, marker="o",
                markersize=6, linewidth=2, color=colour, label=name, zorder=3)
        if annotate_n:
            for _, row in table.iterrows():
                ax.annotate(f"n={int(row['n'])}",
                            (row["n_distinct"], row["accuracy"] * 100),
                            textcoords="offset points", xytext=(0, 9 if i == 0 else -16),
                            ha="center", fontsize=7.5, color=colour, zorder=4)

    ax.set_xticks(range(1, k + 1))
    ax.set_xlabel(f"distinct answers across {k} samples", fontsize=10,
                  color=INK_SECONDARY)
    ax.set_ylabel("transcription accuracy (%)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(-6, 104)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.legend(frameon=False, fontsize=9)
    return ax


# For K=5 the entropy value determines the number of distinct answers exactly
# -- the seven reachable values correspond one-to-one with the partitions of 5
# -- so this inverts the stored column without recomputing labels.
#
# That distinction matters. Recomputing distinct counts from the raw samples
# gives DIFFERENT numbers for the 2026-08-02 Qwen run (92% -> 80% in the
# one-distinct cell), because that run was scored in a Colab session where
# SymPy's LaTeX parser was unavailable and 43/300 items labelled differently
# (see canonicalize.latex_parser_available). Every locked figure -- AUROC
# 0.835, accuracy 39.3% -- comes from the STORED column, so anything reported
# beside them has to come from the same place or the paper contradicts itself.
_K5_ENTROPY_TO_DISTINCT = {
    0.0: 1,        # 5
    0.5004: 2,     # 4+1
    0.6730: 2,     # 3+2
    0.9503: 3,     # 3+1+1
    1.0549: 3,     # 2+2+1
    1.3322: 4,     # 2+1+1+1
    1.6094: 5,     # 1+1+1+1+1
}


def distinct_from_entropy(entropy_values, k: int = 5):
    """Number of distinct answers implied by each K=5 cluster entropy."""
    if k != 5:
        raise ValueError("the exact inversion is only tabulated for k=5")
    out = []
    for value in entropy_values:
        match = min(_K5_ENTROPY_TO_DISTINCT, key=lambda x: abs(x - float(value)))
        if abs(match - float(value)) > 1e-3:
            raise ValueError(
                f"entropy {value} is not a reachable K=5 value; the run was "
                "probably not K=5, so this inversion does not apply")
        out.append(_K5_ENTROPY_TO_DISTINCT[match])
    return out
