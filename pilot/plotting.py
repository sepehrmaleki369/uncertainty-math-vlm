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


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Read a results CSV and validate that the expected columns are present."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
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
