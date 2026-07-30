"""Step 4 scaffold: analysis of results pulled back from a completed Colab run.

Only load_results is implemented now, since it needs no GPU and can be tested
against a synthetic CSV. The rest are stubs to be filled in once a real
results CSV exists from Step 3. Before implementing the plotting functions,
invoke the dataviz skill for chart styling/color guidance.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "perception_entropy",
    "reasoning_entropy",
    "transcription_correct",
    "grading_correct",
    "temp0_entropy_transcription",
    "temp0_entropy_grading",
]


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Read a results CSV and validate that the expected columns are present."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Results CSV is missing expected columns: {missing}")
    return df


def plot_entropy_by_correctness(
    df: pd.DataFrame,
    entropy_col: str,
    correctness_col: str,
    title: str,
    ax: Optional["object"] = None,
) -> "object":
    """Box plot of entropy_col split by correctness_col. TODO: implement in Step 4."""
    raise NotImplementedError("Implement once a real results CSV exists (Step 4)")


def check_temperature_zero_anchor(
    df: pd.DataFrame, temp0_entropy_col: str, tol: float = 1e-6
) -> dict:
    """Report how many items have near-zero entropy across the 2 temp=0 draws.

    Meaningful (not tautological) because the temp-0 anchor uses 2 greedy
    draws per item, not 1: a single-sample entropy is mathematically zero by
    definition, so it could never catch a real bug. With 2 draws, a nonzero
    value flags genuine low-temperature instability (e.g. an accidental
    sampling flag, batching nondeterminism). TODO: implement in Step 4.
    """
    raise NotImplementedError("Implement once a real results CSV exists (Step 4)")


def check_parse_failure_rate(
    df: pd.DataFrame,
    transcription_failure_col: str = "n_transcription_parse_failures",
    grading_failure_col: str = "n_grading_parse_failures",
) -> dict:
    """Aggregate fraction of all samples across the whole run that failed to parse.

    Checked in aggregate, not just per item: an item where all K samples fail
    to parse collapses to a single confident <PARSE_FAILURE> cluster (entropy
    0), which would look like a clean, stable signal while actually hiding a
    systematic regex or prompt-format regression. A high aggregate rate here
    is the tell. TODO: implement in Step 4.
    """
    raise NotImplementedError("Implement once a real results CSV exists (Step 4)")


def summarize_results(df: pd.DataFrame) -> dict:
    """Counts/means feeding the human-written 3-4 sentence summary.

    Should incorporate check_parse_failure_rate's output alongside the
    entropy/correctness split. TODO: implement in Step 4.
    """
    raise NotImplementedError("Implement once a real results CSV exists (Step 4)")


def main(results_csv: str, output_dir: str = "figures") -> None:
    """Orchestrate load -> checks -> plots -> save PNGs. TODO: implement in Step 4."""
    raise NotImplementedError("Implement once a real results CSV exists (Step 4)")
