"""Utilities for the real-model sampling-parameter sensitivity run.

Notebook 30 varies ``K`` and response bias only in a signal-free mathematical
null.  Notebook 32 is different: it runs Qwen2.5-VL-3B on the same 300 FERMAT
pages at several temperatures, then scores nested prefixes of the same draws.
This module keeps the scoring, completeness gates, and public summary out of
the notebook so they can be tested without a GPU or gated data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence

import pandas as pd

from .plotting import aurc, bootstrap_auroc_ci
from .rescore import score_item


TEMPERATURES = (0.3, 0.7, 1.0)
K_VALUES = (3, 5, 10)
K_MAX = max(K_VALUES)
N_ITEMS = 300
BASE_SEED = 20260815

PUBLIC_COLUMNS = (
    "temperature", "k", "n_items", "n_error", "n_correct",
    "accuracy", "auroc", "ci_low", "ci_high", "excludes_chance",
    "aurc", "random_aurc", "n_entropy_values", "parse_failure_rate",
)


def cell_seed(temperature: float, item_id: int,
              base_seed: int = BASE_SEED) -> int:
    """Stable independent seed for one temperature/item generation cell."""
    payload = f"{base_seed}|{temperature:.6f}|{int(item_id)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def score_nested_samples(
    raw_samples: Sequence[str],
    ground_truth: str,
    temperature: float,
    item_id: int,
    k_values: Sequence[int] = K_VALUES,
    rule: str = "strict_v1",
    scorer: Callable = score_item,
) -> list[dict]:
    """Score nested prefixes so different K values reuse identical draws."""
    ks = tuple(sorted(set(int(k) for k in k_values)))
    if not ks or ks[0] < 1:
        raise ValueError("k_values must contain positive integers")
    if len(raw_samples) < ks[-1]:
        raise ValueError(
            f"need at least {ks[-1]} samples, received {len(raw_samples)}")
    rows = []
    for k in ks:
        result = scorer(list(raw_samples[:k]), ground_truth, rule)
        rows.append({
            "item_id": int(item_id),
            "temperature": float(temperature),
            "k": int(k),
            "perception_entropy": float(result["perception_entropy"]),
            "transcription_correct": bool(result["transcription_correct"]),
            "n_transcription_parse_failures": int(
                result["n_transcription_parse_failures"]),
        })
    return rows


def validate_complete_grid(
    scored: pd.DataFrame,
    temperatures: Iterable[float] = TEMPERATURES,
    k_values: Iterable[int] = K_VALUES,
    n_items: int = N_ITEMS,
) -> None:
    """Fail closed unless every planned condition contains every item once."""
    required = {
        "item_id", "temperature", "k", "perception_entropy",
        "transcription_correct", "n_transcription_parse_failures",
    }
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored grid missing columns: {sorted(missing)}")
    if scored.duplicated(["temperature", "k", "item_id"]).any():
        raise ValueError("duplicate temperature/k/item rows")

    expected = {(float(t), int(k)) for t in temperatures for k in k_values}
    actual = {(float(t), int(k)) for t, k in
              scored[["temperature", "k"]].drop_duplicates().itertuples(index=False)}
    if actual != expected:
        raise ValueError(
            f"condition grid mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}")
    counts = scored.groupby(["temperature", "k"])["item_id"].nunique()
    bad = counts[counts != int(n_items)]
    if not bad.empty:
        raise ValueError(f"incomplete conditions: {bad.to_dict()}")


def summarize_grid(
    scored: pd.DataFrame,
    temperatures: Iterable[float] = TEMPERATURES,
    k_values: Iterable[int] = K_VALUES,
    n_items: int = N_ITEMS,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Create the hash-only/public metric table from item-level scores."""
    validate_complete_grid(scored, temperatures, k_values, n_items)
    rows = []
    for (temperature, k), group in scored.groupby(["temperature", "k"], sort=True):
        r = bootstrap_auroc_ci(
            group, "perception_entropy", "transcription_correct",
            n_boot=n_boot, seed=seed,
        )
        curve = aurc(group, "perception_entropy", "transcription_correct")
        rows.append({
            "temperature": float(temperature),
            "k": int(k),
            "n_items": int(len(group)),
            "n_error": int(r["n_error"]),
            "n_correct": int(r["n_correct"]),
            "accuracy": float(group["transcription_correct"].mean()),
            "auroc": float(r["auroc"]),
            "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]),
            "excludes_chance": bool(r["excludes_chance"]),
            "aurc": float(curve["aurc"]),
            "random_aurc": float(curve["baseline_aurc"]),
            "n_entropy_values": int(group["perception_entropy"].round(12).nunique()),
            "parse_failure_rate": float(
                group["n_transcription_parse_failures"].sum() / (len(group) * int(k))),
        })
    return pd.DataFrame(rows)[list(PUBLIC_COLUMNS)]


def reviewer_gate(summary: pd.DataFrame, n_items: int = N_ITEMS,
                  min_class: int = 30) -> dict:
    """State whether a full grid, rather than a smoke run, may enter the paper."""
    complete = len(summary) == len(TEMPERATURES) * len(K_VALUES)
    complete = complete and summary["n_items"].eq(n_items).all()
    powered = summary[["n_error", "n_correct"]].min(axis=1).ge(min_class).all()
    return {
        "complete": bool(complete),
        "minority_at_least_30": bool(powered),
        "paper_eligible": bool(complete and powered),
        "interpretation": (
            "protocol sensitivity across the declared grid; no condition was "
            "selected as a tuned winner" if complete and powered else
            "SMOKE OR UNDERPOWERED: do not cite in the paper"),
    }
