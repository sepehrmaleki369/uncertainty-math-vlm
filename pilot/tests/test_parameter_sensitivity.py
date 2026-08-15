"""Offline acceptance tests for Notebook 32's real-model sensitivity logic."""

import pandas as pd
import pytest

import pilot.parameter_sensitivity as ps


def test_cell_seeds_are_stable_and_condition_specific():
    assert ps.cell_seed(0.7, 4) == ps.cell_seed(0.7, 4)
    seeds = {ps.cell_seed(t, i) for t in ps.TEMPERATURES for i in range(10)}
    assert len(seeds) == len(ps.TEMPERATURES) * 10


def test_nested_scoring_uses_exact_prefixes():
    seen = []

    def fake(samples, truth, rule):
        seen.append(tuple(samples))
        return {
            "perception_entropy": len(samples) / 10,
            "transcription_correct": samples[-1] == truth,
            "n_transcription_parse_failures": 0,
        }

    rows = ps.score_nested_samples(
        list("abcdefghij"), "j", 0.7, 9, scorer=fake)
    assert seen == [tuple("abc"), tuple("abcde"), tuple("abcdefghij")]
    assert [r["k"] for r in rows] == [3, 5, 10]


def test_nested_scoring_refuses_too_few_draws():
    with pytest.raises(ValueError, match="at least 10"):
        ps.score_nested_samples(["a"] * 9, "a", 0.7, 0)


def _synthetic_grid(n=40):
    rows = []
    for t in ps.TEMPERATURES:
        for k in ps.K_VALUES:
            for item in range(n):
                wrong = item % 2 == 0
                rows.append({
                    "item_id": item, "temperature": t, "k": k,
                    "perception_entropy": float(wrong) + (item % 3) / 10,
                    "transcription_correct": not wrong,
                    "n_transcription_parse_failures": 0,
                })
    return pd.DataFrame(rows)


def test_grid_validation_fails_closed_on_a_missing_row():
    grid = _synthetic_grid()
    with pytest.raises(ValueError, match="incomplete"):
        ps.validate_complete_grid(grid.iloc[:-1], n_items=40)


def test_public_summary_contains_no_item_text_or_raw_generations():
    summary = ps.summarize_grid(_synthetic_grid(), n_items=40, n_boot=100)
    assert list(summary.columns) == list(ps.PUBLIC_COLUMNS)
    assert len(summary) == 9
    assert not ({"orig_q", "pert_a", "raw_samples", "item_id"} & set(summary.columns))
    assert ps.reviewer_gate(summary, n_items=40, min_class=10)["paper_eligible"]
