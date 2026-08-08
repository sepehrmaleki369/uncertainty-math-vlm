"""Tests for the ScratchMath loaders and grading prompt (second dataset).

ScratchMath was added 2026-08-08 to test whether the has_error=1 stratified
reasoning result -- confirmed on FERMAT across Qwen-3B/7B and LLaVA-NeXT --
holds on a genuinely different dataset. It differs from FERMAT structurally
in ways that the tests below pin down, because each difference is a way the
analysis could silently go wrong:

  - every item has an error, so no balanced sample and no clean stratum;
  - the question is a separate text field, not part of the image;
  - the student's final answer exists as text and must NOT reach the model,
    or the image stops mattering and the vision claim stops being tested.

No network: a fake loader is injected, matching the style of test_data.py.
"""

import datasets
import pytest

from pilot.data import (
    SCRATCHMATH_CONFIGS,
    SCRATCHMATH_FIELDS,
    load_scratchmath_extra,
    load_scratchmath_sample,
)
from pilot.prompts import (
    SCRATCHMATH_GRADING_SYSTEM_PROMPT,
    SCRATCHMATH_GRADING_USER_PROMPT,
    build_scratchmath_grading_prompt,
)


def _make_fake_config(config: str, n_rows: int) -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "question_id": [f"{config}_q{i}" for i in range(n_rows)],
            "question": [f"{config} question {i}" for i in range(n_rows)],
            "answer": [str(i * 2) for i in range(n_rows)],
            "solution": [f"solution {i}" for i in range(n_rows)],
            "student_answer": [str(i * 2 + 1) for i in range(n_rows)],
            "student_scratchwork": [f"image_{config}_{i}" for i in range(n_rows)],
            "error_category": [i % 7 for i in range(n_rows)],
            "error_explanation": [f"explanation {i}" for i in range(n_rows)],
            "internal_note": ["drop me"] * n_rows,
        }
    )


def _fake_loader_factory(sizes: dict):
    def _fake_loader(name: str, config: str = None, split: str = "train"):
        assert name == "songdj/ScratchMath", name
        return _make_fake_config(config, sizes[config])

    return _fake_loader


def test_sample_pools_both_configs_and_records_study_level():
    loader = _fake_loader_factory({"primary": 60, "middle": 40})
    sample = load_scratchmath_sample(n=100, seed=42, _loader=loader)

    assert len(sample) == 100
    assert set(sample["study_level"]) == set(SCRATCHMATH_CONFIGS)
    # Both configs actually contribute -- a bug that dropped one would still
    # return 100 rows here if the other were large enough.
    assert sample["study_level"].count("primary") == 60
    assert sample["study_level"].count("middle") == 40


def test_sample_drops_unexpected_columns_but_keeps_the_declared_ones():
    loader = _fake_loader_factory({"primary": 20, "middle": 10})
    sample = load_scratchmath_sample(n=30, seed=42, _loader=loader)

    assert "internal_note" not in sample.column_names
    for field in SCRATCHMATH_FIELDS:
        assert field in sample.column_names, field


def test_sample_is_deterministic_for_a_seed():
    loader = _fake_loader_factory({"primary": 60, "middle": 40})
    a = load_scratchmath_sample(n=25, seed=42, _loader=loader)
    b = load_scratchmath_sample(n=25, seed=42, _loader=loader)
    c = load_scratchmath_sample(n=25, seed=7, _loader=loader)

    assert a["question_id"] == b["question_id"]
    assert a["question_id"] != c["question_id"]


def test_sample_warns_and_truncates_when_asked_for_more_than_exists(caplog):
    loader = _fake_loader_factory({"primary": 10, "middle": 5})
    sample = load_scratchmath_sample(n=100, seed=42, _loader=loader)

    assert len(sample) == 15
    assert "only 15 are available" in caplog.text


def test_extra_is_structurally_disjoint_from_the_prior_sample():
    """Same argument as load_fermat_extra_error_items: one seed gives one
    pooled ordering, so [:skip] and [skip:skip+n] cannot overlap. A shared
    RNG that reshuffled per call would break this silently."""
    loader = _fake_loader_factory({"primary": 60, "middle": 40})
    first = load_scratchmath_sample(n=40, seed=42, _loader=loader)
    extra = load_scratchmath_extra(n_extra=30, seed=42, skip=40, _loader=loader)

    assert len(extra) == 30
    assert set(first["question_id"]).isdisjoint(set(extra["question_id"]))


def test_extra_truncates_at_the_end_of_the_pool(caplog):
    loader = _fake_loader_factory({"primary": 60, "middle": 40})
    extra = load_scratchmath_extra(n_extra=50, seed=42, skip=80, _loader=loader)

    assert len(extra) == 20
    assert "Returning 20 instead" in caplog.text


def test_grading_prompt_embeds_the_question_and_keeps_the_parse_contract():
    """The output contract must stay byte-identical to the FERMAT grading
    prompt's, so pilot.parsing.parse_grading and every entropy/scoring path
    downstream apply unchanged -- only the question differs, not the
    scoring path."""
    prompt = build_scratchmath_grading_prompt("一个等腰梯形的周长是120厘米")

    assert "一个等腰梯形的周长是120厘米" in prompt
    assert "**Reasoning:**" in prompt
    assert prompt.rstrip().endswith("**Error:** <0 or 1>")
    assert "{question}" not in prompt  # actually formatted, not left as a template


def test_grading_prompt_never_leaks_the_students_final_answer():
    """The load-bearing design constraint. ScratchMath ships student_answer
    as text; handing it to the model would let it check the arithmetic
    textually and reduce the image to decoration, which would stop testing
    the vision-grounded claim. The prompt takes a question only -- there is
    no parameter through which the answer could reach it."""
    import inspect

    params = inspect.signature(build_scratchmath_grading_prompt).parameters
    assert list(params) == ["question"]

    for text in (SCRATCHMATH_GRADING_SYSTEM_PROMPT, SCRATCHMATH_GRADING_USER_PROMPT):
        lowered = text.lower()
        assert "student_answer" not in lowered
        assert "the student's final answer" not in lowered


def test_grading_prompt_rejects_a_blank_question():
    """A blank question makes the item unanswerable rather than merely
    harder -- the image holds scratchwork only -- but would still parse and
    score as an ordinary result, so it has to fail loudly."""
    for blank in (None, "", "   ", "\n\t "):
        with pytest.raises(ValueError, match="non-empty question"):
            build_scratchmath_grading_prompt(blank)
