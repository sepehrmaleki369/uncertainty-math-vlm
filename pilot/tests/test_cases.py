"""Tests for case snapshots (pilot.cases).

A case bundle is the evidence behind a claim -- the raw samples a reader
needs to see to believe it. These tests cover the properties that make a
bundle trustworthy: that it is recomputed rather than echoed from the CSV,
that a parse failure stays visible as a parse failure, that selection is
deterministic, and that image attachment never damages the text.
"""

import json

import pandas as pd
import pytest
from PIL import Image

from pilot.cases import (
    CASE_CATEGORIES,
    attach_image,
    build_case,
    select_cases,
    write_case,
    write_case_index,
)


def _grading_row(digits, has_error=1, correct=None, entropy=None):
    samples = [
        "**Reasoning:** looks off.\n\n**Error:** " + str(d) if d is not None
        else "**Reasoning:** no verdict given."
        for d in digits
    ]
    return pd.Series({
        "orig_q": "Solve for x.",
        "pert_a": "x = 4",
        "has_error": has_error,
        "grading_correct": correct if correct is not None else True,
        "reasoning_entropy": entropy if entropy is not None else 0.0,
        "all_grading_samples_raw": repr(samples),
        "model_id": "fake/model",
    })


def test_build_case_recomputes_rather_than_echoing_the_csv():
    """The bundle must be derivable from the raw samples alone -- otherwise
    it could not be used to check the CSV it came from."""
    row = _grading_row([1, 1, 1, 1, 1])
    case = build_case(row, "confidently_wrong_grading", arm="grading")

    assert case["model_output"]["k"] == 5
    assert case["model_output"]["parsed"] == ["1"] * 5
    assert case["derived"]["entropy"] == 0.0
    assert case["derived"]["majority_label"] == "1"
    assert case["derived"]["majority_count"] == 5
    assert case["derived"]["n_distinct_labels"] == 1


def test_parse_failures_stay_visible_as_parse_failures():
    """A None must be reported as a parse failure, not silently become just
    another disagreeing sample -- the difference is the parser vs the model."""
    row = _grading_row([1, 1, None, 0, 1])
    case = build_case(row, "high_entropy_wrong", arm="grading")

    assert case["model_output"]["n_parse_failures"] == 1
    assert case["model_output"]["parsed"][2] is None
    assert case["derived"]["entropy"] > 0


def test_build_case_rejects_an_unknown_category():
    """Categories are the reader's guarantee that a case was drawn
    systematically. A typo must fail, not create a new silent bucket."""
    row = _grading_row([1, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="unknown case category"):
        build_case(row, "cherry_picked_nice_example", arm="grading")


def test_every_declared_category_is_accepted():
    row = _grading_row([1, 0, 1, 0, 1])
    for category in CASE_CATEGORIES:
        case = build_case(row, category, arm="grading")
        assert case["category"] == category


def test_at_max_entropy_flag_matches_ln_k():
    row = _grading_row([1, 0, 1, 0, 1])  # 2 distinct labels of 5 -- not max
    assert build_case(row, "high_entropy_wrong")["derived"]["at_max_entropy"] is False

    row_max = pd.Series({
        "orig_q": "q", "pert_a": "a", "has_error": 1, "grading_correct": False,
        "all_grading_samples_raw": repr([f"**Error:** {d}" for d in (0, 1)]),
        "model_id": "fake/model",
    })
    case = build_case(row_max, "max_entropy_degenerate")
    assert case["derived"]["at_max_entropy"] is True


def test_selection_is_deterministic_and_picks_the_actual_extremes():
    """Ranked, not random: 'high entropy' must mean the top of the
    distribution, and the same CSV must always yield the same cases."""
    df = pd.DataFrame({
        "reasoning_entropy": [0.0, 0.2, 0.5, 1.0, 1.6],
        "grading_correct": [True, False, True, False, False],
        "all_grading_samples_raw": [repr(["**Error:** 1"] * 5)] * 5,
        "has_error": [1] * 5,
    })
    hi_wrong = select_cases(df, "high_entropy_wrong", arm="grading", n=1)
    assert hi_wrong.iloc[0]["reasoning_entropy"] == 1.6

    lo_wrong = select_cases(df, "low_entropy_wrong", arm="grading", n=1)
    assert lo_wrong.iloc[0]["reasoning_entropy"] == 0.2

    lo_right = select_cases(df, "low_entropy_correct", arm="grading", n=1)
    assert lo_right.iloc[0]["reasoning_entropy"] == 0.0

    again = select_cases(df, "high_entropy_wrong", arm="grading", n=1)
    assert again.iloc[0].equals(hi_wrong.iloc[0])


def test_selection_accepts_a_text_predicate_for_categories_numbers_cannot_express():
    """Non-engagement is defined by what the sample SAYS, not by entropy."""
    df = pd.DataFrame({
        "reasoning_entropy": [0.0, 0.0],
        "grading_correct": [True, True],
        "has_error": [1, 1],
        "all_grading_samples_raw": [
            repr(["The image does not contain any work.\n**Error:** 1"] * 5),
            repr(["The student divided incorrectly.\n**Error:** 1"] * 5),
        ],
    })
    picked = select_cases(
        df, "non_engagement_says_error", arm="grading", n=1,
        predicate=lambda r: "does not contain" in r["all_grading_samples_raw"],
    )
    assert len(picked) == 1
    assert "does not contain" in picked.iloc[0]["all_grading_samples_raw"]


def test_write_case_roundtrips_unicode_questions(tmp_path):
    """ScratchMath questions are Chinese; a bundle that mangled them would
    be useless for exactly the dataset that needed the most scrutiny."""
    row = _grading_row([1] * 5)
    row["orig_q"] = "一个等腰梯形的周长是120厘米"
    case = build_case(row, "confidently_wrong_grading", note="中文 note")
    d = write_case(case, tmp_path, "case_zh")

    loaded = json.loads((d / "case.json").read_text())
    assert loaded["ground_truth"]["question"] == "一个等腰梯形的周长是120厘米"
    assert loaded["note"] == "中文 note"


def test_attach_image_downscales_and_records_the_original_size(tmp_path):
    row = _grading_row([1] * 5)
    d = write_case(build_case(row, "low_entropy_wrong"), tmp_path, "c")
    big = Image.new("RGB", (4000, 2000), "white")

    path = attach_image(d, big, max_dim=1600)
    meta = json.loads((d / "case.json").read_text())

    assert path.exists()
    assert meta["image"]["width"] == 1600
    assert meta["image"]["original_width"] == 4000
    assert meta["image"]["downscaled"] is True
    assert meta["image"]["bytes"] == path.stat().st_size


def test_attach_image_preserves_the_text_bundle(tmp_path):
    """Attaching an image must only ADD to case.json. If it could clobber
    the samples or the note, a re-run of the image step would destroy the
    substance of the bundle."""
    row = _grading_row([1, 0, 1, 1, 1])
    case = build_case(row, "high_entropy_wrong", note="a human explanation")
    d = write_case(case, tmp_path, "c")
    before = json.loads((d / "case.json").read_text())

    attach_image(d, Image.new("RGB", (100, 50), "white"))
    after = json.loads((d / "case.json").read_text())

    for key in ("note", "category", "ground_truth", "model_output", "derived"):
        assert after[key] == before[key]
    assert "image" in after


def test_attach_image_leaves_no_stale_file_from_the_other_encoding(tmp_path):
    """Format is chosen per image by size, so a re-attach can flip png<->jpg.
    Both must never be present at once, or the bundle has two images and
    case.json names only one."""
    row = _grading_row([1] * 5)
    d = write_case(build_case(row, "low_entropy_wrong"), tmp_path, "c")

    attach_image(d, Image.new("RGB", (80, 40), "white"))
    attach_image(d, Image.new("RGB", (80, 40), "white"))

    images = sorted(p.name for p in d.iterdir() if p.suffix in (".png", ".jpg"))
    assert len(images) == 1
    assert json.loads((d / "case.json").read_text())["image"]["filename"] == images[0]


def test_case_index_lists_every_bundle_and_flags_missing_images(tmp_path):
    """The index is the interface for the manual failure audit, which is a
    reading task -- it has to say which cases still lack an image."""
    row = _grading_row([1] * 5)
    write_case(build_case(row, "low_entropy_wrong", note="n1"), tmp_path, "a")
    d2 = write_case(build_case(row, "high_entropy_wrong", note="n2"), tmp_path, "b")
    attach_image(d2, Image.new("RGB", (60, 30), "white"))

    index = json.loads(write_case_index(tmp_path).read_text())
    by_id = {e["case_id"]: e for e in index}

    assert set(by_id) == {"a", "b"}
    assert by_id["a"]["has_image"] is False
    assert by_id["b"]["has_image"] is True
    assert by_id["a"]["note"] == "n1"
