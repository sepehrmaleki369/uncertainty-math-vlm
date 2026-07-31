from pilot.parsing import (
    grading_cluster_matches_label,
    grading_matches_label,
    parse_grading,
    parse_transcription,
)


def test_parse_transcription_basic_single_line():
    text = "**Question:**2+2\n**Answer:**4"
    assert parse_transcription(text) == "4"


def test_parse_transcription_multiline_latex():
    text = "**Question:**x^2\n**Answer:**\\frac{1}{2}x^2 + C\nextra trailing line"
    result = parse_transcription(text)
    assert result == "\\frac{1}{2}x^2 + C\nextra trailing line"


def test_parse_transcription_missing_marker_returns_none():
    text = "There is no answer marker here."
    assert parse_transcription(text) is None


def test_parse_transcription_empty_string_returns_none():
    assert parse_transcription("") is None


def test_parse_transcription_no_question_field_still_parses():
    text = "**Answer:** 42"
    assert parse_transcription(text) == "42"


def test_parse_transcription_strips_trailing_whitespace():
    text = "**Answer:**   4x + 1   \n\n"
    assert parse_transcription(text) == "4x + 1"


def test_parse_transcription_latex_textbf_bold():
    """Real drift observed in Qwen2.5-VL output: LaTeX bold instead of markdown."""
    text = "\\textbf{Question:} 2+2\n\n\\textbf{Answer:} 4"
    assert parse_transcription(text) == "4"


def test_parse_transcription_latex_textbf_merged_in_braces():
    text = "\\textbf{Question:} pick one\n\n\\textbf{Answer: Option C}"
    assert parse_transcription(text) == "Option C}"


def test_parse_transcription_plain_answer_no_markup():
    text = "Question: 2+2\n\nAnswer: 4"
    assert parse_transcription(text) == "4"


def test_parse_transcription_fullwidth_colon():
    text = "**Question：** 2+2\n**Answer：** 4"
    assert parse_transcription(text) == "4"


def test_parse_transcription_strips_trailing_code_fence():
    text = "```latex\n**Answer:** 4x + 1\n```"
    assert parse_transcription(text) == "4x + 1"


def test_parse_transcription_strips_trailing_end_document():
    text = (
        "\\documentclass{article}\n\\begin{document}\n"
        "**Answer:** 4x + 1\n\\end{document}"
    )
    assert parse_transcription(text) == "4x + 1"


def test_parse_transcription_missing_answer_field_entirely_still_none():
    """Genuine model non-compliance (e.g. '**Solution**' instead) must stay None."""
    text = "**Question:** 2+2\n\n**Solution**\nThe answer is 4."
    assert parse_transcription(text) is None


def test_parse_grading_error_zero():
    assert parse_grading("**Reasoning:** looks fine\n\n**Error:** 0") == 0


def test_parse_grading_error_one():
    assert parse_grading("**Reasoning:** wrong sign\n\n**Error:** 1") == 1


def test_parse_grading_missing_marker_returns_none():
    assert parse_grading("no error field present") is None


def test_parse_grading_invalid_digit_returns_none():
    assert parse_grading("**Error:** 2") is None


def test_parse_grading_ignores_reasoning_text():
    text = (
        "**Reasoning:** The student made an error dividing by zero in step 2, "
        "which invalidates the final answer.\n\n**Error:** 1"
    )
    assert parse_grading(text) == 1


def test_parse_grading_tolerant_of_extra_whitespace():
    assert parse_grading("**Error:**   1  \n") == 1


def test_grading_matches_label_compares_digit_to_bool_label():
    assert grading_matches_label(1, True)
    assert grading_matches_label(0, False)
    assert not grading_matches_label(1, False)
    assert not grading_matches_label(0, True)


def test_grading_matches_label_parse_failure_is_incorrect():
    assert not grading_matches_label(None, True)


def test_grading_cluster_matches_label_compares_majority_cluster_to_bool_label():
    assert grading_cluster_matches_label("1", True)
    assert grading_cluster_matches_label("0", False)
    assert not grading_cluster_matches_label("1", False)


def test_grading_cluster_matches_label_parse_failure_cluster_is_incorrect():
    assert not grading_cluster_matches_label("<PARSE_FAILURE>", True)
