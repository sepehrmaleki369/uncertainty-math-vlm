from pilot.parsing import parse_grading, parse_transcription


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
