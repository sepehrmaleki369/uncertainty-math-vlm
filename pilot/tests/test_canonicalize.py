from pilot.canonicalize import canonicalize_math, extract_final_answer
from pilot.entropy import PARSE_FAILURE_SENTINEL, cluster_entropy


def test_none_maps_to_parse_failure_sentinel():
    assert canonicalize_math(None) == PARSE_FAILURE_SENTINEL


def test_sympy_collapses_bracket_and_spacing_variants():
    """Real drift observed in the pilot run: identical math, different LaTeX spacing."""
    a = canonicalize_math(r"\frac{11}{4} \log |x+1| + C")
    b = canonicalize_math(r"\frac{11}{4} \log|{x+1}| + C")
    assert a == b
    assert a.startswith("sympy:")


def test_sympy_result_differs_for_different_math():
    a = canonicalize_math(r"\frac{1}{2}x")
    b = canonicalize_math(r"\frac{1}{3}x")
    assert a != b


def test_prose_derivation_falls_back_to_structural_text():
    """The transcription prompt asks for the full handwritten answer, which is
    often a multi-step derivation with prose -- SymPy can't parse a sentence,
    so this must fall back rather than raise or silently drop the sample."""
    text = "Let rate = R percent and time = R years. Then the equation holds."
    result = canonicalize_math(text)
    assert result.startswith("text:")
    assert "let rate" in result


def test_structural_fallback_unwraps_text_and_textcolor():
    a = canonicalize_math(r"\textcolor{red}{Option B} \text{ is correct}")
    b = canonicalize_math("Option B is correct")
    assert a == b


def test_structural_fallback_strips_display_math_delimiters():
    a = canonicalize_math(r"\[ Option B \]")
    b = canonicalize_math("Option B")
    assert a == b


def test_structural_fallback_strips_align_environment():
    a = canonicalize_math(r"\begin{align*} some derivation text \end{align*}")
    b = canonicalize_math("some derivation text")
    assert a == b


def test_empty_after_cleanup_maps_to_sentinel():
    assert canonicalize_math(r"\[ \]") == PARSE_FAILURE_SENTINEL


def test_very_long_text_skips_sympy_and_uses_structural_fallback():
    long_prose = "this is a long worked derivation. " * 20
    result = canonicalize_math(long_prose)
    assert result.startswith("text:")


def test_prose_with_stray_equals_sign_falls_back_not_spurious_bool():
    """Real drift observed on pilot data: 'rate = R and time = R' parses
    'successfully' to a bare sympy True/False rather than raising."""
    result = canonicalize_math("rate = R percent and time = R years")
    assert result.startswith("text:")


def test_prose_spelled_out_as_symbols_falls_back():
    """Real drift observed on pilot data: sympy tokenized 'this can be
    simplified to' letter-by-letter into t*h*i*s*c*a*n*... rather than
    raising, since 'to' followed by '(' looks like a function call."""
    result = canonicalize_math("this can be simplified to (zx-y)^2")
    assert result.startswith("text:")
    assert "t*h*i*s" not in result


def test_extract_final_answer_none_returns_none():
    assert extract_final_answer(None) is None


def test_extract_final_answer_empty_string_returns_none():
    assert extract_final_answer("   ") is None


def test_extract_final_answer_prefers_explicit_option_over_everything():
    """MCQ items: the stated option is the substantive answer, even if a
    derivation with its own final equation follows it."""
    text = "Option B is correct.\n\\[ 12R^2 = 432 \\Rightarrow R = 6 \\]"
    assert extract_final_answer(text) == "option b"


def test_extract_final_answer_uses_last_option_if_model_corrects_itself():
    text = "Option A seems right. Wait, re-checking: Option C is correct."
    assert extract_final_answer(text) == "option c"


def test_extract_final_answer_boxed_result():
    text = r"We derive step by step. \boxed{42}"
    assert extract_final_answer(text) == "42"


def test_extract_final_answer_last_display_math_block():
    text = r"First \[ x = 1 \] then substituting \[ y = 2x = 2 \]"
    assert extract_final_answer(text) == "y = 2x = 2"


def test_extract_final_answer_single_dollar_inline_math():
    """Real bug found on pilot data: single-dollar $...$ inline math (as
    opposed to \\[...\\] or $$...$$) wasn't matched by any pattern at all,
    causing the extractor to fall through to a garbled last-line result."""
    text = r"Let $\vec{a} = 3\hat{i}$ and $\vec{b} = \frac{5}{2}\hat{i} - \hat{j}$"
    assert extract_final_answer(text) == r"\vec{b} = \frac{5}{2}\hat{i} - \hat{j}"


def test_extract_final_answer_skips_empty_trailing_align_environment():
    """Real bug found on pilot data: a dangling \\end{align*} with no
    captured content was being returned verbatim as the 'final answer'
    instead of falling back to the actual last meaningful line."""
    text = "\\begin{align*}\nx = 5\n\\end{align*}\nThe answer is 5."
    result = extract_final_answer(text)
    assert result is not None
    assert "end{align" not in result


def test_extract_final_answer_last_line_fallback_for_plain_text():
    text = "We compute step by step.\nThe final result is 17"
    assert extract_final_answer(text) == "The final result is 17"


def test_integrates_with_cluster_entropy_via_normalize_fn():
    """The actual fix this module exists for: these three variants of the
    same integral answer were three separate clusters under exact-string
    matching (contributing to the degenerate ln(5)-for-everything result on
    the real pilot run) and must collapse to one cluster here."""
    samples = [
        r"\frac{11}{4} \log |x+1| + C",
        r"\frac{11}{4} \log|{x+1}| + C",
        r"\frac{11}{4}\log |x+1|+C",
    ]
    assert cluster_entropy(samples, normalize_fn=canonicalize_math) == 0.0
