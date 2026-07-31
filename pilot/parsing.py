"""Regex-based extraction of the transcription answer and grading digit from raw model output."""

import re
from typing import Optional

# Tried in order. The first is the canonical format the prompt asks for;
# the rest recover the real-world drift observed in actual Qwen2.5-VL output
# (see pilot run 2026-07-31): LaTeX \textbf{} bold instead of markdown **, an
# optional parenthesized "(Answer)" label, plain text with no markup, and an
# occasional full-width "：" in place of ":". A response that never contains
# any Answer field at all (e.g. the model wrote "**Solution**" instead)
# correctly falls through all patterns and still returns None.
_ANSWER_PATTERNS = [
    re.compile(r"\*\*Answer\s*[:：]?\s*\*\*\s*(.*)", re.DOTALL),
    re.compile(r"\\textbf\{\(?Answer\)?\s*[:：]?\s*\}?\s*[:：]?\s*(.*)", re.DOTALL),
    re.compile(r"(?:^|\n)\s*Answer\s*[:：]\s*(.*)", re.DOTALL),
]
_ERROR_RE = re.compile(r"\*\*Error:\*\*\s*([01])")

# Trailing junk left over when the answer was wrapped in a code fence or a
# full \documentclass{...}\begin{document}...\end{document} block -- cut
# anything from the first of these onward.
_TRAILING_JUNK_RE = re.compile(r"```|\\end\{document\}")


def parse_transcription(response_text: str) -> Optional[str]:
    """Extract the transcribed answer text, tolerating format drift.

    Per the spec, this pulls everything from the Answer marker to the end of
    the response (the **Question:** field is intentionally never matched
    against, since the question is already known from orig_q) -- but the
    marker itself is matched against several real-world variants, not only
    the exact `**Answer:**` the prompt requests.
    """
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(response_text)
        if match is not None:
            break
    else:
        return None

    text = match.group(1)
    junk = _TRAILING_JUNK_RE.search(text)
    if junk is not None:
        text = text[: junk.start()]
    text = text.strip()
    return text or None


def parse_grading(response_text: str) -> Optional[int]:
    """Extract the single binary digit (0 or 1) after **Error:**.

    Anything other than a literal 0 or 1 (e.g. a missing marker, or an
    out-of-range digit) returns None rather than a guessed value.
    """
    match = _ERROR_RE.search(response_text)
    if match is None:
        return None
    return int(match.group(1))


def grading_matches_label(predicted_error: Optional[int], has_error: bool | int) -> bool:
    """Compare a parsed grading digit against the boolean/integer truth label."""
    if predicted_error is None:
        return False
    return int(predicted_error) == int(has_error)


def grading_cluster_matches_label(majority_cluster: str, has_error: bool | int) -> bool:
    """Compare a majority grading cluster string against the truth label."""
    if majority_cluster not in {"0", "1"}:
        return False
    return grading_matches_label(int(majority_cluster), has_error)
