"""Regex-based extraction of the transcription answer and grading digit from raw model output."""

import re
from typing import Optional

_ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*(.*)", re.DOTALL)
_ERROR_RE = re.compile(r"\*\*Error:\*\*\s*([01])")


def parse_transcription(response_text: str) -> Optional[str]:
    """Extract the text between **Answer:** and the end of the response.

    The **Question:** field is intentionally never matched against, since the
    question is already known from orig_q.
    """
    match = _ANSWER_RE.search(response_text)
    if match is None:
        return None
    return match.group(1).strip()


def parse_grading(response_text: str) -> Optional[int]:
    """Extract the single binary digit (0 or 1) after **Error:**.

    Anything other than a literal 0 or 1 (e.g. a missing marker, or an
    out-of-range digit) returns None rather than a guessed value.
    """
    match = _ERROR_RE.search(response_text)
    if match is None:
        return None
    return int(match.group(1))
