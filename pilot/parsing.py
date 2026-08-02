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
_REASONING_RE = re.compile(r"\*\*Reasoning:\*\*\s*(.*?)\s*\*\*Error:\*\*", re.DOTALL)

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


def parse_grading_reasoning(response_text: str) -> Optional[str]:
    """Extract the free-text explanation between **Reasoning:** and **Error:**.

    Discovered on the real 2026-08-01 pilot run: the parsed **Error:** digit is
    sometimes inconsistent with the model's own reasoning -- 60/494 (12%) of
    samples have digit=0 while the reasoning text opens by flagging a mistake
    ("incorrect", "error", "mistake", ...). This looks like decoding noise on
    the final digit token, independent of the fully-formed judgment already
    written in the reasoning -- verified on real data by checking that samples
    with contradictory digits but semantically clustered reasoning genuinely
    argue the same direction (both say "there is an error", just about
    different specific details), not that the clustering merged opposed
    reasonings. See pilot.semantic for the clustering step this feeds into
    (e.g. semantic_cluster_entropy with nli_cluster_labels), which is not
    vulnerable to this specific last-token noise the way parse_grading alone
    is. Matched on 495/500 (99%) of real grading samples.
    """
    match = _REASONING_RE.search(response_text)
    if match is None:
        return None
    text = match.group(1).strip()
    return text or None


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
