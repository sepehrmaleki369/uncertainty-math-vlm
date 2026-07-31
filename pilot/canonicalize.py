"""Math-aware canonicalization for clustering transcribed answers.

Exact-string clustering (pilot/entropy.py's default normalize_string) treats
`\\log|x+1|` and `\\log |{x+1}|` as different answers, even though they are the
same expression -- on the real pilot run this made perception_entropy sit at
its maximum (ln 5) for 79/100 items, carrying no signal at all.

This module tries three tiers, cheapest and most reliable first:
1. Structural cleanup: strip LaTeX wrappers (\\[ \\], \\(\\), \\text{}, \\textcolor{},
   align environments), collapse whitespace, lowercase.
2. Where the cleaned text is a single mathematical expression, parse it with
   SymPy's LaTeX parser and use the resulting expression's canonical string
   form -- this is what actually collapses `\\log|x+1|` and `\\log |{x+1}|`
   into the same cluster, since both parse to the identical AST.
3. Where SymPy can't parse it (most items here: the transcription prompt asks
   for the FULL handwritten answer, which is often a multi-step derivation
   with prose, not a single expression -- SymPy correctly fails on prose),
   fall back to the structurally-cleaned string from step 1.

Semantic clustering (embedding/NLI-based) is the next upgrade if this proves
insufficient -- not needed for this pass.
"""

import re
from typing import Optional

import sympy
from sympy.parsing.latex import parse_latex

from pilot.entropy import PARSE_FAILURE_SENTINEL

# Above this length, the text is almost certainly a multi-step derivation
# with prose rather than a single expression -- skip the SymPy attempt
# rather than spend time on a parse that will fail anyway.
_MAX_SYMPY_LENGTH = 300

_WRAPPER_PATTERNS = [
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),  # unwrap \text{...}, keep content
    (re.compile(r"\\textcolor\{[^{}]*\}\{([^{}]*)\}"), r"\1"),  # unwrap \textcolor{c}{...}
    (re.compile(r"\\begin\{align\*?\}|\\end\{align\*?\}"), " "),
    (re.compile(r"\\\[|\\\]|\\\(|\\\)"), " "),
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\quad|\\qquad"), " "),
    (re.compile(r"&|\\\\"), " "),
]


def _structural_clean(text: str) -> str:
    """Strip LaTeX display/environment wrappers and collapse whitespace."""
    for pattern, repl in _WRAPPER_PATTERNS:
        text = pattern.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip().lower().rstrip(".")
    return text


def canonicalize_math(text: Optional[str]) -> str:
    """Canonicalize a transcribed answer for clustering.

    None (a parse failure) maps to the same sentinel pilot.entropy uses, so
    it is treated as its own distinguishable cluster rather than dropped --
    consistent with pilot.entropy.normalize_string's handling.
    """
    if text is None:
        return PARSE_FAILURE_SENTINEL

    cleaned = _structural_clean(text)
    if not cleaned:
        return PARSE_FAILURE_SENTINEL

    if len(cleaned) <= _MAX_SYMPY_LENGTH:
        try:
            expr = parse_latex(cleaned)
            # Prose containing stray "=" and word-like tokens (e.g. "rate = R
            # percent and time = R years") can parse "successfully" into a
            # bare True/False rather than raising -- observed on real pilot
            # data. That's not a meaningful math extraction, so treat it the
            # same as a parse failure and fall through to the text fallback.
            if expr in (sympy.true, sympy.false):
                raise ValueError("parsed to a bare boolean, not a math expression")
            # Prose with no LaTeX math at all (e.g. "this can be simplified
            # to ...") gets tokenized letter-by-letter into implicit
            # multiplication of single-character symbols -- observed
            # spelling out "thiscanbesimplifiedto" as t*h*i*s*c*a*n*...
            # Legitimate expressions in this dataset use at most a handful of
            # single-letter variables (x, y, z, C, R, ...), so treat more
            # than a few as the same signature and fall back.
            single_char_symbols = [s for s in expr.free_symbols if len(str(s)) == 1]
            if len(single_char_symbols) > 4:
                raise ValueError("too many single-character symbols -- likely spelled-out prose")
            return f"sympy:{expr}"
        except Exception:
            pass  # not a parseable single expression -- fall through

    return f"text:{cleaned}"
