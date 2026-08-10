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

Even canonicalized, most items here still saturate at max entropy: the
transcription prompt asks for the FULL derivation, and five independent
generations paraphrase it differently ("we get" vs "we find", an echoed
"Answer:" prefix present or absent) even when the underlying math is
identical. structural_clean's output feeds pilot.semantic's NLI/embedding
clustering, which is the actual fix for that -- this module only handles the
syntactic tier.
"""

import logging
import re
from typing import Optional

import sympy
from sympy.parsing.latex import parse_latex

from pilot.entropy import PARSE_FAILURE_SENTINEL, normalize_string

logger = logging.getLogger(__name__)


def latex_parser_available() -> bool:
    """Whether SymPy's LaTeX parser actually works in this environment.

    parse_latex needs the antlr4 runtime at a version matching SymPy's, and
    raises ImportError at call time -- not import time -- when it is missing or
    mismatched. canonicalize_math catches that and falls back to the plain-text
    tier, which is correct behaviour but silent: entropy comes out higher and
    accuracy lower, with no error anywhere.

    That silence cost us real confusion. On the 2026-08-02 n=300 Colab run, 43
    of 300 items scored differently from a local re-run of the identical raw
    samples (118/300 correct vs 141/300), because the Colab environment parsed
    less successfully. Call this at the start of any scoring run and record the
    answer alongside the results.
    """
    try:
        parse_latex("x = 1")
        return True
    except Exception:  # ImportError, antlr version errors, parser internals
        return False


def warn_if_latex_parser_missing() -> bool:
    """Log a loud warning if canonicalization will silently degrade. Returns availability."""
    available = latex_parser_available()
    if not available:
        logger.warning(
            "SymPy's LaTeX parser is unavailable (needs antlr4-python3-runtime==4.11). "
            "canonicalize_math will fall back to plain-text matching for every answer, "
            "which inflates entropy and deflates accuracy. Results will NOT be "
            "comparable to runs where the parser worked. Install the pinned "
            "dependency before scoring."
        )
    return available

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


def structural_clean(text: str) -> str:
    """Strip LaTeX display/environment wrappers and collapse whitespace.

    KNOWN DEFECT, deliberately not fixed here: the \\text{}/\\textcolor{}{}
    patterns above use [^{}]* for the body, so they match only when the body
    contains no braces of its own. \\textcolor{red}{\\hat{b}} is left
    untouched, and "\\textcolor{red}" survives into the cluster label.

    On the 2026-08-02 n=300 run that is 85/300 ground truths, 59 of them
    has_error=1 -- FERMAT marks the *injected error* in red, so the defect
    lands disproportionately on exactly the items the error label depends on.

    It is not fixed in place because this function defines the scoring rule
    that produced every locked result and all of reference/*.json; changing
    it would silently invalidate those and make the next run incomparable to
    the previous ones. unwrap_latex_macro below is the correct implementation,
    and pilot.rescore applies it as an explicit, versioned alternative rule
    that is reported alongside the frozen one rather than replacing it.
    """
    for pattern, repl in _WRAPPER_PATTERNS:
        text = pattern.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip().lower().rstrip(".")
    return text


def unwrap_latex_macro(text: str, macro: str) -> str:
    """Replace every \\macro{...}{...} with its LAST braced argument.

    A real brace counter, so it handles the nested bodies structural_clean's
    regex silently skips: \\textcolor{red}{\\hat{b}} -> \\hat{b}, and
    \\text{cm}^2 -> cm^2. Taking the last argument is what makes one function
    serve both the one-argument form (\\text{B} -> B) and the two-argument
    form (\\textcolor{colour}{B} -> B).

    A macro with no braced argument at all is left verbatim rather than
    dropped -- "\\textcolor" alone is malformed input, and deleting it would
    quietly change the expression instead of preserving the anomaly.
    """
    out: list[str] = []
    token = "\\" + macro
    i = 0
    while i < len(text):
        start = text.find(token, i)
        if start < 0:
            out.append(text[i:])
            break
        # Don't match a longer macro name that merely starts with this one
        # (\\textcolor must not be consumed by macro="text").
        after = start + len(token)
        if after < len(text) and (text[after].isalpha() or text[after] == "*"):
            out.append(text[i:after])
            i = after
            continue
        out.append(text[i:start])

        args: list[str] = []
        cursor = after
        while cursor < len(text) and text[cursor] == "{":
            depth = 0
            body_start = cursor
            while cursor < len(text):
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        cursor += 1
                        break
                cursor += 1
            else:  # unbalanced -- leave the rest alone
                args = []
                break
            args.append(text[body_start + 1:cursor - 1])

        if not args:
            out.append(token)
            i = after
            continue
        out.append(args[-1])
        i = cursor
    return "".join(out)


_OPTION_RE = re.compile(r"option\s*[:\-]?\s*([a-dA-D])\b", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
# $$...$$ and \(...\) must be tried before the single-$ pattern -- Python's re
# tries alternatives in written order at each position, so listing $$ first
# means a "$$...$$" span matches whole rather than as two spurious single-$
# matches (observed on real data: item 17's inline $\vec{...}$ math wasn't
# matched by any pattern at all before this single-$ branch was added).
_DISPLAY_MATH_RE = re.compile(
    r"\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\\((.*?)\\\)|\$([^$]*?)\$", re.DOTALL
)
_ENV_MARKER_RE = re.compile(r"^\\(begin|end)\{[^{}]*\}$")


# The last-line tier splits on "." as a sentence terminator, which also splits
# DECIMAL NUMBERS: "the area is 75.46 cm." yields "46 cm". fix_decimal_split
# uses this instead, which requires a non-digit on at least one side.
_LAST_LINE_SPLIT_RE = re.compile(r"[\n.]")
_LAST_LINE_SPLIT_FIXED_RE = re.compile(r"\n|(?<!\d)\.|\.(?!\d)")


def extract_final_answer(
    text: Optional[str], fix_decimal_split: bool = False
) -> Optional[str]:
    """Pull just the final answer/result out of a full derivation, tiered by
    reliability: an explicit multiple-choice option statement, a \\boxed{}
    result, the last display-math block, or (rarest) the last non-empty line.

    Clustering the WHOLE derivation is what made both canonicalize_math and
    semantic (NLI/embedding) clustering fail on the real pilot corpus:
    syntactic matching is too strict because samples paraphrase the
    scaffolding differently, and semantic matching is too lenient because
    that same shared scaffolding triggers an entailment-model lexical-overlap
    bias, burying the one number that actually differs. Comparing only the
    final answer removes the scaffolding from the comparison entirely.

    KNOWN DEFECT in the last-line tier, off by default: it splits on "." as a
    sentence terminator, so a decimal answer is truncated at the point --
    "the estimated population is 23152.5 square meters." extracts as
    "5 square meters". On the 2026-08-02 n=300 run that hits 2/300 ground
    truths and 13/1500 samples (low only because most items reach an earlier
    tier first). It corrupted BOTH sides of item 101 identically, which then
    scored as agreement under a relaxed rule -- a false pass, not a recovered
    read.

    fix_decimal_split=True is the corrected behaviour. It defaults to False
    because this function defines the frozen scoring rule behind every locked
    result and all of reference/*.json; see structural_clean's docstring for
    the same reasoning. pilot.rescore turns it on from fixed_v2 onward.
    """
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None

    option_matches = list(_OPTION_RE.finditer(cleaned))
    if option_matches:
        return f"option {option_matches[-1].group(1).lower()}"

    boxed_matches = list(_BOXED_RE.finditer(cleaned))
    if boxed_matches:
        return boxed_matches[-1].group(1).strip()

    # Walk matches newest-to-oldest and take the first with real content --
    # the literal last match is sometimes an empty or align-environment
    # block (e.g. a lone "\end{align*}" left dangling with no captured text),
    # which would otherwise silently fall through to the much weaker
    # last-line tier below even though a good earlier match exists.
    for match in reversed(_DISPLAY_MATH_RE.findall(cleaned)):
        content = next((g.strip() for g in match if g and g.strip()), None)
        if content:
            return content

    splitter = _LAST_LINE_SPLIT_FIXED_RE if fix_decimal_split else _LAST_LINE_SPLIT_RE
    lines = [line.strip() for line in splitter.split(cleaned) if line.strip()]
    lines = [line for line in lines if not _ENV_MARKER_RE.match(line)]
    if lines:
        return lines[-1]
    return None


# A run of >=3 plain letters that is not a LaTeX command. \frac, \sin, \circ
# are all backslash-prefixed and so excluded; genuine implicit-multiplication
# variables (xy, dx, dt) are 2 letters and so allowed.
_BARE_WORD_RE = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{3,}")


def sympy_parse_is_trustworthy(cleaned: str, expr) -> bool:
    """Whether a successful parse_latex result actually represents the input.

    parse_latex does not fail loudly on input it only partly understands: it
    parses a PREFIX and silently returns it. Both real failure modes on this
    corpus are that one bug wearing different clothes.

    1. Prose. "hence, the required number of words is 24" parses to
       h*(e*(n*(c*e))) -- SymPy read the word "Hence" as five multiplied
       variables, stopped at the comma, and threw the answer (24) away.
       canonicalize_math's existing guard counts DISTINCT single-character
       symbols and requires more than four; "hence" has exactly four (h, e,
       n, c -- the e repeats), so it slips underneath.

    2. Truncation at an unparseable token. "40^\\circ 20' = \\frac{121\\pi}{540}"
       parses to 40**circ*20: everything from the apostrophe onward, including
       the "=" and the entire answer, is dropped. This one is the more
       damaging, because it is silent AND collapsing -- \\frac{1210}{540} and
       \\frac{121\\pi}{540} are different answers that both reduce to the same
       label, which deflates entropy and can manufacture a false match.

    Two checks, one per mode: reject bare multi-letter words, and reject a
    parse that dropped an "=" the input clearly had. Rejection means falling
    back to the plain-text tier, which compares the string as written -- less
    clever, but never silently wrong.

    On the 2026-08-02 n=300 run this affects 37/300 ground truths and 44/1500
    samples. Not applied to the frozen scoring rule; see structural_clean.
    """
    if _BARE_WORD_RE.search(cleaned):
        return False
    # An input asserting an equation must parse to a relation. If it did not,
    # the "=" and everything after it was discarded.
    if "=" in cleaned and not isinstance(expr, sympy.core.relational.Relational):
        return False
    return True


def canonicalize_math(text: Optional[str], strict_parse: bool = False) -> str:
    """Canonicalize a transcribed answer for clustering.

    None (a parse failure) maps to the same sentinel pilot.entropy uses, so
    it is treated as its own distinguishable cluster rather than dropped --
    consistent with pilot.entropy.normalize_string's handling.
    """
    if text is None:
        return PARSE_FAILURE_SENTINEL

    cleaned = structural_clean(text)
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
            # strict_parse additionally rejects a parse that only covered a
            # PREFIX of the input -- the guard above counts distinct symbols
            # and so misses four-letter words like "hence", and cannot see a
            # dropped "=" at all. See sympy_parse_is_trustworthy.
            if strict_parse and not sympy_parse_is_trustworthy(cleaned, expr):
                raise ValueError("parse does not represent the whole input")
            return f"sympy:{expr}"
        except Exception:
            pass  # not a parseable single expression -- fall through

    return f"text:{cleaned}"


def canonical_answer_label(text: Optional[str]) -> str:
    """Full pipeline from raw transcription text to a perception cluster label.

    extract_final_answer -> canonicalize_math -> normalize_string, in that
    order. Exists so the K sampled transcriptions and the ground-truth answer
    are put through *exactly* the same transformation before being compared.

    That symmetry is the whole point. pilot.entropy.majority_cluster
    normalizes the labels it returns (lowercasing among other things), so
    comparing its output against a ground truth that skipped normalize_string
    silently fails on any answer whose canonical form contains an uppercase
    character. On the real n=100 data that meant every sympy-parsed equation
    ("sympy:Eq(x, 1)" vs. "sympy:eq(x, 1)") counted as a mismatch, scoring
    36/100 correct instead of 42/100 and dropping perception AUROC from 0.788
    to 0.757 -- a silent, plausible-looking wrong answer. Route both sides
    through this function rather than composing the three steps by hand.
    """
    return normalize_string(canonicalize_math(extract_final_answer(text)))
