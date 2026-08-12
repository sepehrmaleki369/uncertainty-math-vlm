"""`strict_v2_display_primary` -- score on the display span, demote SymPy to a warning.

The 2026-08-12 false-pass audit showed why this rule is needed. Of the 20
`strict_v1`-CORRECT items a human ruled unearned, **the failure is almost
always the SPAN, not the normalizer**: 6 were a one-character span the
extractor chose (item 84's span is literally `c`), 8 were a partial span
(item 27 keeps `4x = 3` and drops `y = 33/4`), and only 2 were SymPy
collapsing a long span to one symbol. Two collapsed labels then MATCH and the
item scores correct for no reason at all.

So this rule inverts the authority:

* **PRIMARY** -- the normalized display SPAN, model against truth. That is the
  text a human reads off the page, and it is what the audit adjudicated.
* **HELPER ONLY** -- SymPy. It never decides correctness here. It contributes
  `sympy_match` and three warnings, because a SymPy agreement on top of a span
  agreement is reassuring and a SymPy *dis*agreement is worth a second look.

Everything else is a RISK FLAG, not a verdict. A flag says "this is the shape
of item the automatic scorer gets wrong", so a human pass can be aimed at 30
items instead of 300. `tiny_valid_mcq` and `tiny_suspicious_non_mcq` exist as
separate flags for exactly this reason -- a one-letter span `B` is a perfectly
good answer to a multiple-choice item and near-worthless anywhere else, and
collapsing them into one "short answer" flag would hide that.

**Human visual reading overrides every label in this module**, and where the
display span is richer than the SymPy label, the span is what to believe.

NOTHING HERE TOUCHES THE FROZEN PIPELINE. `rescore.RULES` is unchanged and
`strict_v1` still produces every locked number; this is an alternative
reported alongside, in the same spirit as `pilot.rescore`'s other rules.
"""

import ast
import re
from typing import Optional, Sequence

import pandas as pd

from . import canonicalize, entropy, parsing, rescore

RULE_NAME = "strict_v2_display_primary"

#: Metadata SymPy is still allowed to contribute. None of these decide scoring.
SYMPY_FLAGS = ("sympy_match", "sympy_partial_parse_risk",
               "sympy_malformed_derivative", "multi_answer_collapse")

#: Answer-shape flags. Each marks a kind of item the automatic scorer is known
#: to mishandle; they are diagnostic, never a verdict.
RISK_FLAGS = ("mcq_option", "tiny_valid_mcq", "tiny_suspicious_non_mcq",
              "multi_value_answer", "set_answer", "system_answer",
              "derivative_equation", "text_conclusion")

ALL_FLAGS = SYMPY_FLAGS + RISK_FLAGS

_OPTION_RE = re.compile(r"\boption\b", re.I)
_BARE_OPTION_RE = re.compile(r"^\(?\s*(?:option\s*)?[a-eA-E]\s*\)?[.)]?$")
# Set notation, NOT "any LaTeX braces" -- \frac{a}{b} has braces and is not a
# set. A first version matched bare {...} and fired on 45% of items, which made
# the review queue useless. Detected on the RAW span, because normalization
# turns \{ into { and the distinction is then unrecoverable.
_SET_RE = re.compile(r"\\[{}]|\\(?:cup|cap|setminus|subset|supset|emptyset)\b"
                     r"|\\in\b|(?<![a-zA-Z])\{[^{}]*,[^{}]*\}")
_DERIV_RE = re.compile(r"\\frac\s*\{\s*d(?:\^?\d?)?\s*[a-z]?\s*\}|"
                       r"\bd\^?2?\s*y\s*/\s*d\s*x|\\mathrm\{d\}|"
                       r"derivative\(|\bdy\s*/\s*dx\b", re.I)
_MALFORMED_DERIV_RE = re.compile(r"d\*\*\d\s*\*\s*[a-z]|/\s*\(?\s*dx\s*\*\*|"
                                 r"\bdx\*\*\d", re.I)
# Words that mean the span is a sentence rather than a value. SymPy's own
# function names are excluded -- `eq`, `log`, `tan` are not prose, and treating
# them as prose is the exact over-trigger that once mislabelled item 55.
_SYMPY_WORDS = {"eq", "log", "tan", "sin", "cos", "cot", "sec", "csc", "exp",
                "sqrt", "abs", "integral", "derivative", "limit", "sum", "pi",
                "oo", "re", "im", "conjugate", "matrix", "and", "or", "true",
                "false"}
_WORD_RE = re.compile(r"[A-Za-z]{3,}")

_SPACING_MACROS = re.compile(r"\\(?:,|;|:|!|quad|qquad|ensuremath|displaystyle"
                             r"|left|right|bigl|bigr|hspace\{[^}]*\})")


def normalize_span(span: Optional[str]) -> str:
    """Cosmetic normalization of a display span, and nothing more.

    Deliberately conservative: it removes LaTeX spacing and wrapper macros,
    unifies bracket and multiplication spellings, drops trailing punctuation
    and collapses whitespace. It does NOT reorder terms, evaluate arithmetic
    or canonicalize equations -- those are exactly the operations that let two
    different answers land on the same label, which is what this rule exists
    to stop.
    """
    if span is None:
        return ""
    s = str(span)
    for macro in ("textcolor", "text", "mathrm", "mathbf", "textbf", "mbox",
                  "operatorname"):
        s = canonicalize.unwrap_latex_macro(s, macro)
    s = _SPACING_MACROS.sub(" ", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = re.sub(r"\\[a-zA-Z]+\s*", lambda m: m.group(0).strip() + " ", s)
    s = s.replace("$", "").replace("\\\\", " ")
    s = re.sub(r"[ \t\n\r]+", " ", s)
    s = s.strip().rstrip(".;,")
    s = re.sub(r"\s*([=+\-*/,])\s*", r"\1", s)
    return s.lower().strip()


def _looks_mcq(question: Optional[str], gt_span: Optional[str],
               gt_field: Optional[str]) -> bool:
    blob = " ".join(str(x) for x in (question, gt_span, gt_field) if x)
    if _OPTION_RE.search(blob):
        return True
    return bool(re.search(r"\(\s*[a-d]\s*\)[^\n]{0,60}\(\s*[b-e]\s*\)", blob, re.I))


def _is_prose(span: str) -> bool:
    words = [w.lower() for w in _WORD_RE.findall(span)]
    real = [w for w in words if w not in _SYMPY_WORDS]
    return len(real) >= 3


def answer_flags(span: str, sympy_label: str, is_mcq: bool) -> dict:
    """Answer-shape and SymPy-risk flags for one side of a comparison."""
    norm = normalize_span(span)
    bare = norm.strip()
    n_eq = len(re.findall(r"=", bare))
    # A SYSTEM is several separate equations, not one chain. `a=b=c` has two
    # "=" and is a single statement; `x=3, y=1` is two. Counting raw "=" made
    # this fire on 31% of items and drowned the queue.
    parts = [p for p in re.split(r",|;|\band\b|\\quad", bare) if p.strip()]
    n_eq_parts = sum(1 for p in parts if "=" in p)
    commas = [p for p in re.split(r",", bare) if p.strip()]
    payload = str(sympy_label).split(":", 1)[-1] if sympy_label else ""

    tiny = len(bare) <= 3
    # A bare letter is only an "option answer" when the item actually IS
    # multiple-choice. Without that guard item 289's set-equality answer `B`
    # reads as an option and the tiny-span warning it needs gets contradicted.
    mcq_option = bool(_OPTION_RE.search(bare)) or (
        is_mcq and bool(_BARE_OPTION_RE.match(bare)))
    multi_value = len(commas) >= 2 and n_eq == 0 and len(bare) > 3
    system = n_eq_parts >= 2

    return {
        "mcq_option": mcq_option,
        "tiny_valid_mcq": tiny and is_mcq,
        "tiny_suspicious_non_mcq": tiny and not is_mcq,
        "multi_value_answer": multi_value,
        "set_answer": bool(_SET_RE.search(str(span))),
        "system_answer": system,
        "derivative_equation": bool(_DERIV_RE.search(str(span))),
        "text_conclusion": _is_prose(bare),
        # SymPy risk: the label carries strictly less than the span did.
        "sympy_partial_parse_risk": bool(
            payload and system and payload.count("eq(") <= 1),
        "sympy_malformed_derivative": bool(
            _MALFORMED_DERIV_RE.search(payload)),
        "multi_answer_collapse": bool(
            payload and multi_value and len(payload) <= max(3, len(bare) // 3)),
    }


def score_item_v2(raw_samples: Sequence[str], ground_truth: Optional[str],
                  question: Optional[str] = None) -> dict:
    """Score one item on the display span, with SymPy demoted to metadata."""
    v1 = rescore.trace_item(raw_samples, ground_truth, "strict_v1")

    gt_field = v1["ground_truth"]["answer_field"]
    gt_span = v1["ground_truth"]["final_answer"]
    gt_norm = normalize_span(gt_span)
    is_mcq = _looks_mcq(question, gt_span, gt_field)

    spans = [s["final_answer"] for s in v1["samples"]]
    norms = [normalize_span(s) if s is not None else canonicalize.PARSE_FAILURE_SENTINEL
             for s in spans]
    majority, count = entropy.majority_cluster(norms)
    rep = next((i for i, n in enumerate(norms) if n == majority), 0)

    correct = bool(majority) and majority == gt_norm and gt_norm != ""
    flags = answer_flags(spans[rep] or "", v1["samples"][rep]["label"], is_mcq)
    gt_flags = answer_flags(gt_span or "", v1["ground_truth"]["label"], is_mcq)
    # A risk present on EITHER side is a risk for the comparison.
    for k in RISK_FLAGS + ("sympy_partial_parse_risk",
                           "sympy_malformed_derivative", "multi_answer_collapse"):
        flags[k] = bool(flags.get(k)) or bool(gt_flags.get(k))

    return {
        "correct_strict_v1": v1["correct"],
        "correct_strict_v2_display_primary": correct,
        "span_entropy": entropy.cluster_entropy(norms),
        "perception_entropy": v1["perception_entropy"],
        "model_span": spans[rep],
        "truth_span": gt_span,
        "model_span_norm": majority,
        "truth_span_norm": gt_norm,
        "model_label": v1["samples"][rep]["label"],
        "truth_label": v1["ground_truth"]["label"],
        "model_tier": v1["samples"][rep]["tier"],
        "truth_tier": v1["ground_truth"]["tier"],
        "is_mcq": is_mcq,
        "sympy_match": v1["correct"],
        **{k: bool(flags[k]) for k in ALL_FLAGS if k != "sympy_match"},
    }


def rescore_v2(df: pd.DataFrame,
               samples_col: str = "all_transcription_samples_raw",
               gt_col: str = "pert_a", question_col: str = "orig_q",
               progress: bool = False) -> pd.DataFrame:
    """Apply `strict_v2_display_primary` to a whole run. Input never mutated."""
    rows = []
    it = df.index
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=RULE_NAME)
        except ImportError:
            pass
    for i in it:
        rows.append(score_item_v2(
            ast.literal_eval(df.loc[i, samples_col]), df.loc[i, gt_col],
            df.loc[i, question_col] if question_col in df.columns else None))
    return pd.DataFrame(rows, index=df.index)


def disagreement_and_risk_sheet(scored: pd.DataFrame,
                                path: Optional[str] = None) -> pd.DataFrame:
    """Only the items worth a human look: rules disagree, OR a risk flag fires.

    The point of the whole exercise -- turn "audit 300 items" into "audit the
    ones the two rules cannot agree on plus the ones whose shape is known to
    break the scorer". Ships with an empty `final_label` column, the same
    confirm-or-correct shape as every other coding sheet here.
    """
    risky = scored[list(RISK_FLAGS)].any(axis=1)
    sympy_risky = scored[["sympy_partial_parse_risk",
                          "sympy_malformed_derivative",
                          "multi_answer_collapse"]].any(axis=1)
    disagree = (scored["correct_strict_v1"]
                != scored["correct_strict_v2_display_primary"])
    sel = scored[disagree | risky | sympy_risky].copy()
    sel.insert(0, "item", sel.index)
    sel["rules_disagree"] = disagree[sel.index]
    # MEASURED, not assumed: the full "any flag" queue is 71% of the run and
    # catches 15 of 20 known false passes, which is what a RANDOM 71% would
    # catch. The high tier is 36% of the run and catches the same 15 -- 2.1x
    # enrichment. Sort by it or the queue is no better than reading everything.
    sel["priority"] = review_priority(scored)[sel.index]
    sel = sel.sort_values(
        ["priority", "item"],
        key=lambda c: c.map({"high": 0, "medium": 1, "low": 2}) if c.name == "priority" else c)
    sel["final_label"] = ""
    sel["confidence"] = ""
    sel["note"] = ""
    if path:
        sel.to_csv(path, index=False)
    return sel


def accuracy_summary(scored: pd.DataFrame) -> dict:
    """v1 vs v2 accuracy, the disagreement split, and the flag counts."""
    v1 = scored["correct_strict_v1"].astype(bool)
    v2 = scored["correct_strict_v2_display_primary"].astype(bool)
    dis = v1 != v2
    return {
        "n": len(scored),
        "strict_v1_correct": int(v1.sum()),
        "strict_v1_accuracy": float(v1.mean()),
        "strict_v2_correct": int(v2.sum()),
        "strict_v2_accuracy": float(v2.mean()),
        "n_disagree": int(dis.sum()),
        "v1_only_correct": int((v1 & ~v2).sum()),
        "v2_only_correct": int((~v1 & v2).sum()),
        "flag_counts": {k: int(scored[k].astype(bool).sum())
                        for k in ALL_FLAGS if k in scored},
    }


#: Flags that actually predict a false pass, versus flags that merely describe
#: the item's shape. `mcq_option` is descriptive -- 52 of the 300 items are
#: multiple-choice and most are scored fine -- so putting it in the queue at
#: equal weight buries the signal.
HIGH_PRIORITY_FLAGS = ("tiny_suspicious_non_mcq", "multi_answer_collapse",
                       "sympy_partial_parse_risk", "sympy_malformed_derivative")


def review_priority(scored: pd.DataFrame) -> pd.Series:
    """Triage order for the review queue.

    Exists because "any risk flag" selects ~70% of the run, which is no better
    than reading everything. Measured enrichment for false passes is what
    decides the tiers, not intuition.
    """
    disagree = (scored["correct_strict_v1"]
                != scored["correct_strict_v2_display_primary"])
    high = disagree | scored[list(HIGH_PRIORITY_FLAGS)].any(axis=1)
    medium = scored[["multi_value_answer", "set_answer", "system_answer",
                     "derivative_equation", "text_conclusion"]].any(axis=1)
    out = pd.Series("low", index=scored.index)
    out[medium] = "medium"
    out[high] = "high"
    return out
