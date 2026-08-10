"""Math-Verify as an AUDIT tool, deliberately not as the scorer.

HuggingFace's Math-Verify (https://github.com/huggingface/Math-Verify) parses
two answers and decides mathematical equivalence with SymPy. On isolated
answer pairs it is very good: it resolves every false negative this project's
string rules produce -- `2^3 = 8` vs `2^{3} = 8`, `3x^2 = -4y` vs
`x^2 = -4y/3`, `11130 cm` vs `11130 \\, cm` -- while correctly REJECTING both
of FERMAT's confirmed injected errors (items 55 and 273).

It is still not safe as the scorer HERE, and the reason is specific to this
benchmark rather than a flaw in the tool. FERMAT injects errors that are
frequently a unit, a coefficient, a variable name, or a notation detail --
exactly the differences an equivalence checker exists to normalise away. On
the 2026-08-11 audit, driving the score with Math-Verify moved accuracy from
63.3% to 71.0% (Qwen) and 54.3% to 65.7% (Pixtral), but roughly half of the
newly-accepted items were false positives concentrated on has_error=1 rows:

    item 108   y^2/(9/4) - x^2/(9/4) = 1   vs   y^2/(11/4) - x^2/(25/4) = 1
               different coefficients, matched on the trailing chain
    item  71   3.14159.                    vs   dy/dx = 1. Note pi ~= 3.14159
               the ground truth's answer is 1
    item  47   x = 2 and y = 3             vs   x = 2 cm and y = 3 cm

So the frozen string rule stays the reported metric, and this module is used
to TRIAGE: where the two disagree is a queue for human review, not a
correction. That queue is genuinely useful -- it surfaced item 40, where our
own extractor had emitted a parse failure.

THE GOTCHA, and the reason mv_parse exists rather than calling parse()
directly: Math-Verify's parse() must be given a $-delimited string. On bare
input it silently falls back to plain-number extraction --

    parse("x = 5")   -> [5, '5']            the variable is discarded
    parse("$x = 5$") -> [Eq(x, 5), 'x = 5']

-- so `x = 5` and `z = 5` compare EQUAL on bare input and unequal when
wrapped. That single omission produced every false positive in the first
three runs of this analysis and made the results look far better than they
were. Never call parse() on an unwrapped span.
"""

from typing import Optional, Sequence

import pilot.canonicalize as canonicalize
import pilot.entropy as entropy
import pilot.parsing as parsing


def math_verify_available() -> bool:
    """Whether the optional math-verify dependency is importable.

    Optional on purpose: it is an audit tool, not part of the scoring path,
    and the test suite must stay runnable without it.
    """
    try:
        import math_verify  # noqa: F401
        return True
    except ImportError:
        return False


def mv_parse(span: Optional[str]):
    """Parse one answer span, $-wrapped. None when there is nothing usable.

    Always wraps: see this module's docstring for what bare input does.
    """
    if not span:
        return None
    from math_verify import parse
    try:
        parsed = parse(f"${span}$")
    except Exception:
        return None
    return parsed or None


def mv_equivalent(a, b) -> bool:
    """Mathematical equivalence of two mv_parse results."""
    if not a or not b:
        return False
    from math_verify import verify
    try:
        return bool(verify(a, b))
    except Exception:
        return False


def answer_span(text: Optional[str]) -> Optional[str]:
    """The final-answer span, using this project's corrected extractor.

    Math-Verify has its own answer extraction, but running it end-to-end
    scored WORSE than the string rule (17/30 vs 18/30 on a probe) because its
    heuristics differ from ours on multi-step derivations. Feeding it our span
    keeps extraction and comparison separable, so a disagreement is
    attributable to one or the other.
    """
    if text is None:
        return None
    text = canonicalize.unwrap_latex_macro(text, "textcolor")
    text = canonicalize.unwrap_latex_macro(text, "text")
    return canonicalize.extract_final_answer(text, fix_decimal_split=True)


def mv_score_item(raw_samples: Sequence[str], ground_truth: Optional[str]) -> dict:
    """Majority-cluster correctness under Math-Verify equivalence.

    Mirrors pilot.entropy.majority_cluster so the number is comparable to the
    string rules: group the K samples into mutually-equivalent clusters, take
    the largest, and check that cluster's representative against the truth.

    Grouping by mathematical equivalence is sound because equivalence really
    is transitive -- unlike the NLI union-find cascade documented in
    pilot/semantic.py, where one false link fuses two opposed clusters.

    Note this is a MAJORITY test, not "any of the K matched". The latter is a
    much weaker criterion that inflated an early version of this analysis from
    80.3% to 86.7%.
    """
    spans = [answer_span(parsing.parse_transcription(s)) for s in raw_samples]
    parsed = [mv_parse(s) for s in spans]
    gt_span = answer_span(ground_truth)
    gt = mv_parse(gt_span)

    clusters: list[list[int]] = []
    for i, p in enumerate(parsed):
        for cluster in clusters:
            if mv_equivalent(parsed[cluster[0]], p):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    clusters.sort(key=len, reverse=True)
    biggest = clusters[0]

    return {
        "correct": mv_equivalent(parsed[biggest[0]], gt),
        "majority_span": spans[biggest[0]],
        "gt_span": gt_span,
        "majority_size": len(biggest),
        "n_clusters": len(clusters),
        "gt_parsed": gt is not None,
        "n_samples_parsed": sum(1 for p in parsed if p is not None),
    }


def disagreement_queue(
    df,
    string_correct,
    samples_col: str = "all_transcription_samples_raw",
    gt_col: str = "pert_a",
    progress: bool = False,
):
    """Items where the string rule and Math-Verify disagree -- for REVIEW.

    `string_correct` is the boolean column from the rule being reported (use
    pilot.rescore.rescore_run(...)["transcription_correct"]).

    The `string_wrong_mv_right` rows are the audit candidates: either the
    string rule missed a genuine read, or Math-Verify normalised away the very
    difference FERMAT injected. Both are worth a human look and NEITHER should
    be applied automatically. `string_right_mv_wrong` is the rarer direction
    and usually means the span did not parse.
    """
    import ast

    import pandas as pd

    rows = []
    it = df.iterrows()
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(df), desc="math-verify", leave=False)
        except ImportError:
            pass

    for i, row in it:
        raw = row[samples_col]
        samples = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        mv = mv_score_item(samples, row[gt_col])
        s_ok = bool(string_correct[i]) if hasattr(string_correct, "__getitem__") \
            else bool(string_correct.loc[i])
        if s_ok == mv["correct"]:
            continue
        rows.append({
            "item": i,
            "direction": "string_wrong_mv_right" if mv["correct"]
                         else "string_right_mv_wrong",
            "has_error": bool(row["has_error"]) if "has_error" in row else None,
            # NB: this is MATH-VERIFY's majority cluster, which can pick a
            # different representative sample than the string rule did. A row
            # whose two spans look identical usually means exactly that, not
            # that the canonicaliser failed on equal inputs.
            "mv_majority_span": mv["majority_span"],
            "gt_span": mv["gt_span"],
            "majority_size": mv["majority_size"],
            "n_samples_parsed": mv["n_samples_parsed"],
        })
    return pd.DataFrame(rows)


# Sanity cases the tool must get right before any of the above is trusted.
# Each is a real pair from the n=300 run. The two REJECT cases are FERMAT's
# confirmed injected errors -- if a future version merges those, Math-Verify
# has become unusable here even for triage.
SANITY_CASES = (
    ("cosmetic: brace on exponent", "2^3 = 8", "2^{3} = 8", True),
    ("equation scaled by a constant (item 208)", "3x^2 = -4y", "x^2 = -4y/3", True),
    ("LaTeX thin space in a unit (item 280)", "11130 cm", r"11130 \, cm", True),
    ("spacing inside a tuple (item 9)", "(x,z)", "(x, z)", True),
    ("answer only vs the whole chain (item 49)", "75.46", r"\pi r^2 = 75.46", True),
    ("INJECTED ERROR, sign flip (item 55)",
     r"\frac{\tan x+\tan y}{1-\tan x\tan y}",
     r"\frac{\tan x+\tan y}{1+\tan x\tan y}", False),
    ("INJECTED ERROR, wrong numerator (item 273)", r"\frac{3}{4}", r"\frac{2}{4}", False),
    ("different variable", "x = 5", "z = 5", False),
    ("different exponent (item 94)", "6^n + 5^n", "6^n + 5n", False),
    ("plain different numbers", "24", "25", False),
)


def run_sanity_cases() -> "object":
    """Every SANITY_CASES pair, with the expected verdict alongside."""
    import pandas as pd
    rows = []
    for label, a, b, expected in SANITY_CASES:
        got = mv_equivalent(mv_parse(a), mv_parse(b))
        rows.append({"case": label, "a": a, "b": b,
                     "expected": expected, "got": got, "ok": got == expected})
    return pd.DataFrame(rows)
