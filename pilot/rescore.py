"""Versioned transcription-scoring rules, for reporting the headline result as
a sensitivity analysis rather than as one arbitrary string comparison.

The question this answers, which a reviewer will ask: the perception AUROC is
computed against `transcription_correct`, and that label comes from an exact
match between two canonicalized strings. How much of the "wrong" pile is the
model misreading the page, and how much is the comparison being pedantic?

Measured on the 2026-08-02 n=300 Qwen-3B run, that gap is large -- accuracy
rises from 47.0% to 62.7% as the rule is loosened -- and the answer to the
question is nevertheless reassuring: the AUROC decays gracefully (0.850 ->
0.751) with every interval excluding chance. A signal that only existed
because of a strict comparison would not survive relaxing it.

Four cumulative rules:

  strict_v1     Exactly pilot.canonicalize.canonical_answer_label. The rule
                that produced every locked result and all reference/*.json.
                FROZEN -- never change its behaviour.
  fixed_v2      v1 with both known extractor defects corrected: the nested
                \\textcolor{}{} that survives into the label (see
                canonicalize.structural_clean) and the last-line tier
                splitting decimal numbers at the point (see
                canonicalize.extract_final_answer). Bug fixes, not
                relaxations: they can move items either way.
  relaxed_v3    v2 ignoring formatting the mathematics does not depend on --
                whitespace, LaTeX spacing macros, braces around a single-
                character exponent, currency symbols, trailing punctuation.
  final_term_v4 v3 reduced to the term after the last top-level "=", so a
                model that reports only "= 75.46 cm^2" is compared against a
                ground truth that spells out the whole chain ending there.

v4 changes the LABEL, not just the comparison, so entropy and correctness stay
derived from the same representation -- the symmetry canonical_answer_label
exists to enforce. Scoring correctness leniently while leaving entropy on the
strict labels would compare two different objects.

Nothing here is the headline number. `scoring_sensitivity` reports all four.
"""

import ast
import math
import re
from typing import Callable, Optional, Sequence

import pandas as pd

import pilot.canonicalize as canonicalize
import pilot.entropy as entropy
import pilot.parsing as parsing

RULES = ("strict_v1", "fixed_v2", "relaxed_v3", "final_term_v4")

_SENTINEL = entropy.normalize_string(entropy.PARSE_FAILURE_SENTINEL)

# LaTeX spacing macros carry no mathematical content: "11130 \, cm" and
# "11130 cm" are the same answer.
_SPACING_RE = re.compile(r"\\[,;:!]|\\quad|\\qquad|\\ |\\;|\\thinspace")
_CURRENCY_RE = re.compile(r"[₹$€£¥]|\\rs\.?|\brs\.?\s")
# {3} and 3 are the same exponent; only single tokens, so {x+1} is untouched.
_SINGLE_BRACE_RE = re.compile(r"\{(\w)\}")


def _split_top_level_equals(text: str) -> list[str]:
    """Split on '=' that are not inside braces, brackets or parentheses.

    Depth-aware so "f(x) = 2x" splits but "\\frac{a=b}{c}" does not, and so
    that "<=", ">=", "!=" and "==" are never treated as separators (splitting
    an inequality would silently turn "x <= 5" into the answer "5").
    """
    parts, depth, start = [], 0, 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            prev = text[i - 1] if i else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev in "<>!=" or nxt == "=":
                i += 1
                continue
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return parts


def final_term(label: str) -> str:
    """The term after the last top-level '=', or the label itself if there is
    none. A trailing empty term (an answer that ends in '=') falls back to the
    last non-empty one rather than returning "", which would collapse every
    such item into a single spurious cluster."""
    parts = [p.strip() for p in _split_top_level_equals(label)]
    for part in reversed(parts):
        if part:
            return part
    return label


def _cosmetic(label: str) -> str:
    s = _SPACING_RE.sub("", label)
    s = _CURRENCY_RE.sub("", s)
    s = _SINGLE_BRACE_RE.sub(r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s.strip().rstrip(".,;") or label


def answer_label(text: Optional[str], rule: str = "strict_v1") -> str:
    """Canonical cluster label for one transcription (or for a ground truth)
    under the named rule. Both sides of a comparison must use the same rule."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")

    if rule == "strict_v1":
        return canonicalize.canonical_answer_label(text)

    # fixed_v2 and up: both known extractor defects corrected.
    if text is not None:
        text = canonicalize.unwrap_latex_macro(text, "textcolor")
        text = canonicalize.unwrap_latex_macro(text, "text")
    final = canonicalize.extract_final_answer(text, fix_decimal_split=True)
    label = entropy.normalize_string(canonicalize.canonicalize_math(final))

    if rule == "fixed_v2" or label == _SENTINEL:
        return label

    label = _cosmetic(label)
    if rule == "final_term_v4":
        label = final_term(label)
    return label


def score_item(
    raw_samples: Sequence[str],
    ground_truth: Optional[str],
    rule: str = "strict_v1",
) -> dict:
    """Entropy, majority label and correctness for one item under one rule.

    Takes RAW model output (not pre-parsed answers) so the whole chain --
    parse_transcription -> extract_final_answer -> canonicalize -> compare --
    is exercised, which is where the extractor's own instability lives.
    """
    parsed = [parsing.parse_transcription(s) for s in raw_samples]
    labels = [answer_label(p, rule) for p in parsed]
    majority, _ = entropy.majority_cluster(labels)
    gt = answer_label(ground_truth, rule)
    return {
        "labels": labels,
        "perception_entropy": entropy.cluster_entropy(labels),
        "majority_label": majority,
        "gt_label": gt,
        "transcription_correct": majority == gt,
        "n_transcription_parse_failures": sum(1 for p in parsed if p is None),
    }


def _maybe_tqdm(iterable, total, desc, progress):
    """tqdm when asked for and available, otherwise the bare iterable.

    Rescoring is slow enough (SymPy parses every label) that a silent
    multi-minute cell reads as a hang. tqdm is optional rather than a hard
    dependency so the test suite and any headless caller stay quiet.
    """
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def rescore_run(
    df: pd.DataFrame,
    rule: str = "strict_v1",
    samples_col: str = "all_transcription_samples_raw",
    gt_col: str = "pert_a",
    progress: bool = False,
) -> pd.DataFrame:
    """Recompute entropy and correctness for a whole results CSV under `rule`.

    Returns a NEW frame; the input is never mutated, so a caller cannot
    accidentally overwrite the as-run columns a snapshot was built from.
    """
    records = []
    rows = _maybe_tqdm(df.iterrows(), len(df), rule, progress)
    for _, row in rows:
        raw = row[samples_col]
        samples = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        records.append(score_item(samples, row[gt_col], rule))
    out = pd.DataFrame(records, index=df.index).drop(columns=["labels"])
    for col in ("has_error", "orig_q", "pert_a"):
        if col in df.columns:
            out[col] = df[col]
    return out


def scoring_sensitivity(
    df: pd.DataFrame,
    rules: Sequence[str] = RULES,
    n_boot: int = 10000,
    seed: int = 0,
    k: int = 5,
    progress: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """One row per rule: accuracy, AUROC with a bootstrap CI, max-entropy count.

    This is the table to report. The claim it supports is not "accuracy is
    really 62.7%" -- it is that the AUROC survives every rule, so the
    perception result is a property of the entropy signal rather than of one
    string comparison.
    """
    from pilot.plotting import bootstrap_auroc_ci  # local: avoids a cycle

    rows = []
    for n, rule in enumerate(rules, 1):
        if progress:
            print(f"[{n}/{len(rules)}] {rule}: rescoring {len(df)} items...", flush=True)
        scored = rescore_run(df, rule, progress=progress, **kwargs)
        if progress:
            print(f"          bootstrapping {n_boot} resamples...", flush=True)
        ci = bootstrap_auroc_ci(scored, "perception_entropy",
                                "transcription_correct", n_boot=n_boot, seed=seed)
        n_correct = int(scored["transcription_correct"].sum())
        at_max = int(sum(math.isclose(e, math.log(k))
                         for e in scored["perception_entropy"]))
        rows.append({
            "rule": rule,
            "n_correct": n_correct,
            "accuracy": n_correct / len(scored),
            "auroc": ci["auroc"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "excludes_chance": ci["excludes_chance"],
            "n_at_max_entropy": at_max,
        })
    return pd.DataFrame(rows)


# --- diagnosing WHY an item disagrees --------------------------------------

_TIERS: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("option", lambda t: bool(canonicalize._OPTION_RE.search(t))),
    ("boxed", lambda t: bool(canonicalize._BOXED_RE.search(t))),
    ("display_math", lambda t: bool(canonicalize._DISPLAY_MATH_RE.search(t))),
)


def extraction_tier(answer_field: Optional[str]) -> str:
    """Which branch of extract_final_answer fired: option / boxed /
    display_math / last_line, or parse_fail if there was no Answer field.

    Exists because the branch is invisible in the output but explains a large
    share of the disagreement between samples: on the n=300 run, 153/300 items
    have the extractor firing DIFFERENT branches across the 5 samples, and
    mean entropy rises monotonically with the number of branches used (0.685
    at one, 1.031 at two, 1.243 at three). Some of what the perception arm
    measures as model uncertainty is the extractor changing its mind about
    which line of an unchanged derivation is the answer.
    """
    if answer_field is None:
        return "parse_fail"
    for name, matches in _TIERS:
        if matches(answer_field):
            return name
    return "last_line"


def tier_instability(raw_samples: Sequence[str]) -> dict:
    """How many distinct extractor branches the K samples triggered."""
    tiers = [extraction_tier(parsing.parse_transcription(s)) for s in raw_samples]
    return {"tiers": tiers, "n_distinct_tiers": len(set(tiers))}


def trace_item(
    raw_samples: Sequence[str],
    ground_truth: Optional[str],
    rule: str = "strict_v1",
) -> dict:
    """Every intermediate value of the scoring pipeline for one item.

    The four stages, for the model samples and the ground truth alike:
      1 raw model output
      2 parse_transcription      -> the **Answer:** field
      3 extract_final_answer     -> the final-answer span
      4 answer_label(rule)       -> the string actually compared
    Built for notebook 17; keeping it here rather than in the notebook means
    the inspection view and the scored numbers cannot drift apart.
    """
    def stages(text, is_raw):
        field = parsing.parse_transcription(text) if is_raw else text
        prepared, fix = field, rule != "strict_v1"
        if fix and prepared is not None:
            prepared = canonicalize.unwrap_latex_macro(prepared, "textcolor")
            prepared = canonicalize.unwrap_latex_macro(prepared, "text")
        return {
            "raw": text,
            "answer_field": field,
            "final_answer": canonicalize.extract_final_answer(
                prepared, fix_decimal_split=fix),
            "label": answer_label(field, rule),
            "tier": extraction_tier(field),
        }

    samples = [stages(s, True) for s in raw_samples]
    gt = stages(ground_truth, False)
    labels = [s["label"] for s in samples]
    majority, count = entropy.majority_cluster(labels)
    return {
        "rule": rule,
        "samples": samples,
        "ground_truth": gt,
        "perception_entropy": entropy.cluster_entropy(labels),
        "majority_label": majority,
        "majority_count": count,
        "correct": majority == gt["label"],
        "n_distinct_tiers": len({s["tier"] for s in samples}),
    }
