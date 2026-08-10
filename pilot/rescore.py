"""Versioned transcription-scoring rules, for reporting the headline result as
a sensitivity analysis rather than as one arbitrary string comparison.

The question this answers, which a reviewer will ask: the perception AUROC is
computed against `transcription_correct`, and that label comes from an exact
match between two canonicalized strings. How much of the "wrong" pile is the
model misreading the page, and how much is the comparison being pedantic?

Measured on the 2026-08-02 n=300 Qwen-3B run, that gap is large -- accuracy
rises from 47.0% to 63.3% as the rule is loosened -- and the answer to the
question is nevertheless reassuring: the AUROC decays gracefully (0.850 ->
0.817) with every interval excluding chance. A signal that only existed
because of a strict comparison would not survive relaxing it.

Four cumulative rules:

  strict_v1     Exactly pilot.canonicalize.canonical_answer_label. The rule
                that produced every locked result and all reference/*.json.
                FROZEN -- never change its behaviour.
  fixed_v2      v1 with all three known extractor defects corrected: the
                nested \\textcolor{}{} that survives into the label (see
                canonicalize.structural_clean), the last-line tier splitting
                decimal numbers at the point (see extract_final_answer), and
                parse_latex silently parsing only a PREFIX of its input (see
                sympy_parse_is_trustworthy). Bug fixes, not relaxations: they
                move items both ways and net to no accuracy change here
                (141 -> 141), because the third removes matches that existed
                only because two different answers collapsed to one label.
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
import textwrap
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

    # fixed_v2 and up: all three known extractor defects corrected.
    if text is not None:
        text = canonicalize.unwrap_latex_macro(text, "textcolor")
        text = canonicalize.unwrap_latex_macro(text, "text")
    final = canonicalize.extract_final_answer(text, fix_decimal_split=True)
    label = entropy.normalize_string(
        canonicalize.canonicalize_math(final, strict_parse=True))

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
    really 63.3%" -- it is that the AUROC survives every rule, so the
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


# --- classifying every item by WHY it scored the way it did ----------------
#
# scoring_sensitivity says how many items each rule gets right. It does not
# say what the population is made of, because the four illustrative buckets
# notebook 17 used to draw examples from OVERLAP -- an item can be both a
# cosmetic mismatch and extractor-tier-unstable. These categories are
# mutually exclusive and cover every item, so they can be counted, plotted
# and compared across models.

CATEGORIES = (
    "correct_robust",
    "bug_fix_recovered",
    "cosmetic_mismatch",
    "scope_mismatch",
    "false_pass_removed",
    "broken_by_relaxation",
    "genuinely_wrong",
)

# Each bug fix applied ALONE, for attributing a flip to a named defect.
SINGLE_FIXES = {
    "textcolor": {"unwrap": True},
    "decimal": {"fix_decimal_split": True},
    "sympy_prefix": {"strict_parse": True},
}


def _ablated_label(
    text: Optional[str],
    unwrap: bool = False,
    fix_decimal_split: bool = False,
    strict_parse: bool = False,
) -> str:
    """answer_label with exactly one of the three bug fixes switched on.

    Deliberately mirrors answer_label's non-strict branch rather than calling
    it, because the whole point is to enable the fixes one at a time -- the
    named rules only offer them as a bundle.
    """
    if unwrap and text is not None:
        text = canonicalize.unwrap_latex_macro(text, "textcolor")
        text = canonicalize.unwrap_latex_macro(text, "text")
    final = canonicalize.extract_final_answer(
        text, fix_decimal_split=fix_decimal_split)
    return entropy.normalize_string(
        canonicalize.canonicalize_math(final, strict_parse=strict_parse))


def _correct_under(raw_samples: Sequence[str], ground_truth, **fix) -> bool:
    labels = [_ablated_label(parsing.parse_transcription(s), **fix)
              for s in raw_samples]
    majority, _ = entropy.majority_cluster(labels)
    return majority == _ablated_label(ground_truth, **fix)


def classify_scoring_outcome(
    df: pd.DataFrame,
    samples_col: str = "all_transcription_samples_raw",
    gt_col: str = "pert_a",
    k: int = 5,
    progress: bool = False,
    attribute_bugs: bool = True,
) -> pd.DataFrame:
    """One row per item: which category it falls in, and why.

    Categories are assigned from the verdicts under the four rules. Two of
    them exist because the rules are NOT monotone, which a naive cumulative
    scheme would hide:

      false_pass_removed   correct under the frozen rule, wrong once the
                           extractor bugs are fixed -- the item only looked
                           right because a bug mangled both sides the same
                           way (the item-101 class). This is where relaxing
                           a comparison manufactures a wrong answer, so it is
                           the cell worth reading first.
      broken_by_relaxation correct earlier in the chain, lost at a looser
                           rule because the relaxation changed which cluster
                           won the vote.

    `category` records the FIRST rule that fixed an item; `later_regression`
    separately records that a subsequent rule broke it again. One string
    cannot honestly carry both facts -- on the Qwen run two items are correct
    at relaxed_v3 and wrong at final_term_v4.

    The flag columns (`n_distinct_tiers`, `at_max_entropy`,
    `majority_is_parse_failure`) are ORTHOGONAL to the category, not
    alternatives to it: tier instability holds for 41% of correct items and
    60% of wrong ones on the real run, so it cross-cuts every bar rather than
    defining one.
    """
    scored = {rule: rescore_run(df, rule, samples_col=samples_col,
                                gt_col=gt_col, progress=progress)
              for rule in RULES}
    v1, v2, v3, v4 = (scored[r]["transcription_correct"].values.astype(bool)
                      for r in RULES)

    sentinel = _SENTINEL
    max_h = math.log(k)
    rows = []
    for i in range(len(df)):
        raw = df.iloc[i][samples_col]
        samples = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)

        if v1[i] and v2[i]:
            category, first_fix = "correct_robust", None
        elif v1[i] and not v2[i]:
            category, first_fix = "false_pass_removed", None
        elif v2[i]:
            category, first_fix = "bug_fix_recovered", "fixed_v2"
        elif v3[i]:
            category, first_fix = "cosmetic_mismatch", "relaxed_v3"
        elif v4[i]:
            category, first_fix = "scope_mismatch", "final_term_v4"
        else:
            category, first_fix = "genuinely_wrong", None

        # Correct somewhere in the chain, wrong at the end.
        ever_correct = v1[i] or v2[i] or v3[i] or v4[i]
        later_regression = bool(ever_correct and not v4[i]
                                and category != "false_pass_removed")
        if category == "correct_robust" and not v4[i]:
            category, later_regression = "broken_by_relaxation", False

        attributed = None
        if attribute_bugs and category in ("bug_fix_recovered",
                                           "false_pass_removed"):
            gt = df.iloc[i][gt_col]
            flipped = [name for name, fix in SINGLE_FIXES.items()
                       if _correct_under(samples, gt, **fix) != bool(v1[i])]
            attributed = "+".join(flipped) if flipped else "combined_only"

        rows.append({
            "category": category,
            "first_fixing_rule": first_fix,
            "later_regression": later_regression,
            "attributed_bug": attributed,
            "correct_strict_v1": bool(v1[i]),
            "correct_fixed_v2": bool(v2[i]),
            "correct_relaxed_v3": bool(v3[i]),
            "correct_final_term_v4": bool(v4[i]),
            "entropy": float(scored["strict_v1"]["perception_entropy"].iloc[i]),
            "n_distinct_tiers": tier_instability(samples)["n_distinct_tiers"],
            "at_max_entropy": bool(math.isclose(
                float(scored["strict_v1"]["perception_entropy"].iloc[i]), max_h)),
            "majority_is_parse_failure": bool(
                scored["strict_v1"]["majority_label"].iloc[i] == sentinel),
            "n_parse_failures": int(
                scored["strict_v1"]["n_transcription_parse_failures"].iloc[i]),
        })

    out = pd.DataFrame(rows, index=df.index)
    if "has_error" in df.columns:
        out["has_error"] = df["has_error"].astype(bool)
    unknown = set(out["category"]) - set(CATEGORIES)
    assert not unknown, f"category not in CATEGORIES: {unknown}"
    return out


def scoring_category_summary(
    classified: pd.DataFrame,
    label: str = "",
) -> pd.DataFrame:
    """Counts, share, and within-category diagnostics, in CATEGORIES order.

    Takes the output of classify_scoring_outcome rather than a raw run, so a
    caller can classify once and summarise many ways (e.g. split by
    has_error) without paying for the rescoring again.
    """
    n = len(classified)
    rows = []
    for category in CATEGORIES:
        sub = classified[classified["category"] == category]
        rows.append({
            "category": category,
            "n": len(sub),
            "share": len(sub) / n if n else float("nan"),
            "mean_entropy": float(sub["entropy"].mean()) if len(sub) else float("nan"),
            "frac_multi_tier": float((sub["n_distinct_tiers"] > 1).mean())
            if len(sub) else float("nan"),
            "frac_at_max_entropy": float(sub["at_max_entropy"].mean())
            if len(sub) else float("nan"),
        })
    out = pd.DataFrame(rows)
    if label:
        out.insert(0, "model", label)
    return out


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


def first_difference(a: str, b: str) -> Optional[dict]:
    """Where two labels first diverge, or None if identical.

    The line the inspection view was missing. On a cosmetic mismatch the two
    labels look the same on screen -- "(x,z) \\in r" against "(x, z) \\in r" --
    and a reader cannot tell which character decided the verdict. Reports the
    column and both characters by repr, so a space or a comma is visible.
    """
    if a == b:
        return None
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return {"index": i, "model": repr(ca), "truth": repr(cb),
                    "context_model": a[max(0, i - 15):i + 15],
                    "context_truth": b[max(0, i - 15):i + 15]}
    # One is a prefix of the other.
    i = min(len(a), len(b))
    longer, which = (a, "model") if len(a) > len(b) else (b, "truth")
    return {"index": i, "model": repr(a[i:i + 1]) if len(a) > i else "<end>",
            "truth": repr(b[i:i + 1]) if len(b) > i else "<end>",
            "context_model": a[max(0, i - 15):i + 15],
            "context_truth": b[max(0, i - 15):i + 15],
            "note": f"{which} is longer; the other ends here"}


def _clip(value: Optional[str], limit: int) -> Optional[str]:
    """Truncate for display, marking that it happened. A silently cut string
    reads as the model having stopped there, which is the confusion this whole
    module exists to remove."""
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f" …[+{len(text) - limit} chars]"


def format_trace(
    trace: dict,
    question: Optional[str] = None,
    width: int = 100,
    raw_chars: int = 600,
    category: Optional[str] = None,
) -> str:
    """The scoring pipeline for one item, rendered for reading.

    Lives here rather than in the notebook for the same reason trace_item
    does: the view and the numbers must come from one place or they drift.
    Every line says which function produced it, because the recurring
    confusion is not what the values are but which stage owns them.
    """
    out: list[str] = []

    def head(title, char="="):
        out.append(char * width)
        out.append(title)
        out.append(char * width)

    def field(name, value, indent=5):
        body = "None" if value is None else str(value).replace("\n", " ⏎ ")
        wrapped = textwrap.wrap(body, width - 34) or [""]
        pad = " " * indent
        out.append(f"{pad}{name:<28} {wrapped[0]}")
        for line in wrapped[1:]:
            out.append(f"{pad}{'':<28} {line}")

    rule = trace["rule"]
    verdict = "CORRECT" if trace["correct"] else "WRONG"
    title = f"RULE {rule}   entropy={trace['perception_entropy']:.3f}   " \
            f"majority {trace['majority_count']}/{len(trace['samples'])}   {verdict}"
    if category:
        title += f"   [{category}]"
    head(title)

    if question is not None:
        out.append("")
        out.append("QUESTION  (orig_q, raw — this is what the page asks)")
        field("", str(question)[:raw_chars], indent=5)

    gt = trace["ground_truth"]
    out.append("")
    out.append("─ GROUND TRUTH " + "─" * (width - 15))
    field("raw answer (pert_a):", str(gt["raw"])[:raw_chars])
    field("→ extract_final_answer:", gt["final_answer"])
    field("→ COMPARISON LABEL:", gt["label"])

    for k, s in enumerate(trace["samples"]):
        matches = s["label"] == gt["label"]
        out.append("")
        banner = (f"─ MODEL sample {k + 1} of {len(trace['samples'])}  "
                  f"(extractor branch: {s['tier']}) ")
        out.append(banner + "─" * max(0, width - len(banner)))
        field("raw model output (tail):", str(s["raw"] or "")[-raw_chars:])
        field("→ parse_transcription:", _clip(s["answer_field"], raw_chars))
        field("→ extract_final_answer:", _clip(s["final_answer"], raw_chars))
        field("→ COMPARISON LABEL:",
              f"{_clip(s['label'], raw_chars)}"
              f"{'   ← MATCHES GROUND TRUTH' if matches else ''}")

    out.append("")
    out.append("─ COMPARISON " + "─" * (width - 13))
    field("model majority label:", trace["majority_label"])
    field("ground-truth label:", gt["label"])
    field("verdict:", "MATCH" if trace["correct"] else "DIFFER")
    diff = first_difference(str(trace["majority_label"]), str(gt["label"]))
    if diff is not None:
        field("first difference at col:",
              f"{diff['index']}   model={diff['model']}  truth={diff['truth']}")
        field("  model around it:", diff["context_model"])
        field("  truth around it:", diff["context_truth"])
    out.append("")
    return "\n".join(out)


# --- The \boxed{} experiment ---------------------------------------------
#
# Registered BEFORE the run, per this project's standing rule that a bar
# decided after seeing the numbers is not a bar. See
# prompts.TRANSCRIPTION_USER_PROMPT_BOXED for why the run exists.

BOXED_PREREGISTRATION = {
    # Below this the run measures instruction-following, not uncertainty.
    "min_boxed_compliance": 0.80,
    # The perception AUROC under deterministic extraction.
    "auroc_confirms_signal": 0.75,
    # Extraction must actually have become deterministic, or the manipulation
    # did not work and the comparison is meaningless either way.
    "max_multi_tier_frac": 0.10,
}


def boxed_compliance(raw_samples: Sequence[str]) -> dict:
    """How often the model actually emitted \\boxed{} in its Answer field.

    The gate for the whole experiment. If the model ignores the instruction
    there is nothing to compare: a low-compliance run measures whether a 3B
    model follows a formatting request, which is not the question.
    """
    fields = [parsing.parse_transcription(s) for s in raw_samples]
    boxed = [f is not None and bool(canonicalize._BOXED_RE.search(f)) for f in fields]
    return {
        "n_samples": len(raw_samples),
        "n_boxed": sum(boxed),
        "frac_boxed": sum(boxed) / len(raw_samples) if raw_samples else 0.0,
        "all_boxed": all(boxed) if boxed else False,
    }


def classify_boxed_result(
    compliance_frac: float,
    auroc_ci: dict,
    multi_tier_frac: float,
    prereg: Optional[dict] = None,
) -> str:
    """The registered verdict, as a pure function so the reported label is the
    tested one rather than a post-hoc reading.

    gated_low_compliance    the model ignored \\boxed{}; nothing to conclude
    manipulation_failed     it complied but extraction still varies, so the
                            experiment did not do what it claims
    signal_is_not_extractor the AUROC holds under deterministic extraction --
                            the perception result is the model's uncertainty
    signal_was_extractor    it collapses to chance -- a substantial part of
                            the original signal was extractor instability
    inconclusive            between the two, at this n
    """
    p = prereg or BOXED_PREREGISTRATION
    if compliance_frac < p["min_boxed_compliance"]:
        return "gated_low_compliance"
    if multi_tier_frac > p["max_multi_tier_frac"]:
        return "manipulation_failed"
    if auroc_ci["auroc"] >= p["auroc_confirms_signal"] and auroc_ci["excludes_chance"]:
        return "signal_is_not_extractor"
    if not auroc_ci["excludes_chance"]:
        return "signal_was_extractor"
    return "inconclusive"
