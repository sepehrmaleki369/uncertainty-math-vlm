"""Why a `genuinely_wrong` item is wrong -- a pre-sorter, not a verdict.

`rescore.classify_scoring_outcome` says why an item SCORED the way it did.
Its largest category after `correct_robust` is `genuinely_wrong` (Qwen 104,
Pixtral 130) and nothing subdivides it, so "the model is wrong 35-43% of the
time" carries no explanation. This module proposes a reason from text signals
so a human pass becomes confirm-or-correct on a pre-sorted list rather than a
hundred items read cold.

FIVE proposed labels. The first three are decidable from text; the last two
are explicitly deferred to the image, which is the honest split:

  extraction_issue   the scoring machinery is at fault, not the model -- the
                     majority is a parse failure, or the ground-truth label
                     collapsed to something degenerate. Already known to be
                     real: of the 9 unanimous-and-wrong items read on
                     2026-08-11, ~6 of 8 distinct ones were this.
  copied_wrong_line  the model's answer appears VERBATIM somewhere earlier in
                     the ground truth. It transcribed a real line of the page,
                     just not the final one.
  hallucination      almost no content shared with the ground truth's answer.
  notation_misread   high character overlap, small symbolic difference -- a
                     candidate, flagged rather than asserted, because
                     distinguishing "misread the symbol" from "misread the
                     handwriting" needs the page.
  needs_visual       everything else. Not a failure of the classifier: it is
                     the set that genuinely requires looking.

Deliberately conservative. Every label is a PROPOSAL carrying the evidence
that produced it (`why`), and `needs_visual` is the default rather than a
guess. A pre-sorter that confidently mislabels is worse than none, because
the counts it produces would be quoted.
"""

import difflib
import re
from typing import Optional, Sequence

import pilot.canonicalize as canonicalize
import pilot.entropy as entropy
import pilot.parsing as parsing
import pilot.rescore as rescore

LABELS = (
    "extraction_issue",
    "copied_wrong_line",
    "hallucination",
    "notation_misread",
    "needs_visual",
)

_SENTINEL = entropy.normalize_string(entropy.PARSE_FAILURE_SENTINEL)
_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?|[A-Za-z]+|\\[A-Za-z]+")


# SymPy prints its own function and constant names as bare words, so the
# prose detector fires on them. Without this, sympy:eq(tan(x+y), ...) reads as
# prose and item 55 -- a REAL model error, the confirmed auto-correction case
# -- was mislabelled extraction_issue.
_SYMPY_NAMES = re.compile(
    r"\b(?:eq|ne|lt|le|gt|ge|sin|cos|tan|cot|sec|csc|asin|acos|atan|sinh|cosh"
    r"|tanh|log|ln|exp|sqrt|abs|re|im|conjugate|arg|pi|oo|zoo|nan|true|false"
    r"|integral|derivative|limit|sum|product|factorial|binomial|matrix|max|min"
    r"|floor|ceiling|mod|gcd|lcm|circ|cdot|times|frac|infty)\b")


def _strip_sympy_names(text: str) -> str:
    return _SYMPY_NAMES.sub("", text)


def _tokens(text: Optional[str]) -> set:
    return set(_TOKEN_RE.findall(text or ""))


def _lines(text: Optional[str]) -> list[str]:
    """Non-trivial lines of the ground truth, cleaned the way a label is."""
    if not text:
        return []
    out = []
    for raw in re.split(r"[\n]|\\\\", str(text)):
        cleaned = canonicalize.structural_clean(
            canonicalize.unwrap_latex_macro(
                canonicalize.unwrap_latex_macro(raw, "textcolor"), "text"))
        cleaned = re.sub(r"\s+", "", cleaned)
        if len(cleaned) >= 4:
            out.append(cleaned)
    return out


def classify_failure(
    raw_samples: Sequence[str],
    ground_truth: Optional[str],
    rule: str = "final_term_v4",
) -> dict:
    """Propose a reason, with the evidence that produced it.

    Uses the LOOSEST rule by default: an item still wrong under
    `final_term_v4` has already survived every cosmetic and scope allowance,
    so whatever remains is not a formatting quibble.
    """
    scored = rescore.score_item(raw_samples, ground_truth, rule)
    model_label = str(scored["majority_label"])
    gt_label = str(scored["gt_label"])

    fields = [parsing.parse_transcription(s) for s in raw_samples]
    majority_field = next(
        (f for f, lab in zip(fields, scored["labels"]) if lab == model_label), None)

    def out(label, why):
        return {"label": label, "why": why, "model_label": model_label,
                "gt_label": gt_label, "model_field": majority_field,
                "entropy": scored["perception_entropy"]}

    # 1. the scoring machinery, not the model
    if model_label == _SENTINEL:
        return out("extraction_issue", "majority sample failed to parse")
    if gt_label == _SENTINEL:
        return out("extraction_issue", "ground truth failed to parse")
    stripped = gt_label.split(":", 1)[-1]
    # A short NUMBER is very likely the real answer ("the answer is 4"), so
    # only a stray single LETTER counts as degeneration -- that is the bug-3
    # signature, SymPy pulling one symbol out of prose (sympy:a, sympy:p,
    # sympy:l). An earlier version flagged '4', '25' and '91' here and
    # inflated extraction_issue by 5 items on Qwen alone.
    if re.fullmatch(r"[a-z]", stripped):
        return out("extraction_issue",
                   f"ground-truth label degenerated to a bare symbol: {gt_label!r}")
    if gt_label.startswith("sympy:") and canonicalize._BARE_WORD_RE.search(
            _strip_sympy_names(re.sub(r"\\[A-Za-z]+", "", stripped))):
        return out("extraction_issue",
                   f"ground-truth label is SymPy-parsed prose: {gt_label!r}")

    # 2. transcribed a real line of the page, just not the final one
    model_compact = re.sub(r"\s+", "", model_label.split(":", 1)[-1])
    if len(model_compact) >= 4:
        for i, line in enumerate(_lines(ground_truth)[:-1]):   # exclude the last
            if model_compact and model_compact in line:
                return out("copied_wrong_line",
                           f"model answer appears in an earlier ground-truth "
                           f"line (#{i}): {line[:60]!r}")

    # 3. shares almost nothing with the truth
    mt, gt_toks = _tokens(majority_field), _tokens(ground_truth)
    if mt and gt_toks:
        overlap = len(mt & gt_toks) / len(mt)
        if overlap < 0.20:
            return out("hallucination",
                       f"only {overlap:.0%} of the model's tokens appear in the "
                       "ground truth at all")

    # 4. close but not identical -- a candidate, not a verdict
    ratio = difflib.SequenceMatcher(
        None, model_compact, re.sub(r"\s+", "", stripped)).ratio()
    if ratio >= 0.75:
        return out("notation_misread",
                   f"labels are {ratio:.0%} similar -- small symbolic difference")

    return out("needs_visual", f"labels differ substantially ({ratio:.0%} similar)")


def presort(df, rule: str = "final_term_v4", progress: bool = False):
    """Propose a reason for every `genuinely_wrong` item in a run."""
    import ast

    import pandas as pd

    classified = rescore.classify_scoring_outcome(df, progress=progress)
    idx = classified.index[classified["category"] == "genuinely_wrong"]

    rows = []
    for i in idx:
        row = df.loc[i]
        raw = row["all_transcription_samples_raw"]
        samples = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        res = classify_failure(samples, row["pert_a"], rule)
        rows.append({"item": i, "has_error": bool(row["has_error"]),
                     **{k: v for k, v in res.items() if k != "model_field"}})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["label"] = pd.Categorical(out["label"], categories=LABELS, ordered=True)
    return out


def select_audit_sample(presorted, n_per_stratum: int = 6, seed: int = 0):
    """A reviewable 30-50 item sample spanning the strata that matter.

    Chosen by RULE rather than by hand, so the sample cannot be tuned: the
    same call reproduces it. Strata follow the reviewer's own list -- high and
    low entropy both appear, because a confidently-wrong item and a
    maximally-uncertain one fail for different reasons and a sample of only
    one kind would mislead.
    """
    import math

    import pandas as pd

    if presorted.empty:
        return presorted

    max_h = math.log(5)
    strata = {
        "unanimous_wrong": presorted[presorted.entropy < 1e-9],
        "max_entropy_wrong": presorted[presorted.entropy > max_h - 1e-9],
        "low_entropy_wrong": presorted[(presorted.entropy >= 1e-9)
                                       & (presorted.entropy < 0.7)],
        "likely_scoring_artifact": presorted[
            presorted.label == "extraction_issue"],
        "needs_visual": presorted[presorted.label == "needs_visual"],
    }
    picked = []
    for name, sub in strata.items():
        if sub.empty:
            continue
        take = sub.sample(min(n_per_stratum, len(sub)), random_state=seed)
        picked.append(take.assign(stratum=name))
    out = pd.concat(picked).drop_duplicates(subset="item", keep="first")
    return out.sort_values(["stratum", "item"]).reset_index(drop=True)


def coding_sheet(audit, path=None):
    """The sheet a human fills in while looking at the contact sheets.

    `proposed_label` is this module's guess; `final_label` starts empty and is
    the column that counts. Keeping them separate is the point -- if the
    proposal were edited in place there would be no record of how often the
    pre-sorter was wrong, which is itself worth reporting.
    """
    sheet = audit[["item", "stratum", "has_error", "entropy",
                   "model_label", "gt_label", "why"]].copy()
    sheet = sheet.rename(columns={"why": "proposed_because"})
    sheet.insert(2, "proposed_label", audit["label"].astype(str))
    sheet["final_label"] = ""
    sheet["notes"] = ""
    if path:
        sheet.to_csv(path, index=False)
    return sheet
