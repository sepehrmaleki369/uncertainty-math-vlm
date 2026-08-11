"""Human-audit-corrected labels, and what they do to the metrics.

The 2026-08-11 human pass read every one of Qwen's 104 `genuinely_wrong`
items and found that ~74% of them are the scoring pipeline rather than the
model. That immediately raises two questions this module answers:

  1. If the labels were right, how accurate is the model really?
  2. Does entropy still predict the *remaining* real failures, or was the
     0.835 AUROC partly entropy detecting extractor instability?

**This is a SENSITIVITY / DIAGNOSTIC analysis, never the headline.** Three
constraints are baked in because getting any of them wrong produces a number
that quietly contradicts the paper:

* **The baseline is `strict_v1` (141/300 = 47.0%), not the stored 39.3%.**
  The audit selected its 104 items from a LOCAL rescore, and local rescoring
  relabels 43/300 items versus the stored columns (the original run was
  scored without SymPy's LaTeX parser). Quoting a corrected accuracy against
  the stored 39.3% mixes two labelings.

* **The correction is ONE-SIDED and the accuracy figure is therefore an
  UPPER-BIASED estimate.** Only items the frozen rule called *wrong* were
  audited, so only false negatives could be found. `false_pass_removed`
  already proves false positives exist (6 known Qwen items). Until a sample
  of the *correct* items is audited too, report accuracy as a range and say
  it is one-sided. `correct_item_spot_check_sample` builds that sample.

* **`extraction_issue` does not mean "the model was right".** Where the
  majority label is a parse failure there is no answer to recover, so those
  items are `unknown`, not `true`. Every figure is reported under BOTH
  resolutions of `unknown` rather than picking one.

The AUROC is EXPECTED to fall. Recovered items carry mean entropy ~1.16
against ~0.66 for the already-correct ones, so moving them into the correct
class makes that class noisier. A drop is the honest cost of a cleaner
target, not a failure — and the direction was registered before computing.
"""

from typing import Optional

import numpy as np
import pandas as pd

from . import rescore

#: Human labels that mean the model genuinely got it wrong.
REAL_ERROR_LABELS = ("notation_misread", "copied_wrong_line", "hallucination")

#: Human labels that mean the scoring pipeline, not the model, produced the
#: mismatch -- subject to the parse-failure carve-out below.
SCORING_LABELS = ("extraction_issue",)

#: Human label meaning the coder could not decide from the image.
UNDECIDED_LABELS = ("needs_visual",)


def unrecoverable_items(
    run: pd.DataFrame,
    audit: pd.DataFrame,
    samples_col: str = "all_transcription_samples_raw",
    gt_col: str = "pert_a",
) -> list:
    """`extraction_issue` items with no answer to recover.

    Objective rule: the MAJORITY cluster label is `<parse_failure>`. Defined
    from the data rather than from the coder's prose, because "the model
    produced nothing usable" is exactly the kind of judgement that drifts
    when recalled -- a first pass by memory named 7 items; the rule names 4.
    """
    import ast

    out = []
    for item in audit.index:
        if audit.loc[item, "final_label"] not in SCORING_LABELS:
            continue
        trace = rescore.trace_item(
            ast.literal_eval(run.loc[item, samples_col]), run.loc[item, gt_col])
        if trace["majority_label"] == "<parse_failure>":
            out.append(item)
    return sorted(out)


def human_corrected_correct(
    run: pd.DataFrame,
    audit: pd.DataFrame,
    **kwargs,
) -> pd.Series:
    """Per-item corrected label: True / False / None(unknown), for the audited items.

    True  -- the frozen rule called it wrong, the human says the scorer erred
    False -- a real model mistake
    None  -- undecidable (needs_visual, or nothing parseable to recover)
    """
    bad = unrecoverable_items(run, audit, **kwargs)
    vals = {}
    for item in audit.index:
        label = audit.loc[item, "final_label"]
        if label in REAL_ERROR_LABELS:
            vals[item] = False
        elif label in UNDECIDED_LABELS or item in bad:
            vals[item] = None
        elif label in SCORING_LABELS:
            vals[item] = True
        else:                                    # pragma: no cover - guard
            raise ValueError(f"unhandled audit label {label!r} on item {item}")
    return pd.Series(vals, dtype=object)


def apply_correction(
    run: pd.DataFrame,
    audit: pd.DataFrame,
    unknown_as: bool,
    rule: str = "strict_v1",
    **kwargs,
) -> pd.DataFrame:
    """`run` plus `corrected_correct`, resolving `unknown` the given way.

    Items outside the audit keep their `rule` verdict untouched -- the audit
    only ever looked at items that rule called wrong, which is precisely the
    one-sidedness the module docstring warns about.
    """
    scored = rescore.rescore_run(run, rule=rule)
    base_col = "transcription_correct"
    corrected = scored[base_col].astype(bool).copy()
    hc = human_corrected_correct(run, audit, **kwargs)
    for item, val in hc.items():
        corrected.loc[item] = unknown_as if val is None else bool(val)
    out = scored.copy()
    out["corrected_correct"] = corrected
    out["was_audited"] = out.index.isin(audit.index)
    return out


def corrected_accuracy_bounds(run: pd.DataFrame, audit: pd.DataFrame,
                              rule: str = "strict_v1", **kwargs) -> dict:
    """Baseline and corrected accuracy, with `unknown` resolved both ways."""
    lo = apply_correction(run, audit, unknown_as=False, rule=rule, **kwargs)
    hi = apply_correction(run, audit, unknown_as=True, rule=rule, **kwargs)
    n = len(run)
    base = int(lo["transcription_correct"].astype(bool).sum())
    return {
        "n": n,
        "baseline_correct": base,
        "baseline_accuracy": base / n,
        "corrected_correct_low": int(lo["corrected_correct"].sum()),
        "corrected_correct_high": int(hi["corrected_correct"].sum()),
        "corrected_accuracy_low": float(lo["corrected_correct"].mean()),
        "corrected_accuracy_high": float(hi["corrected_correct"].mean()),
        "one_sided": True,
        "note": ("upper-biased: only items the frozen rule called WRONG were "
                 "audited, so only false negatives could be found"),
    }


def deferral_precision(run: pd.DataFrame, audit: pd.DataFrame,
                       entropy_col: str = "perception_entropy",
                       quantile: Optional[float] = None,
                       threshold: Optional[float] = None,
                       **kwargs) -> dict:
    """Of the AUDITED items entropy flags, how many are REAL model failures?

    The practitioner-facing question behind the whole audit: a deferral rule
    fires on high entropy, but is it catching model mistakes or scorer
    mistakes? Restricted to audited items, where ground truth about *why* the
    item failed actually exists.
    """
    if (quantile is None) == (threshold is None):
        raise ValueError("pass exactly one of quantile or threshold")
    ent = run.loc[audit.index, entropy_col]
    thr = float(ent.quantile(quantile)) if threshold is None else float(threshold)
    flagged = audit.index[ent >= thr]
    hc = human_corrected_correct(run, audit, **kwargs)
    real = sum(1 for i in flagged if hc[i] is False)
    scoring = sum(1 for i in flagged if hc[i] is True)
    unknown = sum(1 for i in flagged if hc[i] is None)
    decided = real + scoring
    return {
        "threshold": thr,
        "n_flagged": len(flagged),
        "n_real_error": real,
        "n_scoring_failure": scoring,
        "n_unknown": unknown,
        "precision_real_error": (real / decided) if decided else float("nan"),
        "base_rate_real_error": float(
            np.mean([hc[i] is False for i in audit.index])),
    }


def correct_item_spot_check_sample(run: pd.DataFrame, n: int = 40,
                                   rule: str = "strict_v1",
                                   seed: int = 20260811) -> pd.DataFrame:
    """A RANDOM sample of items `rule` scored CORRECT, for false-pass auditing.

    Fixes the one-sidedness. Selection is random with a fixed seed rather
    than by hand, so the resulting rate is an estimate rather than a
    collection of interesting cases -- the same discipline `reference/cases`
    uses. `known_false_pass` marks items an automated rule already flags, so
    the human pass can be checked against them.
    """
    scored = rescore.rescore_run(run, rule=rule)
    correct = scored.index[scored["transcription_correct"].astype(bool)]
    cls = rescore.classify_scoring_outcome(run)
    known = set(cls.index[cls["category"] == "false_pass_removed"])
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(np.asarray(correct), size=min(n, len(correct)),
                               replace=False).tolist())
    return pd.DataFrame({
        "item": picked,
        "perception_entropy": run.loc[picked, "perception_entropy"].values,
        "known_false_pass": [i in known for i in picked],
        "final_label": "",
        "confidence": "",
        "note": "",
    })
