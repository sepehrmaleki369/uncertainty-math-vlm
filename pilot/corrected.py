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


# ---------------------------------------------------------------------------
# The OTHER side of the audit: false passes among items the rule called correct
# ---------------------------------------------------------------------------

#: Labels used by the strict_v1-CORRECT spot check. Deliberately NOT
#: `pilot.failures.LABELS` -- that vocabulary describes why a WRONG item
#: failed, and here the question is whether a CORRECT verdict was earned.
SPOTCHECK_LABELS = ("true_correct", "extraction_issue", "needs_visual")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval -- correct near 0 and 1, unlike normal-approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def false_pass_rate(spot: pd.DataFrame) -> dict:
    """Share of `strict_v1`-CORRECT items whose correct verdict was NOT earned.

    `extraction_issue` here means the match was vacuous -- both sides collapsed
    to the same fragment (`c`, `p`, `r`, `i`), or the captured span is only
    part of a multi-part answer. It does NOT always mean the model was wrong;
    it means the CORRECT verdict is not evidence that it was right.
    """
    n = len(spot)
    fp = int((spot["final_label"] == "extraction_issue").sum())
    tc = int((spot["final_label"] == "true_correct").sum())
    lo, hi = _wilson(fp, n)
    return {"n_sampled": n, "n_false_pass": fp, "n_true_correct": tc,
            "n_undecided": n - fp - tc,
            "false_pass_rate": fp / n, "ci_low": lo, "ci_high": hi,
            "true_correct_rate": tc / n,
            "true_correct_ci": _wilson(tc, n)}


def two_sided_accuracy_bounds(run: pd.DataFrame, audit: pd.DataFrame,
                              spot: pd.DataFrame, rule: str = "strict_v1",
                              **kwargs) -> dict:
    """Accuracy after correcting BOTH directions -- the honest figure.

    The one-sided version could only add items, so it ran to 71-74%. Removing
    the false passes that the spot check found pulls it most of the way back:
    the two corrections very nearly cancel. Reporting only the one-sided
    number would have overstated accuracy by roughly twenty points.

    The false-pass count is EXTRAPOLATED from a 40-item sample to all 141
    correct items, so its Wilson interval is carried through -- this is an
    estimate with sampling error, not a census like the wrong-side audit.
    """
    base = corrected_accuracy_bounds(run, audit, rule=rule, **kwargs)
    n_correct = base["baseline_correct"]
    fp = false_pass_rate(spot)

    recovered_lo = base["corrected_correct_low"] - n_correct    # unknown=wrong
    recovered_hi = base["corrected_correct_high"] - n_correct   # unknown=correct

    # verified-correct among the 141, from the sampled true_correct rate
    tc_lo, tc_hi = fp["true_correct_ci"]
    kept_point = n_correct * fp["true_correct_rate"]
    kept_lo, kept_hi = n_correct * tc_lo, n_correct * tc_hi

    n = base["n"]
    return {
        **{f"onesided_{k}": v for k, v in base.items() if "accuracy" in k},
        "n": n,
        "baseline_accuracy": base["baseline_accuracy"],
        "false_pass_rate": fp["false_pass_rate"],
        "false_pass_ci": (fp["ci_low"], fp["ci_high"]),
        "n_recovered_from_wrong": (recovered_lo, recovered_hi),
        "n_false_pass_estimated": (n_correct * fp["ci_low"],
                                   n_correct * fp["ci_high"]),
        "two_sided_accuracy_point": (kept_point + recovered_lo) / n,
        "two_sided_accuracy_low": (kept_lo + recovered_lo) / n,
        "two_sided_accuracy_high": (kept_hi + recovered_hi) / n,
        "note": ("false passes extrapolated from a 40-item random sample of "
                 "the 141 correct items; Wilson CI carried through"),
    }


def audited_subset_labels(run: pd.DataFrame, audit: pd.DataFrame,
                          spot: pd.DataFrame, unknown_as: bool,
                          **kwargs) -> pd.DataFrame:
    """Every item a human actually read, with its corrected label.

    104 wrong (all of them) + 40 correct (a seeded random draw), plus a
    `weight` column that MUST be used when estimating anything from it.

    **The naive unweighted AUROC on this frame is biased LOW and it is a trap.**
    AUROC's invariance to class prevalence is real but does not apply here:
    after correction each class is a MIXTURE of both source strata, and the
    strata were sampled at different rates (100% of the wrong side, 28% of the
    correct side). The corrected-correct class ends up 76% recovered
    high-entropy items where the population would be ~47%, so the correct
    class looks far noisier than it is, so use `weighted_auroc`.

    Correcting the weighting turned out to matter much less than expected --
    0.586 unweighted against 0.572 weighted -- because the false passes pulled
    up into the WRONG class are themselves LOW entropy (mean 0.818 against
    0.530 for verified-correct), so both classes shift down together. The
    weighting is still the right estimator; it simply is not what moves this
    number.
    """
    hc = human_corrected_correct(run, audit, **kwargs)
    rows = {}
    for item, val in hc.items():
        rows[item] = unknown_as if val is None else bool(val)
    for _, r in spot.iterrows():
        lab = r["final_label"]
        if lab == "true_correct":
            rows[int(r["item"])] = True
        elif lab == "extraction_issue":
            rows[int(r["item"])] = False        # the verdict was not earned
        else:
            rows[int(r["item"])] = unknown_as
    idx = sorted(rows)
    out = run.loc[idx, ["perception_entropy"]].copy()
    out["corrected_correct"] = [rows[i] for i in idx]
    out["side"] = ["wrong_side" if i in audit.index else "correct_side"
                   for i in idx]
    # Inverse sampling probability: the wrong side is a census, the correct
    # side is 40 drawn from `n_correct_total`.
    n_correct_total = int(rescore.rescore_run(run, rule="strict_v1")
                          ["transcription_correct"].astype(bool).sum())
    w = n_correct_total / len(spot)
    out["weight"] = [1.0 if sd == "wrong_side" else w for sd in out["side"]]
    return out


def weighted_auroc(frame: pd.DataFrame, entropy_col: str = "perception_entropy",
                   correctness_col: str = "corrected_correct",
                   weight_col: str = "weight", n_boot: int = 4000,
                   seed: int = 0, alpha: float = 0.05) -> dict:
    """AUROC with inverse-sampling weights, plus a STRATIFIED bootstrap CI.

    Estimates P(entropy_wrong > entropy_correct) over the weighted pseudo-
    population, so the differently-sampled strata are put back in their true
    proportions. Ties contribute 0.5, matching the unweighted convention.
    The bootstrap resamples WITHIN each stratum, because the wrong side is a
    census and carries no sampling error of its own.
    """
    ent = frame[entropy_col].to_numpy(float)
    ok = frame[correctness_col].to_numpy(bool)
    wt = frame[weight_col].to_numpy(float)
    side = frame["side"].to_numpy()

    def _auroc(e, o, w):
        ew, ww = e[~o], w[~o]
        ec, wc = e[o], w[o]
        if not len(ew) or not len(ec):
            return float("nan")
        gt = (ew[:, None] > ec[None, :]).astype(float)
        eq = (ew[:, None] == ec[None, :]).astype(float)
        num = (ww[:, None] * wc[None, :] * (gt + 0.5 * eq)).sum()
        return float(num / (ww.sum() * wc.sum()))

    point = _auroc(ent, ok, wt)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(side == s) for s in ("wrong_side", "correct_side")]
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in strata if len(ix)])
        val = _auroc(ent[pick], ok[pick], wt[pick])
        if val == val:
            draws.append(val)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auroc": point, "ci_low": float(lo), "ci_high": float(hi),
            "n": len(frame), "excludes_chance": bool(lo > 0.5),
            "n_boot_ok": len(draws)}


def correct_item_spot_check_extension(run: pd.DataFrame, existing: pd.DataFrame,
                                      n: int = 60, rule: str = "strict_v1",
                                      seed: int = 20260812) -> pd.DataFrame:
    """A further random draw of CORRECT items, disjoint from `existing`.

    Drawing 40 at random and then 60 at random from the remaining 101 leaves
    the UNION a uniform random subset of size 100, so the combined false-pass
    rate stays an unbiased estimate -- it is not two samples that have to be
    reconciled. The n=40 Wilson interval was [26.3%, 55.4%], far too wide to
    separate the two readings of the corrected AUROC (0.57 vs 0.73); n=100
    is what tightens it.
    """
    scored = rescore.rescore_run(run, rule=rule)
    correct = set(scored.index[scored["transcription_correct"].astype(bool)])
    already = set(existing["item"].astype(int))
    missing = already - correct
    if missing:
        raise ValueError(f"existing sheet holds non-correct items: {sorted(missing)}")
    pool = sorted(correct - already)
    cls = rescore.classify_scoring_outcome(run)
    known = set(cls.index[cls["category"] == "false_pass_removed"])
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(np.asarray(pool), size=min(n, len(pool)),
                               replace=False).tolist())
    return pd.DataFrame({
        "item": picked,
        "perception_entropy": run.loc[picked, "perception_entropy"].values,
        "known_false_pass": [i in known for i in picked],
        "final_label": "",
        "confidence": "",
        "note": "",
    })


def label_views(run: pd.DataFrame, item: int, rule: str = "strict_v1",
                samples_col: str = "all_transcription_samples_raw",
                gt_col: str = "pert_a") -> dict:
    """BOTH automatic views of one item: the extracted SPAN and the comparison LABEL.

    A false pass can be manufactured at either of two stages, and the two are
    not distinguishable from the comparison label alone:

      * **extraction picked the wrong span** -- the label faithfully encodes a
        piece of the page that is not the final answer (item 31: the span is
        `2 cm` where the page's answer is 19.2 cm);
      * **normalization collapsed a good span** -- the span is right and long,
        and SymPy reduces it to a single symbol (item 239: a 76-character
        `L.H.S = 9e^{-3x} + ...` becomes `sympy:l`), after which two collapsed
        labels MATCH for no reason.

    Measured on the 16 false passes, and it corrected an earlier claim: only
    **2** are SymPy collapse. **6** are a one-character SPAN the extractor
    chose (item 84's span is literally `c` -- SymPy encoded it faithfully) and
    **8** are a partial span (item 27 keeps `4x = 3` and drops `y = 33/4`).
    So the fix is in `extract_final_answer`, not in the normalizer.

    Showing the span next to the label separates them by eye. The model side
    is the sample that PRODUCED the majority label, not sample 0, so the span
    and the label always describe the same generation.
    """
    import ast

    trace = rescore.trace_item(
        ast.literal_eval(run.loc[item, samples_col]), run.loc[item, gt_col], rule)
    maj = trace["majority_label"]
    rep = next((s for s in trace["samples"] if s["label"] == maj),
               trace["samples"][0])
    return {
        "item": int(item),
        "entropy": trace["perception_entropy"],
        "correct": trace["correct"],
        "model_span": rep["final_answer"],
        "model_label": rep["label"],
        "model_tier": rep["tier"],
        "truth_span": trace["ground_truth"]["final_answer"],
        "truth_label": trace["ground_truth"]["label"],
        "truth_tier": trace["ground_truth"]["tier"],
        "collapsed": (
            len(str(rep["label"]).split(":", 1)[-1]) <= 3
            and len(str(rep["final_answer"] or "")) > 12),
    }
