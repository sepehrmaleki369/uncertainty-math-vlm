"""Scorer-reliability diagnostics across the three completed human audits.

Three passes now exist on the Qwen n=300 run:

  1. `genuinely_wrong` census -- 104 items, every one `strict_v1` called WRONG
  2. `strict_v1`-correct spot check -- 100 items drawn at random from the 141
     it called CORRECT
  3. `strict_v2` high-priority review -- 108 risk-flagged items, MIXED verdicts

This module reports them **per set**, as diagnostics about the scorer. It
deliberately provides no way to collapse them into a corrected model accuracy,
for three reasons that are structural rather than stylistic:

**Two of the three sets agree with the scorer by construction.** Mapping each
vocabulary onto model correctness, set 1's determinate labels are 23 items all
meaning "model wrong" -- on items the scorer already called wrong. Set 2's are
79 items all meaning "model correct" -- on items it already called correct.
Neither vocabulary contains a label that could express disagreement, so their
"agreement" is 100% definitionally. Only set 3, which holds both verdicts and
both directions, carries information. Pooling the three gives 137/170 = 80.6%,
a number that looks measured and is ~60% tautology. **This is the same failure
as the stratum degeneracy already retracted in this project**: where a
stratum's label can take only one value, agreement collapses onto the verdict.

**The sets overlap.** 47 items are shared by sets 1 and 3, 31 by sets 2 and 3.
The union is 234 items against a sum-of-parts of 312, so any count-based
figure double-counts 78 items unless de-duplicated.

**They are targeted, not random.** Set 3 is explicitly the high-risk tier.
Rates from it do not describe the run.

`degenerate` is therefore returned alongside every agreement figure, so a
caller cannot average the three without seeing which are tautological.
"""

from typing import Optional

import pandas as pd

from . import corrected

AUDIT_DIR = "reference/audit"

#: Every audit vocabulary, mapped onto model correctness. `extraction_issue`
#: and `needs_visual` are INDETERMINATE, not wrong: they say the verdict was
#: not earned, which leaves the model's actual answer undecided. Collapsing
#: them to "wrong" is the single easiest way to manufacture a false result
#: here, and the coder asked specifically to keep the two questions apart.
TRUTH_MAP = {
    "true_correct": "correct",
    "true_wrong": "wrong",
    "notation_misread": "wrong",
    "copied_wrong_line": "wrong",
    "hallucination": "wrong",
    "extraction_issue": "indeterminate",
    "needs_visual": "indeterminate",
}

SETS = {
    "genuinely_wrong_census": ["coded_31_qwen_only_qwen_20260811.csv",
                               "coded_73_wrong_on_both_20260811.csv"],
    "v1_correct_spotcheck": ["spotcheck_40_qwen_strict_v1_correct_20260811.csv",
                             "spotcheck_extra60_qwen_strict_v1_correct_20260812.csv"],
    "v2_high_priority": ["strict_v2_high_priority_human_audit_20260812.csv"],
}


def load_audit_sets(audit_dir: str = AUDIT_DIR) -> dict:
    """The three audits as frames indexed by item, with a `truth` column."""
    out = {}
    for name, files in SETS.items():
        frames = []
        for f in files:
            d = pd.read_csv(f"{audit_dir}/{f}")
            if "item_id" in d.columns:
                d = d.rename(columns={"item_id": "item"})
            frames.append(d[["item", "final_label", "note"]])
        d = pd.concat(frames, ignore_index=True).set_index("item")
        unknown = set(d["final_label"]) - set(TRUTH_MAP)
        if unknown:
            raise ValueError(f"{name}: labels not in TRUTH_MAP: {sorted(unknown)}")
        d["truth"] = d["final_label"].map(TRUTH_MAP)
        out[name] = d
    return out


def per_set_diagnostics(v1: pd.Series, v2: pd.Series,
                        audits: Optional[dict] = None) -> pd.DataFrame:
    """Agreement, false passes/fails and extraction rate, PER SET.

    `v1` and `v2` are boolean correctness Series indexed by item. Never
    averaged across sets here -- see `degenerate`, which is True when a set's
    determinate labels run in only one direction and its agreement is
    therefore definitional rather than measured.
    """
    audits = audits if audits is not None else load_audit_sets()
    rows = []
    for name, d in audits.items():
        det = d[d["truth"] != "indeterminate"]
        truth = det["truth"] == "correct"
        row = {"audit_set": name, "n": len(d), "n_determinate": len(det),
               "n_indeterminate": int((d["truth"] == "indeterminate").sum()),
               "extraction_issue": int((d["final_label"] == "extraction_issue").sum())}
        row["extraction_issue_rate"] = row["extraction_issue"] / len(d)
        # Degeneracy is a property of the (set, RULE) pair, not of the set.
        # Set 2's human labels are all "correct", which makes strict_v1's
        # agreement 100% by construction -- the set was SELECTED as v1-correct.
        # strict_v2 was not used to select it, so v2 can and does disagree
        # there, and its 92.4% is a real one-directional measurement of v2's
        # false-fail rate. Collapsing both into one flag would throw that away.
        row["truth_one_directional"] = det["truth"].nunique() < 2
        for tag, scored in (("v1", v1), ("v2", v2)):
            s = scored.reindex(det.index).astype(bool)
            row[f"{tag}_agree"] = int((s == truth).sum())
            row[f"{tag}_agreement"] = float((s == truth).mean()) if len(det) else float("nan")
            row[f"{tag}_false_pass"] = int((s & ~truth).sum())
            row[f"{tag}_false_fail"] = int((~s & truth).sum())
            # Tautological only when BOTH sides are constant: the human label
            # cannot vary and neither can the rule's verdict.
            row[f"{tag}_tautological"] = bool(
                row["truth_one_directional"] and s.nunique() < 2)
            row[f"{tag}_detectable_error"] = (
                "none" if row[f"{tag}_tautological"] else
                ("false_fail_only" if row["truth_one_directional"] and truth.all()
                 else ("false_pass_only" if row["truth_one_directional"]
                       else "both")))
        rows.append(row)
    return pd.DataFrame(rows)


def _pairs(audits: dict):
    names = list(audits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            yield a, b


def overlap_report(audits: Optional[dict] = None) -> dict:
    """Pairwise overlaps, the union, and conflicts split by kind.

    A HARD contradiction is two determinate labels pointing opposite ways --
    a genuine intra-rater disagreement. A SOFT difference is one side
    indeterminate, which usually means the two sheets asked different
    questions with different vocabularies rather than that the coder changed
    their mind.
    """
    audits = audits if audits is not None else load_audit_sets()
    hard, soft, overlaps = [], [], {}
    for a, b in _pairs(audits):
        shared = sorted(set(audits[a].index) & set(audits[b].index))
        overlaps[f"{a}|{b}"] = len(shared)
        for i in shared:
            ta, tb = audits[a].loc[i, "truth"], audits[b].loc[i, "truth"]
            la, lb = audits[a].loc[i, "final_label"], audits[b].loc[i, "final_label"]
            if ta != "indeterminate" and tb != "indeterminate" and ta != tb:
                hard.append({"item": i, "set_a": a, "label_a": la,
                             "set_b": b, "label_b": lb})
            elif la != lb:
                soft.append({"item": i, "set_a": a, "label_a": la,
                             "set_b": b, "label_b": lb})
    union = set().union(*[set(d.index) for d in audits.values()])
    return {
        "pairwise_overlap": overlaps,
        "n_union": len(union),
        "n_sum_of_parts": sum(len(d) for d in audits.values()),
        "n_double_counted": sum(len(d) for d in audits.values()) - len(union),
        "hard_contradictions": hard,
        "soft_differences": soft,
    }


def contradiction_sheet(run: Optional[pd.DataFrame] = None,
                        audits: Optional[dict] = None,
                        path: Optional[str] = None) -> pd.DataFrame:
    """The hard contradictions, both readings side by side, for later adjudication.

    Carries BOTH full notes rather than just the labels: without the coder's
    reasoning from each pass there is nothing to adjudicate later, and the
    point of the sheet is that these four can be settled without re-deriving
    which items they were. `adjudicated_label` ships empty -- nothing here
    resolves them, because a third reading by the same coder is what produced
    the disagreement in the first place.
    """
    audits = audits if audits is not None else load_audit_sets()
    rows = []
    for c in overlap_report(audits)["hard_contradictions"]:
        i = c["item"]
        row = {"item": i,
               "set_a": c["set_a"], "label_a": c["label_a"],
               "note_a": audits[c["set_a"]].loc[i, "note"],
               "set_b": c["set_b"], "label_b": c["label_b"],
               "note_b": audits[c["set_b"]].loc[i, "note"]}
        if run is not None:
            v = corrected.label_views(run, i)
            row.update({"model_span": v["model_span"], "truth_span": v["truth_span"],
                        "model_label": v["model_label"], "truth_label": v["truth_label"],
                        "entropy": v["entropy"]})
        row["adjudicated_label"] = ""
        row["adjudication_note"] = ""
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("item").reset_index(drop=True)
    if path:
        out.to_csv(path, index=False)
    return out


#: Later, more specific passes win. Set 3 asked "is the model right?" with a
#: vocabulary that can say yes; sets 1 and 2 asked narrower questions.
SET_PRECEDENCE = ("v2_high_priority", "v1_correct_spotcheck",
                  "genuinely_wrong_census")


def deduplicated_counts(audits: Optional[dict] = None) -> dict:
    """Label counts over the 234-item UNION, one row per item.

    Summing the three sets counts 78 items twice. Precedence resolves those,
    and the conflicts it papers over are returned alongside rather than
    silently dropped.
    """
    audits = audits if audits is not None else load_audit_sets()
    chosen, source = {}, {}
    for name in SET_PRECEDENCE:
        for i, r in audits[name].iterrows():
            if i not in chosen:
                chosen[i], source[i] = r["final_label"], name
    lab = pd.Series(chosen).sort_index()
    rep = overlap_report(audits)
    return {
        "n_union": len(lab),
        "label_counts": lab.value_counts().to_dict(),
        "truth_counts": lab.map(TRUTH_MAP).value_counts().to_dict(),
        "source_counts": pd.Series(source).value_counts().to_dict(),
        "n_hard_contradictions_masked": len(rep["hard_contradictions"]),
        "precedence": SET_PRECEDENCE,
    }
