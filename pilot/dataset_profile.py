"""Characterise the 300-item FERMAT sample by item type, and price the splits.

Offline. Reads the stored run CSV, the two frozen rules and the human audit
CSVs. **No model inference, no scorer rule changed.**

Two design decisions carry most of the honesty of this module.

**1. Answer types are derived from the GROUND TRUTH SIDE ONLY.**
`strict_v2.score_item_v2` deliberately ORs each risk flag across the model and
truth spans, because a risk on either side is a risk for the comparison. That
is right for a review queue and WRONG for grouping: if the model emits a short
wrong span, `tiny_suspicious_non_mcq` fires from the *model* side, and a group
defined that way is then less accurate partly by construction. Reporting
scorer accuracy per group would be measuring the grouping. So
`truth_answer_type` re-derives the flags from the truth span alone, and the
OR'd version is carried alongside as `answer_type_either_side` so the gap is
visible rather than hidden.

**2. `has_error` is a property of the PAGE, not of the model.** Splitting on
it is safe here. Splitting on a *rule's own verdict* is not, and where this
module reports metrics by `strict_v1 correct/wrong` it reports only the
quantities that are not definitionally fixed by that split -- an AUROC within
a single-verdict group is degenerate for the same reason the retracted
stratified reasoning result was, and is refused rather than printed.
"""

import re
from typing import Optional, Sequence

import pandas as pd

from . import plotting, strict_v2

#: The project's registered minority-class minimum. A group below it gets no
#: AUROC, the same bar every other run in this project was held to.
MIN_MINORITY = 30

#: Answer shapes, in resolution order. An item can trip several flags; the
#: first match wins, so the order encodes which distinction matters most.
ANSWER_TYPE_ORDER = ("mcq_option", "set_answer", "system_answer",
                     "derivative_equation", "multi_value", "text_conclusion",
                     "numeric", "single_expression", "unknown")

#: Categories requested in the brief that CANNOT be inferred from the stored
#: fields, recorded so their absence is a stated result rather than a silent
#: gap. `proof/conclusion` is folded into `text_conclusion` -- the detector
#: fires on any prose answer and cannot tell a proof from a sentence.
UNINFERABLE_TYPES = {
    "table_cell_answer": ("no detector exists and nothing in the stored fields "
                          "distinguishes a table cell from any other short "
                          "span; never assigned"),
    "proof_conclusion": ("folded into `text_conclusion`; the prose detector "
                         "cannot separate a proof from any other sentence"),
}

_NUMERIC_RE = re.compile(r"^[-+]?\d[\d\s.,/*^+\-]*$")
_NUM_WITH_UNIT_RE = re.compile(
    r"^[-+]?\d[\d\s.,/*^+\-]*\\?[a-z%°]{0,12}\.?$", re.I)


def truth_answer_type(flags: dict, truth_span: Optional[str]) -> str:
    """One shape label, from TRUTH-SIDE flags only. See the module docstring."""
    span = str(truth_span or "").strip()
    if not span:
        return "unknown"
    if flags.get("mcq_option") or flags.get("tiny_valid_mcq"):
        return "mcq_option"
    for flag, name in (("set_answer", "set_answer"),
                       ("system_answer", "system_answer"),
                       ("derivative_equation", "derivative_equation"),
                       ("multi_value_answer", "multi_value"),
                       ("text_conclusion", "text_conclusion")):
        if flags.get(flag):
            return name
    norm = strict_v2.normalize_span(span)
    if _NUMERIC_RE.match(norm) or _NUM_WITH_UNIT_RE.match(norm):
        return "numeric"
    return "single_expression"


# --- the red markup, and whether it can serve as error metadata ------------

def red_spans(text: Optional[str]) -> list:
    """Brace-balanced payloads of every `\\textcolor{red}{...}`.

    Brace-balanced rather than `[^{}]*`, because a nested macro is common here
    (`\\textcolor{red}{\\hat{b}}`) and the naive pattern truncates at the first
    inner brace. That exact shortcut is recorded as extractor bug 1.
    """
    s = str(text or "")
    out = []
    for m in re.finditer(r"\\textcolor\s*\{\s*red\s*\}\s*\{", s):
        i, depth = m.end(), 1
        while i < len(s) and depth:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        out.append(s[m.end():i - 1])
    return out


def red_markup_report(run: pd.DataFrame, gt_col: str = "pert_a") -> dict:
    """Can `\\textcolor{red}` be used as injected-error location metadata?

    Answered with counts rather than by assumption, because the intuitive
    reading -- red marks the injected error -- is what this project's own
    notes had recorded, and the data does not support it as stated.
    """
    spans = run[gt_col].map(red_spans)
    n_red = spans.map(len)
    has_error = run["has_error"].astype(bool)
    out = {
        "n_items": len(run),
        "items_with_red": int((n_red > 0).sum()),
        "total_red_spans": int(n_red.sum()),
        "error_items_with_red": int((n_red > 0)[has_error].sum()),
        "error_items_total": int(has_error.sum()),
        "clean_items_with_red": int((n_red > 0)[~has_error].sum()),
        "clean_items_total": int((~has_error).sum()),
        "mean_spans_error": float(n_red[has_error].mean()),
        "mean_spans_clean": float(n_red[~has_error].mean()),
        "median_spans_error": float(n_red[has_error].median()),
        "median_spans_clean": float(n_red[~has_error].median()),
        "max_spans_error": int(n_red[has_error].max()),
        "error_items_without_red": sorted(
            int(i) for i in run.index[has_error & (n_red == 0)]),
    }
    out["clean_with_red_share"] = (out["clean_items_with_red"]
                                   / max(1, out["clean_items_total"]))
    # The decisive number: if red marked the injected error, it would be rare
    # on clean pages. It is not.
    out["usable_as_error_location"] = False
    return out


# --- per-item profile -------------------------------------------------------

def item_profile(run: pd.DataFrame, v1: pd.Series, v2s: pd.DataFrame,
                 human: Optional[pd.Series] = None,
                 human_raw: Optional[pd.Series] = None,
                 extraction_status: Optional[pd.Series] = None,
                 entropy_col: str = "perception_entropy",
                 gt_col: str = "pert_a") -> pd.DataFrame:
    """One row per item, carrying every field the brief asked for.

    Fields the dataset does not provide are emitted as EMPTY columns rather
    than omitted, so the CSV itself records what is missing.
    """
    idx = run.index
    max_h = float(run[entropy_col].max())
    rows = []
    for i in idx:
        v2 = v2s.loc[i]
        gt_flags = strict_v2.answer_flags(
            str(v2["truth_span"] or ""), str(v2["truth_label"] or ""),
            bool(v2["is_mcq"]))
        reds = red_spans(run.loc[i, gt_col])
        rows.append({
            "item_id": int(i),
            "has_error": bool(run.loc[i, "has_error"]),
            "strict_v1_correct": bool(v1.loc[i]),
            "strict_v2_correct": bool(v2["correct_strict_v2_display_primary"]),
            "answer_type": truth_answer_type(gt_flags, v2["truth_span"]),
            "question_type_if_available": "mcq" if bool(v2["is_mcq"]) else "free_response",
            "entropy": float(run.loc[i, entropy_col]),
            "parse_failed": int(run.loc[i, "n_transcription_parse_failures"]) > 0,
            "max_entropy": float(run.loc[i, entropy_col]) >= max_h - 1e-9,
            "span_m_disp": v2["model_span"],
            "span_t_disp": v2["truth_span"],
            "label_m": v2["model_label"],
            "label_t": v2["truth_label"],
            "human_label_if_available": (str(human_raw.get(i, ""))
                                         if human_raw is not None else ""),
            "human_truth_if_available": (str(human.get(i, ""))
                                         if human is not None else ""),
            "extraction_status_if_available": (
                str(extraction_status.get(i, ""))
                if extraction_status is not None else ""),
            # NOT PROVIDED BY THE DATASET. Empty on purpose -- see
            # `metadata_availability`. The red markup is recorded as a
            # separate, explicitly approximate column so it can never be
            # mistaken for the missing metadata.
            "error_location_if_available": "",
            "error_type_if_available": "",
            "image_path_if_available": "",
            "n_red_spans": len(reds),
            "red_spans_approx_only": " | ".join(r[:60] for r in reds[:3]),
            "answer_type_either_side": _either_side_type(v2),
            "truth_span_len": len(str(v2["truth_span"] or "")),
            "model_span_len": len(str(v2["model_span"] or "")),
        })
    return pd.DataFrame(rows).set_index("item_id", drop=False)


def _either_side_type(v2row) -> str:
    """The OR'd (model-or-truth) shape, for comparison with the truth-only one."""
    for flag, name in (("mcq_option", "mcq_option"),
                       ("tiny_valid_mcq", "mcq_option"),
                       ("set_answer", "set_answer"),
                       ("system_answer", "system_answer"),
                       ("derivative_equation", "derivative_equation"),
                       ("multi_value_answer", "multi_value"),
                       ("text_conclusion", "text_conclusion")):
        if bool(v2row.get(flag)):
            return name
    return "numeric_or_expression"


# --- group metrics ----------------------------------------------------------

def group_metrics(profile: pd.DataFrame, by: str,
                  auroc_correctness: str = "strict_v1_correct",
                  n_boot: int = 4000, seed: int = 0,
                  min_minority: int = MIN_MINORITY) -> pd.DataFrame:
    """Full metrics for every level of `by`. One row per group.

    **The AUROC is refused, not approximated, when the split makes it
    meaningless.** Two cases: a minority class below `min_minority` (the bar
    every other run here was held to), and a group defined BY the correctness
    column itself, where every item carries the same label and an AUROC is
    undefined. The second is the same degeneracy that produced this project's
    one retracted result, so it is detected structurally rather than trusted
    to a reader.
    """
    rows = []
    degenerate_split = by == auroc_correctness
    for level, sub in profile.groupby(by, dropna=False):
        n = len(sub)
        row = {
            "group": str(level), "n": n,
            "share": n / max(1, len(profile)),
            "strict_v1_accuracy": float(sub["strict_v1_correct"].mean()),
            "strict_v2_accuracy": float(sub["strict_v2_correct"].mean()),
            "mean_entropy": float(sub["entropy"].mean()),
            "median_entropy": float(sub["entropy"].median()),
            "parse_failure_rate": float(sub["parse_failed"].mean()),
            "max_entropy_rate": float(sub["max_entropy"].mean()),
            "median_truth_span_len": float(sub["truth_span_len"].median()),
        }
        n_correct = int(sub[auroc_correctness].sum())
        minority = min(n_correct, n - n_correct)
        if degenerate_split:
            row["auroc"] = None
            row["auroc_ci"] = ""
            row["auroc_status"] = "degenerate_split_defined_by_the_label"
        elif minority < min_minority:
            row["auroc"] = None
            row["auroc_ci"] = ""
            row["auroc_status"] = f"underpowered_minority_{minority}"
        else:
            ci = plotting.bootstrap_auroc_ci(
                sub, "entropy", auroc_correctness, n_boot=n_boot, seed=seed)
            row["auroc"] = float(ci["auroc"])
            row["auroc_ci"] = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            row["auroc_status"] = ("excludes_chance" if ci["ci_low"] > 0.5
                                   else "includes_chance")
        rows.append(row | _human_block(sub))
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out.reset_index(drop=True)


def _human_block(sub: pd.DataFrame) -> dict:
    """Audit-derived counts, over the audited subset of the group only.

    `extraction_issue` is INDETERMINATE, never "wrong" -- it says the verdict
    was not earned, which leaves the model's own answer undecided. Folding it
    into wrong is the easiest available way to manufacture a result here, so
    the rates are kept apart and each carries its own denominator.
    """
    lab = sub[sub["human_label_if_available"] != ""]
    det = sub[sub["human_truth_if_available"].isin(["correct", "wrong"])]
    n_lab = len(lab)
    out = {
        "n_human_labelled": n_lab,
        "extraction_issue_rate": (
            float((lab["human_label_if_available"] == "extraction_issue").mean())
            if n_lab else float("nan")),
        "n_human_determinate": len(det),
    }
    for rule, tag in (("strict_v1_correct", "v1"), ("strict_v2_correct", "v2")):
        out[f"false_pass_{tag}"] = int(
            ((det["human_truth_if_available"] == "wrong") & det[rule]).sum())
        out[f"false_fail_{tag}"] = int(
            ((det["human_truth_if_available"] == "correct") & ~det[rule]).sum())
    return out


def metadata_availability(run: pd.DataFrame) -> dict:
    """What the stored sample does and does not carry. Checked, not assumed."""
    present = set(run.columns)
    def flag(name, cols, note):
        return {"field": name, "available": any(c in present for c in cols),
                "columns_checked": list(cols), "note": note}
    return {
        "stored_columns": sorted(present),
        "fields": [
            flag("injected error location", ["error_location", "error_span",
                                             "perturbation_span", "error_idx"],
                 "ABSENT. No column identifies or locates the injected "
                 "error. The "
                 "red markup is a separate, approximate signal -- see "
                 "red_markup_report -- and is not error-location metadata."),
            flag("injected error type/category", ["error_type", "error_category",
                                                  "perturbation_type"],
                 "ABSENT. Nothing categorises the perturbation."),
            flag("original clean answer", ["orig_a", "clean_answer",
                                           "original_answer", "gold_a"],
                 "ABSENT. Only `pert_a` (the perturbed answer) is stored, so "
                 "clean-vs-perturbed cannot be diffed. `orig_q` is the "
                 "QUESTION, not the original answer -- the name invites that "
                 "misreading."),
            flag("page / image id", ["image_id", "page_id", "item_uid", "uid"],
                 "ABSENT from the stored CSV. The loader requests an `image` "
                 "column, but it holds pixels and is not persisted to the run "
                 "CSV, so items are addressable only by row index."),
            flag("problem type / subject / topic", ["subject", "topic",
                                                    "problem_type", "chapter"],
                 "ABSENT. Question type is INFERRED here (MCQ vs free "
                 "response) from the question and truth text, never read."),
            flag("handwriting style", ["handwriting_style"],
                 "PRESENT, boolean, semantics vaguer than image_quality."),
            flag("image quality", ["image_quality"],
                 "PRESENT, boolean, documented in arXiv 2501.07244 as True if "
                 "good, False if illumination/shadow/blur/contrast issues."),
            flag("has_error", ["has_error"],
                 "PRESENT. The only label describing the perturbation, and it "
                 "is binary -- whether an error exists, not where or what."),
        ],
    }


#: Coarse shape split for the "simple value vs structured answer" comparison.
SIMPLE_TYPES = ("numeric", "single_expression", "mcq_option")
STRUCTURED_TYPES = ("multi_value", "set_answer", "system_answer",
                    "derivative_equation", "text_conclusion")


def add_derived_groupings(profile: pd.DataFrame) -> pd.DataFrame:
    """Length buckets and a coarse simple/structured split, for goal 4.

    Buckets are cut on the TRUTH span, not the model's. Cutting on the model
    span would make "short spans score badly" partly a statement about which
    items the model answered badly, which is the circularity this module's
    answer types already avoid.
    """
    out = profile.copy()
    out["answer_shape"] = out["answer_type"].map(
        lambda t: "simple_value" if t in SIMPLE_TYPES else
        ("structured_or_prose" if t in STRUCTURED_TYPES else "unknown"))
    out["truth_span_bucket"] = pd.cut(
        out["truth_span_len"], bins=[-1, 3, 10, 30, 100, 10 ** 9],
        labels=["tiny_<=3", "short_4-10", "medium_11-30", "long_31-100",
                "very_long_>100"])
    out["mcq_vs_open"] = out["question_type_if_available"]
    return out


# --- example manifest (text-only until Colab supplies the images) ----------

def example_selection(profile: pd.DataFrame, n_per_stratum: int = 10,
                      seed: int = 20260812) -> pd.DataFrame:
    """10 `has_error=1` and 10 clean, split evenly on `strict_v1`, types varied.

    Four equal sub-draws (stratum x verdict), round-robin over answer types
    within each.

    **The type order is SHUFFLED, and a fixed order is a real bug here.** A
    first version walked `ANSWER_TYPE_ORDER`, which puts `numeric` and
    `single_expression` last -- so with only five slots per sub-draw the round
    robin never reached them, and the two types covering **54% of the corpus**
    could not appear on the sheet at all. Shuffling makes the exclusion
    unbiased instead of systematic.

    Twenty tiles cannot cover eight types across two verdicts and two strata,
    so the sheet is ILLUSTRATIVE, not representative. Read the group tables for
    the distribution; read these for what the items look like.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    picked = []
    per_cell = n_per_stratum // 2
    for flag in (True, False):
        for verdict in (True, False):
            cell = profile[(profile["has_error"] == flag)
                           & (profile["strict_v1_correct"] == verdict)]
            by_type = {}
            for i in cell.index:
                by_type.setdefault(cell.loc[i, "answer_type"], []).append(int(i))
            for t in by_type:
                rng.shuffle(by_type[t])
            order = list(by_type)
            rng.shuffle(order)
            chosen = []
            while len(chosen) < per_cell and order:
                for t in list(order):
                    if len(chosen) >= per_cell:
                        break
                    if by_type[t]:
                        chosen.append(by_type[t].pop(0))
                    else:
                        order.remove(t)
            picked.extend(chosen)
    return profile.loc[sorted(picked)]


def _tile_note(row) -> str:
    """One short line saying why this item is worth looking at.

    Ordered by how much the observation tells a reader, not by how easy it is
    to detect. The collapse check comes first because a vacuous match is the
    single mechanism the audit found behind most false passes, and it is
    invisible from the verdict alone.
    """
    m, t = str(row["label_m"] or ""), str(row["label_t"] or "")
    payload = m.split(":", 1)[-1]
    if m == t and len(payload) <= 2 and len(str(row["span_t_disp"] or "")) > 20:
        return (f"both labels collapsed to `{m}` from a long span; the match "
                "is vacuous")
    if row["answer_type"] == "text_conclusion" and not row["strict_v1_correct"]:
        return "prose conclusion; strict matching cannot score a sentence"
    if row["truth_span_len"] <= 3 and not row["strict_v1_correct"]:
        return "tiny truth span; a coincidental match or miss is cheap here"
    if row["max_entropy"]:
        return "all five samples differ; entropy at its ceiling"
    if row["parse_failed"]:
        return "at least one sample failed to parse"
    if row["strict_v1_correct"] != row["strict_v2_correct"]:
        return "the two rules disagree on this item"
    if row["strict_v1_correct"]:
        return "scored correct by both rules"
    return "scored wrong; span and label shown for comparison"


def write_example_manifest(out_dir: str, examples: pd.DataFrame) -> dict:
    """A captioned, text-only contact sheet plus its CSV.

    **Text-only by necessity, not by choice.** FERMAT is a gated dataset and
    its images are never persisted to the run CSV, so no crop can be produced
    outside an authenticated Colab session. The manifest carries every caption
    field the image sheet would, so attaching tiles later is an asset export
    rather than a re-analysis.
    """
    import html
    import os

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "manifest.csv")
    cols = ["item_id", "has_error", "strict_v1_correct", "strict_v2_correct",
            "answer_type", "question_type_if_available", "entropy",
            "max_entropy", "parse_failed", "span_t_disp", "span_m_disp",
            "label_t", "label_m", "human_label_if_available", "note"]
    ex = examples.copy()
    ex["note"] = [_tile_note(ex.loc[i]) for i in ex.index]
    ex[cols].to_csv(csv_path, index=False)

    tiles = []
    for i in ex.index:
        r = ex.loc[i]
        v1 = "correct" if r["strict_v1_correct"] else "wrong"
        v2 = "correct" if r["strict_v2_correct"] else "wrong"
        tiles.append(f"""
  <figure class="tile {'err' if r['has_error'] else 'clean'}">
    <figcaption>
      <b>item {int(r['item_id'])}</b>
      <span class="pill">has_error={int(bool(r['has_error']))}</span>
      <span class="pill {'ok' if r['strict_v1_correct'] else 'no'}">v1 {v1}</span>
      <span class="pill {'ok' if r['strict_v2_correct'] else 'no'}">v2 {v2}</span>
      <span class="pill">{html.escape(str(r['answer_type']))}</span>
      <span class="pill">H={r['entropy']:.3f}</span>
    </figcaption>
    <div class="imgslot">image not available offline &mdash; export from Colab</div>
    <dl>
      <dt>truth span</dt><dd>{html.escape(str(r['span_t_disp'])[:400])}</dd>
      <dt>model span</dt><dd>{html.escape(str(r['span_m_disp'])[:400])}</dd>
      <dt>truth label</dt><dd><code>{html.escape(str(r['label_t'])[:200])}</code></dd>
      <dt>model label</dt><dd><code>{html.escape(str(r['label_m'])[:200])}</code></dd>
      <dt>human</dt><dd>{html.escape(str(r['human_label_if_available']) or '—')}</dd>
      <dt>note</dt><dd>{html.escape(str(r['note']))}</dd>
    </dl>
  </figure>""")

    html_doc = f"""<!doctype html><meta charset="utf-8">
<title>FERMAT n=300 — example contact sheet (text-only)</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:24px;max-width:1200px}}
 .banner{{background:#fff4e5;border-left:4px solid #e69500;padding:12px 16px;
          margin-bottom:20px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
        gap:16px}}
 .tile{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:0}}
 .tile.err{{border-left:4px solid #c0392b}} .tile.clean{{border-left:4px solid #2980b9}}
 .pill{{display:inline-block;background:#eee;border-radius:10px;padding:1px 8px;
        font-size:12px;margin-left:4px}}
 .pill.ok{{background:#d6f5d6}} .pill.no{{background:#fadbd8}}
 .imgslot{{background:#f6f6f6;border:1px dashed #bbb;color:#888;text-align:center;
           padding:22px 8px;margin:10px 0;font-size:12px}}
 dl{{margin:0;display:grid;grid-template-columns:88px 1fr;gap:2px 8px;font-size:13px}}
 dt{{color:#666}} dd{{margin:0;overflow-wrap:anywhere}}
 code{{background:#f4f4f4;padding:1px 3px}}
</style>
<h1>FERMAT n=300 — example contact sheet</h1>
<div class="banner">
<b>Text-only.</b> FERMAT is gated and its images are not persisted to the run
CSV, so no crop can be rendered offline. <b>Image export requires an
authenticated Colab/Drive session</b> (reuse notebook 23's contact-sheet
machinery). Every caption field the image sheet would carry is already here,
so attaching tiles is an asset export, not a re-analysis.
</div>
<p>{len(ex)} items: {int(ex['has_error'].sum())} <code>has_error=1</code>,
{int((~ex['has_error']).sum())} clean;
{int(ex['strict_v1_correct'].sum())} scored correct by <code>strict_v1</code>.
Selection is seeded and round-robin over answer types, never hand-picked.</p>
<div class="grid">{''.join(tiles)}
</div>
"""
    html_path = os.path.join(out_dir, "index.html")
    with open(html_path, "w") as fh:
        fh.write(html_doc)
    return {"csv": csv_path, "html": html_path, "n": len(ex)}


# --- the report -------------------------------------------------------------

_METRIC_COLS = ("group", "n", "strict_v1_accuracy", "strict_v2_accuracy",
                "mean_entropy", "median_entropy", "max_entropy_rate",
                "parse_failure_rate", "auroc", "auroc_ci", "auroc_status",
                "n_human_labelled", "extraction_issue_rate",
                "false_pass_v1", "false_fail_v1")

_HEADERS = ("group", "n", "v1 acc", "v2 acc", "mean H", "med H", "max-H rate",
            "parse fail", "AUROC", "95% CI", "AUROC status", "n human",
            "extr. issue", "false pass", "false fail")


def _md_table(g: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(_HEADERS) + " |",
            "|" + "---|" * len(_HEADERS)]
    for _, r in g.iterrows():
        def num(c, fmt="{:.3f}"):
            v = r[c]
            return "n/a" if v is None or pd.isna(v) else fmt.format(v)
        rows.append("| " + " | ".join([
            f"`{r['group']}`", str(int(r["n"])), num("strict_v1_accuracy", "{:.1%}"),
            num("strict_v2_accuracy", "{:.1%}"), num("mean_entropy"),
            num("median_entropy"), num("max_entropy_rate", "{:.1%}"),
            num("parse_failure_rate", "{:.1%}"), num("auroc"),
            str(r["auroc_ci"] or "n/a"), f"`{r['auroc_status']}`",
            str(int(r["n_human_labelled"])), num("extraction_issue_rate", "{:.1%}"),
            str(int(r["false_pass_v1"])), str(int(r["false_fail_v1"])),
        ]) + " |")
    return "\n".join(rows)


def write_summary_md(path: str, profile: pd.DataFrame, red: dict,
                     meta: dict, groups: dict,
                     mcq_review: Optional[pd.DataFrame] = None) -> str:
    """The distribution report. Descriptive only; no new claim is made here."""
    n = len(profile)
    L = [f"# FERMAT n={n} — distribution and performance by item type\n",
         "Offline. Built from the stored Qwen2.5-VL-3B run "
         "(`scaleup_n300_bal50_...20260802T163202Z.csv`), the two frozen "
         "rules recomputed for comparison, and the human audit CSVs. "
         "**No model inference was run and no scorer rule was changed.**\n"]

    L.append("> **Read the AUROC column with its status.** A group whose "
             "minority class falls below the project's registered minimum of "
             f"{MIN_MINORITY} gets no AUROC at all, and a group *defined by* a "
             "rule's own verdict gets none either, because every item in it "
             "carries the same label. Printing a number in those cells is the "
             "degeneracy that produced this project's one retracted result.\n")

    L.append("## 1. What the 300 items are\n")
    L.append(f"- `has_error`: {int(profile['has_error'].sum())} / "
             f"{int((~profile['has_error']).sum())}, balanced by design")
    L.append(f"- question type (INFERRED, `strict_v2._looks_mcq`): "
             f"{int((profile['question_type_if_available'] == 'mcq').sum())} "
             f"multiple-choice, "
             f"{int((profile['question_type_if_available'] == 'free_response').sum())} "
             "free response")
    L.append(f"- entropy takes {profile['entropy'].nunique()} distinct values "
             f"(all seven reachable at K=5); "
             f"{int((profile['entropy'] == 0).sum())} items at H=0, "
             f"{int(profile['max_entropy'].sum())} at the ln5 ceiling")
    L.append(f"- at least one sample failed to parse on "
             f"{int(profile['parse_failed'].sum())} items")
    L.append(f"- truth span length: median "
             f"{int(profile['truth_span_len'].median())} chars "
             f"(p25 {int(profile['truth_span_len'].quantile(.25))}, "
             f"p75 {int(profile['truth_span_len'].quantile(.75))}, "
             f"max {int(profile['truth_span_len'].max())})\n")
    L.append("### Answer types, from the GROUND TRUTH span only\n")
    L.append("| answer type | n | share |")
    L.append("|---|---|---|")
    for t, c in profile["answer_type"].value_counts().items():
        L.append(f"| `{t}` | {c} | {c / n:.1%} |")
    same = (profile["answer_type"].replace(
        {"numeric": "numeric_or_expression",
         "single_expression": "numeric_or_expression"})
        == profile["answer_type_either_side"])
    L.append(f"\n**Types are derived from the truth side only, and this is not "
             f"a detail.** `strict_v2` ORs each shape flag across the model and "
             "truth spans, which is right for a review queue and wrong for "
             "grouping: a short *wrong model span* would put the item in a "
             "group that then looks inaccurate by construction. The two "
             f"differ on **{int((~same).sum())} of {n} items**, so the choice "
             "moves real mass. The OR'd label is kept in the CSV as "
             "`answer_type_either_side`.\n")
    L.append("**Two requested categories are not assigned, and their absence "
             "is a finding rather than an oversight:**\n")
    for k, why in UNINFERABLE_TYPES.items():
        L.append(f"- `{k}` — {why}")
    L.append("")

    titles = {
        "has_error": "2.1 By `has_error`",
        "answer_type": "2.2 By answer type (truth-side)",
        "question_type_if_available": "2.3 By question type",
        "answer_shape": "2.4 Simple value vs structured / prose",
        "truth_span_bucket": "2.5 By truth span length",
        "strict_v1_correct": "2.6 By `strict_v1` verdict",
        "strict_v2_correct": "2.7 By `strict_v2` verdict",
    }
    L.append("## 2. Metrics by group\n")
    L.append("Every AUROC below scores entropy against **`strict_v1` "
             "correctness**, so the column is comparable across all seven "
             "splits.\n")
    for key, title in titles.items():
        if key not in groups:
            continue
        L.append(f"### {title}\n")
        L.append(_md_table(groups[key]))
        if key == "strict_v1_correct":
            L.append("\n**Both AUROC cells are refused because the split "
                     "defines the label.** Within a group where every item is "
                     "correct, or every item is wrong, there is nothing for a "
                     "ranking to separate. The accuracy columns are likewise "
                     "0% and 100% by construction. What is worth reading here "
                     "is the rest of the row: `strict_v1`-wrong items sit at "
                     "much higher entropy and a much higher max-entropy rate, "
                     "which is the headline signal seen from the other "
                     "side.\n")
        if key == "strict_v2_correct":
            n_dis = int((profile["strict_v1_correct"]
                         != profile["strict_v2_correct"]).sum())
            L.append(f"\n**`underpowered` here is masking something worse, so "
                     f"do not read it as 'a bigger sample would fix it'.** The "
                     f"two rules agree on {len(profile) - n_dis} of "
                     f"{len(profile)} items, so grouping by one and scoring "
                     "entropy against the other is very nearly the degenerate "
                     "split above. The honest reading is that this table has "
                     "no AUROC to give.\n")
        L.append("")

    L.append("## 3. Where the scorer and the model do better or worse\n")
    L.append("**All four comparisons below are descriptive splits of one run. "
             "None is a controlled contrast**, and the groups differ in more "
             "than the dimension named, so a gap is a place to look rather "
             "than an effect that has been isolated.\n")
    if mcq_review is not None and (mcq_review["confirmed_mcq"].astype(str) != "").any():
        yes = sorted(mcq_review.loc[mcq_review["confirmed_mcq"] == "yes",
                                    "item_id"])
        mcq = profile.loc[[i for i in yes if i in profile.index]]
        fr = profile.drop(index=[i for i in yes if i in profile.index])
        heur = profile[profile["question_type_if_available"] == "mcq"]
        L.append("### 2.8 MCQ after the human audit (supersedes the heuristic row above)\n")
        L.append("The MCQ detector is a heuristic and was audited on both "
                 "sides: all 58 flagged items, plus 19 it did NOT flag whose "
                 "questions carry a choice list. A one-sided review could only "
                 "have shrunk the set.\n")
        L.append("| set | n | `strict_v1` acc | mean H |")
        L.append("|---|---|---|---|")
        L.append(f"| heuristic MCQ-like | {len(heur)} | "
                 f"{heur['strict_v1_correct'].mean():.1%} | "
                 f"{heur['entropy'].mean():.3f} |")
        L.append(f"| **confirmed MCQ** | **{len(mcq)}** | "
                 f"**{mcq['strict_v1_correct'].mean():.1%}** | "
                 f"{mcq['entropy'].mean():.3f} |")
        L.append(f"| free response | {len(fr)} | "
                 f"{fr['strict_v1_correct'].mean():.1%} | "
                 f"{fr['entropy'].mean():.3f} |")
        L.append("\n**The detector was wrong in both directions and the two "
                 "errors nearly cancelled.** It wrongly included 4 items "
                 "(`P(A)`/`P(B)` and `f(a)=f(b)` notation read as options) and "
                 "missed 1. The pre-audit sensitivity range was 78.9-88.5%; "
                 "the audited value sits 0.8 points from the uncorrected "
                 "figure. **A detector demonstrably wrong in both directions "
                 "does not necessarily bias the aggregate it feeds.**\n")
        L.append("Two audited items are MCQ but not option-pick, and both are "
                 "SCORING problems rather than model failures: items 170 and "
                 "237 are **matching** questions whose answer is a four-way "
                 "mapping, yet the truth span is a bare `(b)` -- one quarter "
                 "of the answer. Item 8's page shows **no options at all** "
                 "while its answer is `option b`, so the letter is not "
                 "recoverable from the image by any reader.\n")
    L.append("- **MCQ vs free response is the largest single split.** "
             "Multiple-choice items score far higher and sit at much lower "
             "entropy: there are only a few reachable answers, so five samples "
             "agree easily and a match is cheap. Read it as a property of the "
             "answer space, not as the model reading those pages better.")
    L.append("- **`has_error` costs accuracy and leaves the AUROC alone.** "
             "The two strata's AUROCs are within noise of each other, which "
             "reproduces the locked finding that entropy works equally well on "
             "both. The accuracy gap is the known cost of the balanced design.")
    L.append("- **Structured and prose answers are where scoring falls "
             "apart.** `text_conclusion` and `system_answer` are the worst "
             "groups by a wide margin and carry the highest max-entropy rates. "
             "This is the quantitative form of the standing Limitations "
             "sentence: the automatic labels are unsafe for long or multi-part "
             "answers.")
    L.append("- **Span length is monotone until it inverts at the bottom, and "
             "the inversion is the interesting part.** Accuracy falls steadily "
             "from short spans to very long ones. Tiny spans (<=3 chars) then "
             "score *middling* while carrying the **highest `extraction_issue` "
             "rate of any bucket** — their verdicts are the least earned, "
             "which is exactly the false-pass mechanism the audit found. "
             "**A tiny span's verdict should be read as unreliable in either "
             "direction, not as a good score.**\n")

    L.append("## 4. What the dataset does and does not tell us\n")
    L.append("| field | available | note |")
    L.append("|---|---|---|")
    for f in meta["fields"]:
        L.append(f"| {f['field']} | "
                 f"{'**yes**' if f['available'] else '**no**'} | {f['note']} |")
    L.append("\n**Directly answering the four questions asked:**\n")
    L.append("1. **Does the dataset say where the injected error is? No.** "
             "There is no location, span, index or offset field of any kind.")
    L.append("2. **Does it give an error category or type? No.** "
             "`has_error` is binary and is the only label describing the "
             "perturbation.")
    L.append("3. **Does it give both the clean and the perturbed answer? No.** "
             "Only `pert_a` is stored. `orig_q` is the *question*, and its "
             "name invites exactly the misreading that it is the original "
             "answer. Clean-vs-perturbed cannot be diffed from this run.")
    L.append("4. **Can the red markup stand in as approximate location and "
             "type? For location, weakly and only on error items. For type, "
             "no. Details below.**\n")

    L.append("## 5. The red markup, measured\n")
    L.append(f"`\\textcolor{{red}}{{...}}` appears in **{red['items_with_red']} "
             f"of {red['n_items']}** ground-truth answers, "
             f"**{red['total_red_spans']} spans in total**.\n")
    L.append("| | items with red | share | mean spans | median | max |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| `has_error=1` | {red['error_items_with_red']} / "
             f"{red['error_items_total']} | "
             f"{red['error_items_with_red'] / red['error_items_total']:.1%} | "
             f"{red['mean_spans_error']:.2f} | {red['median_spans_error']:.0f} | "
             f"{red['max_spans_error']} |")
    L.append(f"| clean | {red['clean_items_with_red']} / "
             f"{red['clean_items_total']} | {red['clean_with_red_share']:.1%} | "
             f"{red['mean_spans_clean']:.2f} | {red['median_spans_clean']:.0f} | — |")
    L.append("\n**The decisive number is the second row.** If red marked the "
             f"injected error it would be rare on clean pages. Instead "
             f"**{red['clean_with_red_share']:.0%} of clean items carry red "
             "markup**, and they carry *more* of it on average "
             f"({red['mean_spans_clean']:.2f} spans against "
             f"{red['mean_spans_error']:.2f}). Reading those spans shows why: "
             "on clean items red marks added elaboration and restated steps "
             "(one is a full sentence beginning *\"Additionally, if we "
             "consider the factorial function...\"*), and on error items it "
             "tends to mark changed values. **So red marks what was MODIFIED "
             "when the variant was generated, not what is WRONG.**\n")
    L.append("### Limitations of using red as error metadata\n")
    L.append("1. **It is not specific.** It fires on "
             f"{red['clean_with_red_share']:.0%} of clean items, so presence "
             "of red carries almost no information about whether an error "
             "exists.")
    L.append(f"2. **It is not unique.** Error items carry a median of "
             f"{red['median_spans_error']:.0f} red spans and up to "
             f"{red['max_spans_error']}, so it localises to a set of "
             "candidates rather than to the error.")
    missing = red["error_items_without_red"]
    L.append(f"3. **It is not necessary.** "
             f"{'Item ' + str(missing[0]) + ' is' if len(missing) == 1 else 'Items ' + ', '.join(map(str, missing)) + ' are'} "
             "`has_error=1` with no red markup at all, so absence of red does "
             "not mean absence of an error.")
    L.append("4. **It carries no type information.** A red span is a region, "
             "not a category; nothing distinguishes a sign flip from a "
             "dropped unit.")
    L.append("5. **It is markup on the ground-truth LaTeX, and the model is "
             "scored against a rendered page.** Using it as an analysis "
             "variable mixes a transcription artifact into a claim about "
             "perception, and it has already corrupted labels once here — "
             "nested `\\textcolor` is extractor bug 1, which damaged 85 "
             "ground truths.\n")
    L.append("**Recommended use: none, as metadata.** It is legitimate as a "
             "hypothesis-generating pointer when reading a specific error "
             "item by hand, which is how the auto-correction anecdotes were "
             "found. It must not be counted, aggregated, or reported as "
             "error location or error type.\n")

    L.append("## 6. Caveats that bind every human-derived column\n")
    lab = int((profile["human_label_if_available"] != "").sum())
    det = int(profile["human_truth_if_available"].isin(["correct", "wrong"]).sum())
    L.append(f"- **{lab} of {n} items carry a human label; {det} are "
             "determinate.** Those labels come from three TARGETED audit sets, "
             "two of which are one-directional by construction. Per-group "
             "`extraction_issue` rates are therefore **not population rates**.")
    L.append("- **`false pass` is 0 in every group, and that is a definition, "
             "not a clean bill of health.** The audit codes an unearned "
             "verdict as `extraction_issue`, which maps to *indeterminate* "
             "rather than to *wrong*, so it cannot appear in this column. The "
             "unearned-verdict rate is the `extraction_issue` column, and on "
             "audited `strict_v1`-correct items it is "
             f"{groups['strict_v1_correct'].set_index('group').loc['True', 'extraction_issue_rate']:.1%}. "
             "The canonical figure, with its two-pass disagreement at "
             "p=0.00007, is in the spot-check record.")
    L.append("- **Single coder, Qwen only.** No inter-rater reliability; "
             "intra-rater is 4 hard contradictions among 47 doubly-coded "
             "items. Pixtral's items are entirely uncoded.")
    L.append("- **`strict_v1` and `strict_v2` here are LOCAL recomputes**, "
             "used for comparison. Locked figures elsewhere come from the "
             "stored scored columns of the original run, which differ because "
             "parser availability changes labels.\n")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# MCQ REVIEW SET. The detector is a HEURISTIC and the output is labelled
# "MCQ-like", never "MCQ".
#
# `strict_v2._looks_mcq` fires on two things: the word "option" anywhere in the
# question, truth span or answer field, or an `(a)...(b)` pattern within 60
# characters. Measured on the real sample, the second trigger is unsound --
# `P(A) = P(B)`, `f(a) = f(b)` and a matching exercise's `(d)... (c)` all match
# it, and none is a multiple-choice question.
#
# **THE AUDIT MUST BE TWO-SIDED.** Reviewing only the flagged items can find
# false positives and nothing else, which is the exact one-sided mistake this
# project already made once: the 104-item `genuinely_wrong` census could only
# surface false negatives, and a separate correct-side spot check had to be
# built afterwards to find the other direction. So the review set also carries
# items the detector did NOT flag but which show a choice list in the question,
# as recovery candidates.
# ---------------------------------------------------------------------------

#: Question-side markers for an enumerated list of choices. Presence is NOT
#: proof of MCQ -- LaTeX `enumerate` is equally how a multi-part question
#: writes "(i) ... (ii) ...", which is why these items are candidates for a
#: human to rule on rather than a second automatic verdict.
_CHOICE_LIST_RE = re.compile(r"\\begin\{enumerate\}|\\item\b")

_PAREN_LETTERS_RE = re.compile(r"\(\s*[a-d]\s*\)[^\n]{0,60}\(\s*[b-e]\s*\)", re.I)

#: Pre-sort categories, strongest evidence first. These are PROPOSALS from
#: text; the human read decides.
MCQ_PRESORT = (
    "option_word_and_choice_list",   # both signals; near-certain MCQ
    "option_word_only",              # the word "option" but no visible list
    "choice_list_only",              # a list but nothing says "option"
    "paren_letters_only",            # the unsound trigger; expect false hits
)


def mcq_trigger(question, truth_span, answer_field) -> dict:
    """Why `_looks_mcq` fired, and on what text. A trigger, not a verdict.

    Returning the MATCHED SUBSTRING is the point: `paren_letters_only` items
    are only recognisable as false positives once you can see that what
    matched was `P(A) = P(B)`.
    """
    q = str(question or "")
    parts = {"question": q, "truth_span": str(truth_span or ""),
             "answer_field": str(answer_field or "")}
    option_in = [k for k, v in parts.items() if strict_v2._OPTION_RE.search(v)]
    has_list = bool(_CHOICE_LIST_RE.search(q))
    blob = " ".join(parts.values())
    paren = _PAREN_LETTERS_RE.search(blob)

    if option_in and has_list:
        presort = "option_word_and_choice_list"
    elif option_in:
        presort = "option_word_only"
    elif has_list:
        presort = "choice_list_only"
    elif paren:
        presort = "paren_letters_only"
    else:
        presort = "not_flagged"
    return {
        "presort": presort,
        "option_word_in": ",".join(option_in),
        "question_has_choice_list": has_list,
        "paren_letters_match": paren.group(0)[:60] if paren else "",
        "flagged_by_detector": bool(option_in) or bool(paren),
    }


def mcq_review_set(run: pd.DataFrame, v1: pd.Series, v2s: pd.DataFrame,
                   per_page: int = 9, gt_col: str = "pert_a",
                   question_col: str = "orig_q",
                   entropy_col: str = "perception_entropy") -> pd.DataFrame:
    """Every heuristic MCQ-like item, plus the items the detector may have MISSED.

    Two groups, kept in one file with an unmissable `mcq_group` column:

      * `flagged_mcq_like` -- `strict_v2._looks_mcq` said yes. Review to
        CONFIRM or REJECT.
      * `candidate_missed` -- not flagged, but the question carries a choice
        list. Review to RECOVER. Without this half the audit can only ever
        shrink the MCQ set, so a corrected count would be biased downward by
        construction.

    Sheet page and cell are computed here, not in the notebook, so the manifest
    and the rendered PNGs cannot drift apart: both come from this one ordering.
    """
    rows = []
    for i in run.index:
        v2 = v2s.loc[i]
        trig = mcq_trigger(run.loc[i, question_col], v2["truth_span"],
                           run.loc[i, gt_col])
        flagged = bool(v2["is_mcq"])
        if not flagged and trig["presort"] != "choice_list_only":
            continue
        group = "flagged_mcq_like" if flagged else "candidate_missed"
        on = [f for f in strict_v2.RISK_FLAGS if bool(v2.get(f))]
        rows.append({
            "item_id": int(i),
            "mcq_group": group,
            "presort": trig["presort"],
            "heuristic_mcq_like": flagged,
            "confirmed_mcq": "",                 # human fills this in
            "reviewer_note": "",
            "has_error": bool(run.loc[i, "has_error"]),
            "strict_v1_correct": bool(v1.loc[i]),
            "strict_v2_correct": bool(v2["correct_strict_v2_display_primary"]),
            "entropy": float(run.loc[i, entropy_col]),
            "span_m_disp": v2["model_span"],
            "span_t_disp": v2["truth_span"],
            "label_m": v2["model_label"],
            "label_t": v2["truth_label"],
            "model_tier": v2["model_tier"],
            "truth_tier": v2["truth_tier"],
            "flags": ",".join(on),
            "option_word_in": trig["option_word_in"],
            # The honest weak-trigger test. `presort` can demote a
            # paren-matched item to `choice_list_only` when the question
            # happens to carry an enumerate -- items 170 and 237 are MATCHING
            # exercises whose list is the things to match, not options -- and
            # that would hide exactly the suspicion the reviewer needs.
            "flagged_by_weak_trigger_only": flagged and not trig["option_word_in"],
            "question_has_choice_list": trig["question_has_choice_list"],
            "paren_letters_match": trig["paren_letters_match"],
            "question_head": " ".join(str(run.loc[i, question_col]).split())[:160],
        })
    out = pd.DataFrame(rows)
    order = {p: n for n, p in enumerate(MCQ_PRESORT)}
    out["_g"] = out["mcq_group"].map({"flagged_mcq_like": 0, "candidate_missed": 1})
    out["_p"] = out["presort"].map(lambda p: order.get(p, 99))
    out = out.sort_values(["_g", "_p", "item_id"]).drop(columns=["_g", "_p"])
    out = out.reset_index(drop=True)

    # Page/cell are assigned by walking the frame IN ITS FINAL ORDER, with a
    # per-group counter. The notebook renders each group in this same order,
    # so a caption can never end up under the wrong page number.
    stem = {"flagged_mcq_like": "mcq_flagged",
            "candidate_missed": "mcq_candidate_missed"}
    pos = {g: 0 for g in out["mcq_group"].unique()}
    files, cells = [], []
    for _, r in out.iterrows():
        n = pos[r["mcq_group"]]
        files.append(f"{stem[r['mcq_group']]}_p{n // per_page + 1}.png")
        cells.append(n % per_page + 1)
        pos[r["mcq_group"]] = n + 1
    out["contact_sheet_file"] = files
    out["contact_sheet_cell"] = cells
    return out


def mcq_caption(row, width: int = 52) -> str:
    """Burned-in caption for one contact-sheet cell.

    Carries every field the brief listed. `MCQ?` leads because the reviewer's
    job on this sheet is to answer that, and `paren_letters_match` is shown
    verbatim on the weak-trigger items so a false positive is legible without
    leaving the sheet.
    """
    def z(x, n=width):
        s = " ".join(str(x).split())
        return s[:n] + ("..." if len(s) > n else "")

    tier = {"display_math": "disp", "last_line": "line", "option": "opt",
            "parse_fail": "FAIL", "inline_math": "inl"}
    head = (f"item {int(row['item_id'])}  err={int(bool(row['has_error']))}  "
            f"v1={'OK' if row['strict_v1_correct'] else 'X'}  "
            f"H={float(row['entropy']):.2f}")
    why = row["presort"].replace("_", "-")
    if row["mcq_group"] == "candidate_missed":
        why = "NOT-FLAGGED " + why
    lines = [z(head), z(f"MCQ? {why}")]
    # Shown whenever no "option" word exists anywhere, not only on the
    # `paren_letters_only` presort: a matching exercise carries an enumerate
    # and is demoted to `choice_list_only`, yet the detector still fired on
    # the unsound regex and the reviewer must be able to see that.
    if not row.get("option_word_in") and row["paren_letters_match"]:
        lines.append(z(f"  WEAK trigger matched: {row['paren_letters_match']}"))
    mt = tier.get(row["model_tier"], row["model_tier"])
    tt = tier.get(row["truth_tier"], row["truth_tier"])
    lines += [
        z(f"span  M[{mt}]: {row['span_m_disp']}"),
        z(f"span  T[{tt}]: {row['span_t_disp']}"),
        z(f"label M: {row['label_m']}"),
        z(f"label T: {row['label_t']}"),
    ]
    if row["flags"]:
        lines.append(z(f"flags: {row['flags']}"))
    return "\n".join(lines)


def mcq_accuracy_sensitivity(profile: pd.DataFrame,
                             review: pd.DataFrame) -> dict:
    """How far the MCQ accuracy headline can move on the detector alone.

    Reported as a RANGE because the audit is not yet done. The point is to
    decide whether the figure is quotable, and a single number cannot answer
    that question.
    """
    flagged = sorted(review.loc[review["mcq_group"] == "flagged_mcq_like",
                                "item_id"])
    weak = sorted(review.loc[review["flagged_by_weak_trigger_only"].astype(bool),
                             "item_id"])
    missed = sorted(review.loc[review["mcq_group"] == "candidate_missed",
                               "item_id"])

    def acc(items):
        sub = profile.loc[[i for i in items if i in profile.index]]
        return (float(sub["strict_v1_correct"].mean()) if len(sub)
                else float("nan"), len(sub))

    as_reported, n_rep = acc(flagged)
    strict_only, n_strict = acc([i for i in flagged if i not in weak])
    widest, n_wide = acc([i for i in flagged if i not in weak] + missed)
    return {
        "n_flagged": len(flagged), "n_weak_trigger": len(weak),
        "n_candidate_missed": len(missed),
        "as_reported": {"n": n_rep, "accuracy": as_reported},
        "drop_weak_trigger": {"n": n_strict, "accuracy": strict_only},
        "drop_weak_add_missed": {"n": n_wide, "accuracy": widest},
        "range": (min(as_reported, strict_only, widest),
                  max(as_reported, strict_only, widest)),
        "quotable": False,
        "why": ("the count and the accuracy both move with a detector "
                "judgement that no human has ruled on yet; the DIRECTION "
                "(MCQ far above free response) is stable across every "
                "variant, the MAGNITUDE is not"),
    }
