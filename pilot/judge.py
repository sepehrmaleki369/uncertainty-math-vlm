"""LiveMath-Judge as a scorer, framed as TRANSCRIPTION FIDELITY.

`jnanliu/LiveMath-Judge` (Qwen2.5-3B-Instruct fine-tune, Apache-2.0, arXiv
2412.13147) judges whether a generated math answer matches a standard answer
and emits `\\boxed{yes}` / `\\boxed{no}`.

Motivated by the 2026-08-12 audit: of the false passes a human found, 14 of 16
were `extract_final_answer` choosing a wrong or partial span and only 2 were
SymPy. A judge reading the whole answer against the whole ground truth skips
the extractor, which is the part the audit indicted.

**THE FRAMING IS THE EXPERIMENT.** FERMAT's ground truth is `pert_a` -- the
page WITH a deliberately injected error on 150 of 300 items. The task is
whether the model faithfully transcribed the page, so a faithful copy of a
wrong answer is CORRECT and a silently repaired one is WRONG. A judge asked
"is this correct?" grades mathematics and inverts those 150 items.

Reading the model's published prompt (fetched, not assumed) changes the risk
assessment in both directions:

* **Criterion 3 already helps us** -- *"You do not need to recalculate the
  problem answers... You only need to judge whether the examinee's answer
  matches the standard answer"*. That is close to fidelity as trained.
* **Criterion 2 is the danger** -- *"some formulas may be expressed
  differently but are equivalent, which is also considered correct"*. This is
  the same acceptance that made Math-Verify unsafe here, where ~half its
  newly accepted items were false positives concentrated on `has_error=1`.

So the gate is the experiment's load-bearing step, not a formality.

**THERE IS NO "UNCLEAR" VERDICT FROM THIS MODEL.** Its prompt ends *"If it is
difficult to judge, also output \\boxed{no}"* -- uncertainty is trained to
map onto `no`. `unclear` here therefore means only that WE could not parse a
verdict out of its output, never that the judge abstained. Reporting it as
judge uncertainty would be wrong.

**TWO BUGS IN THE PUBLISHED USAGE SNIPPET**, both of which would waste a GPU
session and neither of which fails loudly:

1. `apply_chat_template(conversations, return_tensors='pt')` returns a TENSOR,
   then the snippet subscripts it as `inputs['input_ids']`. Needs
   `return_dict=True`.
2. `model.generate(...)` is called with no `max_new_tokens`, so it defaults to
   20. The model writes an analysis BEFORE the boxed verdict, so 20 tokens
   almost never reaches it -- every item would parse as `unclear` and look
   like judge failure rather than truncation. Use `MAX_NEW_TOKENS`.
"""

import re
from typing import Callable, Optional, Sequence

import pandas as pd

from . import rescore

#: Any callable str -> str: a local pipeline, an API client, or a stub.
JudgeBackend = Callable[[str], str]

MODEL_ID = "jnanliu/LiveMath-Judge"
VERDICTS = ("correct", "incorrect", "unclear")

#: The model emits an analysis and only then the verdict. The published
#: snippet omits this and defaults to 20, truncating before the verdict.
MAX_NEW_TOKENS = 512

#: Two probes a fidelity judge must return `incorrect` on. They test DIFFERENT
#: things on the Qwen run, and conflating them was an error worth recording:
#:
#: * **item 55 IS a silent correction here.** The page reads
#:   `1 + tan x tan y`, the majority sample reads `1 -`, so the answer handed
#:   to the judge is the model's textbook-correct repair of the injected
#:   error. Rejecting it tests fidelity directly.
#: * **item 273 is NOT a silent correction on this run.** The `3/4` repair is
#:   PIXTRAL's, unanimous across its five samples. On Qwen the majority label
#:   is the bare `}` and the answer field is coin-tossing setup prose --
#:   "We write H for 'head' and T for 'Tail'..." -- not an answer at all.
#:   Rejecting it therefore tests whether the judge accepts a NON-ANSWER, a
#:   weaker premise but a stricter bar: there is no mathematics to be lenient
#:   about, so accepting it cannot be excused as equivalence.
#:
#: Both must come back `incorrect`. A `yes` on 55 means the judge recalculated;
#: a `yes` on 273 means it accepted an unrelated passage as a match.
GATE_ITEMS = (55, 273)

#: The model's own template, verbatim from its card. Deviating from the
#: format a 3B model was fine-tuned on is the fastest way to degrade it, so
#: this is kept exact and the fidelity variant only APPENDS.
NATIVE_PROMPT = """\
Please act as an expert in grading mathematics exam papers, and judge whether \
the following answers match the standard answers, i.e., whether the examinee \
answered correctly. Here are some evaluation criteria:

1. Some answers may contain multiple parts, such as single-choice questions, \
multiple-choice questions, fill-in-the-blank questions, and problem-solving \
questions. As long as the answer matches the standard answer, it is \
considered correct. For multiple-choice questions and fill-in-the-blank \
questions with multiple blanks, the examinee must answer all corresponding \
options or blanks correctly to be considered correct.
2. Some answers may be expressed in different ways; for example, some answers \
may be mathematical expressions, while others may be textual descriptions. As \
long as the meaning conveyed is consistent, it is considered correct. \
Additionally, some formulas may be expressed differently but are equivalent, \
which is also considered correct.
3. You do not need to recalculate the problem answers, as the standard \
answers are already provided. You only need to judge whether the examinee's \
answer matches the standard answer based on the form of the question and \
whether it is correct.

Please judge whether the following answer matches the standard answer \
according to the above criteria. If they match, output \\boxed{{yes}}, \
otherwise output \\boxed{{no}}. If it is difficult to judge, also output \
\\boxed{{no}}.
Original Question: {question}
Standard Answer: {gold_answer}
Examinee's Answer: {answer}

Analysis:"""

#: The fidelity amendment. Inserted as criterion 4 so it reads as part of the
#: same list the model was trained on rather than as a foreign instruction,
#: and worded to override criterion 2 explicitly -- "equivalent is correct" is
#: precisely what must not apply when the standard answer is deliberately
#: wrong.
FIDELITY_CLAUSE = """\
4. IMPORTANT, and this overrides criterion 2: the standard answer is a \
transcription of a handwritten page and may itself contain mistakes, unusual \
notation, or missing units. Judge only whether the examinee reproduced the \
standard answer faithfully. If the standard answer is mathematically wrong \
and the examinee silently corrected it, that does NOT match: output \
\\boxed{{no}}. Do not solve the problem and do not reward corrected \
mathematics."""

_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(yes|no)\s*\}", re.I)
#: The bare fallback requires the final line to BE the verdict, not merely to
#: contain it. A looser `\b(yes|no)\b` reads "no verdict at all" as a `no`,
#: which made `run_gate` PASS on unparseable output -- a broken judge waved
#: through as a strict one. Caught by
#: `test_gate_fails_closed_on_unparseable_output`.
_BARE_RE = re.compile(r"^\**\s*(yes|no)\s*[.!]?\**$", re.I)


def build_prompt(question: Optional[str], gold_answer: Optional[str],
                 answer: Optional[str], fidelity: bool = True) -> str:
    """The judge prompt. `fidelity=False` gives the model's native template."""
    template = NATIVE_PROMPT
    if fidelity:
        template = template.replace(
            "\nPlease judge whether the following answer",
            "\n" + FIDELITY_CLAUSE + "\n\nPlease judge whether the following answer")
    return template.format(question=(question or "").strip(),
                           gold_answer=(gold_answer or "").strip(),
                           answer=(answer or "").strip())


def parse_verdict(text: Optional[str]) -> str:
    """`\\boxed{yes}` -> correct, `\\boxed{no}` -> incorrect, else unclear.

    Takes the LAST boxed verdict: the model restates the instruction before
    answering, so the first match is often the literal `\\boxed{yes}` from the
    prompt echo, which would score almost everything correct. Falls back to a
    bare trailing yes/no only when no boxed form is present at all.
    """
    if not text:
        return "unclear"
    hits = _BOXED_RE.findall(str(text))
    if hits:
        return "correct" if hits[-1].lower() == "yes" else "incorrect"
    tail = str(text).strip().splitlines()[-1].strip() if str(text).strip() else ""
    m = _BARE_RE.match(tail)
    if m:
        return "correct" if m.group(1).lower() == "yes" else "incorrect"
    return "unclear"


def majority_answer(raw_samples: Sequence[str],
                    ground_truth: Optional[str]) -> str:
    """The ANSWER FIELD of the sample that produced the majority label.

    The same generation `strict_v1`/`strict_v2` score, so the three are
    like-for-like -- but the raw answer field, not the extracted span, since
    bypassing `extract_final_answer` is the entire point.
    """
    trace = rescore.trace_item(raw_samples, ground_truth, "strict_v1")
    maj = trace["majority_label"]
    rep = next((s for s in trace["samples"] if s["label"] == maj),
               trace["samples"][0])
    return rep["answer_field"] or ""


def judge_item(backend: JudgeBackend, raw_samples: Sequence[str],
               ground_truth: Optional[str], question: Optional[str] = None,
               fidelity: bool = True) -> dict:
    answer = majority_answer(raw_samples, ground_truth)
    prompt = build_prompt(question, ground_truth, answer, fidelity=fidelity)
    raw = backend(prompt)
    verdict = parse_verdict(raw)
    return {
        "question": (question or "").strip(),
        "ground_truth_answer": (ground_truth or "").strip(),
        "model_answer": answer,
        "livemath_raw_output": raw,
        "livemath_label": {"correct": "yes", "incorrect": "no"}.get(verdict, ""),
        "parse_failed": verdict == "unclear",
        "verdict": verdict,
        "fidelity_prompt": fidelity,
    }


def run_gate(backend: JudgeBackend, run: pd.DataFrame, fidelity: bool = True,
             samples_col: str = "all_transcription_samples_raw",
             gt_col: str = "pert_a", question_col: str = "orig_q") -> dict:
    """Pre-flight on the confirmed silent-correction items. RUN THIS FIRST.

    A green gate does not prove the judge is good; a red one proves it is
    grading mathematics rather than fidelity, which makes the full run
    uninterpretable. Cheap, and this project has twice paid for a pre-flight
    that only printed instead of raising (notebooks 19 and 20).
    """
    import ast

    results, raws = {}, {}
    for item in GATE_ITEMS:
        r = judge_item(backend, ast.literal_eval(run.loc[item, samples_col]),
                       run.loc[item, gt_col],
                       run.loc[item, question_col] if question_col in run else None,
                       fidelity=fidelity)
        results[item] = r["verdict"]
        raws[item] = r["livemath_raw_output"]
    passed = all(v == "incorrect" for v in results.values())
    return {
        "passed": passed, "verdicts": results, "raw": raws,
        "fidelity_prompt": fidelity,
        "required": "incorrect on every item in GATE_ITEMS",
        "verdict_label": "gate_passed" if passed else "gated_judges_mathematics",
        "note": ("" if passed else
                 "The judge accepted a silently corrected injected error, so "
                 "it is grading mathematics rather than transcription "
                 "fidelity. Do NOT quote accuracy from a run behind a failed "
                 "gate; report the gate itself as the result."),
    }


def judge_run(backend: JudgeBackend, run: pd.DataFrame, fidelity: bool = True,
              samples_col: str = "all_transcription_samples_raw",
              gt_col: str = "pert_a", question_col: str = "orig_q",
              progress: bool = False) -> pd.DataFrame:
    """Judge every item. The input frame is never mutated."""
    import ast

    it = run.index
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="LiveMath-Judge")
        except ImportError:
            pass
    rows = []
    for i in it:
        r = judge_item(backend, ast.literal_eval(run.loc[i, samples_col]),
                       run.loc[i, gt_col],
                       run.loc[i, question_col] if question_col in run else None,
                       fidelity=fidelity)
        r["item"] = int(i)
        r["has_error"] = bool(run.loc[i, "has_error"])
        rows.append(r)
    return pd.DataFrame(rows).set_index("item")


def judge_diagnostics(judged: pd.DataFrame, v1: pd.Series, v2: pd.Series,
                      human: Optional[pd.Series] = None) -> dict:
    """Counts, both accuracy conventions, rule disagreement, per-class human agreement.

    `human` maps item -> "correct"/"wrong" for the determinate audited items.
    The per-class split is mandatory, not optional: that set is 128 correct
    against 20 wrong, so a judge answering "correct" every time scores 86.5%
    overall while getting 0% of the class that matters. The overall figure is
    returned under a name that says so.
    """
    v = judged["verdict"]
    n = len(judged)
    counts = {k: int((v == k).sum()) for k in VERDICTS}
    decided = counts["correct"] + counts["incorrect"]
    out = {
        "n": n, "counts": counts,
        "unclear_rate": counts["unclear"] / n if n else float("nan"),
        "unclear_means": "our parse failed; the model has no abstain verdict",
        "accuracy_excluding_unclear": counts["correct"] / decided if decided else float("nan"),
        "accuracy_unclear_as_wrong": counts["correct"] / n if n else float("nan"),
    }
    jc = v == "correct"
    for tag, rule in (("strict_v1", v1), ("strict_v2", v2)):
        r = rule.reindex(judged.index).astype(bool)
        out[f"disagree_{tag}"] = int((jc != r).sum())
        out[f"judge_correct_rule_wrong_{tag}"] = int((jc & ~r).sum())
        out[f"judge_wrong_rule_correct_{tag}"] = int((~jc & r).sum())
    for flag, name in ((True, "has_error"), (False, "clean")):
        sub = judged[judged["has_error"] == flag]
        out[f"{name}_n"] = len(sub)
        out[f"{name}_correct_rate"] = float(
            (sub["verdict"] == "correct").mean()) if len(sub) else float("nan")
    if human is not None:
        h = human.dropna()
        shared = judged.index.intersection(h.index)
        sub, hs = judged.loc[shared], h.loc[shared]
        per_class = {}
        for cls, want in (("correct", "correct"), ("wrong", "incorrect")):
            m = hs == cls
            per_class[cls] = {
                "n": int(m.sum()),
                "judge_agrees": int((sub.loc[m, "verdict"] == want).sum()),
                "rate": float((sub.loc[m, "verdict"] == want).mean()) if m.any() else float("nan"),
            }
        out["human_n"] = len(shared)
        out["human_per_class"] = per_class
        out["human_false_pass"] = int(((hs == "wrong") & (sub["verdict"] == "correct")).sum())
        out["human_false_fail"] = int(((hs == "correct") & (sub["verdict"] == "incorrect")).sum())
        out["human_overall_agreement_DO_NOT_QUOTE_ALONE"] = float(
            (sub["verdict"].map({"correct": "correct", "incorrect": "wrong"}) == hs).mean())
    return out



def write_summary_md(path: str, diag: dict, judged: pd.DataFrame,
                     gate: dict, native_gate: Optional[dict] = None,
                     examples: Optional[pd.DataFrame] = None) -> str:
    """The human-readable summary. Leads with the gate, never with accuracy.

    Ordering is deliberate: a reader who stops after the first section must
    still learn whether the run is usable. Accuracy behind a failed gate is
    not a result, and putting it first would invite exactly that quote.
    """
    L = []
    ok = gate["passed"]
    L.append("# LiveMath-Judge on all 300 — scorer diagnostic\n")
    L.append(f"Model: `{MODEL_ID}` · prompt: fidelity-amended · "
             f"date: 2026-08-12\n")
    L.append("## Gate (run before anything else)\n")
    L.append(f"**{'PASSED' if ok else 'FAILED — ' + gate['verdict_label']}**. "
             f"Items {list(GATE_ITEMS)} are the two confirmed cases where the "
             "model silently repaired FERMAT's injected error while "
             "transcribing. A fidelity judge must return `no` on both.\n")
    L.append("| item | fidelity prompt | native prompt |")
    L.append("|---|---|---|")
    for i in GATE_ITEMS:
        nat = native_gate["verdicts"].get(i, "—") if native_gate else "not run"
        L.append(f"| {i} | `{gate['verdicts'][i]}` | `{nat}` |")
    if native_gate:
        L.append(f"\nThe native prompt is recorded for comparison only and did "
                 f"**not** control the all-300 run "
                 f"(native gate would have {'passed' if native_gate['passed'] else 'failed'}).\n")
    if not ok:
        L.append(f"\n> {gate['note']}\n")
        L.append("**No accuracy is reported below, because the run is gated.**\n")
        pathlib_write(path, "\n".join(L))
        return "\n".join(L)

    c = diag["counts"]
    L.append("\n## Verdict counts\n")
    L.append(f"- yes (correct): **{c['correct']}**")
    L.append(f"- no (incorrect): **{c['incorrect']}**")
    L.append(f"- parse failed: **{c['unclear']}** ({diag['unclear_rate']:.1%})")
    L.append(f"\n`parse_failed` means *our* parse found no verdict. "
             "The model has no abstain option — its own prompt maps "
             "\"difficult to judge\" onto `no` — so this is never judge "
             "uncertainty.\n")
    L.append("## Accuracy (both conventions)\n")
    L.append(f"- excluding parse failures: **{diag['accuracy_excluding_unclear']:.1%}**")
    L.append(f"- parse failures as wrong: **{diag['accuracy_unclear_as_wrong']:.1%}**")
    L.append(f"\nFor reference the frozen rules give `strict_v1` 47.0% and "
             "`strict_v2_display_primary` 46.0% on the same 300.\n")
    L.append("## Disagreement with the frozen rules\n")
    L.append("| rule | disagreements | judge yes / rule wrong | judge no / rule correct |")
    L.append("|---|---|---|---|")
    for tag in ("strict_v1", "strict_v2"):
        L.append(f"| `{tag}` | {diag[f'disagree_{tag}']} | "
                 f"{diag[f'judge_correct_rule_wrong_{tag}']} | "
                 f"{diag[f'judge_wrong_rule_correct_{tag}']} |")
    L.append("\n## has_error split\n")
    L.append(f"- `has_error=1` (injected error on the page): "
             f"{diag['has_error_n']} items, judged correct "
             f"{diag['has_error_correct_rate']:.1%}")
    L.append(f"- clean: {diag['clean_n']} items, judged correct "
             f"{diag['clean_correct_rate']:.1%}")
    L.append("\nA large gap here is the leniency signature: it means the "
             "judge is forgiving the injected error rather than scoring "
             "fidelity to it.\n")
    if "human_per_class" in diag:
        pc = diag["human_per_class"]
        L.append("## Against determinate human labels\n")
        L.append(f"{diag['human_n']} audited items carry a determinate human "
                 "verdict.\n")
        L.append("| human says | n | judge agrees | rate |")
        L.append("|---|---|---|---|")
        for k in ("correct", "wrong"):
            L.append(f"| {k} | {pc[k]['n']} | {pc[k]['judge_agrees']} | "
                     f"{pc[k]['rate']:.1%} |")
        L.append(f"\n- false passes (human wrong, judge yes): "
                 f"**{diag['human_false_pass']}**")
        L.append(f"- false fails (human correct, judge no): "
                 f"**{diag['human_false_fail']}**")
        L.append(f"\n**Do not quote the overall agreement "
                 f"({diag['human_overall_agreement_DO_NOT_QUOTE_ALONE']:.1%}) "
                 f"on its own.** The set is {pc['correct']['n']} correct "
                 f"against {pc['wrong']['n']} wrong, so a judge answering "
                 "\"yes\" every time scores "
                 f"{pc['correct']['n'] / max(1, pc['correct']['n'] + pc['wrong']['n']):.1%} "
                 "while getting none of the class that matters.\n")
    if examples is not None and len(examples):
        L.append("## Example disagreements\n")
        for _, r in examples.iterrows():
            L.append(f"**item {int(r['item'])}** — judge `{r['livemath_label']}`, "
                     f"strict_v1 `{'correct' if r['strict_v1'] else 'wrong'}`, "
                     f"strict_v2 `{'correct' if r['strict_v2'] else 'wrong'}`\n")
            L.append(f"- truth: `{str(r['ground_truth_answer'])[:160]}`")
            L.append(f"- model: `{str(r['model_answer'])[:160]}`\n")
    text = "\n".join(L)
    pathlib_write(path, text)
    return text


def pathlib_write(path: str, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# EXPLORATORY DIAGNOSTIC. Not a scoring path.
#
# Everything below runs the judge AFTER its gate has failed, to characterise
# HOW it fails. Nothing here may produce a headline accuracy, and no function
# returns one -- `diagnostic_summary_md` reports verdict agreement and
# leniency signatures only. The gate in `run_gate` is deliberately not
# consulted here, because the point is to study a judge already known unsafe.
# ---------------------------------------------------------------------------

#: Answer shapes, in priority order. Derived from `strict_v2`'s risk flags so
#: the sample spans the kinds of answer the extractor handles differently --
#: a judge that only ever sees short numerics tells you nothing about sets or
#: prose conclusions.
ANSWER_TYPE_ORDER = ("mcq", "derivative", "set", "system", "multi_value",
                     "text_conclusion", "numeric_or_algebra")


def answer_type(row) -> str:
    """One shape label per item, from the `strict_v2` flags."""
    if bool(row.get("mcq_option")) or bool(row.get("tiny_valid_mcq")):
        return "mcq"
    if bool(row.get("derivative_equation")):
        return "derivative"
    if bool(row.get("set_answer")):
        return "set"
    if bool(row.get("system_answer")):
        return "system"
    if bool(row.get("multi_value_answer")):
        return "multi_value"
    if bool(row.get("text_conclusion")):
        return "text_conclusion"
    return "numeric_or_algebra"


def diagnostic_sample(run: pd.DataFrame, v2_scored: pd.DataFrame,
                      human: Optional[pd.Series] = None,
                      n_per_stratum: int = 20, seed: int = 20260812,
                      force: Sequence[int] = GATE_ITEMS) -> pd.DataFrame:
    """A stratified 40: 20 `has_error=1`, 20 clean, spread over answer shapes.

    `force` items are included first and COUNT toward their stratum, so the
    totals stay exactly `n_per_stratum` -- 55 and 273 are both `has_error=1`,
    and silently adding them on top would make that stratum 22 while the code
    claimed 20.

    Within a stratum the shapes are filled round-robin so no single type
    dominates, and inside each shape human-labelled items are preferred, since
    those are the only ones the judge can be scored against. Selection is
    seeded, never hand-picked.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    types = v2_scored.apply(answer_type, axis=1)
    labelled = set(human.dropna().index) if human is not None else set()
    picked = []

    for flag in (True, False):
        stratum = [i for i in run.index if bool(run.loc[i, "has_error"]) == flag]
        chosen = [i for i in force if i in stratum]
        by_type = {}
        for i in stratum:
            if i in chosen:
                continue
            by_type.setdefault(types.get(i, "numeric_or_algebra"), []).append(i)
        # Human-labelled first inside each shape, then seeded shuffle.
        for t, items in by_type.items():
            lab = [i for i in items if i in labelled]
            unlab = [i for i in items if i not in labelled]
            rng.shuffle(lab)
            rng.shuffle(unlab)
            by_type[t] = lab + unlab
        order = [t for t in ANSWER_TYPE_ORDER if by_type.get(t)]
        while len(chosen) < n_per_stratum and order:
            for t in list(order):
                if len(chosen) >= n_per_stratum:
                    break
                if by_type[t]:
                    chosen.append(by_type[t].pop(0))
                else:
                    order.remove(t)
        picked.extend(chosen[:n_per_stratum])

    out = pd.DataFrame({"item": sorted(picked)}).set_index("item")
    out["has_error"] = [bool(run.loc[i, "has_error"]) for i in out.index]
    out["answer_type"] = [types.get(i, "numeric_or_algebra") for i in out.index]
    out["human_label"] = [human.get(i, "") if human is not None else ""
                          for i in out.index]
    out["forced"] = [i in force for i in out.index]
    return out


#: Phrases that indicate the judge RECALCULATED rather than compared. Its own
#: criterion 3 forbids this, so a hit is direct evidence of the failure mode
#: the gate caught on item 273.
SOLVING_RE = re.compile(
    r"\b(should be|actually|the correct answer|correctly comput|recomput|"
    r"let(?:'s| us) (?:calculate|compute|solve)|we (?:calculate|compute|solve|get)|"
    r"solving (?:this|the)|therefore the answer is|the right answer|"
    r"is wrong|is incorrect because|standard answer (?:is|appears) (?:wrong|incorrect))\b",
    re.I)


def looks_like_solving(text: Optional[str]) -> bool:
    """Heuristic: did the judge work the problem instead of comparing?

    A flag for human review, never a verdict -- it over-triggers on judges
    that merely narrate. Its value is pulling the handful of transcripts worth
    reading out of 80.
    """
    return bool(text) and bool(SOLVING_RE.search(str(text)))


def diagnostic_summary_md(path: str, both: pd.DataFrame,
                          sample: pd.DataFrame) -> str:
    """Summary for the exploratory run. Reports NO accuracy, by construction.

    `both` carries one row per item with `verdict_native`, `verdict_fidelity`,
    the two raw outputs and the solving flags. The header states the gate
    failure first so the file cannot be read as a scoring result.
    """
    L = ["# LiveMath-Judge — EXPLORATORY DIAGNOSTIC (not a scoring run)\n"]
    L.append("> **The gate FAILED before this ran.** On item 273 the judge "
             "accepted a non-faithful / non-useful model output as matching "
             "the ground-truth answer `2/4`. (NOT 'the model corrected 2/4 to "
             "3/4' -- that is a Pixtral fact; on the Qwen run this item's "
             "majority sample is coin-tossing setup prose.) This file "
             "characterises HOW the judge fails. **No accuracy is reported "
             "and none may be derived from it.**\n")
    L.append(f"Model: `{MODEL_ID}` · {len(both)} items · both prompts · "
             "2026-08-12\n")

    L.append("## Sample composition\n")
    L.append(f"- `has_error=1`: {int(sample['has_error'].sum())} · "
             f"clean: {int((~sample['has_error']).sum())}")
    L.append(f"- carrying a determinate human label: "
             f"{int((sample['human_label'] != '').sum())}")
    L.append(f"- forced probes: {sorted(sample.index[sample['forced']].tolist())}\n")
    L.append("| answer type | n |")
    L.append("|---|---|")
    for t, c in sample["answer_type"].value_counts().items():
        L.append(f"| {t} | {c} |")

    nat, fid = both["verdict_native"], both["verdict_fidelity"]
    diff = nat != fid
    L.append("\n## Native vs fidelity prompt\n")
    L.append(f"The two prompts disagree on **{int(diff.sum())} of {len(both)}** items.\n")
    L.append("| prompt | yes | no | parse failed |")
    L.append("|---|---|---|---|")
    for tag, v in (("native", nat), ("fidelity", fid)):
        L.append(f"| {tag} | {int((v == 'correct').sum())} | "
                 f"{int((v == 'incorrect').sum())} | {int((v == 'unclear').sum())} |")
    L.append(f"\nFidelity stricter on {int((nat == 'correct').values.__and__((fid == 'incorrect').values).sum())} "
             f"items, more lenient on "
             f"{int((nat == 'incorrect').values.__and__((fid == 'correct').values).sum())}.\n")

    L.append("## Against determinate human labels\n")
    lab = both[both["human_label"].isin(["correct", "wrong"])]
    if len(lab):
        L.append(f"{len(lab)} of the {len(both)} carry one.\n")
        L.append("| human | n | native agrees | fidelity agrees |")
        L.append("|---|---|---|---|")
        for cls, want in (("correct", "correct"), ("wrong", "incorrect")):
            m = lab["human_label"] == cls
            if not m.any():
                continue
            L.append(f"| {cls} | {int(m.sum())} | "
                     f"{int((lab.loc[m, 'verdict_native'] == want).sum())} | "
                     f"{int((lab.loc[m, 'verdict_fidelity'] == want).sum())} |")
        L.append("\n**Per class only.** The audited pool is ~6:1 correct to "
                 "wrong, so a judge answering yes always looks accurate "
                 "overall while getting none of the class that matters.\n")
    else:
        L.append("None in this draw.\n")

    L.append("## has_error=1 behaviour\n")
    he = both[both["has_error"]]
    L.append(f"On the {len(he)} items carrying an injected error, the fidelity "
             f"prompt says yes on **{int((he['verdict_fidelity'] == 'correct').sum())}**.\n")
    L.append("A faithful transcription of an injected error SHOULD be yes, so "
             "a high rate is not by itself wrong. The signal to read is the "
             "solving flag below: accepting because the maths was repaired is "
             "the failure, accepting because the text matches is not.\n")

    solving = both[both["solving_fidelity"] | both["solving_native"]]
    silent = int(both["raw_fidelity"].astype(str).str.len().le(24).sum())
    L.append(f"## Judge appears to solve rather than compare — "
             f"{len(solving)} of {len(both)}\n")
    L.append("Heuristic flag over the judge's own transcript, for review "
             "rather than as a verdict. Its criterion 3 forbids "
             "recalculating.\n")
    if silent:
        L.append(f"> **READ THIS BEFORE TRUSTING THE COUNT ABOVE.** "
                 f"{silent} of {len(both)} transcripts are a bare verdict with "
                 "no reasoning at all -- the model answers `\\boxed{yes}` or "
                 "`\\boxed{no}` directly despite the prompt ending in "
                 "`Analysis:`. **With no transcript to scan there is nothing "
                 "for the heuristic to find, so a count of 0 is NOT evidence "
                 "the judge did not recalculate** -- it is evidence the judge "
                 "does not explain itself. This diagnostic cannot answer the "
                 "'does it solve the maths' question against this model.**\n")
    for i, r in solving.head(10).iterrows():
        L.append(f"**item {i}** · `has_error={r['has_error']}` · "
                 f"native `{r['verdict_native']}` · fidelity "
                 f"`{r['verdict_fidelity']}`\n")
        L.append(f"- truth: `{str(r['ground_truth_answer'])[:130]}`")
        L.append(f"- model: `{str(r['model_answer'])[:130]}`")
        L.append(f"- judge: {str(r['raw_fidelity'])[:260]}\n")

    pf = both[(nat == "unclear") | (fid == "unclear")]
    L.append(f"## Parse failures — {len(pf)}\n")
    L.append("`unclear` means OUR parse found no verdict. The model has no "
             "abstain option, so this is never judge uncertainty.\n")
    for i, r in pf.head(5).iterrows():
        L.append(f"- item {i}: native `{r['verdict_native']}`, fidelity "
                 f"`{r['verdict_fidelity']}` — {str(r['raw_fidelity'])[:150]}")

    text = "\n".join(L)
    pathlib_write(path, text)
    return text
