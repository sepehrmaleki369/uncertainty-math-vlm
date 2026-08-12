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


# ---------------------------------------------------------------------------
# Omni-Judge -- second open-judge candidate, GATE ONLY.
#
# `KbsdJames/Omni-Judge`, Llama-3.1-8B-Instruct instruction-tuned on GPT-4o
# evaluation data. Tried after LiveMath-Judge failed the same gate.
#
# Three differences from LiveMath-Judge that change how it is used:
#
# 1. **It emits a JUSTIFICATION.** Output is structured -- answer, judgement
#    TRUE/FALSE, justification -- where LiveMath-Judge returned a bare
#    `\boxed{yes}`. So `looks_like_solving` can actually work here, and the
#    "does it solve the maths" question becomes answerable.
# 2. **The prompt is BAKED INTO `tokenizer.get_context()`.** There is no
#    supported way to insert a fidelity clause. Given that the clause did
#    nothing for LiveMath-Judge -- 2 of 40 items, both toward LENIENCY -- the
#    native prompt is used as-is and gated. Amending it is not worth
#    fabricating an unsupported call path for.
# 3. **Custom tokenizer methods mean `trust_remote_code=True`.** That is the
#    shape that broke InternVL3 on this project: a custom `trust_remote_code`
#    class hit `AttributeError: 'InternVLChatModel' object has no attribute
#    'all_tied_weights_keys'` under transformers v5. The risk here is lower --
#    custom TOKENIZER, not a custom model class -- but if loading fails that
#    way, it is a known failure mode and not worth debugging in a paid
#    session.
#
# GATE ONLY. Items 55 and 273, then stop. No 300-item run, per the standing
# rule that a failed gate is the result.
# ---------------------------------------------------------------------------

OMNI_MODEL_ID = "KbsdJames/Omni-Judge"
OMNI_MAX_NEW_TOKENS = 300


def parse_omni_judgement(judgement: Optional[str]) -> str:
    """Omni-Judge's TRUE/FALSE field -> our verdict vocabulary.

    Anything unrecognised is `unclear`, i.e. OUR parse failed -- the same
    convention as the LiveMath adapter, so the two are comparable.
    """
    if judgement is None:
        return "unclear"
    s = str(judgement).strip().upper()
    if s.startswith("TRUE"):
        return "correct"
    if s.startswith("FALSE"):
        return "incorrect"
    return "unclear"


def run_gate_with(judge_fn, run: pd.DataFrame,
                  samples_col: str = "all_transcription_samples_raw",
                  gt_col: str = "pert_a", question_col: str = "orig_q",
                  label: str = "judge") -> dict:
    """Gate ANY judge on `GATE_ITEMS`. `judge_fn(question, gold, answer)`
    returns `(verdict, raw)`.

    Exists so a second candidate can be gated without touching the LiveMath
    path or its tests. The bar is identical, which is the point: comparing
    judges across different gates would prove nothing.
    """
    import ast

    verdicts, raws = {}, {}
    for item in GATE_ITEMS:
        answer = majority_answer(
            ast.literal_eval(run.loc[item, samples_col]), run.loc[item, gt_col])
        q = run.loc[item, question_col] if question_col in run else None
        verdict, raw = judge_fn(q, run.loc[item, gt_col], answer)
        verdicts[item] = verdict
        raws[item] = raw
    passed = all(v == "incorrect" for v in verdicts.values())
    return {
        "judge": label, "passed": passed, "verdicts": verdicts, "raw": raws,
        "solving": {i: looks_like_solving(r) for i, r in raws.items()},
        "required": "incorrect on every item in GATE_ITEMS",
        "verdict_label": "gate_passed" if passed else "gated_unsafe_as_scorer",
        "note": ("" if passed else
                 f"{label} accepted an output that does not faithfully match "
                 "the ground truth. Do NOT use it as a scorer, and do not run "
                 "the 300 behind a failed gate -- report the gate as the "
                 "result."),
    }


#: Byte-BPE marker that leaks when a decode goes wrong.
_BPE_SPACE = "Ġ"


def looks_bpe_mangled(text: Optional[str]) -> bool:
    """Did the decode leak raw BPE tokens instead of producing text?

    Signature seen on Omni-Judge's custom `OmniJudgeTokenizer` under
    transformers v5: output arrives as space-separated tokens with the
    byte-BPE space marker intact -- `E quiv ale Ġn ce Jud gment F AL ĠS ĠE`
    instead of `## Equivalence Judgment FALSE`. `parse_response` then finds
    none of its markers and returns all-None, which reads as a judge failure
    when it is a DECODING failure. Always check before believing a verdict.
    """
    return bool(text) and _BPE_SPACE in str(text)


def demangle_bpe(text: Optional[str]) -> str:
    """Best-effort repair of the mangling above: drop the inserted spaces,
    then turn the byte-BPE marker back into a real space.

    A FALLBACK for reading a transcript that would otherwise be lost, never a
    substitute for decoding correctly -- it cannot restore spaces the marker
    did not record, so `Justification` can come back as `Jus tification`.
    Decode with a plain tokenizer instead where possible.
    """
    if not text:
        return ""
    return str(text).replace(" ", "").replace(_BPE_SPACE, " ")


_OMNI_JUDGEMENT_RE = re.compile(r"judg?e?ment\W{0,4}(true|false)", re.I)


def parse_omni_text(text: Optional[str]) -> str:
    """Recover Omni-Judge's verdict from its raw text, whitespace-insensitively.

    Needed because `tokenizer.parse_response()` returns all-None on this
    model's real output. The generated PROSE decodes perfectly, but the
    structural markers come back space-corrupted --
    `# #Equivale nceJudgmentFAL S E# #Jus tification` for
    `## Equivalence Judgment\\nFALSE\\n## Justification` -- so its own parser
    finds none of the anchors it looks for and reports nothing. Read as a
    judge failure, that would be a false negative against the model: the
    verdict is plainly there.

    Stripping ALL whitespace before matching survives the corruption, since
    the damage is entirely misplaced spaces, never reordered characters. Note
    the pattern tolerates a dropped letter in "Equivalence"/"Judgement" --
    item 273 came back as "Equivalece" -- but requires TRUE or FALSE intact.
    """
    if not text:
        return "unclear"
    squashed = re.sub(r"\s+", "", str(text))
    hits = _OMNI_JUDGEMENT_RE.findall(squashed)
    if not hits:
        return "unclear"
    return "correct" if hits[-1].lower() == "true" else "incorrect"


# ---------------------------------------------------------------------------
# BOTH OPEN JUDGES ON ALL 300 -- EXPLORATORY DIAGNOSTIC. NOT A SCORING PATH.
#
# Both judges have already failed a safety check, and the two failures are
# different in kind:
#
#   * **LiveMath-Judge failed the VERDICT gate** -- on item 273 it accepted a
#     non-answer (coin-tossing setup prose) as matching the ground truth.
#   * **Omni-Judge PASSED the verdict gate and failed the RATIONALE check** --
#     on the same item it returned the desired `FALSE`, but its justification
#     showed it had reconstructed its own reference answer (`3/4`, a string
#     present in none of its three inputs) and graded against that. The right
#     verdict from a comparison it never performed.
#
# So no accuracy may be computed here and no function below returns one. What
# an all-300 pass CAN do is turn those two one-item anecdotes into RATES:
# how often the judges disagree with each other, how often each disagrees with
# a frozen rule, and how often Omni's justification cites a number that appears
# in none of its inputs.
#
# **THE TRAP THIS SECTION IS BUILT AROUND.** Agreement between two unreliable
# scorers measures neither of them. Worse, `strict_v1` calls 47% of items
# correct, so a judge that answers "yes" to everything agrees with it on 47%
# by construction -- and notebook 25 already found LiveMath-Judge sitting
# exactly on that trivial baseline. Every agreement figure below is therefore
# reported next to `always_yes_agreement`, and judge-vs-judge agreement next to
# its chance rate and Cohen's kappa. A raw agreement number quoted alone from
# this file is a misuse of it.
# ---------------------------------------------------------------------------

#: Exactly the requested output schema, in the requested order.
OPEN_JUDGE_COLUMNS = (
    "item_id", "has_error", "ground_truth_answer", "qwen_answer",
    "strict_v1_correct", "strict_v2_correct",
    "livemath_verdict", "livemath_raw_output", "livemath_parse_failed",
    "omni_verdict", "omni_raw_output", "omni_parse_failed",
    "livemath_agrees_strict_v1", "livemath_agrees_strict_v2",
    "omni_agrees_strict_v1", "omni_agrees_strict_v2",
    "judges_agree",
)

#: Appended AFTER the requested block, never interleaved, so the schema above
#: stays exactly as specified while the file still carries what the manual
#: rationale pass needs (a human label to check against, and the flags that
#: choose which transcripts are worth reading).
OPEN_JUDGE_EXTRA_COLUMNS = (
    "human_label", "perception_entropy", "answer_type",
    "livemath_looks_like_solving", "omni_looks_like_solving",
    "omni_invented_reference", "omni_invented_tokens",
)

#: What notebooks 24 and 26 recorded on the two probes, under greedy decoding.
#: The all-300 run contains both items, so it re-derives them for free. A
#: mismatch is not a gate failure -- it means the judge, decoder or library
#: version drifted between sessions, and the rest of the file should be read
#: with that in mind.
GATE_REPRODUCTION = {
    "livemath": {55: "incorrect", 273: "correct"},
    "omni": {55: "incorrect", 273: "incorrect"},
}


def _agreement_column(verdict: pd.Series, rule: pd.Series) -> pd.Series:
    """Nullable agreement between a judge verdict and a rule's boolean.

    **`unclear` becomes NA, never False.** Encoding an unreadable verdict as
    disagreement would assert that the judge contradicted the rule, which is a
    different and stronger claim than "we could not read a verdict at all".
    Every count derived from this column is therefore over the determinate
    subset, and its denominator is reported alongside it.
    """
    rule = rule.reindex(verdict.index)
    vals = []
    for i in verdict.index:
        v = verdict.loc[i]
        if v not in ("correct", "incorrect") or pd.isna(rule.loc[i]):
            vals.append(pd.NA)
        else:
            vals.append((v == "correct") == bool(rule.loc[i]))
    return pd.Series(vals, index=verdict.index, dtype="boolean")


def open_judge_frame(run: pd.DataFrame, answers: pd.Series,
                     livemath: pd.DataFrame, omni: pd.DataFrame,
                     v1: pd.Series, v2: pd.Series,
                     human: Optional[pd.Series] = None,
                     answer_types: Optional[pd.Series] = None,
                     gt_col: str = "pert_a", question_col: str = "orig_q",
                     entropy_col: str = "perception_entropy") -> pd.DataFrame:
    """Assemble the per-item diagnostic frame.

    `livemath` and `omni` are indexed by item with columns `verdict` and
    `raw_output`. `answers` is the SINGLE Qwen majority answer both judges were
    shown -- passed in rather than recomputed per judge, because feeding the
    two judges even slightly different inputs would make every comparison in
    this file meaningless. Mismatched coverage raises rather than silently
    inner-joining, since a short run is a resumable checkpoint problem, not a
    result.
    """
    missing = {"livemath": set(run.index) - set(livemath.index),
               "omni": set(run.index) - set(omni.index)}
    if any(missing.values()):
        raise ValueError(
            "both judges must cover every item before the frame is built; "
            f"missing livemath={sorted(missing['livemath'])[:8]} "
            f"omni={sorted(missing['omni'])[:8]} -- resume the checkpoint "
            "rather than analysing a partial run")

    idx = run.index
    out = pd.DataFrame(index=idx)
    out["item_id"] = [int(i) for i in idx]
    out["has_error"] = [bool(run.loc[i, "has_error"]) for i in idx]
    out["ground_truth_answer"] = [str(run.loc[i, gt_col] or "") for i in idx]
    out["qwen_answer"] = answers.reindex(idx).fillna("").astype(str)
    out["strict_v1_correct"] = v1.reindex(idx).astype(bool)
    out["strict_v2_correct"] = v2.reindex(idx).astype(bool)

    for tag, judged in (("livemath", livemath), ("omni", omni)):
        v = judged["verdict"].reindex(idx)
        out[f"{tag}_verdict"] = v
        out[f"{tag}_raw_output"] = judged["raw_output"].reindex(idx).fillna("")
        out[f"{tag}_parse_failed"] = (v == "unclear")

    for tag in ("livemath", "omni"):
        for rule in ("strict_v1", "strict_v2"):
            out[f"{tag}_agrees_{rule}"] = _agreement_column(
                out[f"{tag}_verdict"], out[f"{rule}_correct"])

    # NA when EITHER judge failed to parse: two unreadable outputs are not two
    # judges agreeing, and counting them as agreement would inflate the one
    # number this file exists to report honestly.
    both = out["livemath_verdict"].isin(("correct", "incorrect")) & \
        out["omni_verdict"].isin(("correct", "incorrect"))
    out["judges_agree"] = pd.Series(
        [bool(out.loc[i, "livemath_verdict"] == out.loc[i, "omni_verdict"])
         if both.loc[i] else pd.NA for i in idx],
        index=idx, dtype="boolean")

    out["human_label"] = [str(human.get(i, "")) if human is not None else ""
                          for i in idx]
    out["perception_entropy"] = run[entropy_col].reindex(idx) \
        if entropy_col in run else float("nan")
    out["answer_type"] = [answer_types.get(i, "") if answer_types is not None
                          else "" for i in idx]

    for tag in ("livemath", "omni"):
        out[f"{tag}_looks_like_solving"] = out[f"{tag}_raw_output"].map(
            looks_like_solving)

    q = run[question_col] if question_col in run else pd.Series("", index=idx)
    invented = [invented_reference_tokens(out.loc[i, "omni_raw_output"],
                                          q.get(i, ""),
                                          out.loc[i, "ground_truth_answer"],
                                          out.loc[i, "qwen_answer"])
                for i in idx]
    out["omni_invented_tokens"] = ["; ".join(t) for t in invented]
    out["omni_invented_reference"] = [bool(t) for t in invented]

    return out[list(OPEN_JUDGE_COLUMNS) + list(OPEN_JUDGE_EXTRA_COLUMNS)]


# --- the item-273 signature, as a countable flag ---------------------------

_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def _numeric_surface(*texts) -> str:
    """Normalise text so numbers compare as substrings.

    `\\frac{3}{4}`, `3 / 4` and `3/4` must all reduce to the same surface, or
    the detector below reports a token as invented merely because the judge
    wrote it in prose while the reference wrote it in LaTeX.
    """
    joined = " ".join(str(t or "") for t in texts)
    prev = None
    while prev != joined:                       # nested \frac
        prev = joined
        joined = _FRAC_RE.sub(r"\1/\2", joined)
    joined = re.sub(r"(?<=\d),(?=\d\d\d)", "", joined)      # 13,000 -> 13000
    return re.sub(r"\s*/\s*", "/", joined)


def invented_reference_tokens(rationale: Optional[str],
                              question: Optional[str],
                              gold: Optional[str],
                              answer: Optional[str],
                              min_len: int = 2) -> list:
    """Numbers the judge cites that appear in NONE of the inputs it was given.

    This is the item-273 signature made countable. There, Omni-Judge wrote
    *"The reference answer is 3/4"* while the reference we passed was `2/4`
    and the model answered a bare `}`; `3/4` occurs in none of the question,
    the reference or the answer. It had re-derived a reference from the
    problem text and graded against that.

    **A review flag, never a verdict, and it errs in both directions.** It
    OVER-triggers on incidental numerals (a judge writing "criterion 12" or a
    year), and UNDER-triggers by construction: tokens shorter than `min_len`
    are skipped because a bare `3` appears in almost any input by chance, so
    a one-digit invented answer is invisible to it. The returned tokens are
    kept in the CSV precisely so a human can dismiss a false hit in seconds.
    """
    if not rationale:
        return []
    hay = _numeric_surface(question, gold, answer)
    said = _numeric_surface(rationale)
    seen, out = set(), []
    for m in _NUMERIC_TOKEN_RE.finditer(said):
        tok = m.group(0)
        if len(tok) < min_len or tok in seen:
            continue
        seen.add(tok)
        if tok not in hay:
            out.append(tok)
    return out


# --- diagnostics ------------------------------------------------------------

def open_judge_diagnostics(frame: pd.DataFrame) -> dict:
    """Counts and agreements. **Returns no accuracy, by construction.**

    Every agreement figure ships with the number that makes it readable:
    `always_yes_agreement` for the rule comparisons (a judge answering yes to
    everything scores exactly that), and `chance_rate`/`kappa` for
    judge-vs-judge (two lenient judges agree often without agreeing about
    anything).
    """
    n = len(frame)
    out = {"n": n, "reports_accuracy": False}

    for tag in ("livemath", "omni"):
        v = frame[f"{tag}_verdict"]
        det = int((v == "correct").sum()) + int((v == "incorrect").sum())
        out[tag] = {
            "correct": int((v == "correct").sum()),
            "incorrect": int((v == "incorrect").sum()),
            "parse_failed": int((v == "unclear").sum()),
            "n_determinate": det,
            "yes_rate_determinate": (float((v == "correct").sum()) / det
                                     if det else float("nan")),
        }

    ja = frame["judges_agree"]
    det = ja.notna()
    nd = int(det.sum())
    lv = frame.loc[det, "livemath_verdict"] == "correct"
    ov = frame.loc[det, "omni_verdict"] == "correct"
    p_l, p_o = (float(lv.mean()), float(ov.mean())) if nd else (float("nan"),) * 2
    pe = p_l * p_o + (1 - p_l) * (1 - p_o) if nd else float("nan")
    po = float((ja[det] == True).mean()) if nd else float("nan")  # noqa: E712
    out["judge_vs_judge"] = {
        "n_both_determinate": nd,
        "agree": int((ja == True).sum()),        # noqa: E712
        "disagree": int((ja == False).sum()),    # noqa: E712
        "undecidable_either_parse_failed": int((~det).sum()),
        "rate": po,
        "chance_rate": pe,
        "kappa": ((po - pe) / (1 - pe)) if nd and pe < 1 else float("nan"),
    }

    for tag in ("livemath", "omni"):
        for rule in ("strict_v1", "strict_v2"):
            a = frame[f"{tag}_agrees_{rule}"]
            d = a.notna()
            nd = int(d.sum())
            jc = frame.loc[d, f"{tag}_verdict"] == "correct"
            rc = frame.loc[d, f"{rule}_correct"].astype(bool)
            out[f"{tag}_vs_{rule}"] = {
                "n_determinate": nd,
                "agree": int((a == True).sum()),      # noqa: E712
                "disagree": int((a == False).sum()),  # noqa: E712
                "rate": float((a[d] == True).mean()) if nd else float("nan"),  # noqa: E712
                "judge_yes_rule_wrong": int((jc & ~rc).sum()),
                "judge_no_rule_correct": int((~jc & rc).sum()),
                # A judge answering "correct" to everything agrees with the
                # rule on exactly the rule's own correct items. Quote the rate
                # above without this and the number means nothing.
                "always_yes_agreement": float(rc.mean()) if nd else float("nan"),
            }

    for flag, name in ((True, "has_error"), (False, "clean")):
        sub = frame[frame["has_error"] == flag]
        block = {"n": len(sub)}
        for tag in ("livemath", "omni"):
            v = sub[f"{tag}_verdict"]
            d = int((v != "unclear").sum())
            block[f"{tag}_yes"] = int((v == "correct").sum())
            block[f"{tag}_yes_rate"] = (float((v == "correct").sum()) / d
                                        if d else float("nan"))
        ja_s = sub["judges_agree"]
        block["judges_agree_rate"] = (float((ja_s[ja_s.notna()] == True).mean())  # noqa: E712
                                      if ja_s.notna().any() else float("nan"))
        block["omni_invented_reference"] = int(sub["omni_invented_reference"].sum())
        out[name] = block

    out["omni_invented_reference_total"] = int(frame["omni_invented_reference"].sum())
    out["livemath_solving_total"] = int(frame["livemath_looks_like_solving"].sum())
    out["omni_solving_total"] = int(frame["omni_looks_like_solving"].sum())
    out["livemath_silent_transcripts"] = int(
        frame["livemath_raw_output"].astype(str).str.len().le(24).sum())

    out["gate_reproduction"] = {
        tag: {i: {"expected": want,
                  "observed": (frame.loc[i, f"{tag}_verdict"]
                               if i in frame.index else "absent"),
                  "matches": (i in frame.index
                              and frame.loc[i, f"{tag}_verdict"] == want)}
              for i, want in probes.items()}
        for tag, probes in GATE_REPRODUCTION.items()
    }
    return out


def rationale_sample(frame: pd.DataFrame, n_has_error: int = 8,
                     n_clean: int = 4, seed: int = 20260812,
                     force: Sequence[int] = GATE_ITEMS) -> list:
    """A FIXED small set of transcripts to read by hand, oversampling `has_error=1`.

    Deterministic and priority-ordered rather than uniform, because a uniform
    draw of 12 from 300 would mostly return items where both judges said yes
    and there is nothing to read. Priority: the forced probes, then items
    flagged as citing an invented reference, then judge-vs-judge
    disagreements, then items carrying a determinate human label, then a
    seeded fill. The probes COUNT toward their stratum so the totals are
    exactly as requested.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    picked = []
    for flag, want in ((True, n_has_error), (False, n_clean)):
        pool = frame[frame["has_error"] == flag]
        chosen = [i for i in force if i in pool.index]
        tiers = [
            pool.index[pool["omni_invented_reference"].astype(bool)],
            # `.fillna(False)` throughout: the subsets below mean
            # "demonstrably disagreed", not "did not demonstrably agree". An
            # undecidable pair belongs in neither, and leaving the NA in the
            # mask makes the answer depend on the pandas version.
            pool.index[(pool["judges_agree"] == False).fillna(False)],  # noqa: E712
            pool.index[pool["human_label"].isin(["correct", "wrong"])],
            pool.index,
        ]
        for tier in tiers:
            remaining = [i for i in tier if i not in chosen]
            rng.shuffle(remaining)
            for i in remaining:
                if len(chosen) >= want:
                    break
                chosen.append(i)
            if len(chosen) >= want:
                break
        picked.extend(chosen[:want])
    return sorted(int(i) for i in picked)


def open_judges_summary_md(path: str, frame: pd.DataFrame, diag: dict,
                           rationales: Sequence[int]) -> str:
    """The human-readable summary. Emits NO accuracy figure, by construction.

    Ordering is deliberate and matches the other summary writers in this
    module: a reader who stops after the first paragraph must still know that
    both judges failed a safety check and that nothing here is a corrected
    accuracy.
    """
    L = ["# Open judges on all 300 — EXPLORATORY DIAGNOSTIC (not a scoring run)\n"]
    L.append("> **Both judges had already failed a safety check before this "
             "ran, in different ways.** LiveMath-Judge failed the *verdict* "
             "gate: on item 273 it accepted a non-answer as matching the "
             "ground truth. Omni-Judge *passed* the verdict gate and failed "
             "the *rationale* check: it returned the desired `FALSE` while "
             "its justification showed it had reconstructed its own reference "
             "answer and graded against that.\n")
    L.append("> **Nothing in this file is a corrected accuracy and none may "
             "be derived from it.** No judge verdict is model accuracy; these "
             "are scorer diagnostics on a fixed set of Qwen outputs.\n")
    L.append(f"`{MODEL_ID}` (fidelity prompt) and `{OMNI_MODEL_ID}` (native "
             f"prompt, baked into its tokenizer) · {diag['n']} items · "
             "2026-08-12\n")

    L.append("## Verdict counts\n")
    L.append("| judge | yes | no | parse failed | yes-rate (determinate) |")
    L.append("|---|---|---|---|---|")
    for tag, name in (("livemath", "LiveMath-Judge"), ("omni", "Omni-Judge")):
        d = diag[tag]
        L.append(f"| {name} | {d['correct']} | {d['incorrect']} | "
                 f"{d['parse_failed']} | {d['yes_rate_determinate']:.1%} |")
    L.append("\n`parse failed` means *our* parse found no verdict. Neither "
             "model has an abstain option — LiveMath-Judge's own prompt maps "
             "\"difficult to judge\" onto `no` — so this is never judge "
             "uncertainty.\n")

    jj = diag["judge_vs_judge"]
    L.append("## Judge vs judge\n")
    L.append(f"- both determinate on **{jj['n_both_determinate']}** items "
             f"(undecidable on {jj['undecidable_either_parse_failed']})")
    L.append(f"- agree **{jj['agree']}** · disagree **{jj['disagree']}** · "
             f"rate **{jj['rate']:.1%}**")
    L.append(f"- agreement expected by chance at these two yes-rates: "
             f"**{jj['chance_rate']:.1%}** · Cohen's kappa **{jj['kappa']:.3f}**")
    L.append("\n**Read the kappa, not the rate.** Two lenient judges agree "
             "often without agreeing about anything: if both say yes to most "
             "items, a high raw agreement follows arithmetically. Kappa near "
             "zero means the agreement is entirely that arithmetic.\n")

    L.append("## Against the frozen rules\n")
    L.append("Neither rule is ground truth, and neither judge is. This table "
             "measures how far apart four unreliable scorers sit, not which "
             "one is right.\n")
    L.append("| judge | rule | n | agree | disagree | judge yes / rule wrong | "
             "judge no / rule correct | rate | always-yes baseline |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tag, name in (("livemath", "LiveMath"), ("omni", "Omni")):
        for rule in ("strict_v1", "strict_v2"):
            d = diag[f"{tag}_vs_{rule}"]
            L.append(f"| {name} | `{rule}` | {d['n_determinate']} | "
                     f"{d['agree']} | {d['disagree']} | "
                     f"{d['judge_yes_rule_wrong']} | "
                     f"{d['judge_no_rule_correct']} | {d['rate']:.1%} | "
                     f"**{d['always_yes_agreement']:.1%}** |")
    L.append("\n**The last column is the one that makes the rest readable.** "
             "A judge answering \"yes\" to every item agrees with a rule on "
             "exactly the rule's own correct items, so it scores the "
             "always-yes baseline for free. Notebook 25 already found "
             "LiveMath-Judge sitting exactly on that baseline against human "
             "labels. An agreement rate at or below it is worth nothing.\n")

    L.append("## Split by `has_error`\n")
    L.append("| stratum | n | LiveMath yes | Omni yes | judges agree | "
             "Omni cites an invented number |")
    L.append("|---|---|---|---|---|---|")
    for name, label in (("has_error", "`has_error=1` (injected error)"),
                        ("clean", "clean")):
        d = diag[name]
        L.append(f"| {label} | {d['n']} | {d['livemath_yes']} "
                 f"({d['livemath_yes_rate']:.1%}) | {d['omni_yes']} "
                 f"({d['omni_yes_rate']:.1%}) | {d['judges_agree_rate']:.1%} | "
                 f"{d['omni_invented_reference']} |")
    L.append("\nA faithful transcription of an injected error *should* be "
             "`yes`, so a high rate on `has_error=1` is not by itself wrong. "
             "The leniency signature is the rate being **higher** there than "
             "on clean items: that direction means the judge is forgiving the "
             "injected error rather than scoring fidelity to it.\n")

    L.append("## Does the judge re-derive the answer instead of comparing?\n")
    L.append(f"- Omni-Judge justifications citing a number present in "
             f"**none** of question, reference or model answer: "
             f"**{diag['omni_invented_reference_total']} of {diag['n']}**")
    L.append(f"- `looks_like_solving` heuristic: LiveMath "
             f"{diag['livemath_solving_total']}, Omni "
             f"{diag['omni_solving_total']}\n")
    L.append("The invented-number flag is the item-273 signature made "
             "countable: there, Omni-Judge wrote *\"The reference answer is "
             "3/4\"* when we had passed `2/4` and the model had answered a "
             "bare `}`. It is a **review flag, not a verdict**, and it errs "
             "both ways — it over-triggers on incidental numerals and cannot "
             "see a one-digit invented answer at all, since a bare digit "
             "appears in almost any input by chance.\n")
    if diag["livemath_silent_transcripts"]:
        L.append(f"> **The `looks_like_solving` count for LiveMath-Judge "
                 f"cannot be read as exoneration.** "
                 f"{diag['livemath_silent_transcripts']} of {diag['n']} of its "
                 "transcripts are a bare verdict with no reasoning at all, "
                 "despite the prompt ending in `Analysis:`. With no transcript "
                 "there is nothing for the heuristic to find, so a low count "
                 "is evidence the judge does not explain itself, not evidence "
                 "it compared properly.\n")

    L.append("## Gate items, re-derived\n")
    L.append("Items 55 and 273 sit inside these 300, so the notebook 24 and "
             "26 gate verdicts come back for free. A mismatch is not a gate "
             "failure; it means the judge, decoder or library version drifted "
             "between sessions.\n")
    L.append("| judge | item | expected | observed | matches |")
    L.append("|---|---|---|---|---|")
    for tag, probes in diag["gate_reproduction"].items():
        for i, r in probes.items():
            L.append(f"| {tag} | {i} | `{r['expected']}` | "
                     f"`{r['observed']}` | {'yes' if r['matches'] else '**NO**'} |")

    agree = (frame["judges_agree"] == True).fillna(False)     # noqa: E712
    disagree = (frame["judges_agree"] == False).fillna(False)  # noqa: E712
    dis = frame[disagree]
    L.append(f"\n## Examples: the judges disagree — {len(dis)} items\n")
    for i, r in dis.head(6).iterrows():
        L.append(f"**item {i}** · `has_error={r['has_error']}` · LiveMath "
                 f"`{r['livemath_verdict']}` · Omni `{r['omni_verdict']}` · "
                 f"`strict_v1` {'correct' if r['strict_v1_correct'] else 'wrong'}\n")
        L.append(f"- truth: `{str(r['ground_truth_answer'])[:150]}`")
        L.append(f"- model: `{str(r['qwen_answer'])[:150]}`")
        L.append(f"- Omni says: {str(r['omni_raw_output'])[:300]}\n")

    both_yes_rule_wrong = frame[agree &
                                (frame["livemath_verdict"] == "correct") &
                                (~frame["strict_v1_correct"])]
    both_no_rule_right = frame[agree &
                               (frame["livemath_verdict"] == "incorrect") &
                               (frame["strict_v1_correct"])]
    L.append("## Examples: both judges agree but differ from `strict_v1`\n")
    L.append(f"- both `yes`, rule says wrong: **{len(both_yes_rule_wrong)}** "
             "— the set the judges would *recover*, and the set most likely "
             "to contain their false passes")
    L.append(f"- both `no`, rule says correct: **{len(both_no_rule_right)}** "
             "— candidate false passes of the rule, independently flagged\n")
    for title, sub in (("both yes / rule wrong", both_yes_rule_wrong),
                       ("both no / rule correct", both_no_rule_right)):
        L.append(f"### {title}\n")
        for i, r in sub.head(4).iterrows():
            L.append(f"**item {i}** · `has_error={r['has_error']}` · human "
                     f"`{r['human_label'] or '—'}`\n")
            L.append(f"- truth: `{str(r['ground_truth_answer'])[:150]}`")
            L.append(f"- model: `{str(r['qwen_answer'])[:150]}`\n")

    L.append("## Fixed sample for manual rationale inspection\n")
    L.append(f"Items {list(rationales)} — seeded and priority-ordered, "
             "oversampling `has_error=1`. Only Omni-Judge writes a "
             "justification, so it is the only one there is anything to "
             "read.\n")
    for i in rationales:
        if i not in frame.index:
            continue
        r = frame.loc[i]
        L.append(f"### item {i} · `has_error={r['has_error']}` · human "
                 f"`{r['human_label'] or '—'}`\n")
        L.append(f"- LiveMath `{r['livemath_verdict']}` · Omni "
                 f"`{r['omni_verdict']}` · `strict_v1` "
                 f"{'correct' if r['strict_v1_correct'] else 'wrong'} · "
                 f"`strict_v2` "
                 f"{'correct' if r['strict_v2_correct'] else 'wrong'}")
        if r["omni_invented_tokens"]:
            L.append(f"- **numbers cited by the judge but absent from every "
                     f"input:** `{r['omni_invented_tokens']}`")
        L.append(f"- truth: `{str(r['ground_truth_answer'])[:300]}`")
        L.append(f"- model: `{str(r['qwen_answer'])[:300]}`")
        L.append(f"- Omni justification:\n\n```\n"
                 f"{str(r['omni_raw_output'])[:900]}\n```\n")

    text = "\n".join(L)
    pathlib_write(path, text)
    return text


# ---------------------------------------------------------------------------
# OMNI-JUDGE DECODE HEALTH.
#
# `looks_bpe_mangled` checks for ONE corruption signature -- the `Ġ` byte-BPE
# marker -- and that turned out to be far too narrow. On the 2026-08-12
# all-300 run it reported "decode clean on all 300" while **94% of the
# transcripts were damaged**, in three shapes it could not see:
#
#   * **marker transposition** -- `Jugdement`, `Eqiuvalence`, `Jusitification`,
#     `FALSe`. The verdict is legible to a human and invisible to the parser.
#     120 of 300.
#   * **shredding** -- one `#` between every character:
#     `#i#f#i#c#a#t#i#o#n#T#h#e#s#t#u#d#e#n#t`. 89 of 300.
#   * **stopping before the verdict** -- output ends mid-marker, e.g. 33
#     characters ending `...radians##Equivale`. 35 of 300.
#
# **19 of the 37 items that PARSED also carried a misspelled marker**, so a
# successful parse was not evidence of a clean decode either.
#
# **THE PARSER IS DELIBERATELY NOT LOOSENED.** Making `parse_omni_text`
# tolerate transposition would "recover" 120 verdicts from a decoder that also
# corrupted the justification text those verdicts were supposed to explain --
# a biased subset with untrustworthy content, which is worse than no data
# because it looks like data. The fix belongs in the GUARD: corruption must
# fail loudly, and a corrupt run must be discarded rather than parsed harder.
# ---------------------------------------------------------------------------

#: Correctly spelled section marker, whitespace-insensitive.
_OMNI_MARKER_RE = re.compile(r"equivalencejudg(?:e)?ment", re.I)

#: Density of `#` above which the output is shredded. A healthy transcript has
#: about six (`## Student's Final Answer`, `## Equivalence Judgment`,
#: `## Justification`); a shredded one has one per character.
_SHRED_RATIO = 0.25
_SHRED_MIN_LEN = 40

OMNI_HEALTH_STATUSES = ("clean", "empty", "bpe_markers_leaked", "shredded",
                        "marker_corrupted", "stopped_before_verdict",
                        "no_marker")


def omni_output_health(text: Optional[str]) -> dict:
    """Classify one Omni-Judge transcript. `corrupt=False` only when intact.

    Written against the 300 real transcripts rather than from the model card,
    because every corruption shape here was discovered in the data and none of
    them was anticipated.
    """
    s = str(text or "")
    squashed = re.sub(r"\s+", "", s)
    out = {"status": "clean", "corrupt": False, "n_chars": len(s),
           "hash_ratio": (squashed.count("#") / len(squashed)) if squashed else 0.0}
    if not s.strip():
        out.update(status="empty", corrupt=True)
        return out
    if _BPE_SPACE in s:
        out.update(status="bpe_markers_leaked", corrupt=True)
        return out
    if len(squashed) >= _SHRED_MIN_LEN and out["hash_ratio"] > _SHRED_RATIO:
        out.update(status="shredded", corrupt=True)
        return out
    if _OMNI_MARKER_RE.search(squashed):
        return out
    # No intact marker. Distinguish "mangled" from "never got there", because
    # the two call for different fixes: a decoder fix versus more tokens.
    if re.search(r"true|false", squashed, re.I):
        out.update(status="marker_corrupted", corrupt=True)
    elif len(squashed) < 120:
        out.update(status="stopped_before_verdict", corrupt=True)
    else:
        out.update(status="no_marker", corrupt=True)
    return out


def omni_decode_report(texts: Sequence[Optional[str]]) -> dict:
    """Health across a whole run, for the notebook's pre-flight assertion."""
    healths = [omni_output_health(t) for t in texts]
    counts = {}
    for h in healths:
        counts[h["status"]] = counts.get(h["status"], 0) + 1
    n = len(healths)
    n_bad = sum(1 for h in healths if h["corrupt"])
    return {
        "n": n, "n_corrupt": n_bad,
        "corrupt_rate": n_bad / n if n else float("nan"),
        "counts": counts,
        "usable": n_bad == 0,
        "note": ("Any corruption at all voids the run. Do NOT parse harder: "
                 "a tolerant parser recovers verdicts from transcripts whose "
                 "CONTENT is also damaged, which looks like data and is not."),
    }


def assert_omni_decode_ok(texts: Sequence[Optional[str]],
                          tolerate: float = 0.0) -> dict:
    """Raise unless the decode is clean. Call BEFORE recording any verdict.

    `tolerate` exists so a caller can state a threshold explicitly rather than
    discover one by accident; it defaults to zero because on this model a
    partial corruption has always meant a broken decoder rather than a few
    unlucky items.
    """
    rep = omni_decode_report(texts)
    if rep["corrupt_rate"] > tolerate:
        raise RuntimeError(
            f"Omni-Judge decode is CORRUPT on {rep['n_corrupt']}/{rep['n']} "
            f"transcripts ({rep['corrupt_rate']:.1%}): {rep['counts']}. "
            + rep["note"])
    return rep
