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

#: The two confirmed silent-correction cases. Item 55's page reads
#: `1 + tan x tan y` and the model wrote `1 -`; item 273's reads `2/4` and
#: Pixtral wrote `3/4` on all five samples. A fidelity judge must return
#: `incorrect` on both -- neither pair is mathematically equivalent, so even
#: a pure equivalence judge should reject them. If it does not, it is
#: recalculating despite being told not to.
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
