# READ BEFORE USING `open_judges_all300_diagnostic_20260812.csv`

> **This file is DIAGNOSTIC EVIDENCE from a partially FAILED run. It contains
> no accuracy of any kind, and no accuracy may be derived from it. It is not a
> corrected accuracy, not a scorer, and not a model result.**
>
> **The `omni_*` columns are VOID.** They record a decoder failure, not judge
> behaviour. Do not analyse them, do not count them, do not re-parse them.

Run: `pilot/27_open_judges_all300.ipynb`, 2026-08-12, Colab, all seven cells
completed without error. Both judges were run over the stored Qwen n=300
outputs (`scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv`).
Qwen was not re-run and no scorer rule was touched.

## Column status

| columns | status | why |
|---|---|---|
| `livemath_verdict`, `livemath_raw_output`, `livemath_parse_failed`, `livemath_agrees_*` | **USABLE** | 300/300 decoded cleanly, 0 parse failures |
| `omni_verdict`, `omni_raw_output`, `omni_parse_failed`, `omni_agrees_*`, `omni_invented_*`, `omni_looks_like_solving` | **VOID** | decoder corrupted 280/300 transcripts |
| `judges_agree` | **VOID** | derived from `omni_verdict` |
| `item_id`, `has_error`, `ground_truth_answer`, `qwen_answer`, `strict_v1_correct`, `strict_v2_correct`, `human_label`, `perception_entropy`, `answer_type` | usable | copied from the run and the frozen rules |

## The usable half: LiveMath-Judge, n=300

`jnanliu/LiveMath-Judge`, fidelity prompt, 300/300 decoded, **0 parse
failures**. Its raw output takes exactly two values, `\boxed{yes}` and
`\boxed{no}`.

| | value |
|---|---|
| yes / no | 261 / 39 |
| yes-rate | **87.0%** |
| agreement with `strict_v1` | **154/300 = 51.3%** |
| **always-yes baseline** | **47.0%** |
| judge yes / rule wrong | **133** |
| judge no / rule correct | **13** |

**Read the agreement against the baseline.** A judge answering "yes" to every
item agrees with `strict_v1` on 47.0% by construction, so 51.3% is a 4.3-point
edge over doing nothing. This **confirms at n=300** what notebook 25 found on
40 items: LiveMath-Judge performs at the trivial baseline. The disagreement is
overwhelmingly one-directional, 133 lenient against 13 strict.

Two things it is NOT evidence of:

* **Leniency is not specific to the injected errors.** The yes-rate is
  *higher* on clean items (89.3%) than on `has_error=1` (84.7%), the opposite
  of the leniency signature. It is not forgiving the perturbation; it says yes.
* **`looks_like_solving` scoring 0 is not exoneration.** All 300 transcripts
  are a bare boxed verdict with no reasoning, despite the prompt ending in
  `Analysis:`. With no transcript there is nothing for the heuristic to find.

## The void half: Omni-Judge decoder corruption

`KbsdJames/Omni-Judge` returned `unclear` on 263/300. That is **not** the
judge abstaining and **not** truncation alone. Classifying all 300 raw
transcripts:

| shape | n | share |
|---|---|---|
| marker transposed (`Jugdement`, `Eqiuvalence`, `FALSe`) | 139 | 46.3% |
| **shredded** (`#i#f#i#c#a#t#i#o#n#T#h#e...`) | 89 | 29.7% |
| stopped before the verdict (33 chars, ends `##Equivale`) | 35 | 11.7% |
| no marker, other | 17 | 5.7% |
| **clean** | **20** | **6.7%** |

**93.3% corrupt.** Two consequences decide the verdict on this half:

1. **A successful parse was not evidence of a clean decode.** Of the 37 items
   that produced a verdict, **19 still carried a misspelled marker**. They
   parsed despite corruption, not because there was none. Only **18** items
   are both clean and determinate; a further 2 decoded cleanly yet still
   scored `unclear`.
2. **The corruption reaches the CONTENT, not only the markers.** **All 36** of
   the `omni_invented_reference` flags sit in corrupt transcripts -- not one
   came from a clean decode. Item 0 was flagged for citing `2.4` when the
   model answered `24`, which is the decoder, not the judge inventing a
   reference. The invented-number rate was the single thing this run existed
   to measure, and it measured decoder noise end to end.

**Therefore the Omni half is discarded rather than repaired.** A
transposition-tolerant parser would recover roughly 120 verdicts from
transcripts whose justifications are also damaged: a biased subset (the mildly
corrupted ones) with untrustworthy content, which is worse than no data
because it looks like data. **The parser is deliberately left strict**, and
`pilot/tests/test_judge.py::test_the_parser_is_NOT_loosened_to_recover_corrupted_output`
exists to stop a future session "fixing" it.

**Omni-Judge is not re-run.** Open-judge testing was closed after notebook 26,
and the LiveMath half already answers the n=300 question.

## The guard that should have caught this

`looks_bpe_mangled` tested for one signature, the `Ġ` byte-BPE marker, and
caught **0 of 300**. The notebook printed *"decode clean on all 300"* while
93% of transcripts were damaged. Fixed by `judge.omni_output_health` /
`omni_decode_report` / `assert_omni_decode_ok`, which classify all the shapes
above and fail closed; validated against these 300 real transcripts rather
than against invented fixtures.

**The general lesson, and it is the third decode failure on this model:** a
judge scoring `unclear` is a decoder bug until proven otherwise, and a guard
written for the last corruption will not see the next one. Check the raw text,
and make the check adversarial to the *content*, not only to a known marker.

## Provenance

The notebook also wrote a summary,
`open_judges_all300_diagnostic_summary_20260812.md`, **before** the corruption
was diagnosed. **It is deliberately NOT committed here**, because its Omni
sections read as findings: the judge-vs-judge agreement and kappa (computed on
the 37 items that happened to decode), the has_error split, and the
invented-number rate are all void. Its LiveMath section is correct but is
reproduced above. The file remains on Drive under
`uncertainty-math-vlm/audit/` for provenance; **this STATUS file supersedes
it, and the numbers here are the ones to use.**
