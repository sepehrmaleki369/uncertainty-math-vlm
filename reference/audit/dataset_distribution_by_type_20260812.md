# FERMAT n=300 — distribution and performance by item type

Offline. Built from the stored Qwen2.5-VL-3B run (`scaleup_n300_bal50_...20260802T163202Z.csv`), the two frozen rules recomputed for comparison, and the human audit CSVs. **No model inference was run and no scorer rule was changed.**

> **Read the AUROC column with its status.** A group whose minority class falls below the project's registered minimum of 30 gets no AUROC at all, and a group *defined by* a rule's own verdict gets none either, because every item in it carries the same label. Printing a number in those cells is the degeneracy that produced this project's one retracted result.

## 1. What the 300 items are

- `has_error`: 150 / 150, balanced by design
- question type (INFERRED, `strict_v2._looks_mcq`): 58 multiple-choice, 242 free response
- entropy takes 7 distinct values (all seven reachable at K=5); 38 items at H=0, 61 at the ln5 ceiling
- at least one sample failed to parse on 45 items
- truth span length: median 19 chars (p25 8, p75 68, max 302)

### Answer types, from the GROUND TRUTH span only

| answer type | n | share |
|---|---|---|
| `single_expression` | 145 | 48.3% |
| `mcq_option` | 53 | 17.7% |
| `text_conclusion` | 38 | 12.7% |
| `numeric` | 17 | 5.7% |
| `system_answer` | 17 | 5.7% |
| `set_answer` | 12 | 4.0% |
| `multi_value` | 11 | 3.7% |
| `derivative_equation` | 7 | 2.3% |

**Types are derived from the truth side only, and this is not a detail.** `strict_v2` ORs each shape flag across the model and truth spans, which is right for a review queue and wrong for grouping: a short *wrong model span* would put the item in a group that then looks inaccurate by construction. The two differ on **26 of 300 items**, so the choice moves real mass. The OR'd label is kept in the CSV as `answer_type_either_side`.

**Two requested categories are not assigned, and their absence is a finding rather than an oversight:**

- `table_cell_answer` — no detector exists and nothing in the stored fields distinguishes a table cell from any other short span; never assigned
- `proof_conclusion` — folded into `text_conclusion`; the prose detector cannot separate a proof from any other sentence

## 2. Metrics by group

Every AUROC below scores entropy against **`strict_v1` correctness**, so the column is comparable across all seven splits.

### 2.1 By `has_error`

| group | n | v1 acc | v2 acc | mean H | med H | max-H rate | parse fail | AUROC | 95% CI | AUROC status | n human | extr. issue | false pass | false fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `False` | 150 | 52.7% | 50.0% | 0.942 | 0.950 | 20.0% | 12.7% | 0.813 | [0.744, 0.875] | `excludes_chance` | 115 | 34.8% | 0 | 16 |
| `True` | 150 | 41.3% | 42.0% | 0.985 | 1.055 | 20.7% | 17.3% | 0.817 | [0.749, 0.880] | `excludes_chance` | 119 | 35.3% | 0 | 17 |

### 2.2 By answer type (truth-side)

| group | n | v1 acc | v2 acc | mean H | med H | max-H rate | parse fail | AUROC | 95% CI | AUROC status | n human | extr. issue | false pass | false fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `single_expression` | 145 | 49.7% | 48.3% | 1.043 | 1.055 | 21.4% | 17.9% | 0.783 | [0.708, 0.851] | `excludes_chance` | 112 | 39.3% | 0 | 11 |
| `mcq_option` | 53 | 86.8% | 86.8% | 0.384 | 0.500 | 1.9% | 13.2% | n/a | n/a | `underpowered_minority_7` | 38 | 18.4% | 0 | 0 |
| `text_conclusion` | 38 | 13.2% | 10.5% | 1.315 | 1.332 | 47.4% | 7.9% | n/a | n/a | `underpowered_minority_5` | 21 | 33.3% | 0 | 3 |
| `numeric` | 17 | 29.4% | 29.4% | 1.031 | 0.950 | 5.9% | 11.8% | n/a | n/a | `underpowered_minority_5` | 17 | 41.2% | 0 | 5 |
| `system_answer` | 17 | 5.9% | 5.9% | 1.093 | 0.950 | 23.5% | 11.8% | n/a | n/a | `underpowered_minority_1` | 17 | 47.1% | 0 | 8 |
| `set_answer` | 12 | 50.0% | 50.0% | 0.936 | 0.950 | 8.3% | 8.3% | n/a | n/a | `underpowered_minority_6` | 12 | 16.7% | 0 | 2 |
| `multi_value` | 11 | 36.4% | 36.4% | 1.092 | 1.055 | 18.2% | 18.2% | n/a | n/a | `underpowered_minority_4` | 10 | 40.0% | 0 | 2 |
| `derivative_equation` | 7 | 28.6% | 28.6% | 1.169 | 0.950 | 42.9% | 28.6% | n/a | n/a | `underpowered_minority_2` | 7 | 42.9% | 0 | 2 |

### 2.3 By question type

| group | n | v1 acc | v2 acc | mean H | med H | max-H rate | parse fail | AUROC | 95% CI | AUROC status | n human | extr. issue | false pass | false fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `free_response` | 242 | 38.4% | 37.6% | 1.086 | 1.055 | 24.4% | 15.3% | 0.760 | [0.698, 0.817] | `excludes_chance` | 192 | 37.5% | 0 | 33 |
| `mcq` | 58 | 82.8% | 81.0% | 0.452 | 0.500 | 3.4% | 13.8% | n/a | n/a | `underpowered_minority_10` | 42 | 23.8% | 0 | 0 |

### 2.4 Simple value vs structured / prose

| group | n | v1 acc | v2 acc | mean H | med H | max-H rate | parse fail | AUROC | 95% CI | AUROC status | n human | extr. issue | false pass | false fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `simple_value` | 215 | 57.2% | 56.3% | 0.879 | 0.950 | 15.3% | 16.3% | 0.824 | [0.767, 0.873] | `excludes_chance` | 167 | 34.7% | 0 | 16 |
| `structured_or_prose` | 85 | 21.2% | 20.0% | 1.176 | 1.332 | 32.9% | 11.8% | n/a | n/a | `underpowered_minority_18` | 67 | 35.8% | 0 | 17 |

### 2.5 By truth span length

| group | n | v1 acc | v2 acc | mean H | med H | max-H rate | parse fail | AUROC | 95% CI | AUROC status | n human | extr. issue | false pass | false fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `long_31-100` | 84 | 31.0% | 31.0% | 1.193 | 1.332 | 29.8% | 20.2% | n/a | n/a | `underpowered_minority_26` | 59 | 30.5% | 0 | 13 |
| `short_4-10` | 78 | 75.6% | 78.2% | 0.573 | 0.500 | 6.4% | 14.1% | n/a | n/a | `underpowered_minority_19` | 60 | 28.3% | 0 | 3 |
| `medium_11-30` | 64 | 54.7% | 51.6% | 1.033 | 1.055 | 18.8% | 10.9% | n/a | n/a | `underpowered_minority_29` | 58 | 29.3% | 0 | 7 |
| `very_long_>100` | 42 | 14.3% | 7.1% | 1.259 | 1.332 | 42.9% | 11.9% | n/a | n/a | `underpowered_minority_6` | 26 | 46.2% | 0 | 4 |
| `tiny_<=3` | 32 | 46.9% | 46.9% | 0.787 | 0.950 | 3.1% | 15.6% | n/a | n/a | `underpowered_minority_15` | 31 | 58.1% | 0 | 6 |

### 2.6 By `strict_v1` verdict

| group | n | v1 acc | v2 acc | mean H | med H | max-H rate | parse fail | AUROC | 95% CI | AUROC status | n human | extr. issue | false pass | false fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `False` | 159 | 0.0% | 8.8% | 1.232 | 1.332 | 34.0% | 18.9% | n/a | n/a | `degenerate_split_defined_by_the_label` | 123 | 54.5% | 0 | 33 |
| `True` | 141 | 100.0% | 87.9% | 0.660 | 0.500 | 5.0% | 10.6% | n/a | n/a | `degenerate_split_defined_by_the_label` | 111 | 13.5% | 0 | 0 |

**Both AUROC cells are refused because the split defines the label.** Within a group where every item is correct, or every item is wrong, there is nothing for a ranking to separate. The accuracy columns are likewise 0% and 100% by construction. What is worth reading here is the rest of the row: `strict_v1`-wrong items sit at much higher entropy and a much higher max-entropy rate, which is the headline signal seen from the other side.


### 2.7 By `strict_v2` verdict

| group | n | v1 acc | v2 acc | mean H | med H | max-H rate | parse fail | AUROC | 95% CI | AUROC status | n human | extr. issue | false pass | false fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `False` | 162 | 10.5% | 0.0% | 1.222 | 1.332 | 32.7% | 16.7% | n/a | n/a | `underpowered_minority_17` | 126 | 54.0% | 0 | 20 |
| `True` | 138 | 89.9% | 100.0% | 0.660 | 0.500 | 5.8% | 13.0% | n/a | n/a | `underpowered_minority_14` | 108 | 13.0% | 0 | 13 |

**`underpowered` here is masking something worse, so do not read it as 'a bigger sample would fix it'.** The two rules agree on 269 of 300 items, so grouping by one and scoring entropy against the other is very nearly the degenerate split above. The honest reading is that this table has no AUROC to give.


## 3. Where the scorer and the model do better or worse

**All four comparisons below are descriptive splits of one run. None is a controlled contrast**, and the groups differ in more than the dimension named, so a gap is a place to look rather than an effect that has been isolated.

### 2.8 MCQ after the human audit (supersedes the heuristic row above)

The MCQ detector is a heuristic and was audited on both sides: all 58 flagged items, plus 19 it did NOT flag whose questions carry a choice list. A one-sided review could only have shrunk the set.

| set | n | `strict_v1` acc | mean H |
|---|---|---|---|
| heuristic MCQ-like | 58 | 82.8% | 0.452 |
| **confirmed MCQ** | **55** | **83.6%** | 0.423 |
| free response | 245 | 38.8% | 1.085 |

**The detector was wrong in both directions and the two errors nearly cancelled.** It wrongly included 4 items (`P(A)`/`P(B)` and `f(a)=f(b)` notation read as options) and missed 1. The pre-audit sensitivity range was 78.9-88.5%; the audited value sits 0.8 points from the uncorrected figure. **A detector demonstrably wrong in both directions does not necessarily bias the aggregate it feeds.**

Two audited items are MCQ but not option-pick, and both are SCORING problems rather than model failures: items 170 and 237 are **matching** questions whose answer is a four-way mapping, yet the truth span is a bare `(b)` -- one quarter of the answer. Item 8's page shows **no options at all** while its answer is `option b`, so the letter is not recoverable from the image by any reader.

- **MCQ vs free response is the largest single split.** Multiple-choice items score far higher and sit at much lower entropy: there are only a few reachable answers, so five samples agree easily and a match is cheap. Read it as a property of the answer space, not as the model reading those pages better.
- **`has_error` costs accuracy and leaves the AUROC alone.** The two strata's AUROCs are within noise of each other, which reproduces the locked finding that entropy works equally well on both. The accuracy gap is the known cost of the balanced design.
- **Structured and prose answers are where scoring falls apart.** `text_conclusion` and `system_answer` are the worst groups by a wide margin and carry the highest max-entropy rates. This is the quantitative form of the standing Limitations sentence: the automatic labels are unsafe for long or multi-part answers.
- **Span length is monotone until it inverts at the bottom, and the inversion is the interesting part.** Accuracy falls steadily from short spans to very long ones. Tiny spans (<=3 chars) then score *middling* while carrying the **highest `extraction_issue` rate of any bucket** — their verdicts are the least earned, which is exactly the false-pass mechanism the audit found. **A tiny span's verdict should be read as unreliable in either direction, not as a good score.**

## 4. What the dataset does and does not tell us

| field | available | note |
|---|---|---|
| injected error location | **no** | ABSENT. No column identifies or locates the injected error. The red markup is a separate, approximate signal -- see red_markup_report -- and is not error-location metadata. |
| injected error type/category | **no** | ABSENT. Nothing categorises the perturbation. |
| original clean answer | **no** | ABSENT. Only `pert_a` (the perturbed answer) is stored, so clean-vs-perturbed cannot be diffed. `orig_q` is the QUESTION, not the original answer -- the name invites that misreading. |
| page / image id | **no** | ABSENT from the stored CSV. The loader requests an `image` column, but it holds pixels and is not persisted to the run CSV, so items are addressable only by row index. |
| problem type / subject / topic | **no** | ABSENT. Question type is INFERRED here (MCQ vs free response) from the question and truth text, never read. |
| handwriting style | **yes** | PRESENT, boolean, semantics vaguer than image_quality. |
| image quality | **yes** | PRESENT, boolean, documented in arXiv 2501.07244 as True if good, False if illumination/shadow/blur/contrast issues. |
| has_error | **yes** | PRESENT. The only label describing the perturbation, and it is binary -- whether an error exists, not where or what. |

**Directly answering the four questions asked:**

1. **Does the dataset say where the injected error is? No.** There is no location, span, index or offset field of any kind.
2. **Does it give an error category or type? No.** `has_error` is binary and is the only label describing the perturbation.
3. **Does it give both the clean and the perturbed answer? No.** Only `pert_a` is stored. `orig_q` is the *question*, and its name invites exactly the misreading that it is the original answer. Clean-vs-perturbed cannot be diffed from this run.
4. **Can the red markup stand in as approximate location and type? For location, weakly and only on error items. For type, no. Details below.**

## 5. The red markup, measured

`\textcolor{red}{...}` appears in **258 of 300** ground-truth answers, **885 spans in total**.

| | items with red | share | mean spans | median | max |
|---|---|---|---|---|---|
| `has_error=1` | 149 / 150 | 99.3% | 2.29 | 2 | 10 |
| clean | 109 / 150 | 72.7% | 3.61 | 1 | — |

**The decisive number is the second row.** If red marked the injected error it would be rare on clean pages. Instead **73% of clean items carry red markup**, and they carry *more* of it on average (3.61 spans against 2.29). Reading those spans shows why: on clean items red marks added elaboration and restated steps (one is a full sentence beginning *"Additionally, if we consider the factorial function..."*), and on error items it tends to mark changed values. **So red marks what was MODIFIED when the variant was generated, not what is WRONG.**

### Limitations of using red as error metadata

1. **It is not specific.** It fires on 73% of clean items, so presence of red carries almost no information about whether an error exists.
2. **It is not unique.** Error items carry a median of 2 red spans and up to 10, so it localises to a set of candidates rather than to the error.
3. **It is not necessary.** Item 142 is `has_error=1` with no red markup at all, so absence of red does not mean absence of an error.
4. **It carries no type information.** A red span is a region, not a category; nothing distinguishes a sign flip from a dropped unit.
5. **It is markup on the ground-truth LaTeX, and the model is scored against a rendered page.** Using it as an analysis variable mixes a transcription artifact into a claim about perception, and it has already corrupted labels once here — nested `\textcolor` is extractor bug 1, which damaged 85 ground truths.

**Recommended use: none, as metadata.** It is legitimate as a hypothesis-generating pointer when reading a specific error item by hand, which is how the auto-correction anecdotes were found. It must not be counted, aggregated, or reported as error location or error type.

## 6. Caveats that bind every human-derived column

- **234 of 300 items carry a human label; 148 are determinate.** Those labels come from three TARGETED audit sets, two of which are one-directional by construction. Per-group `extraction_issue` rates are therefore **not population rates**.
- **`false pass` is 0 in every group, and that is a definition, not a clean bill of health.** The audit codes an unearned verdict as `extraction_issue`, which maps to *indeterminate* rather than to *wrong*, so it cannot appear in this column. The unearned-verdict rate is the `extraction_issue` column, and on audited `strict_v1`-correct items it is 13.5%. The canonical figure, with its two-pass disagreement at p=0.00007, is in the spot-check record.
- **Single coder, Qwen only.** No inter-rater reliability; intra-rater is 4 hard contradictions among 47 doubly-coded items. Pixtral's items are entirely uncoded.
- **`strict_v1` and `strict_v2` here are LOCAL recomputes**, used for comparison. Locked figures elsewhere come from the stored scored columns of the original run, which differ because parser availability changes labels.
