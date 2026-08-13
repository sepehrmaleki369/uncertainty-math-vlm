# uncertainty-math-vlm

Pilot testing whether entropy over a VLM's own repeated samples predicts when
it's wrong, on FERMAT (handwritten math) with Qwen2.5-VL-3B. Two arms:
**perception** (transcription instability) and **reasoning** (grading
instability). See `README.md` for setup/layout basics — this file is
working context for future sessions, not a repeat of that.

## OPEN — what is NOT done (2026-08-11)

Scored against the seven things a reviewer would ask for, in the user's own
priority order. **Items 3–6 are done and locked. Items 1 and 2 are not, and
item 1 is the highest-value remaining work in the project.**

### 1. Manual visual audit — ONE STRATUM READ (2026-08-11). Still the top priority.

*"Show 30–50 real examples: model output, extracted answer, ground truth,
entropy, correct/wrong."*

- **The `only_qwen` stratum is now human-read and coded — 30 of its 31
  items, by the user, from the images.** That is the first genuine human
  pass in this project and it satisfies the *"30–50 real examples"* ask on
  its own terms. See the dated entry below.
- **`wrong_on_both` is COMPLETE — all 73 read** (2026-08-11, same human
  pass), coded in `reference/audit/coded_73_wrong_on_both_20260811.csv`.
  Together that is **all 104 Qwen `genuinely_wrong` items (100%)**,
  so this is effectively a census of that bucket rather than a sample. The
  binding caveats are now single-coder reliability and Qwen-only scope, not
  sampling — see the dated entry.
- **Qwen's `genuinely_wrong` bucket is CLOSED at 104/104. Still unread: all
  57 `only_pixtral` items.**
  `only_pixtral` is the untouched stratum and is the natural next pass, since
  every conclusion so far is Qwen-side and Pixtral's 130 `genuinely_wrong`
  items have never been read.

- **Built:** notebook 17 §6 writes **17 contact-sheet PNGs** to
  `Drive/uncertainty-math-vlm/figures/genuinely_wrong/` — `unanimous_wrong_*`
  (9 items), `wrong_on_both` (73), `only_qwen` (31), `only_pixtral` (57).
  Caption per cell: item index, entropy, which model(s) failed, model label,
  truth label. `show_item` + `rescore.format_trace` give the full per-item
  view including **raw model output** and the column where the two labels
  first diverge.
- **NOT done: nobody has read them and coded them.** That is a human task and
  it is what converts "the tooling exists" into evidence.
- **Partial exception:** the 9 unanimous-and-wrong items WERE read
  (2026-08-11, via the Drive OCR of the sheets) and ~6 of 8 distinct items
  turned out to be scoring failures, not misreads. That result is recorded
  above and is the reason this audit matters — the unread 73 may contain
  more of the same.
- **How to read them without Colab:** `mcp__claude_ai_Google_Drive__search_files`
  returns an OCR `contentSnippet` per sheet carrying the transcribed
  handwriting and every caption. Do NOT download the PNGs (0.5–1.2 MB each →
  ~180K+ tokens).

### 2. Failure-category counts — PRE-SORTER BUILT, human pass outstanding.

*"Categorize: bad handwriting / notation misread / copied wrong line /
extraction issue / hallucination."*

- `classify_scoring_outcome` categorizes every item by **why it scored**
  (`correct_robust` … `genuinely_wrong`). **Within** `genuinely_wrong` (Qwen
  104, Pixtral 130) there is **no breakdown at all** — `CATEGORIES` has seven
  entries and none of them subdivides it.
- This is what turns "it works" into "we understand why it fails", and it is
  cheap: several categories are text-detectable offline —
  **copied-wrong-line** (the model's answer matches a *different* line of
  `pert_a`), **extraction issue** (already separable via the tier/label
  machinery), **hallucination** (no token overlap with `pert_a`). Only
  **bad handwriting** and **notation misread** need the image.
- Suggested shape: a first-pass classifier proposing a label from text
  signals, plus a coding sheet, so the human pass is confirm-or-correct on a
  pre-sorted list rather than 104 items from scratch.

### 3–6. Done and locked

| ask | status |
|---|---|
| verbalized confidence baseline | notebook 20; `test_verbalized_confidence.py` (skips until the CSV is downloaded) |
| token confidence baseline | `test_confidence_and_variants.py`, +0.297 [+0.214, +0.380] |
| extractor-confound control | one-tier-only analysis, 0.845 vs 0.835 / 0.853 vs 0.828 |
| second model family | Pixtral-12B, 0.828 [0.782, 0.871], robust |

### 7. Reasoning — deliberately closed

Null/artifact story. **No more GPU there.** See the RETRACTION section.

### Not on the reviewer's list, but the actual critical path

**Writing.** `paper/refs.bib` has **3 entries** and Related Work is a `\todo`
that says "VERIFY EVERY CITATION AGAINST THE REAL PAPER". Introduction,
Conclusion and two figures are also `\todo`. And **`report/report.tex`
contradicts itself** — the Retraction section sits at line ~150 while the body
still asserts the retracted Phase 4/5 claims as findings.

WACV Round 2: enrol Aug 21, submit Aug 28. **Two blockers to fix before any
further analysis: Related Work, and the report's self-contradiction.**

### One CSV still Drive-only

**`confidence_perception_full_n300_*` was downloaded 2026-08-11 and all 7
verbalized-confidence tests pass** — the notebook-20 result is now test-
protected, and the run's own accuracy (45.0%) is recorded rather than
borrowed from notebook 19. Only `boxed_perception_full_n300_*` (the gated
one) is still Drive-only, and nothing may be quoted from it anyway.

## Current findings

### 2026-08-12: DISTRIBUTION BY ITEM TYPE — and the red markup is NOT error metadata

`pilot/dataset_profile.py`, `pilot/dryruns/build_dataset_profile.py`,
`pilot/tests/test_dataset_profile.py` (18 tests). Offline, no inference.
Artifacts: `reference/audit/dataset_distribution_by_type_20260812.{md,csv}`
and `.../dataset_examples_contact_sheets_20260812/` (20 captioned tiles,
**text-only** — FERMAT is gated and its images never reach the run CSV, so
crops need Colab).

- **`\textcolor{red}` CANNOT be used as injected-error location or type, and
  the earlier note in this file that "FERMAT marks the injected error in red"
  is too strong.** Measured: red appears in **258/300** ground truths,
  **885 spans**. It is on **149/150 error items but ALSO 109/150 clean ones
  (72.7%)**, and clean items carry *more* of it on average (3.61 spans vs
  2.29). Reading the spans says why — on clean items red marks added
  elaboration and restated steps (one is a whole sentence starting
  *"Additionally, if we consider the factorial function…"*), on error items it
  marks changed values. **So red marks what was MODIFIED when the variant was
  generated, not what is WRONG.** Not specific (fires on 73% of clean), not
  unique (median 2 spans, up to 10), not necessary (**item 142 is
  `has_error=1` with no red at all**), and carries no type information.
  Legitimate only as a pointer when reading one item by hand; never counted
  or aggregated.
- **The dataset provides NO error location, NO error type, NO clean answer,
  NO page/image id and NO subject/topic.** `has_error` is binary and is the
  only label describing the perturbation. **`orig_q` is the QUESTION — its
  name invites reading it as the original answer**, which would turn "no
  clean answer stored" into a false positive; pinned in
  `test_orig_q_is_not_mistaken_for_the_clean_answer`.
- **ANSWER TYPES MUST BE DERIVED FROM THE TRUTH SIDE ONLY.**
  `strict_v2.score_item_v2` deliberately ORs each shape flag across model and
  truth (right for a review queue). Grouping that way makes a group's accuracy
  partly a statement about *which items the model answered badly* — a short
  wrong model span puts the item in the group that then looks inaccurate by
  construction. Truth-only vs OR'd differ on **26 of 300**, so it moves real
  mass. Both are in the CSV (`answer_type`, `answer_type_either_side`).
- **The splits, `strict_v1` / `strict_v2`:**

  | split | n | v1 acc | mean H | AUROC |
  |---|---|---|---|---|
  | `has_error=0` | 150 | 52.7% | 0.942 | 0.813 [0.744, 0.875] |
  | `has_error=1` | 150 | 41.3% | 0.985 | 0.817 [0.749, 0.880] |
  | MCQ | 58 | **82.8%** | 0.452 | underpowered (minority 10) |
  | free response | 242 | 38.4% | 1.086 | 0.760 [0.698, 0.817] |
  | simple value | 215 | 57.2% | 0.879 | 0.824 [0.767, 0.873] |
  | structured/prose | 85 | **21.2%** | 1.176 | underpowered (minority 18) |

  Worst types: **`system_answer` 5.9% (1/17)** and **`text_conclusion` 13.2%**
  with a **47.4% max-entropy rate**. Best: `mcq_option` 86.8%.
  **The AUROC is flat across `has_error`** — reproduces the locked finding.
- **Span length is monotone and then INVERTS, and the inversion is the
  finding.** 75.6% (4–10 chars) → 54.7% → 31.0% → **14.3% (>100 chars)**. But
  **tiny spans (≤3) score a middling 46.9% while carrying the HIGHEST
  `extraction_issue` rate of any bucket, 58.1%** — their verdicts are the
  least earned. A tiny span's verdict is unreliable in *either* direction, not
  a good score.
- **`false_pass` is 0 in every group and that is a DEFINITION, not a clean
  bill of health.** The audit codes an unearned verdict `extraction_issue` →
  *indeterminate*, so it can never reach that column. The unearned rate is the
  `extraction_issue` column: **13.5% among audited `strict_v1`-correct items**.
  The canonical figure with its p=0.00007 two-pass disagreement stays in the
  spot-check entry.
- **AUROC is REFUSED, not approximated, in two situations** — minority class
  below the registered 30, and a group *defined by* the correctness label
  (accuracy is then 0%/100% by construction). Grouping by `strict_v2` verdict
  reports `underpowered`, which **masks something worse**: the two rules agree
  on 269/300, so that split is very nearly degenerate too and simply has no
  AUROC to give. Both are locked in tests.
- **A real selection bug, caught by looking at the output rather than by an
  assert.** The contact-sheet sampler round-robined over a FIXED type order
  that puts `numeric` and `single_expression` last, so with five slots per
  cell the two types covering **54% of the corpus could never appear**.
  Shuffling the order made the exclusion unbiased; pinned in
  `test_the_dominant_answer_type_can_reach_the_sheet`.
- Question type is **inferred** (`strict_v2._looks_mcq`) and gives **58 MCQ**,
  against the **52** recorded in the 2026-08-11 entry from a different
  detector. See the MCQ entry below — it is both over- AND under-inclusive.

### 2026-08-12: ANSWER FORMAT — a DIAGNOSTIC, agreed with the user 2026-08-12

**KEEP THIS AS A DIAGNOSTIC ANSWER-FORMAT ANALYSIS. It is NOT a headline
result and must not be presented as one.** It is evidence that *answer format
strongly affects apparent accuracy* — a property of the answer space and the
scorer, not of the model reading multiple-choice pages better.
**`paper/main.tex` is NOT to carry this unless the user explicitly asks.**

**THE TWO NUMBERS TO USE, copy them rather than re-deriving:**

| set | n | `strict_v1` accuracy | mean H |
|---|---|---|---|
| **confirmed MCQ** | **55** | **83.6%** (46/55) | **0.423** |
| **free response** | **245** | **38.8%** (95/245) | **1.085** |

**The heuristic detector was manually checked, on both sides:** of **58
flagged**, **54 are true MCQ and 4 are false positives**; of **19 missed
candidates**, **1 was recovered**. All 77 coded in
`reference/audit/mcq_like_review_20260812.csv` (`confirmed_mcq` / `mcq_type` /
`coding_depth` / `reviewer_note`).

**THE THREE CAVEATS THAT TRAVEL WITH THE NUMBERS — do not drop them:**

1. **Items 170 and 237 are MCQ-like but NOT ordinary single-option MCQ.** They
   are *matching* questions whose answer is a four-way mapping
   `(i)→(d), (ii)→(c), (iii)→(a), (iv)→(b)`, while the truth span is a bare
   `(b)` — **one quarter of the answer**. Scored wrong for a reason that is
   not a misread.
2. **Item 8's options are not visible in the crop.** Its answer is `option b`
   while the page shows no option list, so the letter is **visually
   unrecoverable by any reader**, model or human. Scored wrong at max entropy.
3. **Not a headline.** The gap is about answer format; an MCQ answer space has
   few reachable values, so five samples agree easily and a string match is
   cheap.

- **The pre-audit sensitivity range was 78.9–88.5% and the truth landed at
  83.6%, 0.8 points off the uncorrected figure.** The detector was wrong in
  both directions and **the two errors very nearly cancelled**: it wrongly
  included 4 items scoring 50%, and missed 1 item that was wrong.
  **Lesson worth keeping: a detector being demonstrably wrong in both
  directions does not mean the aggregate it feeds is wrong.** The range was
  the honest thing to report before the audit; it was also much wider than the
  error turned out to be. `mcq_accuracy_sensitivity` still returns
  `quotable=False` because it prices the HEURISTIC's uncertainty; its
  `superseded_by` field points at the coded manifest.
- **The 4 false positives are exactly the ones the weak-trigger flag caught,
  and its precision was 4/6.** Confirmed not-MCQ: **82, 117, 195, 234** — all
  `P(A)`/`P(B)`/`f(a)=f(b)` notation. **170 and 237 ARE MCQ**, both *matching*
  questions (left `(i)-(iv)` to right `(a)-(d)`), so the regex fired for a
  wrong reason on a right answer.
- **`mcq_type` is a new distinction the coder introduced and it matters for
  scoring.** Item 170 is a **matching** question whose answer is a 4-way
  mapping `(i)→(d), (ii)→(c), (iii)→(a), (iv)→(b)`, but its **truth span is
  the bare `(b)` — one quarter of the answer**, and the model's
  `Finally, (iv) matches (b)` is scored wrong. The scorer checks a quarter of
  a matching answer; that is a `multi_answer_collapse` case, not a model
  failure. Same for 237.
- **Item 8 carries `options_not_visible_in_crop`:** its answer is `option b`
  while the page shows no option list, so *neither the model nor a human could
  recover the letter from the image*. It is the only `option_word_only` item
  and it is scored WRONG at max entropy — an unwinnable item, not a misread.
- **ID RECONCILIATION — CONFIRMED by the user 2026-08-12.** The coder wrote
  "item 41", but **item 41 of this run is an ellipse-equation item**
  (`x²/75 + y²/100 = 1`, no options, `strict_v1` correct) **that was never on
  any sheet.** The description matched item 8 on four independent points (the
  coder's own phrase "option-word-only" is that item's unique presort, truth
  span literally `option b`, no visible choice list). Recorded against
  **item 8**; the user confirmed it **"unless another sheet proves
  otherwise"**, and that standing condition is written into the row's
  `reviewer_note` so a later sheet can reopen it. **The count is unaffected
  either way** — item 8 was already `yes` from the sweep.
- **CODING DEPTH IS RECORDED, because this project has already measured that
  it matters.** 25 items were read one by one; **52 were ruled by a block
  sweep** (*"the rest were mcq exactly"*) — the same sweep pattern whose
  first-40-vs-extra-60 disagreement hit **p=0.00007** on the false-pass audit.
  **Two reasons it is far safer here:** the judgement is objective (options
  are visible or they are not), and **51 of the 52 swept items carry BOTH a
  choice list in the question AND the word "option" in the answer**, an
  independent text signal agreeing with the sweep. Only **item 237** is swept
  without corroboration, and it is a matching question like 170.
- Free-response accuracy moves 38.4% → **38.8%**; the MCQ-vs-free gap is
  **44.8 points** and was never in doubt.

### 2026-08-12: the MCQ detector, and why it needed auditing (superseded by the entry above)

`pilot/dataset_profile.py` (`mcq_trigger`, `mcq_review_set`, `mcq_caption`,
`mcq_accuracy_sensitivity`), `pilot/28_mcq_contact_sheets.ipynb`,
`pilot/dryruns/dryrun_nb28.py`, 12 more tests (file now 30).
Manifest: `reference/audit/mcq_like_review_20260812.csv` (77 rows).
**Everything is labelled "MCQ-like", never MCQ. Nobody has read the sheets.**

- **THE OVER-COUNT IS PROVEN AND THE MATCHED TEXT NAMES THE CAUSE.**
  `_looks_mcq`'s second trigger is `\(\s*[a-d]\s*\)[^\n]{0,60}\(\s*[b-e]\s*\)`
  with `re.I`, so it matches **probability and function notation**:
  item 82 `P(A) = P(B)`, 117 `P(A) = \frac{7}{13} …P(B)`, 234 `P(A) + P(B)`,
  195 `f(a) = f(b)`, and 170/237 a **matching exercise**
  (`(d)$. Similarly, $(ii)$ matches $(c)`). **6 of 58 fired with no "option"
  word anywhere.** None is multiple choice.
- **AND IT ALSO UNDER-COUNTS — this is the half I initially missed and it
  reverses the framing.** **19 items NOT flagged carry a choice list
  (`\begin{enumerate}`/`\item`) in the question.** So the corrected count can
  go UP as well as down. **An audit of only the flagged items could find false
  positives and nothing else** — the exact one-sided mistake the 104-item
  `genuinely_wrong` census made, which needed a whole separate correct-side
  spot check afterwards. `mcq_review_set` therefore ships **two groups**:
  `flagged_mcq_like` (58, confirm/reject) and `candidate_missed` (19, recover).
- **THE 82.8% MCQ ACCURACY IS DIAGNOSTIC ONLY. Do not quote the magnitude.**

  | variant | n | `strict_v1` acc |
  |---|---|---|
  | as reported | 58 | **82.8%** |
  | drop the 6 weak-trigger items | 52 | **88.5%** |
  | drop weak + add all 19 candidates | 71 | **78.9%** |

  A **~10-point swing on a detector judgement no human has ruled on.** The
  **direction is stable** — MCQ far above free response (38.4%) under every
  variant — so *"MCQ items score much higher"* is safe; the number is not.
  `mcq_accuracy_sensitivity` returns `quotable=False` and says why.
- **A presort can HIDE the suspicion it was built to surface.** Items 170/237
  carry an enumerate, so `presort` demotes them to `choice_list_only` even
  though the detector fired on the unsound regex. `flagged_by_weak_trigger_only`
  (flagged AND no "option" word anywhere) is the honest test, and the caption
  prints `WEAK trigger matched: …` off that, not off `presort`. Locked in
  `test_weak_trigger_is_flagged_even_when_a_choice_list_demotes_the_presort`.
- **Page and cell are computed in `mcq_review_set`, not the notebook**, so the
  committed manifest and the rendered PNGs cannot drift; the notebook asserts
  the rebuilt set matches the committed CSV before rendering, and the dry run
  checks every named sheet exists. `--break-alignment` proves the
  image-alignment guard fires (298/300 mismatches refused).
- **TO RUN: `pilot/28_mcq_contact_sheets.ipynb` in Colab** (no GPU). Writes
  `Drive/uncertainty-math-vlm/figures/mcq_like_review/` — `mcq_flagged_p1..7`
  and `mcq_candidate_missed_p1..3`, captions burning in item id, `has_error`,
  `strict_v1`, H, both spans, both labels, flags and the weak-trigger match.
  Code `confirmed_mcq` = yes/no/unclear in the manifest.

### 2026-08-13: BUILT, NOT RUN — notebook 29, scorer-vs-human examples for review

`pilot/29_confusion_examples_for_professor.ipynb`, `dataset_profile`
(`confusion_category` / `confusion_examples` / `confusion_caption` /
`write_confusion_readme`), `dryruns/dryrun_nb29.py`, 8 tests.
**Supplementary review material, not an experiment and not a measurement.**
Renders captioned PNG sheets to
`Drive/uncertainty-math-vlm/figures/confusion_examples_for_professor/`.

- **The prediction is the SCORER's verdict, the truth is the human read, so
  TP/FP/FN/TN describe the SCORER and not the model.** Audited population
  (234 of 300, precedence-deduplicated): **TP 95, INDETERMINATE 71, FN 33,
  TN 20, FP 15**.
- **FIVE GROUPS, NOT FOUR, AND THIS IS THE LOAD-BEARING DECISION.**
  `extraction_issue` cuts two ways: on a **pass** it is a **false pass (FP)**;
  on a **fail** it leaves the model's answer **undecided**, so it is neither
  TN nor FN. **Folding those 71 into TN would inflate the scorer's apparent
  accuracy on the largest audited group** — and they are the project's own
  headline finding. Pinned in
  `test_an_unearned_FAIL_is_never_a_true_negative`, and the dry run asserts
  `extraction_issue` lands on BOTH sides of the matrix.
- Requested cases are forced in and appear first: **180, 289, 291** (FP),
  **144** (FN — prose answer against a collapsed `sympy:2`), **132, 147** (TP),
  **299, 273** (INDETERMINATE, *not* TN, which is what they turned out to be).
- **TP EXEMPLARS MUST SHOW THE ANSWER (2026-08-13).** Item 77 is a *genuine*
  true pass — the coder's own note reads *"tiny span but a valid table cell"* —
  but its span is the single letter `m` while the page holds two coefficient
  tables, so as an illustration it is indistinguishable from the collapse
  cases on the FP sheet. **9 of the 95 TP-eligible items are like that**
  (10, 77, 91, 102, 127, 138, 223, 244, 254), so excluding one by name would
  leave the draw free to pick a sibling. TP now prefers spans ≥
  `MIN_EXEMPLAR_SPAN` (4 chars), verified to hold across 8 seeds, with tiny
  spans kept only as a fallback. **This is a choice of EXAMPLE, not a
  recategorisation** — those items remain true passes in the population
  counts. `confusion_examples` also gained `exclude=` for by-name rejections.
- **Selection round-robins over the HUMAN LABEL, rarest first.** A uniform
  draw on TN returned six `notation_misread` items and hid `copied_wrong_line`
  and `true_wrong` entirely; six copies of one mechanism teach a reviewer
  nothing.
- **SIX groups, not five (2026-08-13).** `needs_visual` was originally routed
  to INDETERMINATE regardless of verdict, so **item 5 — scorer PASSED, coder
  could not read the page — rendered under a sheet titled "scorer said
  WRONG" while its own caption read `v1=CORRECT`.** A reader without context
  sees a contradiction in the DATA, not a routing bug.
- **ALL `needs_visual` items now share ONE group and that group claims NO
  verdict** (`CONFUSION_VERDICT["NEEDS_VISUAL"] is None`). Splitting them by
  verdict fixed the contradiction but scattered three related items over two
  sheets and left a one-tile group. They belong together because the shared
  fact is about the **evidence** — the page could not be read — not about the
  verdict. **A heading that asserts nothing cannot contradict a caption**, and
  each tile prints its own `v1`. Holds items **5, 45, 126**.
  `assert_confusion_groups_match_titles` skips groups whose declared verdict
  is `None` and enforces the rest before rendering.
- **A LEGEND is now drawn once per SHEET** (`CONFUSION_LEGEND`, via a new
  optional `footer=` on `plotting.contact_sheet`, empty by default so
  notebooks 17/23/28 render unchanged). It defines v1/v2, span M/T, label M/T,
  `sympy:`/`text:`, `human`, `H` and `err`. **Not per tile** — the fields are
  identical on every cell.
- **FOUR defects only RENDERING A PAGE AND LOOKING AT IT caught; the asserts
  passed every time.** (1) the `why:` line truncated mid-word (`the match is
  vacu...`), now wrapped; (2) the FN note was one generic sentence on every
  tile, now names the mechanism (prose-vs-collapsed-symbol, trailing
  punctuation, LaTeX spacing, residual `\textcolor`); (3) item 5 under a
  contradicting title; (4) **the NEEDS_VISUAL note fell through to "human
  confirms the model was correct" under a title reading "correctness
  UNKNOWN"** — the same contradiction the group existed to remove.
  **Render the sheet and read it.**
- **TO RUN: notebook 29 in Colab** (no GPU). Writes 5 sheets + `manifest.csv`
  + `README.md`, the last defining the groups before showing any count.

### RUN 2026-08-12: notebook 27 — LiveMath n=300 VALID, Omni n=300 VOID (decoder)

`pilot/27_open_judges_all300.ipynb`, all 7 cells completed, no errors.
Evidence: `reference/audit/open_judges_all300_diagnostic_20260812.csv` +
**`..._20260812_STATUS.md`, which is the authoritative reading — read it
first.** No accuracy anywhere; no scorer rule touched.

- **LiveMath-Judge, n=300, USABLE. 300/300 decoded, 0 parse failures**, raw
  output takes exactly two values (`\boxed{yes}` / `\boxed{no}`).

  | | |
  |---|---|
  | yes / no | 261 / 39 → **87.0% yes-rate** |
  | agreement with `strict_v1` | **154/300 = 51.3%** |
  | **always-yes baseline** | **47.0%** |
  | judge yes / rule wrong | **133** |
  | judge no / rule correct | **13** |

  **CONFIRMS AT n=300 what notebook 25 found at n=40: it performs at the
  trivial baseline** (51.3% against 47.0%). Disagreement is one-directional,
  133 lenient vs 13 strict. **Two things it is NOT evidence of:** leniency is
  *higher on clean items* (89.3%) than on `has_error=1` (84.7%), the opposite
  of the leniency signature, so it is not forgiving the perturbation — it just
  says yes; and `looks_like_solving`=0 is not exoneration, since all 300
  transcripts are a bare boxed verdict with nothing to scan.
- **Omni-Judge, n=300, VOID. 263/300 came back `unclear` and it is a DECODER
  failure, not judge behaviour.** All 300 transcripts classified:
  marker transposed (`Jugdement`, `Eqiuvalence`, `FALSe`) **139**, shredded
  (`#i#f#i#c#a#t#i#o#n#T#h#e...`) **89**, stopped before the verdict **35**,
  no marker **17**, **clean only 20 — 93.3% corrupt.**
- **TWO FACTS THAT DECIDE IT IS UNSALVAGEABLE, not merely mis-parsed.**
  (1) **19 of the 37 items that PARSED still carried a misspelled marker** —
  a successful parse was never evidence of a clean decode. (2) **ALL 36
  `omni_invented_reference` flags sit in corrupt transcripts**; item 0 was
  flagged for citing `2.4` when the model answered `24`, which is the decoder.
  **The invented-number rate was the one thing the run existed to measure and
  it measured decoder noise end to end.**
- **THE PARSER IS DELIBERATELY NOT LOOSENED, and there is a test to keep it
  that way.** Tolerating transposition would recover ~120 verdicts from
  transcripts whose justifications are also damaged: a biased subset (the
  mildly corrupted ones) with untrustworthy content, **which is worse than no
  data because it looks like data**. See
  `test_the_parser_is_NOT_loosened_to_recover_corrupted_output`. **Omni is not
  re-run either** — open-judge testing closed after notebook 26 and the
  LiveMath half answers the n=300 question.
- **THE GUARD FAILED AND THAT IS THE TRANSFERABLE LESSON.**
  `looks_bpe_mangled` tested for ONE signature (the `Ġ` byte-BPE marker),
  caught **0 of 300**, and the notebook printed *"decode clean on all 300"*
  while 93% was damaged. **This is the third distinct decode corruption on
  this model, and each time the guard written for the previous one was blind
  to the next.** Replaced by `judge.omni_output_health` /
  `omni_decode_report` / `assert_omni_decode_ok`, which classify every shape
  above and fail closed — **validated against the 300 real transcripts, not
  invented fixtures** (280/300 flagged). `dryrun_nb27.py --corrupt-decode`
  proves it aborts. **Standing rule, now with a second instance: a judge
  scoring `unclear` is a decoder bug until proven otherwise, and make the
  check adversarial to the CONTENT, not only to a known marker.**

### 2026-08-12: BUILT, NOT RUN — notebook 27, both open judges on all 300 (superseded above)

`pilot/27_open_judges_all300.ipynb`, `pilot/judge.py` (new
`open_judge_*` section), `pilot/dryruns/dryrun_nb27.py`, 15 new tests.
**Exploratory diagnostic only. Nothing it produces may be reported as
corrected accuracy, and no function in it returns one.**

**Why run judges that already failed.** Both failures are currently
*single items*. LiveMath-Judge failed the verdict gate on 273; Omni-Judge
failed the rationale check on 273. An all-300 pass converts two anecdotes
into **rates** — and the one worth having is
`invented_reference_tokens`, which flags a justification citing a number
present in **none** of question, reference or model answer. That is the
item-273 signature (*"The reference answer is 3/4"* when we passed `2/4`)
made countable.

- **THE TRAP THE WHOLE NOTEBOOK IS BUILT AROUND, and it is the same one
  the cross-audit entry records: agreement between two unreliable scorers
  measures neither of them.** `strict_v1` calls 47.0% of items correct, so
  **a judge answering "yes" to everything agrees with it on 47.0% by
  construction** — and notebook 25 already found LiveMath-Judge sitting
  exactly on that trivial baseline against human labels. Every agreement
  figure therefore ships with `always_yes_agreement`, and judge-vs-judge
  agreement with its chance rate and **Cohen's kappa**. **A raw agreement
  number quoted alone out of this file is a misuse of it.** Pinned in
  `test_the_always_yes_baseline_equals_the_rules_own_correct_rate` and
  `test_kappa_exposes_agreement_that_is_only_arithmetic` (80% raw
  agreement, kappa **below zero**).
- **`unclear` is NA, never `False`.** Recording an unreadable verdict as
  disagreement asserts the judge *contradicted* the rule, which is a
  different and stronger claim than "we could not read a verdict". Two
  parse failures are likewise **undecidable, not two judges agreeing** —
  counting them as agreement would inflate the single number the file
  exists to report honestly.
- **Both judges are fed ONE shared majority answer, computed once.**
  Letting each recompute it is one refactor away from feeding them
  different inputs, which would still produce a plausible-looking
  comparison table. The dry run's stubs record what each was shown and
  compare element by element.
- **Sequential model loads, not concurrent.** 3B then 8B with an explicit
  free between, so it fits a 16 GB runtime. Both loops checkpoint every 25
  items and **refuse to resume across a different model id or prompt** —
  a stale checkpoint would silently mix two judges into one column.
- **The gate items come back for free**, since 55 and 273 are inside the
  300. `GATE_REPRODUCTION` records what notebooks 24/26 saw; a mismatch is
  reported as **version drift, not a gate failure**, and does not raise.
- **The invented-number flag errs BOTH ways and the docstring says so.**
  It over-triggers on incidental numerals and **cannot see a one-digit
  invented answer at all**, since a bare digit appears in almost any input
  by chance. The flagged tokens are kept in the CSV so a human dismisses a
  false hit in seconds. Pinned in
  `test_the_flag_cannot_see_a_one_digit_invented_answer`.
- **A low `looks_like_solving` count for LiveMath-Judge is NOT
  exoneration** — it emits a bare `\boxed{}` with no reasoning, so there
  is no transcript to scan. The summary writer says so inline whenever
  silent transcripts are present, the same guard notebook 25 needed.
- Outputs: `reference/audit/open_judges_all300_diagnostic_20260812.csv`
  (the 17 requested columns in order, plus 7 clearly-separated extras for
  the manual rationale pass) and a summary `.md` whose no-accuracy
  property is asserted by both the tests and the dry run.
- `strict_v1`, `strict_v2` and the audit CSVs are **read only**; no scorer
  rule is touched.

### RUN 2026-08-12: Omni-Judge passes the VERDICT gate and fails the RATIONALE check

`pilot/26_omni_judge_gate.ipynb`, gate only. `KbsdJames/Omni-Judge` —
Llama-3.1-8B-Instruct instruction-tuned on GPT-4o evaluation data. Second
open-judge candidate after LiveMath-Judge failed the same gate. **Open-judge
testing STOPS here.**

| probe | verdict | rationale |
|---|---|---|
| item 55 | `FALSE` ✅ | **correct** — cites the sign discrepancy in the denominator |
| item 273 | `FALSE` ✅ | **WRONG REASON** — substitutes its own reference |

- **THE PAPER SENTENCE, agreed with the user 2026-08-12. Copy it:**
  *"A verdict-only gate is insufficient for open math judges. Omni-Judge
  returned the desired FALSE verdict on both probes, but on item 273 its
  justification showed that it reconstructed a mathematically correct
  reference answer from the problem text rather than comparing against the
  provided perturbed answer. Thus, even when verdicts match, the judge may
  not be performing the intended fidelity task."*
- **Item 273 in full.** It wrote *"The student's answer of 2/4 is incorrect …
  The reference answer is 3/4."* **We passed `2/4` AS the reference, and the
  student answered a bare `}`.** Verified: `3/4` appears in **none** of the
  three inputs — question (150 chars), reference (652), model answer (614).
  It read the reference's own prose (*"the number of outcomes favourable to E
  is 3"*), computed `3/4`, promoted that to the reference, and attributed the
  real reference's answer to the student. **Right verdict from a comparison
  it never performed.**
- **THIS IS A LIMITATION OF THE GATE I BUILT, and it generalises.** A
  verdict-only gate cannot detect a right-answer-wrong-reason failure. On
  **LiveMath-Judge, which emits a bare `\boxed{yes}` with no reasoning, this
  identical failure would have been invisible and scored as a clean pass.**
  Only a judge that explains itself makes the distinction observable. **Any
  future judge must be gated on rationale as well as verdict.**
- **THE STOPPING CONCLUSION:** LiveMath-Judge fails the verdict gate;
  Omni-Judge passes it while not doing the comparison task. **Open math
  judges are not reliable fidelity scorers for perturbed-answer
  transcription.** The sharper reason is not that they return wrong verdicts
  — it is that **they re-derive the answer and grade against that**, which is
  exactly wrong when the ground truth is deliberately perturbed.
- **TWO FALSE NEGATIVES AGAINST THE MODEL BEFORE THIS WAS READABLE, both
  mine, both silent.** (1) The custom `OmniJudgeTokenizer` decode leaked
  byte-BPE markers (`F AL ĠS ĠE`), so `parse_response` returned all-None and
  the gate recorded `unclear`. Fixed by building a plain
  `PreTrainedTokenizerFast` from the repo's own `tokenizer.json` —
  `AutoTokenizer` cannot help here, it refuses without `trust_remote_code`
  and returns the same broken class with it. (2) Even after that, the PROSE
  decoded perfectly while the STRUCTURAL markers stayed space-corrupted
  (`# #Equivale nceJudgmentFAL S E`), so its own `parse_response` still found
  no anchors. `judge.parse_omni_text` reads the verdict whitespace-
  insensitively as a fallback. **Lesson: a judge scoring `unclear` is a
  decoder bug until proven otherwise — check the raw text before recording a
  verdict.**
- The verdicts above were derived by applying the fallback parser **offline to
  the raw text captured in the notebook's own outputs**, not from a fresh GPU
  run. That is legitimate here because the fallback is deterministic text
  parsing and the rationale finding *is* that text — but it is why the
  notebook's stored cells show `unclear` while this entry says `FALSE`.

### RUN 2026-08-12: LiveMath-Judge FAILS the fidelity gate — do not use it as the scorer

`pilot/judge.py`, `pilot/24_livemath_judge.ipynb` (gated scoring path),
`pilot/25_livemath_judge_diagnostic.ipynb` (exploratory), 32 tests.
`jnanliu/LiveMath-Judge` — Qwen2.5-3B-Instruct fine-tune, Apache-2.0,
arXiv 2412.13147, emits `\boxed{yes}`/`\boxed{no}`.

**Motivation.** The audit found 14 of 16 confirmed false passes were
`extract_final_answer` picking a wrong or partial span. A judge reading the
whole answer against the whole ground truth skips the extractor entirely.

**Result: GATED.** The 300-item run was never spent.

| probe | verdict | required |
|---|---|---|
| item 55 | `incorrect` | ✅ |
| item 273 | **`correct`** | ❌ |

- **THE HEADLINE SENTENCE:** *on item 273 the judge accepted a
  non-faithful / non-useful model output as matching the ground-truth answer
  `2/4`.* **LiveMath-Judge failed the fidelity gate and must not be used as
  the scorer.**
- **DO NOT SAY "the model corrected 2/4 to 3/4" ABOUT THIS RUN.** That is a
  PIXTRAL fact — Pixtral gave `3/4` unanimously. The gate runs on the **Qwen**
  CSV, where item 273's majority label is the bare `}` and the answer field
  handed to the judge is coin-tossing setup prose (*"We write H for 'head'
  and T for 'Tail'…"*), not `3/4`. Only sample s1 said `3/4`. The same
  Qwen/Pixtral conflation was caught once before on this item; it recurred
  here.
- **The two probes test DIFFERENT things on the Qwen run, and the gate
  comment originally implied otherwise.** Item 55 *is* a silent correction on
  Qwen (page `1 +`, majority sample `1 -`), so it tests fidelity directly and
  the judge passed it. Item 273 is not — it tests whether the judge accepts a
  **non-answer**, which is a weaker premise but a **stricter bar**: there is
  no mathematics to be lenient about, so accepting an unrelated passage
  cannot be excused as equivalence. Fixed in `GATE_ITEMS`' docstring.
- **Reading the model card rather than assuming it found three things.** Its
  criterion 3 (*"You do not need to recalculate"*) already helps; criterion 2
  (*"equivalent … is also considered correct"*) is the danger, the same
  acceptance that made Math-Verify unsafe here. And **it has no abstain
  verdict** — its prompt maps *"difficult to judge"* onto `no` — so `unclear`
  in our output can only ever mean OUR parse failed.
- **TWO BUGS IN THE PUBLISHED USAGE SNIPPET, both silent.**
  `apply_chat_template(return_tensors='pt')` returns a TENSOR that the snippet
  then subscripts as `inputs['input_ids']` (needs `return_dict=True`); and
  `generate()` is called with no `max_new_tokens`, defaulting to 20, which
  truncates before the boxed verdict — every item would have parsed as a
  failure looking like judge breakage rather than truncation.
- **Two bugs caught by tests before any GPU time.** The verdict parser must
  take the **last** boxed answer, since the prompt contains a literal
  `\boxed{yes}` and a first-match parser would score nearly everything
  correct. And the bare yes/no fallback matched `"no verdict at all"` as a
  `no`, which made `run_gate` **pass on unparseable output**.
- **NEXT CANDIDATE, user's recommendation: Omni-Judge** (~8B, trained from
  GPT-4o judgments, reported stronger than LiveMath-Judge). **Test it on the
  SAME gate, items 55 and 273, before anything else.** Also raised:
  `Qwen2.5-Math-7B-Instruct` prompted as a fidelity comparator, and
  DeepSeek-Math / DeepSeek-R1-Distill-Qwen-7B — the latter noted as *likely to
  solve the maths*, which is precisely the hazard here.
  **Verify any candidate's model card before writing the adapter** — doing so
  for LiveMath-Judge found the two snippet bugs above.
  **If Omni-Judge also accepts item 273, the honest conclusion is that open
  math judges are unsafe for a perturbed-answer fidelity task**, which is a
  Limitations finding rather than a failed experiment.
- **WHAT EACH PROBE ACTUALLY TESTS, and this is the clean framing to use:**
  - **item 55 — can the judge reject a mathematically corrected but
    non-faithful answer? PASSED.** The page reads `1 +`, the majority sample
    reads `1 -`, and the judge said `no`.
  - **item 273 — can the judge reject a NON-ANSWER? FAILED.** The answer
    handed to it is coin-tossing setup prose, and it said `yes`.

  **That split makes the failure cleaner, not weaker:** the judge can spot a
  wrong formula but cannot spot that it was handed no answer at all.

- **RUN 2026-08-12, notebook 25: the 40-item diagnostic (A100, 0 parse
  failures).** Exploratory, never a scoring run.

  | | native | fidelity |
  |---|---|---|
  | yes / no | 31 / 9 | **33 / 7** |
  | agrees with human `correct` (n=32) | 28 | 29 |
  | agrees with human `wrong` (n=5) | 4 | 3 |

  - **THE JUDGE PERFORMS EXACTLY AT THE TRIVIAL BASELINE.** On the 37 items
    with a determinate human label, answering "yes" to everything scores
    **32/37 = 86.5%**; both prompts also score **32/37 = 86.5%**. It is not
    literally always-yes — it catches 3 of the 5 wrong items — but it pays
    for them with 3 false fails, landing on the do-nothing baseline. **This
    is exactly why per-class reporting is mandatory here**; the overall
    figure looks respectable and means nothing.
  - **THE FIDELITY CLAUSE DID NOT WORK, AND WHAT IT DID WAS BACKWARDS.** The
    two prompts disagree on **2 of 40**, and on both the fidelity version is
    **more lenient**, never stricter (yes 31 → 33). A 3B fine-tune does not
    honour an added criterion that contradicts its training. **Do not assume
    a prompt amendment can re-purpose a fine-tuned judge.**
  - **More permissive on the perturbed items: 90% yes on `has_error=1`
    against 75% on clean** — the direction leniency predicts.
  - **THE "DOES IT SOLVE THE MATHS" QUESTION IS UNANSWERABLE AGAINST THIS
    MODEL, and a 0 count must not be read as exoneration.** `looks_like_solving`
    fired on **0 of 40** — but the model emits a bare `\boxed{yes}` / `\boxed{no}`
    with **no reasoning at all**, despite the prompt ending in `Analysis:`.
    With no transcript there is nothing to scan. The summary writer now says
    so inline; the earlier version would have let "0 of 40" read as evidence.

- **A failed gate is a result.** It cost one Colab session and establishes
  that this benchmark's scoring problem is not solved by swapping in an
  off-the-shelf judge.

### 2026-08-12: CROSS-AUDIT DIAGNOSTICS — and why there is no pooled number

`pilot/audit_diagnostics.py`, `pilot/tests/test_audit_diagnostics.py`
(12 tests), `reference/audit/contradictions_20260812.csv`. Offline.
Scorer reliability only — **no corrected model accuracy is computed, and none
may be derived from this section.**

| audit set | n | determinate | v1 agreement | v2 agreement | ext. issue |
|---|---|---|---|---|---|
| `genuinely_wrong` census | 104 | 24 | **95.8%** (see below) | 95.8% | 77 (74.0%) |
| `v1`-correct spot check | 100 | 79 | ~~100%~~ tautological | **92.4%** | 20 (20.0%) |
| `v2` high-priority | 108 | 68 | **51.5%** | **48.5%** | 39 (36.1%) |

- **TWO OF THE THREE AGREEMENTS ARE DEFINITIONS, NOT MEASUREMENTS. DO NOT
  AVERAGE THEM.** Set 1 is items `strict_v1` called WRONG and its determinate
  labels all mean "model wrong"; set 2 is items it called CORRECT and its
  determinate labels all mean "model correct". Neither vocabulary contains a
  label that could express disagreement, so 100% is forced. Pooling the three
  gives **137/170 = 80.6%, roughly 60% tautology.** **This is the same
  structural failure as the RETRACTED stratum result** — where a stratum's
  label can take only one value, agreement collapses onto the verdict.
- **DEGENERACY IS PER (SET, RULE), NOT PER SET — and one real number survives
  because of it.** Set 2 is tautological for `strict_v1` (it was selected as
  v1-correct) but **not** for `strict_v2`, which had no part in selecting it.
  So v2's **92.4% with 6 false fails and 0 false passes on set 2 is a genuine
  one-directional measurement**. A single per-set flag would have thrown it
  away. `per_set_diagnostics` returns `v1_tautological`, `v2_tautological`
  and `detectable_error` per rule for exactly this reason.
- **NEITHER RULE EVER PRODUCED A FALSE PASS on any audited set.** Where a
  human could decide, `strict_v1` and `strict_v2` never called correct
  something the human called wrong — **0 false passes across all three sets**.
  Every disagreement is a false FAIL (v1: 33, v2: 35+6). Note this is *not*
  in tension with the 20 "false passes" in the spot-check entry below: those
  are `extraction_issue`, i.e. the verdict was **unearned**, which here maps
  to *indeterminate* rather than to *wrong*. The two are different claims and
  must not be conflated.
- **THE SETS OVERLAP — 78 items are double-counted.** 47 shared between the
  census and the high-priority pass, 31 between the spot check and it. Union
  is **234 of 300**, sum-of-parts is 312. Any count summed across sets is
  wrong. `deduplicated_counts` resolves by precedence (later, more specific
  pass wins) over the union: 128 `true_correct`, 82 `extraction_issue`, 16
  `notation_misread`, 4 `needs_visual`, 3 `true_wrong`, 1 `copied_wrong_line`.
- **FOUR HARD CONTRADICTIONS — the first intra-rater reliability measurement
  in this project.** Items **108, 149, 222, 230** were coded
  `notation_misread` (model wrong) in the census and `true_correct` in the
  high-priority pass. Both determinate, opposite directions. A further **27
  differences are soft** — one side indeterminate, because the two sheets
  asked different questions with different vocabularies. **This replaces the
  standing "no inter-rater reliability measured" limitation with a number:
  4 hard contradictions among the 47 items coded determinately in both.**
- **The four are NOT random coder noise, and the spans say why.** In at least
  three of them the census reading was made from OCR and the SymPy label,
  while the later pass had the display SPAN. Item 222's census note says "the
  model says 18" — but the span is `18, 81, 42, 24`, all four orders; the
  `18` came from the collapsed SymPy label. Item 230's census note says the
  model gave `x^2 = 4z^2` — the span reads `x^2 = 4(z)z, i.e., x^2 = 8z`,
  matching truth. **So the later pass had strictly better evidence, which
  makes this a story about information available to the coder rather than
  about the coder being unreliable.** Recorded, NOT adjudicated — settling
  them now would mean a third reading by the same coder, which is what
  produced the disagreement.
- **Nothing here is a population rate.** Set 3 is the high-risk tier by
  construction; sets 1 and 2 are one-sided by construction. Report these as
  scorer diagnostics, never as model performance.

### 2026-08-12: THE FALSE-PASS CHECK — 40% of "correct" verdicts were not earned

`pilot/corrected.py`, `pilot/tests/test_corrected.py` (19 tests),
`reference/audit/spotcheck_40_qwen_strict_v1_correct_20260811.csv`,
`pilot/23_spotcheck_contact_sheets.ipynb`. Offline.

The 104-item audit could only find false NEGATIVES. A human then read a
**seeded random sample of 40 of the 141 `strict_v1`-CORRECT items**:

| | n | share |
|---|---|---|
| `true_correct` | 23 | 57.5% |
| `extraction_issue` (verdict not earned) | **16** | **40.0%** [Wilson 26.3%, 55.4%] |
| `needs_visual` | 1 | 2.5% |

**All 4 seeded calibration items (31, 117, 239, 294) were caught**, so the
pass detects what it was built to detect.

- **THE ONE-SIDED NUMBER WAS WRONG BY ~20 POINTS AND MUST NOT BE USED.**

  | | accuracy |
  |---|---|
  | `strict_v1` baseline | 47.0% |
  | one-sided (wrong side only) | ~~71.3–74.0%~~ **retracted** |
  | **two-sided** | **44.5–60.6%, point 51.7%** |

  The two corrections very nearly cancel: ~73 items recovered from the wrong
  side, ~37–78 false passes removed from the correct side. **The honest
  headline is that human correction moves accuracy from 47.0% to ~51%, not to
  74%.** The false-pass count is extrapolated from 40 items, so its Wilson
  interval is carried through — the wrong-side audit is a census, this is not.
- **The mechanism, and it is the worst case for the method: false passes are
  LOW ENTROPY.** Mean 0.818 against 0.530 for verified-correct and 1.370 for a
  real misread. **A false pass is a CONFIDENTLY wrong item, which is precisely
  what entropy cannot flag by construction.**
- **THE CORRECTED AUROC IS NOT RESOLVED, AND THE RANGE MUST BE REPORTED, NEVER
  AN ENDPOINT.** On the 144 fully-audited items, weighted:

  | reading of a false pass | AUROC [95% CI] | excludes chance |
  |---|---|---|
  | model was WRONG | **0.572** [0.456, 0.689] | **no** |
  | undecidable (excluded) | **0.724** [0.576, 0.848] | yes |
  | model still right | 0.728 [0.582, 0.854] | yes |

  The audit did not settle which reading is right. Some are unambiguous (item
  31: model 2 cm against a handwritten 19.2 cm); the collapsed-to-one-symbol
  cases (84, 117, 239, 250, 282) leave the model's actual answer unknown,
  because only the *match* was shown to be vacuous. **Say 0.57–0.73,
  unresolved, pending a larger correct-side audit.**
- **THE ESTIMATOR TRAP, and the naive number is the one that looks publishable
  in the wrong direction.** After correction each class is a MIXTURE of both
  strata, sampled at 100% (wrong) and 28% (correct), so the corrected-correct
  class is 76% recovered high-entropy items where the population would be
  ~47%. **Use `corrected.weighted_auroc` with inverse-sampling weights and a
  stratified bootstrap, never a plain `bootstrap_auroc_ci` on the subset.**
  Here it moved 0.586 → 0.572 — small, because the false passes pulled into
  the wrong class are themselves low-entropy, so both classes shift together.
  The weighting is still the correct estimator; it just is not what moves this
  number.
- **The generalisable lesson, and it is a Limitations sentence: the automatic
  labels are unsafe as the final judge for long or multi-part answers.** They
  are fine for a simple numeric or algebraic final answer, and they fail when
  the answer is a sentence or proof, has several parts, is a
  matrix/vector/set, or is multiple-choice plus a worked value. The whole
  thing reduces to one symbol (`c`, `p`, `r`, `l`, `f`), and **two reduced
  labels then MATCH, scoring the item correct for no reason at all.**
- **CORRECTION (2026-08-12) — an earlier version of this entry blamed SymPy,
  and the data says otherwise.** `corrected.label_views` shows the extracted
  SPAN next to the comparison LABEL. Across the 16 false passes:

  | where it goes wrong | n | example |
  |---|---|---|
  | extractor picked a **1-character span** | **6** | item 84's span is literally `c`; 127 `f`; 250 `R`. SymPy encoded it faithfully. |
  | extractor picked a **partial span** | **8** | item 27 keeps `4x = 3` and drops `y = 33/4`; 215 keeps one of three direction cosines |
  | **SymPy collapse** (long span → one symbol) | **2** | 117 (118 chars → `sympy:p`), 239 (76 chars → `sympy:l`) |

  **So 14 of 16 are `extract_final_answer` choosing the wrong or partial
  span, not the normalizer.** Fixing SymPy would not touch them. Say
  *"final-answer extraction is the weak link"*, not *"SymPy is unsafe"* —
  though 117 and 239 show the collapse is real too, and the symptom (a
  vacuous match) is identical either way, which is why the two were confused
  in the first place. **Both views are now on the contact sheets so the
  distinction is visible while coding.**
- **What this does NOT overturn.** The distinct-answers table (92.1% → 6.6%)
  and the cross-family replication are untouched — they rest on the stored
  columns of the frozen rule, which is unchanged. What is now in question is
  how well the *frozen rule's labels* track truth, which is a Limitations
  matter, not a retraction. **Do not withdraw the perception claim on this;
  do state the corrected range.**
- **THE THREE SENTENCES TO USE, agreed with the user 2026-08-12. Copy these
  rather than re-deriving phrasing:**
  1. *"Human audit shows the frozen scorer has substantial label noise in
     both directions; corrected accuracy is uncertain, roughly 44–61%, with a
     point estimate about 51%."*
  2. *"Corrected AUROC ranges from 0.57 to 0.73 depending on how false passes
     are treated."*
  3. *"The automatic scorer is noisy, especially for long and multi-part
     answers. Entropy predicts frozen-scorer failures; human truth is
     harder."*
  **Do not present corrected accuracy as a clean improvement, and do not
  claim the corrected AUROC.**
- **THE PAPER DELIBERATELY OMITS THE ~51% POINT ESTIMATE, AND THIS IS NOT AN
  OVERSIGHT TO "FIX".** `paper/main.tex` reports corrected accuracy ONLY as
  *"an audit-sensitive range, roughly 44–61% under conservative false-pass
  assumptions."* Sentence 1 above keeps the point estimate because this file
  is the internal record; the paper drops it because the figure moves with a
  coding judgement — the first 40-item pass implies ~51%, the extra 60 implies
  ~68%, and the two disagree at **p=0.00007**. The range is the finding; any
  point inside it is not. **Do not reintroduce a single corrected-accuracy
  number into the paper**, and note that `paper/check_numbers.py` will not
  catch it, because the guard is a verification step in the writing plan, not
  an assertion in code.
### 2026-08-12: THE EXTENSION BACKFIRED — the two coder passes disagree at p=0.00007

`reference/audit/spotcheck_extra60_qwen_strict_v1_correct_20260812.csv`
(60 items, all coded). Coverage is now **100 of the 141** correct items.

**It did not tighten the estimate. It showed that coder variation, not
sampling, is the dominant uncertainty — which is more useful and less
convenient.**

| pass | false passes | rate | Wilson |
|---|---|---|---|
| first 40 | 16/40 | **40.0%** | [26.3%, 55.4%] |
| extra 60 | 4/60 | **6.7%** | [2.6%, 15.9%] |
| pooled | 20/100 | 20.0% | [13.3%, 28.9%] |

- **Fisher exact on the 2×2: odds ratio 9.3, p = 0.00007.** Two seeded random
  draws from the same 141 items cannot differ this much by chance. **It
  survives removing the 4 known `false_pass_removed` items** (33.3% vs 6.7%,
  p = 0.0013), so those are not the explanation either.
- **The likely cause is visible in how the passes were run:** the first 40 was
  read item by item with reasoning recorded for each; the 60 was swept as
  *"all are okay except the below ones"*. That is the single-coder
  reliability limitation this file already lists, now **measured** rather
  than hypothesised.
- **DO NOT QUOTE THE POOLED 20% AS SETTLED.** Pooling two passes that
  demonstrably disagree hides the disagreement inside a tighter interval.
  What the rate does to the headline:

  | rate used | two-sided accuracy |
  |---|---|
  | first 40 (40.0%) | 44.5–60.6%, point **51.7%** |
  | extra 60 (6.7%) | 63.8–72.8%, point **68.2%** |
  | pooled (20.0%) | 57.2–67.3%, point 61.5% |

  **A 17-point swing on a coding choice.** Baseline is 47.0%.
- **The AUROC is more robust to it.** Pooled: 0.628 [0.537, 0.719] if a false
  pass means the model was wrong, 0.754 [0.656, 0.839] if undecidable — both
  now exclude chance, where the n=40 version's first reading did not (0.572).
  **Across both the interpretive choice and the pass choice, say 0.57–0.75.**
- **THE CHEAP FIX, and it is the next thing to do:** re-read ~15 of the extra
  60 at the depth the first 40 got, prioritising captions showing a
  one-symbol span. If the rate stays near 7% the first 40 was anomalous; if
  it climbs to 30–40% the sweep was, and the pooled figure should be built
  from careful passes only. Either way it yields an **intra-rater agreement
  number, which this project currently has none of.**
- The 4 false passes found in the 60 are 4 (span `y^{(3)}`, page says the
  degree is not defined), 180 (span `zx` against the full polynomial), 289
  (span `B` against a set equality), 291 (span `x` against "infinitely many
  solutions"). All four are span-selection, consistent with the 14-of-16
  finding above. Items 132, 147 and 183 are `true_correct` but carry SymPy
  partial-parse caveats worth keeping.

- **SUPERSEDED — the n=60 extension, originally logged as the fix for the
  wide interval. It ran, and the entry above is what it found.**
- **IN FLIGHT: the n=60 extension, and it is the highest-value open work —
  ahead of the Pixtral audit and ahead of any new GPU.** The n=40 sample is
  what makes both intervals wide. `correct_item_spot_check_extension` draws
  **60 more correct items, disjoint from the first 40**
  (`reference/audit/spotcheck_extra60_qwen_strict_v1_correct_20260812.csv`,
  seed 20260812), taking coverage to **100 of the 141** correct items.
  Drawing 40 then 60 from the remaining 101 leaves the union a uniform random
  sample of 100, so the two draws pool directly rather than needing
  reconciliation. At the same 40% rate the Wilson interval tightens from
  **[26.3%, 55.4%] to [30.9%, 49.8%]** — width 29.1 → 18.9 points, which is
  what could separate the 0.57 and 0.73 readings of the AUROC. Sheets:
  notebook 23 writes both draws to **separate** Drive folders
  (`figures/spotcheck_strict_v1_correct/` and
  `.../_extra60/`); the dry run asserts they are separate, because a shared
  folder would silently overwrite the first draw's pages.
  **The new 60 contain no seeded calibration items** — all 6 known
  `false_pass_removed` items live outside this draw, so unlike the first 40
  it carries no built-in check on the coder.

### 2026-08-11: `wrong_on_both` COMPLETE — and Qwen's `genuinely_wrong` is now 99% coded

The user read the images for **all 73 `wrong_on_both` items** (Qwen and
Pixtral both `genuinely_wrong` on the same page). Coded in
`reference/audit/coded_73_wrong_on_both_20260811.csv`, 47 high / 26 medium
confidence. `pilot.failures.LABELS` has no `parse_failure`, so item 126 →
`needs_visual` and items 142/236 → `extraction_issue`.

| final label | n | share of the 73 |
|---|---|---|
| `extraction_issue` | 51 | 69.9% |
| `notation_misread` | 18 | 24.7% |
| `needs_visual` | 3 | 4.1% |
| **`true_correct`** | **1** | **1.4%** (recoded 2026-08-13) |
| `copied_wrong_line` | 0 | 0% |
| **`hallucination`** | **0** | **0%** |

### 2026-08-13: item 95 RECODED — the census is no longer one-directional

The coder re-read **item 95** and changed `needs_visual` → **`true_correct`**:
the model span `(2x - y + z)(2x - y + z)` and the truth `(2x - y + z)^2` are
equivalent and the handwritten answer is right. The truth span is itself
damaged — `\textcolor{red}{(2x - y + z(2x - y + z)}` is missing a bracket — so
`label T` fell to the text tier and the comparison failed. Applied at source in
`coded_73_wrong_on_both_20260811.csv`, so it survives regeneration.

- **It is an FN, not a TP.** `strict_v1` scores item 95 **WRONG**, and TP means
  *the scorer said correct*. Putting it on the TP sheet would recreate the
  item-5 contradiction (a caption reading `v1=WRONG` under a heading saying
  "scorer said CORRECT"); `assert_confusion_groups_match_titles` refuses it.
- **THE CENSUS IS NO LONGER TAUTOLOGICAL, and this is the interesting part.**
  The cross-audit finding was that set 1's vocabulary *could not express*
  disagreement with the rule, forcing 100% agreement. It now contains one
  `true_correct`, so `v1_tautological` flips **True → False** and agreement
  reads **23/24 = 95.8%**. **The caveat survives in substance:** the set was
  *drawn* one-directionally and 103 of 104 items still carry labels that could
  only agree, so **the pooled agreement figure stays unquotable**. That is
  asserted separately rather than inferred from the flag.
- **`corrected.py` gained `MODEL_CORRECT_LABELS`**, deliberately NOT subject to
  the parse-failure carve-out: a coder who read the page and confirmed the
  answer outranks a heuristic about the majority label.
- **The paper's number does not move.** Two-sided corrected accuracy goes
  44.2–60.6% → **44.5–60.6%** (point 51.4% → 51.7%), and `paper/main.tex` says
  *"roughly 44–61%"*, which still holds. **`main.tex` was not touched.**

**Pooled with the `only_qwen` pass: ALL 104 of Qwen's `genuinely_wrong`
items are human-coded (100%). This is a complete census of the bucket, not
a sample.**

| pooled, 104 items | n | share |
|---|---|---|
| `extraction_issue` | 77 | **74.0%** |
| `notation_misread` | 22 | 21.2% |
| `needs_visual` | 3 | 2.9% |
| `copied_wrong_line` | 1 | 1.0% |
| **`true_correct`** | **1** | **1.0%** (item 95, recoded 2026-08-13) |
| `hallucination` | 0 | 0% |

- **THE CAVEAT HAS CHANGED SHAPE, AND THIS MATTERS.** At **100% coverage
  sampling is not a limitation at all** — this is a complete census of Qwen's
  `genuinely_wrong` bucket, not a targeted sample, so the earlier "targeted
  audit, not a population rate" wording no longer describes the binding
  weakness. **The real limitations now are: (1) a SINGLE CODER with
  no second rater — inter-rater reliability is still unmeasured, but
  INTRA-rater now is: 4 hard contradictions among the 47 items coded
  determinately in two passes, see the cross-audit entry above; (2) it is
  Qwen-only — Pixtral's 130 `genuinely_wrong` items are entirely uncoded;
  (3) `genuinely_wrong` itself comes from a LOCAL rescore, which relabels
  43/300 items vs the stored columns.** State those three, not a sampling
  caveat.
- **`extraction_issue` does NOT mean "the model was right".** It means the
  failure is attributable to the scoring pipeline. Several items (39, 40,
  126, 142, 166, 188, 236) are parse failures or mangled labels where the
  model produced nothing usable. **Never convert 74.0% into recovered
  accuracy 1:1.**
- **`extraction_issue` does NOT mean "the model was right".** It means the
  failure is attributable to the scoring pipeline. Six of the 33 (39, 40,
  126, 142, 166, 188) are parse failures or mangled labels where the model
  produced nothing usable. **Never convert this share 1:1 into recovered
  accuracy.**
- **The mechanism here is DIFFERENT from `only_qwen`, and worse.** In
  `only_qwen` the key was reachable — Pixtral hit it on 21 of 30. In
  `wrong_on_both` **the ground-truth label itself is often broken, so no
  model can pass.** Verified on five: item 81 key is `eq(r, 10*cm)` (the
  radius) where the answer is `40π cm²/s`, which Qwen gave correctly; 148 key
  is bare `sympy:n`; 184 key is the interest step `15/100 × 10000 = ₹1500`
  rather than the final ₹13000, which Pixtral gave; 203 key is `eq(r,8)`, one
  root of two; 35 key is bare `sympy:7`, the radius only.
- **Item 35 is the cleanest demonstration in the whole audit:** both models
  independently emitted the *identical* answer `(-4, -5)` and both are scored
  wrong, because the key kept only `7`. Item 203 shows bug 3 firing on both
  models at once (`sympy:h*(e*(n*(c*e)))` — "Hence").
  **Consequence: a broken key biases every model equally, so it corrupts
  cross-model comparisons, not just Qwen's accuracy.** That is a stronger
  reason to care than the `only_qwen` span-selection story.
- **Real misreads roughly double vs `only_qwen`** (13.3% → 27.5%), which is
  what two architecturally independent models failing the same page should
  look like. That direction is the sanity check that the coding is tracking
  something real.
- **`hallucination` is 0 for the fourth independent time.**
- **`copied_wrong_line` has nearly vanished — 1 item across all 81 coded.**
  Items 55, 67 and 151 all moved out of it under the human read.
- **OPEN, and it bears on the auto-correction status: item 55.** The user's
  image read makes it `notation_misread` (page has `1 + tan x tan y`, model
  wrote `1 -`), contradicting the `copied_wrong_line` correction recorded
  below. **Verified against the data: `pert_a` is 84 chars and contains the
  plus form once and the minus form zero times; the minus form appears only
  in `orig_q`'s derivation.** So the claim "the page carries both lines" is
  false of `pert_a`. Whether the rendered page shows `orig_q`'s derivation
  decides it. **If it does not, item 55 returns as an auto-correction
  candidate and the "rests on 273 alone" sentence below needs revisiting.**
- **Item 151 is flagged the other way:** coded `extraction_issue` by the
  human read, but the majority sample `eq(15/2,(q+8)/2)` is verbatim an
  earlier line of `pert_a` while s0/s3 gave `q=7` correctly — the string
  evidence favours the older `copied_wrong_line`.
- **AUTO-CORRECTION: the "unsupported" verdict below rests on a FACTUAL
  ERROR about item 55, and both anecdotes survive verification.** Checked on
  both runs this session:
  - **Item 55.** `pert_a` is **84 characters**, containing `1 + \tan x \tan y`
    **once** and `1 - \tan x \tan y` **zero times**. The minus form exists
    only in `orig_q`'s derivation. So the recorded correction — *"the page's
    derivation ends `1 -` while its Answer line carries `1 +`"* — **is not
    true of `pert_a`, and `copied_wrong_line` has no line to point at.**
    Qwen wrote the minus in 4/5 samples, Pixtral in 4/5, **and Pixtral's s0
    wrote the PLUS faithfully — proving the page's plus is legible.**
  - **Item 273 is NOT retracted by the human pass.** The user coded it
    `extraction_issue` with the note "not auto-correction", but that reading
    is **Qwen-scoped** and correct for Qwen (majority label is the bare `}`;
    s4 gives a faithful 2/4, s1 gives 3/4). **The auto-correction anecdote
    was always about PIXTRAL, where all five samples give `3/4` unanimously
    against the page's `2/4`** — and the page's own prose says "the number of
    outcomes favourable to E is 3", so the model followed the page's
    reasoning over its stated answer.
  - **Honest status: back to two well-evidenced anecdotes on two model
    families with a NULL aggregate** (+6.7% [−4.0%, +17.3%]) — i.e. *real
    anecdotes, not a demonstrated mechanism*, which is the pre-correction
    position. **Do NOT write "unsupported".** Do not write "demonstrated"
    either.
- **The whole stratum is read — nothing outstanding here.** Remaining audit
  work is the 57 `only_pixtral` items; Qwen is complete.

### 2026-08-11: the `only_qwen` stratum, HUMAN-READ — and the rate is stratum-bound

The user read the contact-sheet images for **30 of the 31 `only_qwen` items**
(all but 276) and ruled on every label. Coded in
`reference/audit/coded_31_qwen_only_qwen_20260811.csv` (item, final_label,
confidence, note), 23 high confidence / 7 medium. **This is a human reading of
the images, not an OCR reading of the sheets** — unlike the 31-item audit
above, which is my reading of the Drive OCR.

| final label | n | share of this stratum |
|---|---|---|
| `extraction_issue` | 25 | 83.3% |
| `notation_misread` | 4 | 13.3% |
| `copied_wrong_line` | 1 | 3.3% |
| **`hallucination`** | **0** | **0%** |

- **DO NOT QUOTE 83% AS A POPULATION RATE. It is not one, and this stratum
  cannot produce one.** `only_qwen` is *by construction* the items where Qwen
  is `genuinely_wrong` and Pixtral is **not** — i.e. the pages a second model
  read successfully. That selects hard for "the page is legible and the key
  is reachable, so Qwen's failure is stylistic rather than perceptual". The
  honest population sentence is still the existing range: *a substantial
  share — plausibly a third to a half — of what the frozen rule counts as
  model error is the scoring pipeline.* This stratum tells you **where in
  that range the mass concentrates**, not what the range is.
- **The keys are reachable, which is the finding that reframes the rest.**
  **Pixtral scored correct on 21 of these same 30 items against the identical
  ground-truth labels.** So these are not unreachable garbage keys. Qwen is
  systematically choosing a *different but also valid* span of the page than
  the key does: the value instead of the option letter, the full sentence
  instead of the fragment, the chain instead of the final term. Write it as a
  **span-selection mismatch between model and answer key**, not as "the
  scorer is broken".
- **A concrete recurring sub-pattern worth its own sentence: multiple-choice
  answer keys.** Items 25, 93 and 173 all have GT `text:option d` while the
  model transcribes the page's derived value (10 years, 2540, 2x=10). In each
  case **one of the five samples emitted `option d` and matched**. With 52 MCQ
  items in the sample (17.3%), this is a systematic interaction, and it is
  distinct from the known `_OPTION_RE` bug (items 14, 218) — here the regex
  fired correctly and the mismatch is value-vs-letter.
- **`hallucination` is 0 for the third independent time** — text pre-sorter,
  OCR-coded 31, and now a human image pass. Safe to state plainly.
- **Two named bugs were confirmed live on specific items:** bug 1 (nested
  `\textcolor` unresolved) destroyed item 240's key into
  `eq(textcolor*((3k/2)*(r*d*e)), pm*3)` — "red" parsed as `r*e*d` — while
  sample s4 held the correct `eq(3k/2, pm*3)`; and bug 2 (decimal split)
  truncated item 52's `5887.32 cm^3` to `32 cm^3`.
- **Item 131 is a new auto-correction CANDIDATE and is not claimed.** The
  model omitted the red injected step `cos(-1710° + 1700°)` entirely while
  transcribing. Recorded in the note only. Per the standing status,
  auto-correction remains **unsupported** — item 55 was retracted to
  `copied_wrong_line`, leaving 273; this is a third anecdote, not evidence.
- **Where the text-only reading and the image disagreed, the image won, and
  it moved a label.** Item 193: the strings show three of five samples
  containing `= 75.46 m^2` with 4.9 correctly, which reads as an extraction
  failure; the image shows the model's own label using `49 * 49` where the
  page needs `4.9 * 4.9`, so it is coded **`notation_misread`**. The
  countervailing string evidence is preserved in the note rather than
  deleted. **General lesson: string evidence systematically under-detects
  notation misreads**, because a correct value elsewhere in the sample masks
  a wrong one in the answer span.

### 2026-08-11: the distinct-answers table — the result without an AUROC

`accuracy_by_distinct_answers`, `plot_accuracy_by_distinct_answers`,
`distinct_from_entropy` in `pilot/plotting.py`; figure at
`report/figures/accuracy_by_distinct_answers.png`; 5 tests.

**Accuracy vs. how many different answers the model gave (K=5):**

| distinct answers | Qwen-3B | Pixtral-12B |
|---|---|---|
| 1 (unanimous) | **92.1%** (n=38) | **87.5%** (n=56) |
| 2 | 62.9% (n=62) | 61.4% (n=57) |
| 3 | 38.0% (n=79) | 32.9% (n=76) |
| 4 | 16.7% (n=60) | 26.7% (n=60) |
| 5 (all differ) | **6.6%** (n=61) | **0.0%** (n=51) |

**Lead with this, not the AUROC.** A 13× accuracy range with no threshold, no
calibration and no free parameters, monotone on two architecturally
independent models. The AUROC is the summary statistic; this is the behaviour.
Pixtral's 4-distinct cell sits slightly above its 3-distinct one (n=60) —
noise, and the tests assert monotonicity only where it is claimed.

**THE TRAP, and it nearly went into the figure. Derive the distinct count
from the STORED `perception_entropy`, never by recomputing labels.** For K=5
the seven reachable entropies map one-to-one onto the partitions of 5, so
`distinct_from_entropy` is exact. Recomputing from raw samples gives a
**different** table for the 2026-08-02 Qwen run — **80% instead of 92% in the
one-distinct cell** — because that run was scored in a Colab session without
SymPy's LaTeX parser and 43/300 items labelled differently
(`canonicalize.latex_parser_available`). Every locked figure (AUROC 0.835,
accuracy 39.3%) comes from the stored column, so a figure built on a
recompute would contradict the paper's own headline. **Pixtral is unaffected
— its recompute matches exactly — which is precisely why the discrepancy is
easy to miss.** Locked in
`test_recomputing_distinct_counts_would_contradict_the_reported_numbers`.

### 2026-08-11: what the 300 items actually are

| field | value |
|---|---|
| `has_error` | 150 / 150 (by design) |
| `image_quality` | 205 good / **95 degraded (31.7%)** |
| `handwriting_style` | 266 / 34 |
| question type | **52 multiple-choice (17.3%)** / 248 free-response |
| question length | median 138 chars (p25 88, p75 215) |
| **answer length** | **median 449 chars** (p25 290, p75 611, max 1767) |

- **These are multi-step derivations, not short answers** — median 449
  characters — which is why final-answer extraction was needed at all.
- **Entropy is well spread**: all 7 reachable K=5 values populated, 12.7% at
  H=0 and 20.3% at ln5. **Compare InternVL3's 67% at the ceiling**, the
  degeneracy that invalidated its 0.915.

### 2026-08-11: "did you pick easy cases?" — answered, with one real caveat

`pilot.data.sample_vs_corpus`, `pilot/22_sample_representativeness.ipynb`,
`pilot/dryruns/dryrun_nb22.py`, 3 tests in `test_data.py`. No GPU.

**The structural answer: you cannot have.** `load_fermat_balanced` shuffles
with a fixed seed and takes the first 150 of each `has_error` stratum. It
**never inspects handwriting, image quality, length or model output**, so
difficulty is not selected on. Notebook 22 shows it by comparing the drawn
300 against the full corpus on every observable field — everything ≈0 except
the intended `frac_has_error` delta of −0.37.

**The one deviation, and it does cost something.** Error items are genuinely
harder to transcribe, so our 50/50 against a ~87/13 corpus flatters accuracy:

| | error items | clean items | our 50/50 | **corpus-reweighted** |
|---|---|---|---|---|
| Qwen-3B acc | 34.0% | 44.7% | 39.3% | **35.4%** |
| Pixtral acc | 34.7% | 48.7% | 41.7% | **36.5%** |
| Qwen AUROC | 0.830 | 0.839 | 0.835 | — |
| Pixtral AUROC | 0.840 | 0.824 | 0.828 | — |

- **Accuracy is ~4 points optimistic. Report the reweighted figure too.**
- **AUROC is flat across strata**, so the balance does not touch the headline.
  That asymmetry is the whole answer: the design choice affects the number we
  do not lead with, and not the one we do.
- **The clean pool is the binding constraint on any balanced design**, not our
  choice of 300 — `fermat_census`'s `max_balanced_n` is the ceiling.
- **`shuffle(seed=42)` is NOT stable across dataset revisions.** Two items once
  overlapped between draws that should have been disjoint because FERMAT's Hub
  copy drifted. **Reproducibility needs the seed AND a pinned revision, which
  we do not pin** — one line for Limitations.
- The dry run runs the REAL `load_fermat_balanced` against a stub corpus whose
  difficulty flags are independent of `has_error`, so the stratification is
  exercised rather than faked; `test_it_would_detect_difficulty_coupled_to_the_stratifier`
  proves the comparison can fail.

### 2026-08-11: the failure audit is CODED — most "model errors" are scoring

31-item audit sample from notebook 21, read via the Drive OCR of the four
contact sheets. Labels in `reference/audit/coded_31_qwen_20260811.csv` with a
per-item confidence and note. **This is my reading of the OCR, not a human
reading of the images — 22/31 high confidence, 7 medium, 2 low.**

| final label | all 31 | excluding the artifact stratum (27) |
|---|---|---|
| `extraction_issue` | 18 (58.1%) | **14 (51.9%)** |
| `notation_misread` | 7 (22.6%) | 7 (25.9%) |
| `copied_wrong_line` | 4 (12.9%) | 4 (14.8%) |
| `needs_visual` (still) | 2 (6.5%) | 2 (7.4%) |
| `hallucination` | **0** | **0** |

- **DO NOT quote 58% as a population rate.** The sample deliberately
  oversamples `likely_scoring_artifact` (a whole stratum by construction).
  Excluding it gives ~52%, which is still not a clean population estimate
  because the remaining strata are also purposive. The defensible statement
  is a RANGE: the text-only pre-sorter says **≥15%** and the coded audit says
  roughly **half**, so the truth is somewhere between and the honest sentence
  is *"a substantial share — plausibly a third to a half — of what the
  frozen rule counts as model error is the scoring pipeline."*
- **`hallucination` stays 0** under human coding too.
- **The pre-sorter agreed with the coding on only 42%.** It is a triage tool,
  not a labeller — and the disagreement is almost entirely one-directional:
  16 of 18 were `needs_visual` → a real label, i.e. it deferred rather than
  guessed wrong. That is the designed behaviour and the reason
  `proposed_label` and `final_label` are separate columns.

**THIS CORRECTION IS ITSELF WITHDRAWN (2026-08-11) — it was factually
wrong.** It read: *"item 55 is NOT auto-correction — the page's derivation
ends `1 - tan x tan y` while its Answer line carries the injected
`1 + tan x tan y`, so the model copied a real line of the page, the wrong
one"*, and concluded that auto-correction rests on item 273 alone and should
be treated as **unsupported**.

**Verified against the run data: `pert_a` for item 55 is 84 characters and
contains `1 + \tan x \tan y` ONCE and `1 - \tan x \tan y` ZERO times.** The
minus form exists only in `orig_q`'s derivation. There is therefore no minus
line on the page for the model to have copied, and `copied_wrong_line` is not
available as an explanation. **Pixtral's s0 transcribed the plus faithfully,
so the page's plus is legible**; Qwen wrote the minus in 4/5 samples and
Pixtral in 4/5. **Item 273 also stands** — the human pass coded it
`extraction_issue`, but that reading is Qwen-scoped, and the anecdote was
always about Pixtral, where 5/5 samples give `3/4` against the page's `2/4`.

**Honest status: two well-evidenced anecdotes on two model families, with the
aggregate still null (+6.7% [−4.0%, +17.3%]) — real anecdotes, not a
demonstrated mechanism.** Do not write "unsupported"; do not write
"demonstrated". See the 2026-08-11 `wrong_on_both` entry for the full check.

**Item 161 is the nearest thing to a real auto-correction:** the page's
condition repeats `x` (`{(x,y,x)}`) and the model wrote `z`, normalising
FERMAT's internal inconsistency. Coded `notation_misread`, medium confidence.

**Still outstanding:** a human pass on the images themselves. The OCR is
lossy and 2 items (240, 228) could not be decided from it at all.

### 2026-08-11: `genuinely_wrong` broken down — and hallucination is ZERO

`pilot/failures.py`, `pilot/21_failure_audit.ipynb`,
`pilot/tests/test_failures.py` (8 tests), `pilot/dryruns/dryrun_nb21.py`.
Offline. Answers the reviewer question *"are these real model mistakes or
parser artifacts?"*

| proposed label | Qwen (104) | Pixtral (130) |
|---|---|---|
| `extraction_issue` | 16 (15.4%) | 19 (14.6%) |
| `copied_wrong_line` | 6 (5.8%) | 1 (0.8%) |
| `hallucination` | **0** | **0** |
| `notation_misread` | 23 (22.1%) | 35 (26.9%) |
| `needs_visual` | 59 (56.7%) | 75 (57.7%) |

- **`hallucination` is 0 on both models** — the model never invents content
  unrelated to the page. State it; it is the failure a reviewer most fears
  from a VLM asked to transcribe.
- **~15% of `genuinely_wrong` is STILL the scoring machinery**, after all
  four rules, and it replicates across families (15.4% vs 14.6%). **A lower
  bound** — text signals only see what text can see.
- **~57% `needs_visual` on both.** The honest remainder; that is what the
  notebook's coding sheet is for.
- **These are PROPOSALS, not verdicts.** `coding_sheet` keeps
  `proposed_label` and `final_label` in separate columns on purpose, so the
  pre-sorter's own error rate is measurable and reportable. Report counts
  from `final_label`.

**Two over-triggers found and fixed, both on real data — the number moved
34% → 23% → 15% for `extraction_issue`:**

1. **Short numeric ground truths.** Flagging any label ≤2 chars as degenerate
   called `'4'`, `'25'`, `'91'` scoring failures, but a short *number* is
   plausibly the real answer. Only a stray single *letter* is the bug-3
   signature (`sympy:a`, `sympy:p`).
2. **SymPy's own function names read as prose.** `eq`, `tan`, `log`,
   `integral` are bare words, so the prose detector fired on SymPy's output
   and labelled **item 55 — the confirmed auto-correction case, a REAL model
   error — as a parser artifact.** That direction flatters the method, which
   is why it matters. Fixed with a `_SYMPY_NAMES` whitelist; item 55 now
   lands in `notation_misread` (96% similar, a sign flip).

**That sensitivity is the point:** a pre-sorter that confidently mislabels is
worse than none, because its counts get quoted. Both fixes are pinned in
`test_failures.py`.

**Also fixed by the dry run:** the final-counts cell read the empty
`final_label` column back as `NaN`, and `str(NaN)` is `"nan"`, so every
uncoded row counted as coded and it reported "coded 31/31" against an empty
table. `fillna("")` before the comparison.

**STILL NOT DONE: the human pass.** Notebook 21 writes a 31-item coding sheet
and captioned contact sheets to
`Drive/uncertainty-math-vlm/figures/failure_audit/`. Nobody has filled in
`final_label`.

### RUN 2026-08-11: verbalized confidence FAILS on perception; the \boxed{} control GATED

Both n=300, Qwen2.5-VL-3B, same sample/seed/K as the reference run. CSVs are
Drive-only (`confidence_perception_full_n300_...`, `boxed_perception_full_n300_...`);
the push 403'd as always.

**Notebook 20 — the result. Asking the model does not work.**

| score | AUROC [95% CI] |
|---|---|
| perception entropy | **0.807** [0.757, 0.854] |
| −verbalized confidence | **0.528** [0.463, 0.593] |
| **paired difference** | **+0.279 [+0.194, +0.359], resolved** |

- **Headline sentence:** *the model states a median 95% confidence while its
  transcription self-report carries no signal at all (AUROC 0.528); entropy
  beats it by +0.279 [+0.194, +0.359].*
  **CORRECTION:** an earlier version of this entry said "right 42% of the
  time". **That 42.0% is notebook 19's accuracy, from the gated boxed run —
  a different generation.** Notebook 20 never printed its own accuracy.
  **FILLED IN 2026-08-11 from the downloaded CSV: 45.0% (135/300), and
  45.8% (135/295) on the items the confidence comparison actually uses** —
  all 5 unparsed-confidence items are wrong ones, so restricting to the
  usable subset flatters accuracy slightly. Against the unconstrained
  reference run's 47.0%: asking for a confidence number costs ~2 points of
  transcription accuracy, well inside noise at this n and NOT a finding.
  Pinned in `test_accuracy_is_recorded_so_it_is_never_taken_from_another_run`.
- **Not a degeneracy artifact** — 37 distinct values, range 68–100. The model
  varies its self-report; it just varies uninformatively. The pre-written
  "near-constant self-report" caveat does NOT apply here.
- Confidence parsed on 93.8% of samples, 295/300 items usable.
- **Converges with the token-logprob result** (+0.297 [+0.214, +0.380]): two
  independent cheap-confidence baselines both land at ~0.52–0.53. Together
  they answer "why not just ask / just use logprobs?" for the arm the paper
  leads with.

**Notebook 19 — GATED, and its AUROC must not be used.**

- `\boxed{}` compliance **33.5%** against the registered 80% bar →
  `gated_low_compliance`.
- **The manipulation backfired: multi-tier extraction rose to 72.7%**, worse
  than the 51.0% baseline, because some samples box the answer and some do
  not — a NEW tier switch.
- It printed AUROC 0.810 [0.762, 0.855]. **Do not quote it.** The run is
  gated; using a number from it is exactly what the registered bar exists to
  prevent.
- **A pre-flight that only prints can be ignored, and was.** The cell reported
  `contains \boxed{}: False` and the session still ran 300 items. It now
  probes 5 items and RAISES below 60% compliance. Fixed for 7B, which is the
  retry.
- **7B ALSO GATED: 2/5 greedy probes.** The pre-flight refused and the
  session was never spent, which is the fix working. **Box-constrained
  extraction is unavailable on models this size** — write it exactly that
  way: *attempted on Qwen-3B and 7B, failed its compliance gate on both*.

**And then the confound was addressed for free, without the run.** Restrict
to items where the extractor fired a SINGLE tier across all five samples, so
tier-switching cannot contribute to the entropy by construction:

| | n | AUROC [95% CI] |
|---|---|---|
| Qwen, all items | 300 | 0.835 [0.787, 0.879] |
| Qwen, one tier only | 147 | **0.845** [0.778, 0.904] |
| Pixtral, all items | 300 | 0.828 [0.782, 0.871] |
| Pixtral, one tier only | 206 | **0.853** [0.799, 0.902] |

**The signal is not weaker where extraction could not vary — it is marginally
stronger, on both models.** If extractor instability were generating it,
removing the switching items would degrade it. Locked in
`test_the_signal_holds_where_extraction_was_already_deterministic`.

**State it as evidence against the confound, not as its elimination.** It is
a subset analysis rather than a manipulation: it cannot rule out WITHIN-tier
variation (item 9 has all five samples in `display_math` and still
disagrees), and the subsets differ in difficulty (accuracy 46.9% vs 32.0% on
Qwen). **Lesson: the cheap conditional version of an experiment is worth
computing BEFORE proposing the GPU version** — here it answered the question
better than the run would have, since the run gated twice.

### 2026-08-11: two offline controls — one useful, one that stays a hypothesis

Both free, no GPU, on the existing n=300 runs for Qwen-3B and Pixtral-12B.

**#1 `image_quality` is a CONTROL, and it passes.** FERMAT's field is
documented (arXiv 2501.07244) as *True if good, False if illumination,
shadow, blurring or contrast issues*. The flags are identical across both
runs and near-independent of `has_error` (r = 0.02, 0.06), so stratifying is
not confounded.

| | acc (Qwen / Pixtral) | AUROC Qwen | AUROC Pixtral |
|---|---|---|---|
| `image_quality=True` (n=205) | 39.0% / 42.0% | 0.843 [0.785, 0.894] | 0.803 [0.743, 0.859] |
| `image_quality=False` (n=95) | 40.0% / 41.1% | 0.818 [0.729, 0.896] | 0.883 [0.813, 0.943] |

**Degraded images are transcribed just as accurately as clean ones, and
entropy works equally well on both.** Every interval overlaps; the two models
even lean in opposite directions. `handwriting_style` is the same story
(266/34 split, nothing resolves) but its semantics are vaguer, so lean on
`image_quality`. **Use this to foreclose "is your signal just detecting
blurry images?" — it is not.** That is worth more than the "entropy helps
most when the input is hard" story we went looking for, which is simply not
there.

**#2 The auto-correction hypothesis is NOT established. Do not report it.**
Direct test: for `has_error=1` items, pull the `\textcolor{red}{}` payloads
(brace-balanced) and ask whether the model reproduces the *injected* numeric
token as faithfully as a non-red control token from the same page.

- **A first version scored 97.5% and was meaningless** — searching for a
  1-character token like `2` as a substring hits almost any transcription by
  chance. Restrict to distinctive tokens.
- ≥3 characters: Qwen **−8.5% [−17.0%, −2.1%], resolved**; Pixtral −1.6%
  [−9.5%, +6.0%], not resolved.
- ≥4 characters: neither resolves (n falls to 28 vs 20).

So one of four analyses resolves, on one model, **at a length threshold
chosen after seeing that ≥1 was contaminated**. That is the
multiple-comparisons trap this project keeps guarding against. Combined with
the earlier aggregate null (+6.7% [-4.0%, +17.3%] on both models), the honest
status is unchanged: **items 55 and 273 are real anecdotes, not a
demonstrated mechanism.** What is new is a sharp instrument — red-span token
reproduction with a matched control — for a *pre-registered* test on a future
run, at ≥3 characters, fixed in advance.

### 2026-08-11: Math-Verify — a better checker, and still not the scorer

`pilot/mathverify.py`, `pilot/18_mathverify_audit.ipynb`,
`pilot/tests/test_mathverify.py` (21 tests), `pilot/dryruns/dryrun_nb18.py`.
Offline, no GPU. `math-verify` is an OPTIONAL dependency; the tests skip
without it.

**LIMITATIONS SENTENCE, agreed with the user:** *symbolic equivalence tools
are valuable for audit but unsafe as automatic scoring on this benchmark,
because the injected errors are often units, coefficients, or notation
details that an equivalence checker is designed to erase.*

- **On isolated `$`-wrapped pairs it is excellent.** Passes all 10 sanity
  cases: resolves the string rule's false negatives (`2^3=8` vs `2^{3}=8`,
  `3x^2=-4y` vs `x^2=-4y/3`, `11130 cm` vs `11130 \, cm`) **and rejects both
  confirmed injected errors** (items 55, 273). That rejection is the property
  that makes it usable for triage at all — re-run
  `test_the_injected_errors_are_rejected` before trusting any new version.
- **As a scorer it is not safe here.** Driving the score with it: Qwen
  63.3% → 71.0%, Pixtral 54.3% → 65.7% — but **~half the newly-accepted
  items are false positives, concentrated on `has_error=1`.** Item 108
  (`y²/(9/4)` vs `y²/(11/4)`, different coefficients, matched on the trailing
  chain), item 71 (truth's answer is `dy/dx = 1`, matched a `π` in an aside),
  items 30/47 (units dropped). Locked in
  `test_math_verify_accepts_items_it_should_not`.
- **THE GOTCHA — `parse()` must get a `$`-delimited string.** Bare input
  silently falls back to plain-number extraction: `parse("x = 5")` → `[5,'5']`,
  so `x = 5` and `z = 5` compare **EQUAL**. `parse("$x = 5$")` → `[Eq(x,5),…]`
  and they compare unequal. **This produced every false positive in the first
  three runs of the analysis and made the numbers look far better than they
  were.** Always use `mathverify.mv_parse`, never `parse()` directly.
- **Two more self-inflicted errors worth remembering.** (1) An early version
  asked whether *any* of the K samples matched rather than the majority —
  inflating 80.3% → 86.7%; `mv_score_item` mirrors `majority_cluster`.
  (2) Running Math-Verify end-to-end (its extraction *and* comparison) scored
  **worse** than the string rule, 17/30 vs 18/30 — its answer-extraction
  heuristics differ from ours on multi-step derivations. Feed it *our* span.
- **The supported use is `disagreement_queue`** — 64 rows (Qwen) / 59
  (Pixtral), written to `Drive/figures/mathverify_audit/`. Review, never
  apply. It has already earned its keep: it surfaced item 40, where our own
  extractor emitted a parse failure. **`mv_majority_span` is Math-Verify's
  majority cluster, not the string rule's** — a row whose two spans look
  identical usually means the two picked different representative samples.

### 2026-08-10: the `genuinely_wrong` audit (offline half) — where the real errors are

Decided with the user: extractor work is **done and closed**; the remaining
explanatory mass is the `genuinely_wrong` bucket (Qwen 104, Pixtral 130),
which nobody had read. Offline analysis on both n=300 runs, no GPU.

- **73 items (24% of the sample) are wrong on BOTH models.** 31 Qwen-only,
  57 Pixtral-only, 139 wrong on neither. The 73 are the high-value set: two
  architecturally independent models failing the same page is either genuinely
  hard handwriting or a scoring problem that survived all four rules.
- **On those 73, the two models agree with each other on only 22.** So they
  mostly misread the same page in *different* ways — that is a genuine
  perception failure, not a shared scoring artifact. Ground-truth length is
  the same for pages both models fail (median 439 chars) and pages neither
  fails (426), so it is **not** simply "longer answers are harder".
- **Confidently wrong items exist and are the ones to read: 3 (Qwen) and 6
  (Pixtral) are unanimous across all 5 samples AND wrong.** Plus 20/24 at
  entropy < 0.7. These are the cases entropy cannot help with by construction.
- **NEW MECHANISM, confirmed on a specific case: the model silently
  "corrects" the injected error while transcribing.** Item 55's `pert_a` is
  `\textcolor{red}{\frac{\tan x + \tan y}{1 + \tan x \tan y}}` — FERMAT's
  injected error is the sign flip. Both models transcribed
  `1 - \tan x \tan y`, the textbook-correct identity. **The model's prior over
  a familiar formula overrode what is actually on the page.** This is the
  perception-arm analogue of the reasoning arm's documented
  "pattern-match the procedure instead of recomputing" failure.
- **But the aggregate version is NOT resolved, and must not be reported as a
  finding.** `genuinely_wrong` rate on has_error=1 vs clean items: Qwen 38.0%
  vs 31.3%, Pixtral 46.7% vs 40.0% — **both +6.7% [-4.0%, +17.3%], spanning
  zero.** Direction is consistent across two model families, which is
  suggestive, but per this project's standing rule that is a hypothesis to
  pre-register, not a measured effect. Do not write "the model systematically
  auto-corrects injected errors."
- **A FOURTH extractor bug, found by the audit and NOT fixed: `_OPTION_RE`
  misses LaTeX-wrapped option letters.** It needs `option` then a *bare*
  letter, so `Option $\text{A}$` (item 218) and `Option \textbf{D}` (item 14)
  do not match, the ground truth falls through to a later tier, and item
  218's truth becomes `sympy:a` while both models correctly answered
  `option a`. **Only 2/300 items — but both sit in the unanimous-and-wrong
  set, i.e. 2 of the 9 highest-value pages in the audit.** Small in
  aggregate, large in the population we most wanted to understand. Left
  unfixed by decision (extractor work is closed); recorded so the two items
  are not mistaken for model failures.
- **Contact sheets** (`pilot.plotting.contact_sheet`, notebook 17 §6) write
  17 PNG pages to `Drive/uncertainty-math-vlm/figures/genuinely_wrong/`:
  `unanimous_wrong_*` (1 page each), `wrong_on_both` (7), `only_*` (3 + 5).
  **Item 218 is unanimous-and-wrong on BOTH models** — 10/10 samples, zero
  entropy — and is the single most informative page, precisely because it
  turned out to be the `_OPTION_RE` bug rather than a misread.
- **READ 2026-08-11 via the Drive OCR of the contact sheets: the
  unanimous-and-wrong set is MOSTLY SCORING, NOT THE MODEL.** Of the 8
  distinct items (218 appears on both models), roughly 6 are scoring
  failures. **All three of Qwen's are:** 176 (`eq(x,4)` vs bare `4`), 208
  (see below), 218 (`_OPTION_RE`). Pixtral's: 14 (`Option D` — and option D
  *is* 4.5), 218, 280 (`11130` vs `11130cm`) are scoring; 138 (read `f+g`
  as `f∘g`) and 180 (garbled) are genuine misreads. **Entropy was silent on
  most of these because the models were RIGHT and agreed with each other.**
  Good for the method, bad for the label: `genuinely_wrong` is somewhat
  inflated.
- **A FIFTH scoring issue, from item 208: SymPy does not normalise an
  equation scaled by a constant.** `eq(3*x**2,-4*y)` and `eq(x**2,-4*y/3)`
  are the same parabola and compare unequal. Scanned across all
  `genuinely_wrong` items: **Qwen 3 (99, 151, 208), Pixtral 0.** Locked in
  `test_mathematically_equivalent_labels_are_scored_wrong`.
  **Gotcha that cost a wrong first count:** `normalize_string` lowercases
  labels, so a label is `eq(...)`, not SymPy's `Eq(...)`. An `isinstance(...,
  sympy.Eq)` check silently never fires and the scan under-reports (it said
  1). Parse the `eq(a,b)` form by hand.
- **SECOND confirmed case of the auto-correction mechanism: item 273.**
  Ground truth `P(E) = \frac{\textcolor{red}{2}}{4}` — the red 2 is the
  injected error, and the page's own working says "the number of outcomes
  favourable to E is 3". Pixtral transcribed **3/4**. So the mechanism now
  has two independent instances on two model families (55 on both, 273 on
  Pixtral). **Still not a resolvable aggregate effect** (+6.7% [-4.0%,
  +17.3%]) — a better-evidenced hypothesis to pre-register, nothing more.
- **How the images were finally read, since inline rendering is lost:**
  `mcp__claude_ai_Google_Drive__search_files` returns an OCR
  `contentSnippet` for each contact sheet, which carries the transcribed
  handwriting and every caption. That is enough to audit without
  downloading. **Do NOT download the sheets** — at 0.5–1.2 MB the base64
  runs to ~180K+ tokens.
- **Still unmet: nobody has looked at a handwritten page.** Every conclusion
  above is from strings. See the image-sync note below.

**Colab image outputs do not survive the sync to the local checkout.** The
2026-08-10 notebook-17 run has zero `image/png` outputs in the file — every
`display_data` is a tqdm/HF widget view — even though `plt.show()` is called
and no cell errored. Verified not a repo cause: no `.gitattributes`, no git
filters, and no commit of that notebook has ever contained an image. So
**never rely on inline rendering to carry evidence out of Colab** — write
PNGs to Drive (`show_item` already saves per item) and prefer a single
contact-sheet file for anything meant to be reviewed later.

### 2026-08-10: every item classified by WHY it scored — and bug 3's real damage

`pilot.rescore.classify_scoring_outcome` + `scoring_category_summary`,
`pilot.plotting.plot_scoring_categories`, `pilot.rescore.format_trace`.
Notebook 17 rebuilt around them, now on **both Qwen-3B and Pixtral-12B**
(same 300 images). Offline, no GPU.

Notebook 17's original four buckets **overlapped** (an item could be both
`cosmetic` and `tier_unstable`), so they could show an example but not say
what the population is made of. Seven mutually exclusive categories, total:

| category | Qwen-3B | Pixtral-12B |
|---|---|---|
| `correct_robust` | 134 | 118 |
| `genuinely_wrong` | 104 | 130 |
| `scope_mismatch` | 27 | 25 |
| `cosmetic_mismatch` | 22 | 11 |
| `bug_fix_recovered` | 6 | 9 |
| `false_pass_removed` | **6** | **6** |
| `broken_by_relaxation` | 1 | 1 |

- **What replicates across model families, stated precisely.**
  `scope_mismatch` is comparable (27 vs 25) and is the real cross-family
  result — a property of the **scoring pipeline**, not of one model.
  `false_pass_removed` is 6 on both models but **only 2 items overlap**
  (Qwen `[2, 31, 37, 117, 239, 294]`, Pixtral `[2, 17, 106, 107, 117, 200]`),
  so the matching count is mostly coincidence. **An earlier version of this
  file called it an "identical 6-item" category, which reads as the same six
  items and is wrong.** Do not cite the count as replication evidence.
- **`false_pass_removed` is the cell to read first, and it shows bug 3 was
  worse than the two example strings suggested.** Qwen items are
  `[2, 31, 37, 117, 239, 294]`. Item 31's model text is about the rectangle's
  *length* and the ground truth about its *perimeter* — **both canonicalize
  to `sympy:2*(c*m)`** and the frozen rule scored it CORRECT. Items 117 and
  239 collapse to `sympy:p` and `sympy:l`. SymPy grabbed one letter out of
  prose and everything matched. Locked in
  `test_false_pass_items_are_the_bug_3_collapse`.
- **Correction to an earlier assumption: item 101 is `genuinely_wrong`, not
  `false_pass_removed`.** It was the motivating false pass, but only under a
  relaxed rule applied *before* the decimal fix existed; with the fix it is
  wrong under every rule. `false_pass_removed` means *the frozen rule scored
  it correct*, which item 101 never did.
- **Two categories exist only because the rules are NOT monotone** —
  `false_pass_removed` and `broken_by_relaxation`. A cumulative scheme folds
  both into "correct" and hides the two places scoring goes backwards.
  Separately, `later_regression` is its own column (Qwen items 129, 206):
  one label cannot honestly carry both "first fixed at v3" and "broken at v4".
- **Attribution by single-fix ablation** (each fix applied alone): Qwen
  `textcolor` 6, `sympy_prefix` 5, `decimal` 1; Pixtral 8 / 6 / 1.
- **Tier instability is a FLAG, not a category** — it cross-cuts everything.
  **Both denominators are pinned in the test because they are easy to
  confuse: 41.8% vs 66.7% grouping by the loosest rule's verdict, 41.1% vs
  59.8% grouping by the frozen rule's.** Quoting one against the other
  denominator is the specific mistake to avoid.
- **The tier-instability confound does NOT fully replicate on Pixtral, which
  weakens it as a Limitation.** Multi-tier share among items that end
  correct vs wrong: Qwen **42% vs 67%**, Pixtral **33% vs 29%** — flat, and
  Pixtral's 3-tier group is its *most* accurate (0.500 vs 0.417 at one tier).
  Pixtral also has far less of it: 94/300 multi-tier vs Qwen's 153/300.
  **What replicates is entropy rising with tier count** (Pixtral 0.822 →
  1.023 → 1.130); what does not is the coupling to *correctness*. Say
  "extractor instability appears in Qwen and does not fully replicate in
  Pixtral", not "part of the perception signal is extractor noise".
- `format_trace` names every stage and adds the missing line: **the column
  where the two labels first diverge**, with both characters by `repr`. On a
  cosmetic mismatch the labels look identical on screen otherwise.

### 2026-08-10: E-AURC + Coverage@Risk — the deferral comparison, offline

Added `oracle_aurc`, `e_aurc`/`e_aurc_on_grid` (inside `aurc()`'s dict) and
`coverage_at_risk` to `pilot/plotting.py`. No GPU, no new run — scored on
`results/confidence_k10_...20260804T203938Z.csv`. Suite 385 → 411.

**Why E-AURC:** raw AURC is dominated by the base error rate (0.493 here), so
it cannot compare models with different accuracies. E-AURC subtracts what a
perfect ranking would score at that same error rate (`r + (1-r)ln(1-r)`).

| score | AUROC | AURC | E-AURC | E-AURC (on grid) | op pts |
|---|---|---|---|---|---|
| perception entropy K=10 | 0.806 | 0.242 | **0.093** | 0.106 | 34 |
| perception entropy K=5 | 0.761 | 0.329 | 0.180 | 0.114 | 7 |
| −mean token logprob | 0.521 | 0.510 | 0.361 | 0.360 | 299 |
| −min token logprob | 0.423 | 0.576 | 0.427 | 0.426 | 300 |

- **Deferring by token confidence is worse than deferring at random** —
  `improvement` (base error − AURC) is *negative* for both logprob scores
  (−0.017, −0.083). Entropy is +0.251. Sharpest form of that negative result.
- **Coverage@Risk is the practitioner-facing version.** Entropy hits every
  target: risk≤10% → 21.3% coverage (64 items, actual 9.4%); ≤20% → 43.7%
  (131); ≤30% → 57.0% (171); ≤40% → 74.7% (224). **`-mean-logprob` cannot
  hold error under 40% at ANY coverage.**
- **New nuance, and the reason `e_aurc_on_grid` exists: K=10's gain over K=5
  is resolution, not a better ordering.** Against the *continuous* oracle
  E-AURC halves (0.180 → 0.093), but against an oracle held to each score's
  own coverage grid the two are nearly identical (0.114 vs 0.106). More
  samples buy operating points (7 → 34), not discrimination. Consistent with
  the AUROC result, since ties mechanically cap AUROC too. **Don't describe
  K=10 as ranking better — describe it as offering finer control.**
- Validation: a perfect ranker scores `e_aurc_on_grid` **exactly 0**; the
  n=300 perception AURC reproduces the report's 0.410 / 0.607 unchanged.
- `coverage_at_risk` scans *all* operating points, not the first crossing —
  `risk_kept` is not monotone in coverage on a coarse grid. Returns
  `achievable=False` rather than the nearest point, which is a real and
  common K=5 outcome, not an error.

### RUN 2026-08-10: the scoring rule undercounts correct reads — three real extractor bugs

Prompted by the user asking to see the 4-stage pipeline (model output →
`parse_transcription` → `extract_final_answer` → compare against `pert_a`)
on real examples. It does not hold up as a point estimate of accuracy, and
it does hold up as a ranking signal. New: `pilot/rescore.py`,
`pilot/17_scoring_inspection.ipynb`, `pilot/tests/test_rescore.py` (44
tests), `pilot/dryruns/dryrun_nb17.py`. Suite 342 → 385, no existing test
touched.

**Four versioned rules, all measured on the n=300 Qwen-3B run:**

| rule | correct | acc | AUROC [95% CI] | at max-H |
|---|---|---|---|---|
| `strict_v1` (frozen, as-run) | 141/300 | 47.0% | 0.850 [0.806, 0.890] | 50 |
| `fixed_v2` (all 3 bugs fixed) | 141/300 | 47.0% | 0.838 [0.792, 0.881] | 53 |
| `relaxed_v3` (+ cosmetic) | 162/300 | 54.0% | 0.802 [0.752, 0.849] | 37 |
| `final_term_v4` (+ last `=` term) | 190/300 | 63.3% | 0.817 [0.765, 0.863] | 21 |

- **The headline survives.** Every rule excludes chance, every `ci_low` >
  0.70. A signal that existed only because of a strict string comparison
  would not decay this gracefully. **This is the reviewer answer to "isn't
  your accuracy just a scoring artifact?"**
- **FRAMING, decided with the user 2026-08-10. The extractor is NOT the
  story.** Write it as *"strict scoring underestimates accuracy, but the
  AUROC result survives every scoring rule"* — never *"the main problem was
  the extractor"*. The numbers force this: cosmetic + scope + bug-recovered
  is ~18% of items, while `genuinely_wrong` is **34.7% (Qwen) / 43.3%
  (Pixtral)**, the largest category after correct. The model really is
  wrong that often and no rule touches it.
- **ONE main accuracy number: `strict_v1`, 47.0%** — the frozen rule every
  other locked result already uses. Quote the relaxed rules as *sensitivity*
  ("relaxed scoring raises accuracy to 63.3%; AUROC stays 0.80–0.85"), not
  as a 47–63% range presented as the finding. A 16-point range offered as
  the headline invites the sharpest available attack: *"your correctness
  labels are rule-dependent, so the AUROC is measured against a noisy
  target."* The sensitivity table is the answer to that, but only if a
  single number leads.
- **Do not explain `fixed_v2` moving up on Pixtral (0.828 → 0.851) and down
  on Qwen (0.850 → 0.838).** The intervals overlap heavily; this is noise,
  and the project's standing rule is to check resolvability before
  reaching for a mechanism.
- **Report transcription accuracy as a RANGE, 47–63%, never 63.3%.** 9 of
  the 30 items `final_term_v4` newly scores correct rest on a ≤2-character
  match (`1`, `5`, `24`), where a wrong answer can coincide. The other 21
  are substantive. Locked in
  `test_final_term_v4s_gain_is_partly_short_label_coincidence`.
- **Bug 1: nested `\textcolor{}{}`.** `structural_clean`'s unwrap patterns
  use `[^{}]*` for the body, so `\textcolor{red}{\hat{b}}` survives into the
  cluster label. **85/300 ground truths, 59 of them `has_error=1`** — FERMAT
  marks the *injected error* in red, so it concentrates on exactly the items
  the error label depends on. Correct implementation:
  `canonicalize.unwrap_latex_macro` (brace-balanced, guards against a longer
  macro name that merely starts with the target — `macro="text"` must not
  eat `\textcolor` or the colour becomes the answer).
- **Bug 2: decimal splitting.** `extract_final_answer`'s last-line tier
  splits on `.` as a sentence terminator, so `"the area is 75.46 cm."`
  extracts as `"46 cm"`. Fix: `extract_final_answer(..., fix_decimal_split=True)`.
- **Bug 3 (found by the user reading notebook 17's output): `parse_latex`
  silently parses a PREFIX and returns it.** `"Hence, the required number of
  words is 24"` → `sympy:h*(e*(n*(c*e)))` — SymPy read "Hence" as five
  multiplied variables, stopped at the comma, discarded the 24.
  `"40^\circ 20' = \frac{121\pi}{540}"` → `sympy:40**circ*20`, dropping the
  `=` and the whole answer. **37/300 ground truths, 44/1500 samples.**
  `canonicalize_math`'s existing guard counts *distinct* single-char symbols
  and needs >4; "hence" has exactly 4 (h,e,n,c — the e repeats), so it slips
  under. Fix: `canonicalize_math(..., strict_parse=True)` →
  `sympy_parse_is_trustworthy`, which rejects bare ≥3-letter words and any
  parse that dropped an `=` the input had.
- **Bug 3 is worse in kind than the others because it COLLAPSES.**
  `\frac{1210}{540}` and `\frac{121\pi}{540}` — different answers — both
  reduce to `sympy:40**circ*20`, so it deflates entropy as well as
  manufacturing matches. **Watch for any `sympy:` label containing
  spelled-out words; that is always this bug.**
- **Fixing them LOWERS accuracy (144 → 141 at `fixed_v2`, 166 → 162 at v3)
  and RAISES max-entropy (50 → 53).** That is the honest direction: the
  removed matches only existed because two answers had collapsed onto one
  truncated label. `fixed_v2` is the one step not guaranteed monotone.
- **Why bug 2 had to be fixed rather than noted, and the general lesson:
  relaxing a comparison is not uniformly generous.** It truncated *both*
  the ground truth and two model samples of item 101 to `"5 square meters"`,
  and a looser rule then scored that shared mangling as **correct** — a
  false pass, not a recovered read. Before trusting any relaxed number,
  check the extractor feeding it. Regression test:
  `test_item_101_is_a_false_pass_that_the_decimal_fix_removes`.
- **Open confound, SOFTENED 2026-08-10 — it does not fully replicate.** On
  Qwen, `extract_final_answer` fires a different tier on **153/300 items**
  and mean entropy rises monotonically with the count (0.685 → 1.031 →
  1.243 at 1/2/3 tiers), with multi-tier concentrated on wrong items (42%
  correct vs 67% wrong). **On Pixtral the entropy relationship replicates
  but the correctness coupling vanishes** (33% vs 29%; 94/300 multi-tier).
  So the Limitations sentence is *"extractor instability appears in Qwen and
  does not fully replicate in Pixtral"* — not "the perception signal is
  partly extractor noise".
  **153 is a lower bound** — item 9 has all five samples in the same
  `display_math` tier, three returning the conclusion and two an
  intermediate step, entropy 1.332 from samples that agree mathematically.
  Separating extractor noise from model uncertainty needs a run where the
  extractor cannot vary (e.g. requiring `\boxed{}` in the prompt), not a
  rescoring.

**Freeze discipline — the reason this is structured as versioned rules.**
`canonical_answer_label` / `structural_clean` / `extract_final_answer`'s
defaults produced every locked number and all 12 `reference/*.json`
snapshots. Changing them in place would silently invalidate those *and*
make the next run incomparable to previous ones. So the frozen entry points
keep the old behaviour, each with a docstring saying why, and
`pilot.rescore` applies the corrections as explicit alternatives reported
alongside. `test_strict_v1_is_bit_identical_to_the_frozen_pipeline`
(0/300 mismatches) is the invariant that lets the other rules exist.
Nothing in `paper/`, `report/`, `slides/` or `reference/` changed.

### RUN 2026-08-09: Pixtral-12B replicates the perception result

`pilot/16_pixtral_perception.ipynb`, same n=300 balanced sample, bf16.
`results/pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv`
(Drive-only). Locked in `pilot/tests/test_pixtral_perception.py`, snapshot
`reference/pixtral_perception_n300_20260809.json`. Verified: 0/300
mismatches recomputing entropy/correctness/parse-failures from raw samples.

- **Perception AUROC 0.828 [0.782, 0.871]** vs Qwen-3B's 0.835
  [0.787, 0.879]. Intervals overlap almost entirely — not resolvably
  different. Accuracy 41.7% vs Qwen's 39.3%.
- **Survives the control that killed InternVL3**: 0.758 excluding
  max-entropy items, 0.772 excluding both cuts, `robust=True`. Only 17% of
  items at the ceiling (InternVL3: 67%).
- **So the perception claim is now two independent families, both robust** —
  the single biggest strengthening available to the paper, and the reason
  the InternVL3 contrast lands: it scored *higher* raw (0.915) and
  collapsed to 0.556.
- Model chosen on evidence: the FERMAT paper benchmarks nine VLMs and finds
  Pixtral-124B among the strongest at reading the handwriting. Mistral is
  genuinely independent of Qwen (unlike MiniCPM-V / olmOCR, both Qwen2-VL
  fine-tunes). Llama-3.2-11B rejected as `gated=manual`; Phi-3.5-vision as
  `trust_remote_code` + custom architecture, the shape that crashed
  InternVL3.
- **Abstention precision is 51/51 = 100% here.** Do NOT report that as
  beating Qwen's 57/61 without the caveat — this project already watched
  14/14 at n=100 decay to 93.4% at n=300. Treat it as small-sample optimism
  about the tail.
- Interface verified against the published chat template before writing
  code: Pixtral supports a system role (so the Qwen system+user shape is
  used, not LLaVA's folded-in workaround), the system message must be a
  plain string, and image chunks carry no payload.
- **Notebook gap worth fixing if reused:** the gate cell suppresses the
  AUROC unconditionally, so even the n=300 run printed none — it was
  computed offline from the CSV.
- Scored into `paper/main.tex` §"Does it replicate on a second model
  family?", Setup, Limitations and the abstract; slide 2 gained a
  cross-model strip.

### RETRACTION (2026-08-09) — read this before citing any reasoning number

**The has_error=1 stratified reasoning result (0.775–0.854, "confirmed
across three model families") is withdrawn. It is an artifact.**

Within a stratum where every item has the same true label, `grading_correct`
is the *same column* as "what the model answered" — they agree on 100% of
items at 7B. Reasoning entropy is computed from the very votes that produce
that answer, so the stratified AUROC is near-circular. A signal-free biased
coin (`pilot.plotting.bias_only_null_auroc`) reaches a **higher** AUROC than
any model here: null medians 0.901 / 0.868 / 0.831 against observed
0.854 / 0.801 / 0.775.

The "sign reversal" between strata, previously the headline finding, is the
same identity with the sign flipped: in the clean stratum correctness is the
exact *complement* of the verdict, so the two stratum AUROCs sum to 1.

**What survives:**
- **Perception is unaffected**, and the reason is label cardinality. Both
  arms have ground truth; grading's is binary, so stratifying by it makes
  the truth *constant* within a group and correctness reduces to
  `majority == that constant` — a function of the vote. Transcription truth
  is a math expression from an effectively unbounded set: five samples can
  agree on `5/2(x-1)` and still be wrong because the truth was `5/2(x+1)`.
  38 items are unanimous and 3 of them are wrong; 4 max-entropy items are
  right. Note it is the *stratifying* that collapses it, not the label type
  — grouping transcription items by exact reference answer would degenerate
  identically. Low cardinality is what makes stratification both tempting
  and total, which is why the trap is specific to binary-decision tasks.
- **The honest reasoning result is the pooled one on a balanced sample:
  ~0.52, no signal.**
- ScratchMath's gate is reinforced — it is 100% error items, i.e. entirely a
  single-label stratum, so its AUROC was degenerate on this ground too.

**What none of the usual safeguards caught:** the pre-registered 0.70
threshold (the null's 2.5th percentile clears it), adequate power (both
strata had it), or replication (it reproduced the artifact three times).

Locked in `pilot/tests/test_stratum_degeneracy.py` and
`reference/stratum_degeneracy_20260809.json`. Test files asserting the old
numbers carry a RETRACTED INTERPRETATION header — their numbers are correct,
their meaning was not. `report/report.tex` Phases 4–5 still describe the old
reading and are flagged but not yet rewritten; `paper/main.tex` is correct.


**The n=300 balanced run (2026-08-02) is the current source of truth, and
`report/report.tex` has NOT been updated with it** — the user updates the
report on request only (see the conventions section). The report still
describes the n=100 state.

### n=300 balanced run — the decisive result

`results/scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv`
(Drive-only; the push 403s). 150 error / 150 clean, K=5 both arms.

- **Perception: replicated and strengthened. AUROC 0.835 [0.787, 0.879]
  pooled**, 0.830 [0.760, 0.893] in the has_error=1 stratum (the like-for-like
  comparison against n=100's 0.762), 0.839 [0.772, 0.898] on clean items.
  Robust to every artifact cut — 0.796 [0.736, 0.852] excluding both parse
  failures and max-entropy items. Pre-registered verdict: **replicated**.
- **Reasoning: dead. AUROC 0.522 [0.462, 0.582] pooled — no signal.** The
  reason is now unambiguous: **the model cannot do the grading task.** At a
  50/50 split its accuracy is 51.7% against a 0.500 baseline. It answers
  "there is an error" on 93% of items — 94.7% correct on error items, 8.7% on
  clean ones. Its old 75% was the 87/13 class imbalance flattering it.
  Reasoning entropy is also near-degenerate: 42% of items sit at entropy 0 and
  only 3 values occur in practice.
- **The stratified hypothesis could not be tested.** The model's bias is so
  extreme that only 8 of 150 error items were misgraded, below the registered
  minimum of 30. Verdict: `inconclusive_underpowered`, not the 0.836 point
  estimate. Same for the clean-stratum inversion (137 wrong / 13 right) —
  direction well supported (0.239 [0.128, 0.374]) but underpowered by the
  registered standard.
- **Replicated side findings:** digit/reasoning self-contradiction 11.0% at
  n=300 vs 12% at n=100. Parse failures 3.6% / 0.9% vs 4.0% / 1.0%. Temp-0
  anchor 0/50 non-zero. The max-entropy abstention rule degraded from 14/14
  (n=100) to 57/61 = 0.934 precision — the perfect version was small-sample
  optimism, but 93% on 61 items is still usable.
- **Takeaway for any write-up:** perception entropy works (~0.83, tight CI,
  robust). Reasoning entropy is a clean negative — *entropy over a model's
  samples cannot predict correctness when the model performs at chance*.
  That's a real finding, not a failure to measure.

### Earlier n=100 state (superseded, kept for context)

Don't re-derive findings from git history; the report is the compiled record
of the n=100 work (`report/report.pdf` is the rendered version, kept in sync
with `report/report.tex`).

**Every AUROC now carries a bootstrap 95% CI, and they changed the
conclusions.** At n=100 a single AUROC has a CI of roughly ±0.12 and a paired
difference roughly ±0.15, so most of the differences this pilot chased are
inside sampling error. Always report a CI alongside an AUROC here, and use
`bootstrap_auroc_difference_ci` (paired, same resampled items) to compare two
conditions — never "one CI excludes 0.5 and the other doesn't", which is the
difference-in-significance fallacy and is exactly how the reasoning-text
result got overstated the first time.

- **Perception entropy: working and verified, AUROC 0.788 [0.697, 0.869].**
  Two prior approaches failed first (exact-string clustering on the full
  derivation — too strict, 83% pinned at max entropy; naive NLI/embedding
  clustering — over-merged, 95% implausible correctness). Fix: extract just
  the final answer span (`pilot/canonicalize.py:extract_final_answer`) before
  clustering, not the whole derivation. Note this final path uses exact-string
  `cluster_entropy`, **not** NLI — the headline result never touches
  `pilot/semantic.py`. Survives every artifact cut (`auroc_sensitivity`):
  0.707 [0.587, 0.819] excluding both parse failures and max-entropy items.
  Side finding: all 14 items where the 5 transcriptions fully disagreed were
  wrong (precision 1.00, recall 0.24 as an abstention rule).
- **Reasoning entropy: not statistically resolved at n=100.** Digit-only is
  0.618 [0.489, 0.741] at K=5 — the CI *includes chance*. K=15 gives 0.665
  [0.514, 0.800], but the paired gain over K=5 is −0.047 [−0.178, +0.090]:
  more sampling bought nothing measurable. Text-clustering is 0.702
  [0.579, 0.817] at K=5, which beats chance on its own but **not** the
  digit-only baseline it was meant to improve on (paired +0.083
  [−0.064, +0.233]). Don't describe it as a validated improvement.
- **Reasoning entropy, text-clustering at K=15: fails, AUROC 0.520
  [0.378, 0.664].** The paired K=5 → K=15 degradation (+0.182 [+0.037, +0.330]
  in favour of K=5) is the *only* reasoning-arm comparison that resolves at
  this n. Root cause: NLI + union-find transitive cascade merging — see
  `pilot/semantic.py:nli_cluster_labels`'s docstring for the mechanism and a
  real example. **Do not use `nli_cluster_labels`/`semantic_cluster_entropy`
  at K > 5 without either re-verifying for that specific use case or replacing
  the union-find step with a stricter merge criterion** (e.g. minimum
  intra-cluster pairwise agreement fraction, not any single transitive chain).
- **Root cause of the reasoning arm's weakness (found by manual case
  inspection, 2026-08-02): class imbalance + a systematic model bias, NOT
  clustering.** The sample is 87/13 on `has_error`, so a constant "there is an
  error" predictor scores 0.87 and beats the model's 0.75 (K=5) / 0.82 (K=15).
  The model says "error" for 86/100 items and gets only 1 of 13 clean answers
  right — on 6 of those 13 all five samples agree, so entropy has no
  disagreement to detect. Reading those cases, they're arithmetic/recall
  failures (asserting `b=-9` for `2x²-4x+3`, mis-stating the distance
  formula), not prompt ambiguity. Consequence: entropy's direction *reverses*
  between strata — within gt=1 it's 0.756 [0.618, 0.876] at K=5 and 0.891
  [0.813, 0.955] at K=15, within gt=0 it's inverted, and pooling cancels them
  to 0.618/0.665. Use `stratified_auroc` before interpreting any pooled AUROC
  here. **This stratification is post-hoc** — a hypothesis to pre-register for
  the next run, not a finding.
- **n=100 is the binding constraint on everything else.** The agreed next step
  is scaling the existing 3B pipeline to 300–500 items — *before* fixing the
  cascade bug, trying 7B, or testing token-level confidence, since none of
  those can be evaluated against a ±0.15 error bar. **Rebalance the grading
  subset toward ~50/50 on `has_error` when scaling**, or the extra items just
  buy precision around a class-imbalance artifact. Also dump the pairwise NLI
  entailment matrix on the next GPU run: it makes every future clustering
  variant a free offline experiment.
- **Independent of all the above: the model self-contradicts in 12% of grading
  samples** (60/494 have `**Error:** 0` while their own `**Reasoning:**` opens
  by flagging a mistake). That's a direct count, not an AUROC comparison, so
  the CI work doesn't touch it.

### 2026-08-04 follow-up run — closes the token-confidence gap, tests prompt fixes

`results/confidence_k10_qwen25-vl-3b-instruct_20260804T203938Z.csv` (300 rows)
and `results/grading_variants_qwen25-vl-3b-instruct_20260804T203938Z.csv` (400
rows), both Drive-only. Locked in `pilot/tests/test_confidence_and_variants.py`.
Produced by `pilot/04_confidence_and_prompts.ipynb`, run on an A100 (the token-
confidence capture needs ≥24 GiB VRAM — `output_scores=True` forces eager
attention through the whole model including the vision tower, which OOMs a T4
regardless of batch size; the notebook auto-detects VRAM and skips capture
below the threshold).

- **Entropy decisively beats token confidence as a baseline. AUROC 0.835 vs
  -mean-logprob 0.537, paired +0.297 [+0.214, +0.380], resolved.** This closes
  the biggest remaining gap in the perception story — entropy isn't just
  redundant with the model's own next-token probabilities. `-min-logprob`
  comes out *below* chance (0.460) but don't report that as "token confidence
  is actively misleading" — 81% of items cluster in a narrow band that
  plausibly reflects batched-generation padding rather than real content
  uncertainty (checked: splitting on that band gives two wide, chance-
  overlapping CIs, so it's not a clean artifact story either — the honest
  conclusion is only that `-min-logprob` isn't usable, not confidently why).
  `-mean-logprob` is the trustworthy half of this comparison.
- **K=10 modestly sharpens perception, confirming the K=5 subsample
  prediction.** Scored against the *K=10-recomputed* correctness label (the
  fair comparison — scoring against the old K=5 label instead gives a
  different, less meaningful number since it's asking whether more samples
  predict a stale target): K=5 0.761 → K=10 0.806, paired +0.045
  [+0.005, +0.087], resolved. 7 reachable entropy values at K=5 → 34 at K=10.
  Side finding: the "ground truth" majority vote itself isn't fully stable at
  K=5 — some items flip `transcription_correct` between K=5 and K=10.
- **No grading prompt variant fixes the yes-bias. Screened at n=100, none
  clear the pre-registered bar.** Tested restate-the-result-first, an
  explicit "half are correct" framing, and eliciting a 0–100 confidence,
  against the FERMAT baseline. All land within a few points of the baseline's
  0.510 accuracy. The interesting negative: `restate` collapses the
  says-error rate from 94% to 38% — a huge behavioral shift — while accuracy
  barely moves (0.510 → 0.550). The model just picks a different near-constant
  answer rather than discriminating, which argues *against* "yes-bias is a
  fixable prompt artifact" and *for* a genuine capability ceiling on this
  3B model. Strengthens the case for the 7B capability check as the next
  real lever on the reasoning arm.
- **Notebook engineering note, if reused:** checkpoint filenames are keyed by
  `CAPTURE_LOGPROBS` and tag every `skipped_oom` entry with the mode it
  happened under. A skip from the (expensive, OOM-prone) capturing path is
  not evidence the (cheap, working) plain path would also fail — without the
  mode tag, a resumed run silently trusts stale all-skipped checkpoints from
  a different mode and never retries them. Found by reading a genuinely
  confusing "resuming from 22 completed items" log line, not by inspection.

### Deferral write-up + 7B capability check notebook (2026-08-05)

- **`report/report.tex` §5.4 formalizes the risk-coverage/AURC/conformal
  analysis** with a real figure (`pilot.plotting.plot_risk_coverage`,
  `report/figures/risk_coverage_n300.png`). Correct current numbers: AURC
  **0.410** against a 0.607 random-deferral baseline. **A stale 0.3417 figure
  circulated earlier in chat from an ad hoc, order-dependent computation
  before `aurc()` was fixed for ties — never use that number, it does not
  appear in the report or any test.** Conformal at a 30% risk target: 13.7%
  observed risk, 2.4% violation rate, but two-thirds of 1000 calibration
  splits couldn't reach the target at all (K=5 gives too few operating
  points — ties directly to the K=10 result).
- Dry-run gotcha worth remembering: `load_fermat_balanced` **shuffles its
  final selection** (`pilot/data.py`), so a stub/fixture must never assume
  sample order correlates with `has_error` (e.g. "first half are error
  items") — read ground truth from the real sample object instead. Caught
  by the dry run disagreeing with itself, not by inspection.

### 7B capability check — RUN 2026-08-05, result is MARGINAL

`pilot/05_7b_capability_check.ipynb` ran clean on an A100, full bf16 (not
quantized). `results/grading_7b_n300_qwen2.5-vl-7b-instruct_20260805T115859Z.csv`
(300 rows, Drive-only — push 403'd as always, downloaded via Drive API since
the push failed). Locked in `pilot/tests/test_7b_capability_check.py`.

- **Gate verdict: `marginal`.** Grading accuracy 59.0% vs a 50.0% baseline —
  better than 3B's 51.7%, but under the report's 0.65 bar. Doesn't cleanly
  resolve "bigger model fixes it" or "task is unanswerable"; `entropy_result_meaningful`
  is `False`, so the pooled AUROC (0.520, no signal) is diagnostic only.
- **The stratified pattern replicates 3B almost exactly, same sign_reversal,
  same pooled_understates.** has_error=1 stratum AUROC 0.761 [0.646, 0.860]
  (3B: 0.756) — but **still underpowered** (17 wrong vs the registered
  minimum of 30, same standard applied, not a new one). Point estimates
  matching across model size is suggestive of a stable mechanism, not a
  confirmed magnitude at either size.
- **New: the clean stratum is adequately powered at 7B for the first time.**
  3B's clean stratum was hopeless (13 items total). 7B's balanced sample
  gives it 150 items / 44 wrong — clears the n=30 minimum. Result: AUROC
  **0.280 [0.200, 0.366]**, resolvably below chance. **The inversion
  (entropy runs backwards on clean items) is now a genuinely confirmed
  finding, not just a suggestive direction** — the first stratum in this
  whole project to both invert *and* clear the power bar.
- Side improvements at 7B, real but partial: parse failures 0.1% (vs 3B's
  0.9%), says-error rate 80% (vs 3B's 93%) — same bias, less extreme, not
  gone. Consistent with the sign-reversal pattern still holding.
- **Next open question, if pursued further:** a rebalanced *grading* sample
  specifically for 7B (skewed toward has_error=1, the underpowered stratum)
  would be the direct fix, mirroring the 3B→n=300 rebalancing move. Not
  started — no GPU session run for this yet, and not clearly worth it before
  deciding whether the reasoning arm is being written up as a closed
  negative or pursued further.

### Manual case inspection (2026-08-05) + has_error=1 stratum powered (2026-08-05/06)

The user asked, before running more GPU: *why* can't the 7B model find more
mistakes — not just the aggregate rate. `pilot/06_7b_error_stratum_power.ipynb`
was rewritten to carry raw grading text through to the final CSV (checkpoints
always had it; only the final scoring step used to strip it down to
digit+entropy+correctness). Manual inspection of the 17 wrong has_error=1
cases from the reference run found two real mechanisms — digit/reasoning
self-contradiction, and pattern-matching a solution's procedural structure
instead of independently recomputing an arithmetic step (e.g. accepting
`19 × -18 × 23 = -7966` at face value; the correct value is -7866) — but also
turned up **two cases (items 40 and 161) with no findable error anywhere in
the text** — initially read as possible label noise. `pilot/07_manual_case_inspection.ipynb`
pulls the actual handwritten images for every wrong has_error=1 case
(recomputed from the checkpoints, not hardcoded) so this could be checked
visually, not just textually. It also corrected an overcount from the
initial ad hoc chat inspection: of the reference run's 17, only **2 are
truly unanimous / zero-entropy** (items 40, 161) — the other 15 had at least
one disagreeing or failed sample but the majority vote was still wrong, so
entropy did have signal available there, it just wasn't enough to flip the
vote.

**RUN 2026-08-06: notebook 07 executed, and the "label noise" reading was
wrong — both have real errors, just not visible in the text excerpt used to
screen them.** Downloaded the actual PNGs from Drive and viewed them
directly (not re-reading the text more carefully — the text truly doesn't
show it, e.g. item 40's `pert_a` field literally ends "The required number
is $y$" and that IS the answer line, not a truncation artifact). Findings:
- **Item 40**: algebra is correct (y=7), but the final line is verbatim "The
  required number is y" — the computed value is never substituted into the
  answer statement.
- **Item 161**: defines R×R×R as `{(x,y,z) : x,y,x ∈ R}` — the tuple
  introduces z, but the membership condition repeats x instead of z. An
  internal inconsistency in the reference solution itself.
- **A third unanimous case appeared** after the 200-item extension merged in
  (extra_idx185, not in the original 17 — total wrong has_error=1 is now 38,
  3 unanimous): correct arithmetic (√45 = 3√5 for a 3D distance calc)
  followed by labeling the answer "meters" — a unit the problem never gave
  for two bare coordinate points.

**Net correction: zero label noise found across all 3 unanimous cases —
all real, findable errors, just not arithmetic ones.** This sharpens the
two known mechanisms (self-contradiction, pattern-matching without
recomputing) into four: the model appears to verify a solution's *core
computation* and stop there, without checking whether the final answer
statement is complete (unsubstituted variable), internally consistent
(mismatched notation), or licensed by the problem (fabricated unit). Fixed
in `report/report.tex` §7.3 and CLAUDE.md — the report briefly stated the
label-noise reading as an open question before this was checked; that
version is wrong and has been corrected, not just appended to.

**Then the stratum-powering run itself completed and fully resolved the
has_error=1 stratum.** 200 more has_error=1 items (disjoint by construction,
`pilot.data.load_fermat_extra_error_items`) merged with the 300-item
reference run: `results/grading_7b_stratum_powered_n500_qwen2.5-vl-7b-instruct_20260805T135021Z.csv`
(500 rows, Drive-only as always). Locked in `pilot/tests/test_7b_stratum_powered.py`.

- **has_error=1: CONFIRMED, not just a point estimate anymore.** n_wrong grew
  17 → **38** (clears the registered minimum of 30 for the first time), and
  the AUROC itself rose 0.761 → **0.834 [0.768, 0.891]** — clears the
  pre-registered 0.70 threshold with a CI that excludes chance outright.
- **has_error=0 (clean) stratum is unchanged, as expected** (no new clean
  items were drawn): still 150 items / 44 wrong, AUROC 0.280 [0.200, 0.366].
- **This is the first point in the entire project where both grading strata
  are simultaneously adequately powered, each fully resolved, in opposite
  directions.** `sign_reversal=True`, `pooled_understates=True`, and both
  hold with real power now, not suggestively.
- **Don't be misled by the pooled AUROC on this specific CSV: it now reads
  0.612, above chance** — but that is an artifact of the sample no longer
  being 50/50 (350 has_error=1 vs 150 clean), which mechanically favors the
  larger, stronger has_error=1 stratum in the pooled mix. It is not evidence
  the sign-reversal problem resolved itself; `stratified_auroc` is still the
  only correct read, per the project's standing rule on this.
- Side note: says-error rate rose to 83.6% (vs the reference-only run's
  79.7%) simply because the 200 new items are all has_error=1 and the
  model's existing bias scores well on them — not a new behavioral finding.
- **Net effect on the reasoning-arm story:** within the has_error=1 stratum,
  entropy is now a confirmed, sizable (0.834) predictor of grading
  correctness at 7B. The arm is not a clean negative anymore at this
  stratum — the earlier "reasoning is dead" framing (from the pooled/3B
  numbers) was correct about *pooled* entropy on an error-biased model, not
  about entropy's usefulness in general. The honest updated framing: entropy
  works within-stratum at 7B, in both directions, but the model's own
  response bias makes the *pooled* statistic actively misleading — a
  measurement-methodology finding as much as a capability one.

### RUN 2026-08-06: the 7B result is model-size-independent — 3B confirms too

Open question after the 7B stratum-powering run: is 0.834 a 7B-specific
behavior, or does the same effect show up at 3B once it's given enough
misgraded items? `pilot/08_3b_error_stratum_power.ipynb` mirrors notebook 06
for 3B — **N_EXTRA=500, not 200**, because 3B's has_error=1 error rate
(8/150=5.3%) is roughly half 7B's (17/150=11.3%): 3B says "error" even more
often (93% vs 80%), so it needed more than double the extra items to reach
the same power. `results/grading_3b_stratum_powered_n800_qwen2.5-vl-3b-instruct_20260806T150044Z.csv`
(800 rows, Drive-only). Locked in `pilot/tests/test_3b_stratum_powered.py`.

- **CONFIRMED, and close to 7B's number (n=350).** n_wrong grew 8 → **42**
  (clears the registered minimum of 30). AUROC: 0.836 (unconfirmed point
  estimate) → **0.854 [0.796, 0.902]** — clears the pre-registered 0.70
  threshold, CI excludes chance. **7B (n=350): 0.834 [0.768, 0.891]. 3B
  (n=650): 0.854 [0.796, 0.902].** Two independent model sizes, both
  confirmed. (This "within 0.02" comparison used different total n per
  model — see the 2026-08-06 matched-n650 follow-up below for the
  apples-to-apples version, which tells the same story with a bit more
  daylight between the numbers.)
- **This answers the open question decisively: the has_error=1 stratified
  effect is model-size-independent, not a 7B-specific behavior.** It isn't
  "bigger model unlocks the signal" — both model sizes had the signal all
  along; they just needed enough misgraded items to measure it.
- Clean stratum unchanged and still underpowered at 3B (13 correct items,
  same as the original n=300 run — no new clean items were drawn; unlike
  7B, 3B's clean stratum has never cleared the power bar).
- Same pooled-AUROC trap as the 7B n=500 CSV: pooled reads 0.610 here
  (above chance) purely because the sample is no longer 50/50 (650 vs
  150) — not evidence the sign-reversal problem resolved, `stratified_auroc`
  is still the only correct read.
- Infra note: the save cell's git commit failed on this run with "Author
  identity unknown" (git user.name/email never configured on this fresh
  Colab runtime) — a different failure mode than the usual 403, caught one
  step earlier. **Fixed** in the same commit as the matched-n650 notebook
  below (`git config user.email/user.name` added to notebook 08's save
  cell, matching what notebook 06's already had).

### RUN 2026-08-06 follow-up: matching 7B to 3B's exact n=650 — still confirmed, gap wider than it looked

User's own push-back on the comparison above: 7B was confirmed at n=350,
3B needed n=650 to confirm — is that mismatch itself hiding something?
Checked for free first (no GPU): scoring 3B's own combined CSV restricted
to just the 350 items 7B had already seen gave 0.858 [0.768, 0.924],
consistent with both 3B's full-650 result and 7B's 0.834 — so the
different totals weren't distorting anything. User asked to extend 7B to
n=650 anyway for a cleaner paper table. `pilot/09_7b_match_3b_n650.ipynb`
draws 300 more has_error=1 items at `skip=350` (past 7B's existing 150+200)
and merges THREE checkpoints. `results/grading_7b_matched_n650_qwen2.5-vl-7b-instruct_20260806T172004Z.csv`
(798 rows — see below). Locked in `pilot/tests/test_7b_matched_n650.py`.

- **Real bug caught by the dry run before this ever ran on GPU:** the
  merge cell's precision check only compared `entries[0]` of the
  reference and round-1 checkpoints against the session's `QUANTIZED` —
  it never actually checked the round-2 data being merged in. A
  deliberately-corrupted test entry slipped through silently. Fixed: now
  checks every entry across all three checkpoints, not just the first
  entry of two of them.
- **Real overlap hit on the actual run, diagnosed and handled correctly:**
  2 items overlapped between round-1 (drawn under notebook 06, weeks
  earlier) and round-2 (drawn fresh this session) despite both using
  seed=42 with disjoint `skip` values — FERMAT's Hub copy appears to have
  drifted slightly in the interim, shifting `shuffle(seed=42)`'s exact
  ordering near the boundary. Diagnosed by printing the 2 overlapping
  questions (ordinary items, nothing anomalous) before dropping them from
  the round-2 side. Final n is 648, not 650.
- **Result: AUROC 0.801 [0.751, 0.846], n_wrong=73, still CONFIRMED** —
  clears the threshold and power bar. Sits inside the original n=350
  result's own CI [0.768, 0.891], so this is a refinement, not a
  contradiction.
- **Corrected framing vs. the earlier comparison:** 7B (n=648) 0.801 vs 3B
  (n=650) 0.854 — CIs still overlap (0.846 > 0.796), so the two remain
  statistically indistinguishable, but the point estimates are 0.05 apart,
  not 0.02. The "within 0.02" framing from the first comparison overstated
  how tight the agreement is once both are actually measured on
  (essentially) the same items. Conclusion unchanged either way: both
  model sizes confirm the has_error=1 effect.
- Verified independently against the real downloaded CSV before writing
  the test, same as every other result this project reports.

### RUN 2026-08-06: second model family (LLaVA-NeXT) — perception gated, reasoning promising

WACV push started (target: Round 2, enrollment Aug 21, submission Aug 28
2026 — see the plan at `.claude/plans/pilot-kickoff-instructions-glittery-eclipse.md`
for the full roadmap). Biggest reviewer-facing gap identified: every result
so far is one model family (Qwen2.5-VL, 3B/7B). `pilot/10_llava_next_fermat.ipynb`
re-ran both headline measurements on **LLaVA-NeXT** (`llava-hf/llava-v1.6-mistral-7b-hf`)
at the identical n=300 sample. The adapter (no system role — folded into
one user turn; LLaVA-NeXT's chat templates vary by base LLM and never
demonstrate system-role support in the public docs — plus the combined
`apply_chat_template(tokenize=True, return_dict=True, ...)` call, verified
against current `transformers` docs before writing any code) worked on the
first real run, no fallback needed.
`results/scaleup_n300_bal50_llava-v16-mistral-7b-hf_20260806T231143Z.csv`
(300 rows, Drive-only). Locked in `pilot/tests/test_llava_next_fermat.py`.

- **Perception: gated by a real capability gap, not usable.** Transcription
  accuracy is 3.0% (9/300) vs Qwen's ~42%. Diagnosed properly before
  concluding anything (the user specifically pushed back asking for a full
  diagnosis, not a guess): independently recomputing `transcription_correct`
  from raw text matches the stored CSV on all 300/300 rows (not a scoring
  bug); parse failures are only 9.6% (144/1500), so it's not a
  format-following failure either. **The real cause: LLaVA can read a
  multiple-choice letter but essentially cannot transcribe free-response
  handwritten derivations** — MCQ accuracy 12.3% (7/57) vs free-response
  0.8% (2/243), a ~15x gap that accounts for the entire effect. This is
  the same "capability gate" concept as the 7B reasoning check, just
  applied to the perception arm for the first time: **perception_entropy's
  AUROC (0.710, only 9 correct items) is not reportable** — an order of
  magnitude below the registered minimum of 30 used everywhere else, CI
  barely excludes chance.
- **Reasoning: promising, replicates Qwen-7B's exact pattern, underpowered.**
  Grading accuracy 50.0% (chance, same story as Qwen). Clean-stratum point
  estimate **0.283 [0.178, 0.398] — within 0.01 of Qwen-7B's CONFIRMED
  0.280.** Same bias mechanism (says "error" ~91% of the time here). Not
  confirmed: only 14 misgraded per stratum (has_error=1: 0.766
  [0.652, 0.883]; clean: 0.283), same starting point Qwen-7B was in before
  notebook 06 extended it.
- **Decision (with the user): pursue the reasoning-arm extension on LLaVA
  now (cheap, promising); hold off on perception until a different
  second-model choice is made** (LLaVA-NeXT-7B specifically doesn't clear
  the bar for that arm — needs a model with real dense-OCR competence,
  not yet chosen).
- `pilot/11_llava_error_stratum_power.ipynb` mirrors notebooks 06/09
  exactly: draws 250 more `has_error=1` items (sized off the observed
  9.3% error rate on this stratum, Wilson CI [6.1%, 14.0%], same
  point-estimate-based planning that worked for both prior extensions),
  merges with the notebook 10 reference checkpoint. Dry-run verified
  against REAL reference data (reconstructed from the already-downloaded,
  already-verified n=300 CSV — not synthetic stubs) — reproduces the
  verified 14/150 figure exactly before any GPU time is spent on the
  extension itself.

**RUN 2026-08-07: notebook 11 completed — LLaVA-NeXT's has_error=1 stratum
CONFIRMED, first cross-model-family replication in this project.**
`results/grading_llava_stratum_powered_n550_llava-v1.6-mistral-7b-hf_20260807T181623Z.csv`
(550 rows: 300 reference + 250 extra). Locked in
`pilot/tests/test_llava_stratum_powered.py`.

- n_wrong grew 14 → **34** (clears the registered minimum of 30). AUROC:
  0.766 (unconfirmed) → **0.775 [0.694, 0.848]** — clears the 0.70
  threshold, CI excludes chance. Overlaps Qwen-7B's confirmed
  [0.751, 0.846] almost entirely — the two model families are not
  resolvably different on this measurement.
- Clean stratum unchanged (14 correct, still underpowered — the
  extension targeted only has_error=1) but its point estimate (0.283) is
  itself close to Qwen-7B's confirmed 0.280. Response bias replicates
  too: says-error 91.1% (vs Qwen-3B 93%, Qwen-7B 80%).
- **This is the first confirmed replication across model families in the
  whole project.** Every prior confirmation (7B vs 3B) compared model
  *sizes* within Qwen2.5-VL; this compares architecturally distinct
  families. Locked into `report/report.tex` §Phase 5.

### RUN 2026-08-07/08: third model family (InternVL3-8B) — perception result is real but fails its own sensitivity cut

`pilot/12_internvl3_fermat.ipynb`, same n=300 balanced sample. **Built
twice.** v1 used the `trust_remote_code` recipe from InternVL3's own model
card (`OpenGVLab/InternVL3-8B`, custom `.chat()` interface, custom
image-tiling code) and crashed on the user's real run:
`AttributeError: 'InternVLChatModel' object has no attribute
'all_tied_weights_keys'` — root-caused via search (not guessed) as a
`transformers` v4-vs-v5 breaking change: v5 requires custom
`trust_remote_code` model classes to initialize `all_tied_weights_keys` via
`post_init()`, which pre-v5 custom code (InternVL3's included, also hits
Molmo/moondream/InternVisionModel per search results) never does. **Fixed
by switching to the checkpoint transformers itself now ships natively**
(`OpenGVLab/InternVL3-8B-hf`, `AutoModelForImageTextToText` +
`AutoProcessor`) rather than pinning an old transformers version — verified
against current docs (WebFetch) before rewriting, confirming the same
message-format and combined `apply_chat_template(...)` call as the LLaVA
adapter, and critically restoring standard batched `generate()` sampling
the old `.chat()` interface never had. All ~230 lines of custom tiling code
deleted. Committed as `34d5534`.

`results/scaleup_n300_bal50_internvl3-8b-hf_20260807T205407Z.csv` (300
rows, Drive-only). Locked in `pilot/tests/test_internvl3_fermat.py`
(12 tests, including a full raw-sample recompute with 0 mismatches).

- **Transcription accuracy 21.0% (63/300)** — about half Qwen's ~42%, but
  unlike LLaVA-NeXT's 0.8%, well above the 30-item power minimum, so this
  is NOT a capability gate.
- **Perception AUROC printed as 0.915 [0.880, 0.944] — higher than Qwen's
  confirmed 0.835 despite half the base accuracy.** That mismatch doesn't
  happen with a genuine graded signal, so it got the project's standard
  surprising-result verification before being reported as a finding.
  Every derived column (`perception_entropy`, `reasoning_entropy`,
  `transcription_correct`, `grading_correct`, both parse-failure counts)
  recomputes from raw samples with **0/300 mismatches** — real model
  behavior, not a scoring bug.
- **But it does NOT survive this project's standard sensitivity cut.**
  `auroc_sensitivity` (excluding max-entropy items): 0.915 → **0.556
  [0.495, 0.614], chance** — versus Qwen's 0.835 → 0.796 under the
  identical cut, which survives. **Do not report 0.915 as "InternVL3 beats
  Qwen on perception."**
- **Mechanism (found by categorizing all 300 items, not eyeballing a
  sample): 202/300 (67%) pin at exact max entropy ln(5); accuracy within
  that group is a 0.99% floor (2/202) vs 62.2% (61/98) in the rest** — a
  bimodal coherent-vs-incoherent split driving the whole headline number,
  not a graded relationship. **Correction to an initial chat claim:** 1-2
  examples showing degenerate repeated-phrase generation ("where the word
  ROSE" ×40) were read as *the* mechanism; a full-population check found
  that pattern in only ~30% of the max-entropy-wrong group (61/200). The
  dominant mechanism (~70%, 139/200, confirmed by reading concrete
  examples) is different: the model's 5 retries each land on a different
  *incomplete slice* of a multi-step derivation (one restates a garbled
  question, another jumps into a mid-derivation step, another states an
  unfinished fragment as its "Answer") rather than converging on a stated
  final result the way Qwen typically does even when wrong. Every field is
  non-empty and canonicalizes to a real label — the parser is behaving
  correctly on genuinely incomparable content, not failing.
- **Reasoning: has_error=1 stratum is a new verdict type for this
  project** — powered (n_wrong=45, clears the 30 minimum) and CI excludes
  chance (0.548 lower bound), but AUROC **0.628 [0.548, 0.706] does not
  clear the 0.70 confirmation threshold**. Weaker than every other
  model/family measured (0.775–0.854). Clean stratum unaffected by any of
  this: 0.369 [0.290, 0.454], n_wrong=97, powered, inverted as expected.
  One wrong has_error=1 case's own grading text defaults to "no error"
  explicitly because it judges the image unreadable — plausibly the same
  generation instability from the perception arm leaking into reasoning's
  inputs, not a separate mechanism.
- **Decision (with the user, after explicitly rejecting "iterate until a
  fix clears 0.70" as a p-hacking risk — same failure class as the K=15
  cascade episode): run one small, pre-registered, single-shot screen of 3
  evidence-grounded ideas** (illegible→don't-default-to-no-error prompt,
  retest the 3B restate-first variant, K=10 grading resample), capped at
  n=100-150, report all three results honestly even if none pass. No
  further InternVL3 GPU work planned beyond that screen. See
  `.claude/plans/pilot-kickoff-instructions-glittery-eclipse.md` for the
  full spec (`pilot/13_internvl3_grading_screen.ipynb`, not yet run).
- Locked into `report/report.tex` §Phase 6 (new section, plus Summary
  table, Limitations, and Recommended Next Steps updates) — compiled
  clean, zero new warnings.

### RUN 2026-08-08: the InternVL3 screen — none of the 3 fixes resolves, and the pre-registered bar itself was too blunt

`pilot/13_internvl3_grading_screen.ipynb` ran the full n=150 screen.
`results/internvl3_grading_screen_n150_20260807T231004Z.csv` (450 rows =
3 variants × 150 items, Drive-only). Locked in
`pilot/tests/test_internvl3_grading_screen.py`. Verified independently:
0/450 mismatches recomputing entropy/correctness/K from raw samples.

- **Baseline on the same 150 items: 0.628 [0.548, 0.706], acc 0.700,
  n_wrong=45.**
- **No variant resolvably beats it — all three paired intervals span
  zero:** commit +0.069 [−0.028, +0.166]; k10 +0.074 [−0.024, +0.171];
  restate −0.008 [−0.112, +0.095].
- **`k10` technically PASSES the pre-registered bar** (AUROC 0.7023 ≥
  0.70, CI excludes chance, minority 34 ≥ 30) — reported as written, bar
  not moved after the fact. **But it clears 0.70 by 0.0023 with a CI of
  [0.616, 0.781]** (lower bound far below the bar it passed), and the
  0.005 separating it from `commit` (which "fails" at 0.6976) is noise.
  k10 also changes the *target* (21 items flip correctness label at
  K=10), a confound the paired interval can't remove.
- **Methodological lesson worth carrying forward: the 0.70 bar was
  inherited from the confirmation runs (0.775–0.854), where CIs were
  tight and unambiguously above it. Applied to an n=150 screen with a
  ±0.08 interval it's a much weaker statement.** A better-specified
  screen registers on the *paired difference* (variant beats baseline on
  the same items, CI excluding zero), not an absolute point-estimate
  threshold. Recorded in the report's Limitations and Next Steps.
- **Decision (with the user): no confirmation run. 0.628 stands as
  InternVL3's reported reasoning result.** Screen documented as a
  good-faith attempt that found no resolvable fix.
- **Two real side findings:** (1) k10's *grading accuracy* does resolve
  where its AUROC doesn't — 0.700 → 0.773, +0.073 [+0.013, +0.133], 16
  fixed / 5 broke; more samples sharpen the majority vote, same direction
  as the confirmed K=5→K=10 perception result. (2) **`restate`
  reproduces its Qwen-3B failure signature on a different model family**:
  heavy churn (31 fixed / 36 broke) with accuracy moving *down* (0.700 →
  0.667) — the model swapping which near-constant answer it gives rather
  than discriminating. A clean cross-family replication of a prior
  negative.
- **Two real bugs caught on this run, both now guarded in notebook 13**
  (see the conventions section for the general rules they produced):
  a stale `pilot.*` module surviving a re-clone (`KeyError: 'commit'` for
  a variant that was demonstrably on disk — the auth cell now purges
  `sys.modules` first), and a **silent 4-bit fallback** that would have
  confounded the entire screen against its bf16 baseline (the sample cell
  now compares the reference CSV's `quantized` column against the session
  and refuses to run on a mismatch). The 4-bit one is the more dangerous:
  it would have produced normal-looking, uninterpretable numbers.

### RUN 2026-08-08: second dataset (ScratchMath) — GATED, not comparable

`pilot/14_scratchmath_qwen7b.ipynb`, Qwen2.5-VL-7B held fixed so only the
dataset varies. `results/scratchmath_sizing_n100_qwen25-vl-7b-instruct_20260808T121016Z.csv`
(100 rows, Drive-only). Locked in `pilot/tests/test_scratchmath_gate.py`.
Verified: 0/100 mismatches recomputing from raw samples, bf16, 0% parse
failures.

- **Scouting finding worth keeping: both viable public benchmarks of
  handwritten student errors are 100% error items.** Checked against real
  data, not their papers — ErrorRadar 0/2500 rows without an error;
  ScratchMath `student_answer` never equals `answer` (0/1479). So neither
  supports a balanced sample, and **the confirmed clean-stratum inversion
  cannot be tested on either.** ErrorRadar also excluded because its
  images are the *problem figures* (student work is a text field) — no
  handwriting to grade.
- **GATED: misgrade rate 4.0% (4/100), under the pre-registered 5% bar.**
  Accuracy 96% is meaningless here — all items are errors and the model
  says "error" 90% of the time.
- **The decisive evidence is qualitative, quantified over all 500 samples:
  24% (120/500) explicitly state the model cannot read/use the image, and
  82% of those (98/120) emit `Error: 1` anyway** — which scores as correct
  on an all-error set. Much of the 96% is response bias firing through
  explicit non-engagement.
- **Why "not comparable" rather than "underpowered":** non-engagement is
  concentrated on misgraded items (0.40 vs 0.23), so a powered run (~750
  items) would substantially measure scratchwork legibility, not
  uncertainty about the mathematics — a different construct from the
  FERMAT result. Plus 70/100 items sit at exactly zero entropy.
- **Decision (user): stop. 0.628/FERMAT-only stands; move to writing.**
- Qualitative replication that did survive: 2 of the 4 misgrades are the
  documented "verify the student's own computation instead of recomputing"
  mechanism (accepting 522x5 where the problem needed 522x3x5; 50-12 where
  it needed 50-28). The failure mode travels even where the metric cannot.
- Locked in `report/report.tex` §Phase 7 + Summary/Limitations.

### Frozen artifacts: `reference/` (2026-08-08)

Two kinds, for two failure modes. **Read `reference/README.md` first.**

- **Metrics snapshots** (`reference/*.json`) — one per claim-bearing run,
  recomputed from raw columns via `pilot.snapshot`. `snapshot_metrics` for
  full perception+reasoning runs; **`snapshot_grading_metrics` (new) for
  grading-only runs**, which also handles the all-error case (clean stratum
  → `not_measured`, distinct from `inconclusive_underpowered`), flags a
  pooled AUROC on an unbalanced sample, and records whether the entropy
  distribution is degenerate.
- **Case snapshots** (`reference/cases/<id>/`) — the *evidence*: ground
  truth, every raw sample verbatim, parsed values, entropy/correctness, a
  `category` recording why the case was selected (selection is by rank, not
  by hand), and a human-written `note`. Seven cases covering the four
  entropy×correctness quadrants plus InternVL3 degeneration, ScratchMath
  non-engagement, and confidently-wrong grading.
- **Only claim-bearing runs get snapshotted** — not debug/smoke/interrupted
  runs. If a number appears in the report, it gets frozen.
- **Images need auth** (FERMAT is gated): text bundles build offline,
  `pilot/15_attach_case_images.ipynb` attaches FERMAT images in Colab
  (idempotent). ScratchMath is ungated and already attached. A bundle
  without an image is a normal state, never an error.
- **Real bug caught while building this:** `classify_stratum_result`
  initially checked `excludes_chance` before the inversion case. That field
  is defined as `ci_low > 0.5` — **one-sided** — so a resolvably *inverted*
  stratum reports `excludes_chance=False`, and Qwen-7B's CONFIRMED clean
  inversion (0.280 [0.200, 0.366]) was being labelled `no_signal`. Caught
  because the snapshot disagreed with the report. Regression test in
  `test_plotting.py`.

## Repo-specific conventions

- **Reported numbers come from the STORED scored columns of the original run;
  local rescoring is for sensitivity and audit ONLY.** Parser availability
  changes labels — the 2026-08-02 Qwen run was scored in a Colab session
  without SymPy's LaTeX parser, and recomputing locally relabels **43/300
  items** (118 vs 141 correct). So anything quoted next to a locked figure
  must be derived from the same CSV columns that produced it, or the paper
  contradicts itself. This is why `plotting.distinct_from_entropy` exists:
  it inverts the stored entropy rather than re-deriving labels, and using a
  recompute instead moves the headline cell of the distinct-answers table
  from 92% to 80%.
  **The transparent phrasing if asked:** *we use the stored scored columns
  from the original run for all locked reported numbers; local rescoring is
  used only for sensitivity and audit analyses, because parser availability
  can change labels.*
  **Pixtral is unaffected** (its recompute matches exactly), so a
  one-model check will not reveal the problem — verify on the Qwen run.

- **Don't touch `report/` unless explicitly asked.** As of 2026-08-02 the user
  updates the report on request only. Run the analysis, write and test the
  code, report findings in chat — then say a finding would change the report
  and let them decide. This means `report/report.tex` may lag behind what's
  been found; that's intentional, not a gap to close.
- **Notebooks**: use the `NotebookEdit` tool, not `Edit` — `pilot.ipynb` and
  `pilot/02_reasoning_text_entropy.ipynb` will reject direct `Edit` calls.
  For multi-cell inserts, a small Python script (`json.load` → mutate
  `cells` → `json.dump`) is more reliable than chained `NotebookEdit` insert
  calls — see git history for examples.
- **`pilot.ipynb` diffs against a clean tree are usually not yours.** The
  user runs it live (Colab or VS Code's Jupyter connection), and execution
  outputs sync back to the file on disk. Check `git diff --stat` and scan
  for leaked tokens before committing, but don't strip legitimate output
  diffs or revert cells the user added themselves.
- **A dry run must execute the SAVE cell too.** Found the hard way on
  2026-08-11: notebooks 19 and 20 both failed on their last cell in Colab
  with `NameError: name 'df' is not defined` (the gate builds `scored_df`),
  and the same cell still carried notebook 16's hardcoded `pixtral` filename,
  so had it not crashed it would have written a CSV labelled as a Pixtral
  run. Both were invisible because the dry run stopped at the gate. Stub
  `subprocess.run` with a fake process, `chdir` into a temp dir, exec the
  cell, and assert on the filename it produces — including that swapping
  `MODEL_ID` changes it, or two runs collide on Drive.
- **Dry-run notebook cell logic with stubs before handoff.** Extract the
  cell source, monkeypatch GPU-dependent calls (`generate()`, image
  processing) with fakes, and exec it locally to catch bugs before the user
  burns Colab time on it. This caught real bugs multiple times (checkpoint
  threshold gaps, `re.sub` backslash-escaping corrupting injected stubs).
- **Push new `pilot/` library code BEFORE handing a notebook off — a green
  dry run does not cover this.** Every notebook's auth cell does
  `rm -rf repo` → `git clone` → `pip install -e repo/`, so Colab runs
  against the *remote* `pilot/`, while `dryrun_nb*.py` inserts the *local*
  working tree on `sys.path`. Uncommitted library code therefore passes the
  dry run and fails in Colab. Hit for real on 2026-08-08: notebook 13's new
  `GRADING_USER_PROMPT_COMMIT` wasn't pushed, dry run was green, Colab died
  with `KeyError: 'commit'`. Verify with
  `git show origin/main:pilot/<file>.py | grep <new symbol>` before saying
  a notebook is ready. Recovery is cheap if it does happen: push, re-run
  only the auth cell plus the failing cell — the loaded model stays in
  memory, no GPU work is lost.
- **Every real bug found on actual pilot data gets a regression test that
  reproduces it**, not just a fix. Search `pilot/tests/` for docstrings
  citing specific dates/numbers (e.g. "found on the 2026-08-02 K=15 run") —
  these are real failure modes, not synthetic edge cases.
- **Tokens (HF/GitHub): never hardcode, even temporarily.** The notebooks
  use `getpass()`-based entry specifically so the tracked file never
  contains a real secret. If a real token ever appears in a diff, chat, or
  notebook output, treat it as compromised — tell the user to revoke it
  immediately, don't just quietly fix the code.
- **`results/` is intentionally untracked in git** (not gitignored, just
  never `git add`ed) — CSVs come from Google Drive, downloaded manually by
  the user into `results/` when needed for local analysis. Large/expensive
  reproducible outputs (`figures/` at repo root) genuinely are gitignored;
  `report/figures/` is not (anchored `/figures/` pattern in `.gitignore` —
  don't widen that pattern back to bare `figures/`, it'll swallow both).
- **Verify surprising metric improvements against real data before
  reporting them as findings**, especially from NLI/embedding-based
  clustering — it has two confirmed failure modes on this project
  (lexical-overlap over-merging, transitive cascade merging). A plausibility
  check against other known numbers, plus spot-checking individual
  clustering decisions, catches both.
- **Check whether a difference is resolvable before explaining it.** A large
  amount of effort here went into mechanistically explaining AUROC gaps that
  turned out to be inside the error bar. Compute the CI first; if the paired
  interval spans zero, the honest finding is "unresolved at this n", and any
  mechanism you find is a hypothesis about the direction, not a cause of a
  measured effect. (The cascade bug is real and reproduces as a unit test —
  but the 0.702→0.520 *magnitude* attributed to it is poorly determined.)

## Testing

```
source .venv/bin/activate
pytest pilot/tests/ -v
```

Local, no GPU needed. All `pilot/` modules are unit-tested with synthetic
and (for regression tests) real-data-derived fixtures — no network, no
model downloads required to run the suite.
