# Second-rater instructions (second_rater_v1_20260814)

You are the **second** reader. Someone has already coded these pages; you will
not see their labels, and that is deliberate. If you could see them you would
anchor on them, and the agreement number would measure suggestion rather than
reliability.

## What you are shown

The handwritten page, the model's transcription, and the dataset's reference
answer. Nothing else. In particular you are NOT shown the automatic
correct/wrong verdict, the first rater's label or note, the entropy, or whether
the page carries a deliberately injected error.

**You must open the image.** Several of these cannot be decided from the text
alone; that is the point of a second read.

## The fields

**`model_correctness`** -- did the model read the page correctly?
  * `correct` -- the model's answer matches what is written on the page.
  * `wrong` -- it does not.
  * `indeterminate` -- you cannot tell from the page. Use this freely; it is a
    real answer, not a failure to decide.

**`reference_fidelity`** -- does the dataset's reference answer match the page?
  * `faithful` / `unfaithful` / `indeterminate`. The reference is sometimes
    wrong or truncated, and that is worth recording separately from the model.

**`failure_category`** -- only if `model_correctness = wrong`, else
`not_applicable`:
  * `notation_misread` -- misread a symbol or digit on the page.
  * `copied_wrong_line` -- copied a real line of the page, the wrong one.
  * `hallucination` -- produced content not on the page at all.
  * `extraction_issue` -- the model's answer looks right but only part of it
    was captured, or the comparison used a fragment.
  * `reference_issue` -- the reference answer, not the model, is at fault.
  * `ambiguous_multianswer` -- several answers on the page, unclear which counts.
  * `other` -- with a note.

**`confidence`** -- `high`, `medium`, `low`.

**`evidence_note`** -- REQUIRED. One line naming what on the page decided it.

**`rater_pseudonym`** -- any stable string that is not your name.

## Two rules that matter for the analysis

1. `extraction_issue` and `indeterminate` are **not** ways of saying the model
   was wrong. They will never be converted into "wrong" downstream.
2. Do not go back and revise earlier rows after forming a theory. If you change
   your mind about a criterion, say so in the note and keep coding forward.

Leave a row blank rather than guessing. A blank row is analysable; a guessed
one is not.
