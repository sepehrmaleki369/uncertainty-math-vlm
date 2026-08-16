# Figure 2 panel verification (2026-08-15)

Every panel of `paper/figures/four_quadrant.png` traced to the run item that
produced it, with what the page shows, what the scorer did, what the human audit
found, and what the raw model samples show. Built offline from the frozen run
CSV, `pilot.wacv_artifact.audit_labels_long()` and the professor review folder on
Drive. No inference, no new labels, no numeric result changed.

**The rule applied throughout: a panel may only assert something that survives
checking the raw samples, not just the extracted span.** Three otherwise-obvious
candidates failed that check, including one whose human coding turned out to be
mistaken. Details under "Rejected candidates".

## Final panels

Laid out as a contingency table: left column low entropy, right column high
entropy, top row what the scorer passed, bottom row what it failed. Low entropy
does not always mean unanimous agreement: item 160 is a four-of-five split.

| position | item | H | `strict_v1` | audit verdict | headline | sub-line |
|---|---:|---:|---|---|---|---|
| top left | 229 | 0.000 | CORRECT | `true_correct` x2 | `low entropy, scored correct` | `item 229, H = 0.00, audit: correct` |
| top right | 92 | 1.609 | CORRECT | `true_correct` | `high entropy, scored correct` | `item 92, H = 1.61, audit: correct` |
| bottom left | 160 | 0.500 | WRONG | `notation_misread` + `true_wrong` | `low entropy, scored wrong` | `item 160, H = 0.50, audit: model wrong` |
| bottom right | 251 | 1.332 | WRONG | `notation_misread` | `high entropy, scored wrong` | `item 251, H = 1.33, audit: model wrong` |

All four are human-coded, each carries one consistent verdict across every audit
pass it appears in, and each was re-checked against its five raw samples.

**Pages**: gated FERMAT content, not committed. See
`paper/figures/pages/README.md`; export with
`pilot/31_export_figure2_pages.ipynb`.

---

## Panel 1, top left. Item 229

- **Stored**: entropy 0.000, 1 distinct label, `transcription_correct` True,
  `has_error` False.
- **Page**: "Find all the points of local maxima and local minima of
  `f(x) = 2x^3 - 6x^2 + 6x + 5`". The handwritten solution uses `t` as the
  variable throughout, which is FERMAT's modification, and concludes `t = 1`.
- **Model**: five identical transcriptions, all ending "the point `t = 1` is
  neither a point of local maxima nor a point of local minima. Hence `t = 1` is
  a point of inflexion." Extracted span `t = 1`.
- **Scorer**: truth span `\textcolor{red}{t} = 1`, model span `t = 1`; CORRECT.
- **Audit**: `true_correct` in **both** the `v1_correct_spotcheck` and the
  `v2_high_priority` passes.
- **Why it belongs**: an earned pass. The samples agree, the scorer passes, and
  the model reproduced the page's own substituted variable rather than
  normalising it back to `x`, which is exactly the fidelity being asked for.

## Panel 2, top right. Item 92

- **Stored**: entropy 1.609, 5 distinct labels, `transcription_correct` True,
  `has_error` False.
- **Page**: a binomial-theorem proof that `6^n - 5n` always leaves remainder 1
  when divided by 25.
- **Model**: five different transcriptions of a long derivation; majority span
  `6^n - 5n`.
- **Scorer**: truth span `6^n - 5n`; CORRECT.
- **Audit**: `true_correct` (`v1_correct_spotcheck`).
- **Why it belongs**: the false alarm. The samples disagree across a long
  derivation while the majority answer is right, which is the cost any deferral
  rule pays. Deferring here would waste a correct transcription.

## Panel 3, bottom left. Item 160

- **Stored**: entropy 0.500 (four of five samples agree),
  `transcription_correct` False, `has_error` False.
- **Page**: "Solve `2x - 3 = x + 2`". The handwritten solution uses `z`
  throughout, which is FERMAT's modification, and ends `z = 5`.
- **Model**: four of five samples wrote `x` and concluded `x = 5`; only s2 wrote
  `z = 5`. Extracted span `x = 5`.
- **Scorer**: truth span `\textcolor{red}{z} = 5`; WRONG.
- **Audit**: `notation_misread` in the census ("Model wrote x = 5 where image
  says z = 5") and `true_wrong` at high confidence in the v2 pass ("MODEL
  WRONG"). Two independent passes, same direction.
- **Verified against the samples**: the failure is real. The model normalised the
  page's `z` back to the conventional `x`, so it did not transcribe what is
  written. This is the auto-correction mechanism the project documents
  elsewhere, and it is a genuine fidelity failure rather than a scoring artifact.
- **Why it belongs**: the lowest-entropy genuine model failure in this run. Four
  of five samples agree on the wrong answer, so self-disagreement barely fires.

## Panel 4, bottom right. Item 251

- **Stored**: entropy 1.332, `transcription_correct` False, `has_error` True.
- **Page**: "Write the solution set of `x^2 + x - 2 = 0` in roster form". The
  handwritten solution carries FERMAT's injected `y`: `\textcolor{red}{y} = 1,
  -2`, and the roster form reads `{y, -2}`.
- **Model**: the majority wrote `x = 1, -2` and `{1, -2}`, the mathematically
  correct set; sample s1 did write `y`, which is where the entropy comes from.
- **Scorer**: truth span `\{\textcolor{red}{y}, -2\}`, model span `\{1, -2\}`;
  WRONG.
- **Audit**: `notation_misread` (census), "Model says {1, -2}, truth/image says
  {y, -2} -- symbol mismatch".
- **Why it belongs**: the signal working as intended. The model failed to
  reproduce the page and the samples disagree about it, so entropy flags exactly
  the item that should be deferred.

---

## Rejected candidates, and why

These were checked and dropped. Each is a case where the extracted span looks
like one thing and the samples say another.

- **Item 176** (H = 0.000, scored wrong) was in the original figure as "five
  identical transcriptions, all wrong". Its unanimous `x = 4` is the page's own
  final answer, and **two** independent passes both coded it `extraction_issue`
  at high confidence, the census note reading "Model correct." The reference span
  had swallowed the entire display line, including a `\textcolor{red}` prose
  insertion. The model was right and the scorer was wrong.
- **Item 271** (H = 0.000, scored wrong) is coded `true_wrong` with the note
  "MODEL WRONG: misread the final result. Span 25 against the full power
  expression." **The coding is mistaken.** All five samples wrote
  `\frac{9}{4} = 2.25`, matching the page. The span `25` is the documented
  decimal-split extractor bug truncating `2.25`. The coder read the span, not the
  samples. This is the one place where checking the raw output overturned a
  human label rather than confirming it.
- **Item 166** (H = 1.332, scored wrong) is `extraction_issue` in the census and
  `true_wrong` in the v2 pass. Both are high confidence and they disagree on the
  verdict. A panel asserting either would be taking a side in an open
  disagreement.
- **Item 5** (H = 0.000, scored correct) was the original top-left panel,
  captioned "the signal working: correct". Its page states "Option A" while its
  own arithmetic gives `\textcolor{red}{231444}`, which is option D. Working the
  problem: `LCM(21,36,66) = 2772`, so the least perfect square multiple is
  `(2*3*7*11)^2 = 462^2 = 213444`, which **is** option A. The perturbation
  changed the computed value and left the option letter untouched, making the
  reference internally inconsistent; the extractor then scored on the letter, the
  one fragment the injection never reached. The audit recorded `needs_visual`,
  "possible false pass / key mismatch ... Cannot resolve from text", and the
  user's later visual reading confirmed the page's value is option D. Kept out of
  the figure because no single short label states this honestly.
- **Item 1** (H = 1.609, scored wrong) was the original bottom-left panel. It
  appears in no audited set at all, so nothing human-verified could be printed on
  it.

## A finding that came out of this check

**At exactly `H = 0` this run has three scored-wrong items, 176, 218 and 271, and
every one of them is a scoring artifact rather than a model failure** (176 and
218 extraction issues, 271 the decimal-split bug). There is no genuine unanimous
model failure in the 300 items to put in the figure, which is why the bottom-left
panel is the lowest-entropy *real* failure at `H = 0.50` rather than a `H = 0`
case.

This does not contradict the paper's argument at
\S"Failure Analysis", which is a statement about construction: as the samples
concentrate on one wrong answer, self-disagreement has less to use. It does mean
the run contains no worked example of a genuine all-five-agree model failure, and
the three items that look like one are all scorer failures. Worth stating rather
than leaving a reviewer to discover it.

## What changed in the paper

- Panel items replaced: 5, 92, 176, 1 becomes 229, 92, 160, 251.
- Panel labels simplified to `low/high entropy, scored correct/wrong` with an
  item, entropy and audit line beneath each.
- Panels reordered into a contingency table, so the caption and the body can
  refer to the low-entropy *column* rather than to a diagonal.
- Caption shortened; the per-panel analysis lives in this file instead. Two
  wordings were then corrected against the final panels: "cannot flag unanimous
  failures" became "weakest when the samples mostly agree on the same wrong
  reading", because the bottom-left panel is a four-of-five split rather than a
  unanimous one, and "human-audit notes are shown where available" became "each
  panel gives its item, entropy and human-audit verdict", because all four are
  now audited.
- `paper/main.tex` body pointer changed from "bottom right" to "left column",
  and "a confidently wrong item is one where all five samples agree" became "a
  low-entropy wrong item is one where the samples mostly agree", for the same
  four-of-five reason.
- Figure canvas resized from 12in to 6.5in. At `\linewidth` in a single column
  every font is scaled by `3.25/FIG_W`, so the old 13pt titles printed at 3.5pt
  and the sub-lines at 2.6pt, which is unreadable.

No numeric result was touched. `strict_v1` verdicts, entropies and every reported
figure are unchanged; `paper/check_numbers.py` passes.

## Reproducing the figure

    # once, in Colab, no GPU:  pilot/31_export_figure2_pages.ipynb
    python paper/build_four_quadrant.py

The builder asserts each panel's entropy and scorer verdict against
`reference/wacv_evaluation_artifact/fermat_n300_public_manifest.csv`, so a label
cannot drift away from the item it describes. Before this pass the figure had no
builder in the repository and could not be regenerated or checked at all.
