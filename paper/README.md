# WACV 2027 submission

`main.tex` is a **content skeleton**, not a submission-ready document.

## Before anything else

**It is WACV 2027, not 2026.** WACV 2026's deadlines were in 2025 and that
conference has already happened. Looking at the wrong CFP is an easy and
expensive mistake.

| | Date (AoE) |
|---|---|
| Paper enrollment | **21 Aug 2026** |
| Paper submission | **28 Aug 2026** |
| Supplementary material | **30 Aug 2026** |

Supplementary is due **two days after** the paper — the 31-page
`report/report.pdf` can go in as an appendix with two extra days to prepare.

## Track: Evaluations & Datasets

WACV 2027 has three tracks. Submit to **Evaluations & Datasets**, which
explicitly invites *"negative results and critical analyses"*,
*"stress-testing and audit studies"*, *"analysis of benchmark limitations
and failure modes"*, and *"reproducibility studies"*.

This matters more than it sounds. Under **Algorithms**, the paper is judged
on *"algorithmic novelty"* — and entropy over repeated samples is not novel;
that is a likely reject on its own. Under Evaluations & Datasets, the things
that are unusual about this work stop being weaknesses and become the
contribution. Say the track's framing out loud in the abstract.

## Hard rules

- **8 pages excluding references.** Over-length is desk-rejected without review.
- **Use the official template.** A paper not using it is rejected without
  review. Download the WACV 2027 Author Kit and move this content into it —
  do not hand-roll the style file. `main.tex`'s preamble is a placeholder.
- **Double-blind.** No names, no grant IDs, no identifying repo links —
  including in supplementary material and in any submitted code.
- *"Reviewers are encouraged to check the submitted code."* That is an
  advantage here: 322 tests, frozen snapshots, and case bundles are exactly
  what makes a claim checkable. Budget time to anonymise the repo.

## Structure

Follows the FERMAT paper (same dataset, same task family), which is the
closest available model for what this venue accepts:

- Introduction — narrative prose, **no bulleted contributions list**
- Related Work — unnumbered thematic paragraphs, not subsections
- Setup
- **Results, organised as questions** (FERMAT does this; it suits a paper
  whose results *are* a sequence of questions)
- Failure Analysis
- Limitations
- Conclusion

## Numbers

Every figure in `main.tex` comes from `reference/*.json`, which are
recomputed from raw model output and asserted by `pilot/tests/`.

**Run this after every edit:**

```bash
python paper/check_numbers.py
```

It does two things: verifies every snapshot figure appears in the paper
(catches a stale paper), and verifies every three-decimal number in the body
*is* a snapshot value (catches a typo). Exit code 1 on failure.

The second check exists because the first is not sufficient — corrupting
`0.835` to `0.853` in the prose still passed the existence test, since
`0.835` also appears in Table 1. Tested by deliberately corrupting a number;
do not weaken it to a plain existence check.

Never retype a figure from memory or from the slides. If a number is not in
`reference/`, snapshot it first — that is how the token-confidence baselines
came to be missing and got added.

## What is still to write

Marked `\todo{}` in `main.tex`. In rough order of effort:

1. **Related Work** — the largest genuinely new task. Nothing exists yet.
   Every citation must be verified against the real paper; `refs.bib` is a
   stub and its entries are unverified.
2. **Introduction** — argument is sketched in the comments.
3. **Conclusion** — short; the closing line is drafted in a comment.
4. **Failure analysis at scale** — three mechanisms are established from real
   cases; the section wants 50–100 coded cases. Reading time, not compute.
   Tooling exists in `pilot/07_manual_case_inspection.ipynb`.
5. **Two figures** — risk–coverage (asset exists at
   `report/figures/risk_coverage_n300.png`) and the four-quadrant page figure
   (assets exist at `reference/cases/01`–`04`).

Results, tables, Setup and Limitations are drafted from real data and should
need editing rather than writing.
