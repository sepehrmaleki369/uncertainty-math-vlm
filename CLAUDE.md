# uncertainty-math-vlm

Pilot testing whether entropy over a VLM's own repeated samples predicts when
it's wrong, on FERMAT (handwritten math) with Qwen2.5-VL-3B. Two arms:
**perception** (transcription instability) and **reasoning** (grading
instability). See `README.md` for setup/layout basics — this file is
working context for future sessions, not a repeat of that.

## Current findings — read `report/report.tex` for full detail

Don't re-derive findings from git history; the report is the maintained,
compiled source of truth (`report/report.pdf` is the rendered version, kept
in sync with `report/report.tex`). Status as of the last update:

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

## Repo-specific conventions

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
- **Dry-run notebook cell logic with stubs before handoff.** Extract the
  cell source, monkeypatch GPU-dependent calls (`generate()`, image
  processing) with fakes, and exec it locally to catch bugs before the user
  burns Colab time on it. This caught real bugs multiple times (checkpoint
  threshold gaps, `re.sub` backslash-escaping corrupting injected stubs).
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
