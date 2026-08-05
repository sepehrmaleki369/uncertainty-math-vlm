# uncertainty-math-vlm

Pilot testing whether entropy over a VLM's own repeated samples predicts when
it's wrong, on FERMAT (handwritten math) with Qwen2.5-VL-3B. Two arms:
**perception** (transcription instability) and **reasoning** (grading
instability). See `README.md` for setup/layout basics — this file is
working context for future sessions, not a repeat of that.

## Current findings

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
