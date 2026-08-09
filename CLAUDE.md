# uncertainty-math-vlm

Pilot testing whether entropy over a VLM's own repeated samples predicts when
it's wrong, on FERMAT (handwritten math) with Qwen2.5-VL-3B. Two arms:
**perception** (transcription instability) and **reasoning** (grading
instability). See `README.md` for setup/layout basics — this file is
working context for future sessions, not a repeat of that.

## Current findings

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
