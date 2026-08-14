# WACV evaluation artifact — FERMAT n=300, audited subset, and a null

**Status: two parts are complete and one is deliberately unfinished.**
`alignment_status = PENDING_AUTHENTICATED_RUN` and `SECOND-RATER STATUS: PENDING`.
Neither is an oversight; each names a step only a human with credentials can
take, and the code refuses to fake either.

## What is here

| file | what it is | release |
|---|---|---|
| `null_grid.csv`, `null_grid_config.json` | the binary-stratification null, 285 cells | public |
| `figures/stratification_null.png` | the null across K and p, with the three locked cases | public |
| `fermat_n300_public_manifest.csv` | 300 items: ids, hashes, pinned revision, derived numbers | **public** |
| `fermat_n300_private_manifest.csv` | gated text and raw generations | **do not release** |
| `audit_labels_long.csv` | 312 rows, one per item per audit pass | public |
| `second_rater_queue.csv`, `..._template_blank.csv`, `..._instructions.md` | the blinded packet | public |
| `second_rater_blinding_key_PRIVATE.csv` | review id to item id | **do not release** |
| `provenance.json` | hashes, prompts, protocol, claims not supported | public |

## The null is a reviewer control, not a tuned result

**K=5 is the frozen evaluation protocol.** It was fixed before any of these
results existed and is the K behind every reported number in this project. The
sweep over K in {3,5,7,9,15} and p in 0.05..0.95, across y=0, y=1 and balanced
pooling, exists to answer "is this artifact specific to your K, or to your
model's bias?". It is neither: the effect appears at every K, grows with p,
vanishes at p=0.5, and disappears under balanced pooling.

Every AUROC in `null_grid.csv` is produced by a responder with **no item-level
information at all**. It is a stress test, not a measurement of any model.

## Reconstruction, and why it says PENDING

FERMAT is gated. Metadata and data have different access levels: the commit
revision is readable anonymously (`80ff9934c38615bb8d3a33c24252db02e21774f0`), while the rows
require authentication. So the manifest pins the revision but **has not
verified the 300-row alignment**, and says so in every row rather than
implying otherwise.

`shuffle(seed=42)` is **not stable across dataset revisions** — this project
already had two items overlap between draws that should have been disjoint —
so an index alone does not reproduce the sample. That is why per-item content
hashes are published: they identify the items even if row order moves.

**To complete it:** run notebook 30 Stage B once in an authenticated session.
`verify_reconstruction` requires all 300 of `orig_q`, `pert_a` and `has_error`
to match; a partial or order-insensitive match is reported as FAILED.

## The audit, and what it cannot support

312 rows over **234 unique items**, with
**78 overlaps** preserved rather than merged. Collapsing
them would destroy the only intra-rater evidence available. The four hard
contradictions (108, 149, 222, 230) are kept
visible and **unadjudicated**.

Two of the three audit sets are **one-directional by construction** — they drew
only items the scorer called wrong, or only items it called correct — so their
agreement with the scorer is partly definitional. **These are targeted sets,
not an IID sample of the 300.**

`extraction_issue` and `indeterminate` mean *the verdict was not earned* or
*the reader could not decide*. Neither is ever converted into "the model was
wrong", in this artifact or downstream.

## Claims this artifact does NOT support

* human-level accuracy;
* reliable inter-rater agreement (there is one rater so far);
* a representative error rate for FERMAT as a whole;
* a corrected benchmark score;
* that the 234 audited items are an IID evaluation set.

## Licensing and ethics

FERMAT is gated and contains real student work. The public manifest carries no
page images, no question or reference text, and no model output derived from a
gated page — only ids, hashes and derived numbers. Redistribution of the gated
content is not assumed to be permitted, so it lives in the private manifest and
is excluded from release. The blinding key is named `PRIVATE` for the same
reason.

## Intended use

Reproducing the evaluation trap, checking which items the audit selected and
why, and running a second rating. It is **not** a corrected benchmark and must
not be used to report a model score.
