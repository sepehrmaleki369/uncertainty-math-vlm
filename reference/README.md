# `reference/` — frozen results

Two kinds of artifact, for two different failure modes.

**Metrics snapshots** (`*.json`, one per run) freeze the *numbers*. They exist
because `results/` is untracked and arrives from Drive by hand: if a CSV is
lost or replaced, the figures we reported would go with it. Each snapshot is
recomputed from the raw columns, never copied from the report, so it can be
used to check the report rather than merely echo it.

**Case snapshots** (`cases/`, one directory per case) freeze the *evidence*.
Several findings here rest on a handful of specific items, and re-deriving
them from scratch each time is how the label-noise misreading survived as
long as it did.

## What is snapshotted, and what is not

Only claim-bearing runs — anything whose numbers appear in `report/report.tex`
or will appear in the paper. Debug runs, smoke tests, interrupted checkpoints
and one-off parsing checks are deliberately absent: snapshotting them would
make the directory a log rather than a reference.

| Snapshot | Run | Headline |
|---|---|---|
| `n300_balanced_20260802.json` | Qwen2.5-VL-3B, FERMAT n=300 | Perception **0.835** [0.787, 0.879], robust to every cut |
| `qwen7b_fermat_n300_20260805.json` | Qwen2.5-VL-7B, FERMAT n=300 | Capability gate *marginal* (59.0% vs 50.0%); clean stratum inverted |
| `qwen3b_stratum_powered_n800_20260806.json` | Qwen-3B, `has_error`=1 powered to n=650 | **Confirmed 0.854** [0.796, 0.902] |
| `qwen7b_matched_n650_20260806.json` | Qwen-7B, matched to 3B's items (n=648) | **Confirmed 0.801** [0.751, 0.846] |
| `llava_fermat_n300_20260806.json` | LLaVA-NeXT-7B, FERMAT n=300 | Perception **capability-gated** (0.8% free-response OCR) |
| `llava_stratum_powered_n550_20260807.json` | LLaVA-NeXT-7B, `has_error`=1 powered to n=400 | **Confirmed 0.775** — first cross-family replication |
| `internvl3_fermat_n300_20260807.json` | InternVL3-8B, FERMAT n=300 | Perception 0.915 raw but **fails the max-entropy cut** (→0.556) |
| `scratchmath_gated_n100_20260808.json` | Qwen-7B on ScratchMath n=100 | **Gated / not comparable** — see `extra` |

`n300_balanced_20260802.md` is a human-readable companion to the first one.

## Reading a metrics snapshot

Full runs (perception + reasoning) and grading-only runs have different
shapes; `arm: "grading_only"` marks the latter. Three fields are there
specifically to stop a number being misread:

- `reasoning.pooled_is_misleading_unless_balanced` — on a non-50/50 sample the
  pooled AUROC mechanically favours the larger stratum. The n=500/n=800 CSVs
  read ~0.61 pooled while their strata run in *opposite* directions. Read the
  strata.
- `reasoning.*_verdict` — `confirmed`, `confirmed_inverted`,
  `resolved_below_threshold`, `inconclusive_underpowered`, `no_signal`, or
  `not_measured`, from `pilot.plotting.classify_stratum_result`. Note
  `not_measured` (the stratum does not exist — ScratchMath has no clean items)
  is *not* the same as `inconclusive_underpowered` (too few of it).
- `entropy_distribution.is_degenerate` — both InternVL3 and ScratchMath were
  invalidated by their entropy distribution rather than by their CI. A run can
  have a tight, above-chance interval and still be measuring nothing.

Diff two runs with:

```bash
python -m pilot.snapshot --compare reference/<old>.json reference/<new>.json
```

## Case snapshots (`cases/`)

`cases/index.json` lists every bundle. Each `cases/<id>/` holds `case.json`
and, when available, the image. A bundle carries the ground truth, **every
raw model sample verbatim**, what each parsed to (so a parse failure is
visible as a parse failure rather than as model disagreement), the derived
entropy and correctness, a `category`, and a human-written `note`.

`category` is what makes a case evidence rather than an anecdote — it records
why the case was selected, and selection is by rank on entropy, not by hand:

| Category | Shows |
|---|---|
| `low_entropy_correct` | the signal working — confident and right |
| `high_entropy_wrong` | the signal working — uncertain and wrong |
| `high_entropy_correct` | a false alarm; the cost of the abstention rule |
| `low_entropy_wrong` | **confidently wrong** — what entropy is structurally blind to |
| `max_entropy_degenerate` | InternVL3's five incompatible fragments |
| `non_engagement_says_error` | ScratchMath: "cannot read it" → `Error: 1` |
| `confidently_wrong_grading` | grading unanimously wrong on a real error |

### Images

Text bundles build offline from the CSVs; **images need an authenticated
session** because FERMAT is gated. ScratchMath (ungated) is attached locally.
Run `pilot/15_attach_case_images.ipynb` in Colab to attach the FERMAT ones —
it is idempotent. A bundle without an image is a normal, useful state, never
an error.

Images are downscaled to 1600px on the long edge and stored in whichever of
PNG/JPEG encodes smaller, with the original dimensions recorded. The
full-resolution image is always re-derivable from the dataset via the
recorded ground truth.

## Regenerating

```python
from pilot.snapshot import write_snapshot, write_grading_snapshot
write_snapshot("results/<full-run>.csv", "reference/<name>.json", label="...")
write_grading_snapshot("results/<grading-run>.csv", "reference/<name>.json", label="...")

from pilot.cases import build_case, write_case, select_cases, write_case_index
```

Numbers in these files are asserted by `pilot/tests/` — `test_reported_numbers.py`
for the n=300 reference, and one `test_*.py` per run for the rest. If a code
change moves a number, those tests fail before the snapshot silently drifts.
