"""Build the dataset distribution artifacts. Offline, no GPU, no inference.

    python pilot/dryruns/build_dataset_profile.py

Writes:
  reference/audit/dataset_distribution_by_type_20260812.csv   (one row/item)
  reference/audit/dataset_distribution_by_type_20260812.md    (the report)
  reference/audit/dataset_examples_contact_sheets_20260812/   (manifest + HTML)

Reads only stored artifacts: the run CSV, the two frozen rules recomputed for
comparison, and the human audit CSVs. **No scorer rule is modified.**
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pilot import audit_diagnostics as ad          # noqa: E402
from pilot import dataset_profile as dp            # noqa: E402
from pilot import rescore, strict_v2               # noqa: E402

RUN = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")
OUT = ROOT / "reference" / "audit"
SHEETS = OUT / "dataset_examples_contact_sheets_20260812"

if not RUN.exists():
    sys.exit(f"missing run CSV: {RUN}")

run = pd.read_csv(RUN)
assert len(run) == 300, f"expected 300 rows, got {len(run)}"
print(f"run: {len(run)} rows")

v1 = rescore.rescore_run(run, "strict_v1")["transcription_correct"].astype(bool)
v2s = strict_v2.rescore_v2(run, progress=True)
print(f"strict_v1 {int(v1.sum())}/300 · "
      f"strict_v2 {int(v2s['correct_strict_v2_display_primary'].sum())}/300")

# Human labels, deduplicated by the audit sets' own precedence. `final_label`
# is kept RAW alongside the mapped truth: `extraction_issue` is indeterminate,
# not wrong, and collapsing the two is the easiest way to manufacture a
# result here.
audits = ad.load_audit_sets(str(ROOT / "reference" / "audit"))
raw, truth = {}, {}
for name in ad.SET_PRECEDENCE:
    for i, r in audits[name].iterrows():
        raw.setdefault(i, r["final_label"])
        truth.setdefault(i, r["truth"])

hp = pd.read_csv(OUT / "strict_v2_high_priority_human_audit_20260812.csv")
key = "item_id" if "item_id" in hp.columns else "item"
extraction = (pd.Series(dict(zip(hp[key], hp["extraction_status"])))
              if "extraction_status" in hp.columns else None)

profile = dp.add_derived_groupings(dp.item_profile(
    run, v1, v2s, human=pd.Series(truth), human_raw=pd.Series(raw),
    extraction_status=extraction))

csv_path = OUT / "dataset_distribution_by_type_20260812.csv"
profile.to_csv(csv_path, index=False)
print(f"per-item CSV -> {csv_path}  ({len(profile)} rows, "
      f"{len(profile.columns)} columns)")

groups = {by: dp.group_metrics(profile, by) for by in (
    "has_error", "answer_type", "question_type_if_available", "answer_shape",
    "truth_span_bucket", "strict_v1_correct", "strict_v2_correct")}

red = dp.red_markup_report(run)
meta = dp.metadata_availability(run)
md_path = OUT / "dataset_distribution_by_type_20260812.md"
dp.write_summary_md(str(md_path), profile, red, meta, groups)
print(f"report     -> {md_path}")

examples = dp.example_selection(profile)
info = dp.write_example_manifest(str(SHEETS), examples)
readme = SHEETS / "README.md"
readme.write_text(
    "# Example contact sheets — TEXT ONLY\n\n"
    "Open `index.html`. Every caption field the image sheet would carry is "
    "already present: item id, `has_error`, both rule verdicts, answer type, "
    "entropy, the truth and model spans, both labels, any human label, and a "
    "one-line note.\n\n"
    "**Images are not attached, and cannot be offline.** FERMAT is a gated "
    "dataset and its `image` column is never persisted to the run CSV, so no "
    "crop exists locally. **Image export requires an authenticated "
    "Colab/Drive session** — reuse notebook 23's contact-sheet machinery, "
    "keyed on `manifest.csv`'s `item_id`. That is an asset export, not a "
    "re-analysis: no number here changes when the tiles arrive.\n\n"
    f"{info['n']} items: "
    f"{int(examples['has_error'].sum())} `has_error=1`, "
    f"{int((~examples['has_error']).sum())} clean, "
    f"{int(examples['strict_v1_correct'].sum())} scored correct by "
    "`strict_v1`. Selection is seeded and round-robin over answer types.\n")
print(f"sheets     -> {SHEETS}/  ({info['n']} examples, text-only)")

print("\n--- headline splits ---")
for by in ("has_error", "question_type_if_available", "answer_shape"):
    g = groups[by].set_index("group")
    for lvl in g.index:
        print(f"{by:28s} {lvl:22s} n={int(g.loc[lvl,'n']):3d}  "
              f"v1={g.loc[lvl,'strict_v1_accuracy']:.1%}  "
              f"v2={g.loc[lvl,'strict_v2_accuracy']:.1%}  "
              f"AUROC={g.loc[lvl,'auroc_status']}")
