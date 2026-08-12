"""Dry-run pilot/28_mcq_contact_sheets.ipynb with the gated dataset stubbed.

Notebook 28 renders captioned PNG contact sheets for the heuristic MCQ-like
items. FERMAT is gated, so the images only exist in Colab; here a stub dataset
supplies blank pages of the right shape and the REAL caption/paging code runs
against them.

What is worth asserting:

  * **the manifest and the sheets describe the same set, in the same order.**
    Page and cell come from `mcq_review_set`, and the notebook renders each
    group in that order; a drift between them would send a reviewer to the
    wrong page, which is worse than no sheet at all.
  * every sheet file the manifest names is actually written.
  * both groups render -- dropping `candidate_missed` would make the audit
    one-sided and bias the corrected MCQ count downward by construction.
  * captions carry every field the brief listed, and the WEAK-trigger warning
    appears on items the detector caught with no "option" word anywhere.
  * the index-alignment assert really fires when the sample order is wrong.

    python pilot/dryruns/dryrun_nb28.py

Also: `python pilot/dryruns/dryrun_nb28.py --break-alignment` proves the
notebook's alignment guard is not decorative.
"""
import argparse
import json
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "28_mcq_contact_sheets.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")
MANIFEST = ROOT / "reference" / "audit" / "mcq_like_review_20260812.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--break-alignment", action="store_true",
                    help="shuffle the stub sample so the guard must fire")
args = parser.parse_args()

for path in (CSV, MANIFEST):
    if not path.exists():
        sys.exit(f"SKIP: not present: {path}")

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import pilot.canonicalize  # noqa: E402
# Imported BEFORE the huggingface_hub stub goes in: `datasets` pulls real
# symbols (CommitInfo and friends) out of that package at import time, and a
# bare stub makes `import pilot.data` fail with an unrelated ImportError.
import pilot.data  # noqa: E402

# --- stub the Colab environment -------------------------------------------
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda p: print(f"[stub] drive.mount({p})"))
google = types.ModuleType("google")
google.colab = colab
sys.modules.setdefault("google", google)
sys.modules["google.colab"] = colab

hub = types.ModuleType("huggingface_hub")
hub.login = lambda token=None: print("[stub] huggingface_hub.login")
sys.modules["huggingface_hub"] = hub

# --- stub the gated dataset ------------------------------------------------
_run = pd.read_csv(CSV)


class _StubSample:
    """Stands in for load_fermat_balanced's return: indexable, same order."""

    def __init__(self, rows, shuffle=False):
        from PIL import Image
        self._rows = list(rows)
        if shuffle:                       # --break-alignment
            self._rows = self._rows[1:] + self._rows[:1]
        self._img = Image.new("RGB", (120, 160), "white")

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return {"orig_q": self._rows[i], "image": self._img}


def _fake_balanced(n=300, seed=42, target_error_frac=0.5, **kw):
    return _StubSample([str(q) for q in _run["orig_q"]],
                       shuffle=args.break_alignment)


pilot.data.load_fermat_balanced = _fake_balanced

# --- scratch Drive; confined to nb28, since parts of .dryrun_scratch are
# TRACKED IN GIT and wiping the shared path deletes tracked files.
SCRATCH = ROOT / ".dryrun_scratch" / "nb28"
DRIVE = str(SCRATCH / "drive")
shutil.rmtree(SCRATCH, ignore_errors=True)
Path(DRIVE, "results").mkdir(parents=True, exist_ok=True)
shutil.copy(CSV, Path(DRIVE, "results", CSV.name))

SHELL_RE = re.compile(r"^\s*[!%]")


def prepare(src: str) -> str:
    lines = [f"print({json.dumps('[stub] ' + ln.strip())})"
             if SHELL_RE.match(ln) else ln for ln in src.splitlines()]
    out = "\n".join(lines)
    out = out.replace('"/content/drive/MyDrive/uncertainty-math-vlm"', f'"{DRIVE}"')
    out = re.sub(r'with open\(f"\{PROJECT_DIR\}/\.tokens\.json"\) as f:\n'
                 r'\s*HF_TOKEN = json\.load\(f\)\["HF_TOKEN"\]',
                 'HF_TOKEN = "stub-token-never-real"', out)
    out = out.replace('sys.path.insert(0, os.path.abspath("repo"))', "pass")
    out = re.sub(r"^for _name in \[m for m in sys\.modules.*?\n    del sys\.modules\[_name\]",
                 "pass", out, flags=re.M | re.S)
    out = out.replace('"repo/reference/audit', f'"{ROOT}/reference/audit')
    return out


nb = json.loads(NB.read_text())
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells; MCQ contact sheets"
      f"{' (ALIGNMENT DELIBERATELY BROKEN)' if args.break_alignment else ''}\n")

ns = {"__name__": "__main__"}
raised = None
for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 60)
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns)
    except AssertionError as e:
        raised = e
        print(f"AssertionError: {str(e)[:200]}")
        break
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

if args.break_alignment:
    assert raised is not None and "images are index-aligned" not in str(raised), (
        "the notebook accepted a MISALIGNED sample -- captions would be "
        "attached to the wrong handwritten pages and the whole audit would be "
        "silently wrong")
    assert "match the CSV" in str(raised), str(raised)
    print("\n" + "=" * 74)
    print("GUARD PROVEN -- a shuffled sample is refused before any sheet renders")
    print("=" * 74)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    sys.exit(0)

assert raised is None, f"unexpected assertion: {raised}"

# --- post-conditions -------------------------------------------------------
import os as _os  # noqa: E402

import pilot.dataset_profile as dp  # noqa: E402

review, sheet_dir = ns["review"], ns["SHEET_DIR"]
committed = pd.read_csv(MANIFEST)

assert len(review) == len(committed), (
    f"rebuilt {len(review)} rows against a committed manifest of "
    f"{len(committed)} -- regenerate the manifest")
assert list(review["item_id"]) == list(committed["item_id"]), (
    "the rebuilt review set and the committed manifest disagree on ORDER, so "
    "the page/cell columns point at different cells")

groups = set(review["mcq_group"])
assert groups == {"flagged_mcq_like", "candidate_missed"}, (
    f"both halves must render; got {groups}. Reviewing only the flagged "
    "items can find false positives and nothing else, which biases the "
    "corrected MCQ count downward by construction.")

# Every sheet the manifest names exists, and each holds the cells claimed.
for fname, sub in review.groupby("contact_sheet_file"):
    path = _os.path.join(sheet_dir, fname)
    assert _os.path.exists(path), f"manifest names a missing sheet: {fname}"
    assert _os.path.getsize(path) > 5000, f"{fname} is suspiciously small"
    assert sorted(sub["contact_sheet_cell"]) == list(range(1, len(sub) + 1)), (
        f"{fname}: cell numbers are not 1..n")

# Paging must follow the manifest's own order, not the sheet-writing loop's.
for group, sub in review.groupby("mcq_group", sort=False):
    for n, (_, r) in enumerate(sub.iterrows()):
        page, cell = n // 9 + 1, n % 9 + 1
        assert r["contact_sheet_file"].endswith(f"_p{page}.png"), r["item_id"]
        assert r["contact_sheet_cell"] == cell, r["item_id"]

# Captions carry every field the brief listed.
sample_row = review.iloc[0]
cap = dp.mcq_caption(sample_row)
# Single-spaced: the caption helper collapses runs of whitespace, so the
# source's "span  M[" reaches the sheet as "span M[".
for token in (f"item {int(sample_row['item_id'])}", "err=", "v1=", "H=",
              "span M[", "span T[", "label M:", "label T:", "MCQ?"):
    assert token in cap, f"caption is missing {token!r}:\n{cap}"

weak = review[review["flagged_by_weak_trigger_only"].astype(bool)]
assert len(weak) >= 4, f"expected the weak-trigger items, got {len(weak)}"
for _, r in weak.iterrows():
    assert "WEAK trigger matched" in dp.mcq_caption(r), (
        f"item {r['item_id']} was flagged with no 'option' word anywhere, so "
        "the caption must show what the regex actually matched")

# The committed manifest is now CODED (2026-08-12). A freshly built review set
# still ships its human columns empty -- pinned in
# test_the_human_columns_ship_empty -- but the file on disk carries the human
# read, and asserting it is empty was a stale check that broke the moment the
# coder filled it in. Pin the coding instead.
assert (committed["confirmed_mcq"].astype(str) != "").all(), (
    "the committed manifest is coded; a blank verdict means a lost row")
assert set(committed["confirmed_mcq"]) <= {"yes", "no", "unclear"}, (
    sorted(set(committed["confirmed_mcq"])))
n_yes = int((committed["confirmed_mcq"] == "yes").sum())
assert n_yes == 55, f"confirmed MCQ moved from 55 to {n_yes}"
# 52 of the 58 flagged items were ruled by a block sweep rather than one by
# one. Recorded, because this project measured a sweep and an item-by-item
# pass disagreeing at p=0.00007 on the false-pass audit.
assert set(committed["coding_depth"].dropna()) <= {"item_by_item", "block_sweep", ""}
assert int((committed["coding_depth"] == "block_sweep").sum()) == 52

manifest_out = _os.path.join(sheet_dir, "mcq_like_review_20260812.csv")
assert _os.path.exists(manifest_out), "the sheet folder must carry its own CSV"

print("\n" + "=" * 74)
print("DRY RUN PASSED -- manifest and sheets agree, both halves rendered")
print("=" * 74)
print(review.groupby(["mcq_group", "presort"]).size().to_string())
print(f"\nsheets written: {sorted(_os.listdir(sheet_dir))}")
print(f"weak-trigger items: {sorted(weak['item_id'])}")

shutil.rmtree(SCRATCH, ignore_errors=True)
