"""Dry-run pilot/29_confusion_examples_for_professor.ipynb, dataset stubbed.

Notebook 29 renders captioned PNG sheets showing TP / FP / FN / TN and the
INDETERMINATE group. FERMAT is gated, so the images exist only in Colab; here
a stub supplies blank pages of the right shape and the REAL selection,
captioning and paging code runs against them.

What is worth asserting, given a professor will read these at face value:

  * **the five groups are all rendered.** Dropping INDETERMINATE would present
    a clean 2x2 that silently omits the largest audited group and inflates the
    scorer's apparent accuracy.
  * **`extraction_issue` lands on the correct side of the matrix**: FP when
    the scorer passed, INDETERMINATE when it failed. Collapsing the two is the
    single easiest way to misrepresent this audit.
  * every requested item is present and first in its group.
  * captions carry every field asked for, and the `why:` line is WRAPPED, not
    truncated -- it was truncated mid-word until a rendered page was looked at.
  * manifest page/cell matches the render order.
  * the README defines the groups before it shows any count.

    python pilot/dryruns/dryrun_nb29.py
"""
import json
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "29_confusion_examples_for_professor.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")

if not CSV.exists():
    sys.exit(f"SKIP: not present: {CSV}")

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import pilot.canonicalize  # noqa: E402
# Imported BEFORE the huggingface_hub stub: `datasets` pulls real symbols out
# of that package at import time and a bare stub breaks it.
import pilot.data  # noqa: E402

colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda p: print(f"[stub] drive.mount({p})"))
google = types.ModuleType("google")
google.colab = colab
sys.modules.setdefault("google", google)
sys.modules["google.colab"] = colab

hub = types.ModuleType("huggingface_hub")
hub.login = lambda token=None: print("[stub] huggingface_hub.login")
sys.modules["huggingface_hub"] = hub

_run = pd.read_csv(CSV)


class _StubSample:
    def __init__(self, rows):
        from PIL import Image
        self._rows = list(rows)
        self._img = Image.new("RGB", (240, 320), "white")

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return {"orig_q": self._rows[i], "image": self._img}


pilot.data.load_fermat_balanced = (
    lambda n=300, seed=42, target_error_frac=0.5, **kw:
    _StubSample([str(q) for q in _run["orig_q"]]))

# Confined to nb29: parts of .dryrun_scratch are TRACKED IN GIT, so wiping the
# shared path deletes tracked files.
SCRATCH = ROOT / ".dryrun_scratch" / "nb29"
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
print(f"{len(code_cells)} code cells; confusion examples for review\n")

ns = {"__name__": "__main__"}
for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 60)
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

# --- post-conditions -------------------------------------------------------
import os as _os  # noqa: E402

import pilot.dataset_profile as dp  # noqa: E402

ex, sheet_dir, pop = ns["examples"], ns["SHEET_DIR"], ns["population"]
v1, audit = ns["v1"], ns["audit"]

# All five groups, and INDETERMINATE is not silently dropped.
groups = set(ex["category"])
assert groups == set(dp.CONFUSION_ORDER), (
    f"expected all five groups, got {sorted(groups)}. Omitting INDETERMINATE "
    "presents a clean 2x2 that drops the largest audited group and inflates "
    "the scorer's apparent accuracy.")

# `extraction_issue` must land on BOTH sides of the matrix, never one.
ei = audit[audit["final_label"] == "extraction_issue"].index
sides = {dp.confusion_category(v1.loc[i], "extraction_issue") for i in ei}
assert sides == {"FP", "INDETERMINATE"}, (
    f"extraction_issue collapsed onto {sides}; an unearned PASS is a false "
    "pass and an unearned FAIL leaves the model's answer undecided")
assert dp.confusion_category(True, "extraction_issue") == "FP"
assert dp.confusion_category(False, "extraction_issue") == "INDETERMINATE"
assert dp.confusion_category(False, "true_correct") == "FN"
assert dp.confusion_category(False, "notation_misread") == "TN"

# Requested items present, and FIRST in their group.
for cat, wanted in ns["PREFER"].items():
    got = ex.loc[ex["category"] == cat, "item_id"].tolist()
    for i in wanted:
        assert i in got, f"requested item {i} missing from {cat}"
    assert got[:len(wanted)] == list(wanted), (
        f"{cat}: requested items must come first, got {got[:len(wanted)]}")

# Every field the brief asked to burn in.
cap = dp.confusion_caption(ex.iloc[0])
for token in ("[", "item ", "err=", "H=", "scorer v1=", "v2=", "human=",
              "span M:", "span T:", "label M:", "label T:", "why:"):
    assert token in cap, f"caption missing {token!r}:\n{cap}"

# The `why:` line is WRAPPED, not truncated. It was cut mid-word ("the match
# is vacu...") until a rendered page was actually looked at, and it is the one
# line a reader outside this project needs.
long_note = ex.loc[ex["note"].str.len() > 58]
assert len(long_note), "fixture: expected at least one note longer than a line"
for _, r in long_note.iterrows():
    lines = dp.confusion_caption(r).splitlines()
    why = " ".join(x for x in lines if x.startswith("why:") or lines.index(x) > 5)
    assert not why.rstrip().endswith("..."), f"note truncated on item {r['item_id']}"

# Mechanism diversity: a group built from several human labels must show more
# than one, or six identical tiles teach a reviewer nothing.
tn_labels = set(ex.loc[ex["category"] == "TN", "human_label"])
assert len(tn_labels) >= 2, f"TN shows only {tn_labels}"

# Paging follows the manifest's own order.
for cat, sub in ex.groupby("category", sort=False):
    for n, (_, r) in enumerate(sub.iterrows()):
        assert r["contact_sheet_file"] == f"confusion_{cat}_p{n // 6 + 1}.png"
        assert r["contact_sheet_cell"] == n % 6 + 1

# Files really written.
for fname in ex["contact_sheet_file"].unique():
    p = _os.path.join(sheet_dir, fname)
    assert _os.path.exists(p) and _os.path.getsize(p) > 5000, fname
for name in ("manifest.csv", "README.md"):
    assert _os.path.exists(_os.path.join(sheet_dir, name)), name
assert len(pd.read_csv(_os.path.join(sheet_dir, "manifest.csv"))) == len(ex)

readme = open(_os.path.join(sheet_dir, "README.md")).read()
assert "describe the SCORER, not the model" in readme
assert "five groups" in readme.lower()
assert "not a population rate" in readme
# Bold markers sit inside this sentence (`**not** a measurement`), so match
# the surrounding words rather than the phrase.
assert "a measurement" in readme and "should be counted" in readme
# The groups must be DEFINED before any count appears.
assert readme.index("| **TP** |") < readme.index("items audited"), (
    "the README shows counts before defining what they count")

print("\n" + "=" * 74)
print("DRY RUN PASSED -- five groups, extraction_issue split correctly, "
      "captions intact")
print("=" * 74)
print(pd.crosstab(ex["category"], ex["human_label"]).to_string())
print(f"\nsheets: {sorted(_os.listdir(sheet_dir))}")

shutil.rmtree(SCRATCH, ignore_errors=True)
