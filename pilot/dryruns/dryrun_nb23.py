"""Dry-run pilot/23_spotcheck_contact_sheets.ipynb locally, Colab/Drive/FERMAT stubbed.

Notebook 23 renders contact sheets for the 40 randomly drawn `strict_v1`-CORRECT
items so they can be read for FALSE PASSES -- the other half of the one-sided
2026-08-11 audit.

No GPU and no model, so nearly all of it runs for real here. Only the
environment is faked: google.colab, the Drive mount, the HF login, the shell
escapes, and matplotlib's display. The FERMAT sample is stubbed (the dataset is
gated) but built FROM the real CSV, so the order-alignment assert is exercised
against real question text rather than a synthetic stand-in.

**The SAVE cell runs too, and its output filenames are asserted.** Notebooks 19
and 20 both shipped with a save cell that had never executed -- one crashed on
an undefined name, and both still carried notebook 16's hardcoded `pixtral`
filename, so a green dry run that stopped early would have written a CSV
labelled as a different model's run.

    python pilot/dryruns/dryrun_nb23.py
"""

import json
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "23_spotcheck_contact_sheets.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")
SPOT = (ROOT / "reference" / "audit"
        / "spotcheck_40_qwen_strict_v1_correct_20260811.csv")

for p in (CSV, SPOT):
    if not p.exists():
        sys.exit(f"SKIP: not present: {p}")

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import pilot.canonicalize  # noqa: E402
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.show = lambda *a, **k: None

# --- stub the gated dataset, using REAL question text ----------------------
_real_df = pd.read_csv(CSV)

from PIL import Image  # noqa: E402

# A REAL PIL image: contact_sheet hands it to plt.imshow, which needs actual
# pixel data. Sizes vary so the grid layout is exercised on ragged input.
_IMAGES = [Image.new("RGB", (700 + 10 * (i % 7), 520 + 8 * (i % 5)), "white")
           for i in range(len(_real_df))]


def _fake_load_fermat_balanced(n, seed, target_error_frac):
    assert (n, seed, target_error_frac) == (300, 42, 0.5), (
        f"notebook asked for n={n} seed={seed} frac={target_error_frac}; the "
        "stub only mirrors the run the CSV came from")
    return [{"orig_q": q, "pert_a": a, "has_error": bool(h), "image": _IMAGES[i]}
            for i, (q, a, h) in enumerate(zip(
                _real_df["orig_q"], _real_df["pert_a"], _real_df["has_error"]))]


pilot.data.load_fermat_balanced = _fake_load_fermat_balanced

# --- rewrite the cells that cannot run verbatim ---------------------------
DRIVE = str(ROOT / ".dryrun_scratch" / "drive")
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
    # The notebook reads the spot-check CSV out of the cloned repo; locally it
    # lives in the working tree.
    out = out.replace('"repo/reference/audit/', f'"{ROOT}/reference/audit/')
    return out


nb = json.loads(NB.read_text())
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells\n")

ns = {"__name__": "__main__"}
for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 60)
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

print("\n" + "=" * 74)
print("DRY RUN PASSED -- every cell executed against the real n=300 data")
print("=" * 74)

# --- post-conditions -------------------------------------------------------
import os as _os  # noqa: E402

sheet_dir = ns["SHEET_DIR"]
written = ns["written"]
assert sheet_dir.endswith("/figures/spotcheck_strict_v1_correct"), sheet_dir
assert len(written) == 4, f"40 items at 12/page should be 4 pages, got {len(written)}"
for p in written:
    assert _os.path.exists(p) and _os.path.getsize(p) > 10_000, p
    assert _os.path.basename(p).startswith("spotcheck_correct_p"), p
assert _os.path.exists(f"{sheet_dir}/coding_sheet.csv")

spot = ns["spot"]
assert len(spot) == 40 and spot["item"].is_unique
assert int(spot["known_false_pass"].sum()) == 4, "the 4 calibration items must survive"

# The captions are the whole point of the sheet: every one must name its item,
# both labels and the entropy, and the 4 calibration items must be marked.
caps = [ns["caption_for"](r) for _, r in spot.iterrows()]
assert len(caps) == 40
for (_, r), cap in zip(spot.iterrows(), caps):
    assert f"item {int(r['item'])}" in cap and "H=" in cap
    assert "model:" in cap and "truth:" in cap
    assert ("KNOWN FALSE PASS" in cap) == bool(r["known_false_pass"]), cap
print(f"post-conditions OK: {len(written)} pages, 40 items, "
      f"{int(spot['known_false_pass'].sum())} flagged calibration items")
print("sample caption:\n" + caps[0])

shutil.rmtree(ROOT / ".dryrun_scratch", ignore_errors=True)
