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
SPOT60 = (ROOT / "reference" / "audit"
          / "spotcheck_extra60_qwen_strict_v1_correct_20260812.csv")

for p in (CSV, SPOT, SPOT60):
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

written = ns["written"]
sheets = ns["sheets"]
assert set(written) == {"first40", "extra60"}, written
EXPECT = {"first40": ("spotcheck_strict_v1_correct", "spotcheck_correct_p", 40, 4),
          "extra60": ("spotcheck_strict_v1_correct_extra60",
                      "spotcheck_correct_extra_p", 60, 0)}
for name, (subdir, stem, n_items, n_cal) in EXPECT.items():
    df, sub, _ = sheets[name]
    assert sub == subdir, f"{name}: {sub}"
    assert len(df) == n_items and df["item"].is_unique
    assert int(df["known_false_pass"].sum()) == n_cal
    paths = written[name]
    assert len(paths) == -(-n_items // 9), f"{name}: {len(paths)} pages"
    for p in paths:
        assert _os.path.exists(p) and _os.path.getsize(p) > 10_000, p
        assert _os.path.basename(p).startswith(stem), p
        assert f"/figures/{subdir}/" in p, p
    assert _os.path.exists(_os.path.join(_os.path.dirname(paths[0]),
                                         "coding_sheet.csv"))

# Separate folders, or the second draw would overwrite the first draw's pages.
d1 = _os.path.dirname(written["first40"][0])
d2 = _os.path.dirname(written["extra60"][0])
assert d1 != d2, "both draws wrote to the same Drive folder"

# The captions are the whole point of the sheet.
for name, (df, _, _) in sheets.items():
    for _, r in df.iterrows():
        cap = ns["caption_for"](r)
        assert f"item {int(r['item'])}" in cap and "H=" in cap
        # BOTH automatic views must be present: the extracted span and the
        # comparison label, for model and truth alike.
        # NB: the caption helper collapses runs of whitespace, so the
        # rendered text is "span M[...]", single-spaced.
        assert "span M[" in cap and "span T[" in cap, cap
        assert "label M:" in cap and "label T:" in cap, cap
        assert max(len(l) for l in cap.split(chr(10))) <= 56, (
            "caption line overflows its cell and collides with the next one; "
            "found by RENDERING a sheet and looking at it, not by asserts")
        assert ("KNOWN-FALSE-PASS" in cap) == bool(r["known_false_pass"]), cap
print(f"post-conditions OK: {sum(len(v) for v in written.values())} pages across "
      f"2 folders, {sum(len(s[0]) for s in sheets.values())} items")
print("sample caption:\n" + ns["caption_for"](sheets["extra60"][0].iloc[0]))

shutil.rmtree(ROOT / ".dryrun_scratch", ignore_errors=True)
