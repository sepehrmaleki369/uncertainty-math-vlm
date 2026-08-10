"""Dry-run pilot/17_scoring_inspection.ipynb locally, with Colab/Drive/FERMAT stubbed.

Notebook 17 is inspection-only -- no GPU, no model -- so almost all of it can
run for real here. What must be faked is only the environment: google.colab,
the Drive mount, the HF login, the `!git clone` shell escapes, and
matplotlib's display. The FERMAT sample is stubbed too (the dataset is gated),
but built FROM the real CSV so the order-alignment assert in cell 2 is
exercised against real question text rather than a synthetic stand-in.

Everything that matters -- rescore_run, scoring_sensitivity, the bucket
selection, trace_item, show_item's formatting -- runs on the real n=300 data.

    python pilot/dryruns/dryrun_nb17.py [--fast]

--fast lowers n_boot so the run takes ~2 min instead of ~6; use the full form
before an actual handoff.
"""

import argparse
import ast
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "17_scoring_inspection.ipynb"
CSVS = {
    "Qwen2.5-VL-3B": ROOT / "results"
    / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv",
    "Pixtral-12B": ROOT / "results"
    / "pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv",
}
CSV = CSVS["Qwen2.5-VL-3B"]  # the sample stub is built from this one

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true")
args = parser.parse_args()

missing = [str(p) for p in CSVS.values() if not p.exists()]
if missing:
    sys.exit(f"SKIP: not present (download from Drive): {missing}")

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

# matplotlib: headless, and never actually render
import matplotlib  # noqa: E402
matplotlib.use("Agg")

# --- stub the gated dataset, using REAL question text ----------------------
_real_df = pd.read_csv(CSV)


from PIL import Image  # noqa: E402

# A REAL PIL image, not a duck-typed stand-in: show_item passes it to
# plt.imshow, which needs actual pixel data. A fake object passes .save/.width
# and then fails inside matplotlib -- caught by this dry run on first pass.
_STUB_IMAGE = Image.new("RGB", (800, 600), "white")


def _fake_load_fermat_balanced(n, seed, target_error_frac):
    assert (n, seed, target_error_frac) == (300, 42, 0.5), (
        f"notebook asked for n={n} seed={seed} frac={target_error_frac}; the "
        "stub only mirrors the run the CSV came from")
    return [{"orig_q": q, "pert_a": a, "has_error": bool(h), "image": _STUB_IMAGE}
            for q, a, h in zip(_real_df["orig_q"], _real_df["pert_a"],
                               _real_df["has_error"])]


pilot.data.load_fermat_balanced = _fake_load_fermat_balanced

# --- rewrite the cells that cannot run verbatim ---------------------------
SHELL_RE = re.compile(r"^\s*[!%]")


def prepare(src: str, idx: int) -> str:
    lines = []
    for line in src.splitlines():
        if SHELL_RE.match(line):
            lines.append(f"print({json.dumps('[stub] ' + line.strip())})")
        else:
            lines.append(line)
    out = "\n".join(lines)

    # Drive paths -> a local scratch dir; the token file does not exist here.
    out = out.replace('"/content/drive/MyDrive/uncertainty-math-vlm"',
                      f'"{DRIVE}"')
    out = re.sub(r'with open\(f"\{PROJECT_DIR\}/\.tokens\.json"\) as f:\n'
                 r'\s*HF_TOKEN = json\.load\(f\)\["HF_TOKEN"\]',
                 'HF_TOKEN = "stub-token-never-real"', out)
    # sys.path/clone bookkeeping is meaningless locally and would shadow the
    # working tree we are actually testing.
    out = out.replace('sys.path.insert(0, os.path.abspath("repo"))', "pass")
    out = re.sub(r"^for _name in \[m for m in sys\.modules.*?\n    del sys\.modules\[_name\]",
                 "pass", out, flags=re.M | re.S)
    if args.fast:
        out = out.replace("n_boot=10000", "n_boot=300")
    # Only render the first item of each display cell -- the formatting is
    # what is being tested, not repetition.
    out = re.sub(r"\[:(\d)\]:\n    show_item", "[:1]:\n    show_item", out)
    return out


DRIVE = str(ROOT / ".dryrun_scratch" / "drive")
Path(DRIVE, "results").mkdir(parents=True, exist_ok=True)
import shutil  # noqa: E402
for _csv in CSVS.values():
    shutil.copy(_csv, Path(DRIVE, "results", _csv.name))

nb = json.loads(NB.read_text())
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells\n")

ns = {"__name__": "__main__"}
import matplotlib.pyplot as plt  # noqa: E402
plt.show = lambda *a, **k: None

for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 60)
    try:
        exec(compile(prepare(src, i), f"<cell {i}>", "exec"), ns)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

print("\n" + "=" * 74)
print("DRY RUN PASSED -- every cell executed against the real n=300 data")
print("=" * 74)

# --- post-conditions the notebook is supposed to establish ----------------
from pilot.rescore import CATEGORIES  # noqa: E402

qwen = ns["sens"]["Qwen2.5-VL-3B"].set_index("rule")
assert bool(qwen["excludes_chance"].all()), "a rule stopped excluding chance"
assert qwen.loc["strict_v1", "n_correct"] == 141
assert qwen.loc["fixed_v2", "n_correct"] == 141
assert qwen.loc["relaxed_v3", "n_correct"] == 162
assert qwen.loc["final_term_v4", "n_correct"] == 190
assert bool(ns["sens"]["Pixtral-12B"]["excludes_chance"].all())

EXPECTED = {
    "Qwen2.5-VL-3B": {"correct_robust": 134, "bug_fix_recovered": 6,
                      "cosmetic_mismatch": 22, "scope_mismatch": 27,
                      "false_pass_removed": 6, "broken_by_relaxation": 1,
                      "genuinely_wrong": 104},
    "Pixtral-12B": {"correct_robust": 118, "bug_fix_recovered": 9,
                    "cosmetic_mismatch": 11, "scope_mismatch": 25,
                    "false_pass_removed": 6, "broken_by_relaxation": 1,
                    "genuinely_wrong": 130},
}
for model, expected in EXPECTED.items():
    counts = ns["classified"][model]["category"].value_counts().to_dict()
    assert counts == expected, f"{model}: {counts} != {expected}"
    assert sum(counts.values()) == 300
    # Every category the notebook renders an example from must be non-empty.
    for category in ("false_pass_removed", "cosmetic_mismatch",
                     "scope_mismatch", "genuinely_wrong"):
        assert counts.get(category), f"{model}: no {category} to show"
    # The taxonomy must reconcile with the sensitivity table it sits beside.
    n_v1 = ns["sens"][model].set_index("rule").loc["strict_v1", "n_correct"]
    assert (counts["correct_robust"] + counts["false_pass_removed"]
            + counts["broken_by_relaxation"]) == n_v1, (
        f"{model}: categories disagree with strict_v1's n_correct")

assert set(ns["summary"]["category"]) <= set(CATEGORIES)
print("post-conditions OK: category counts match on both models, and reconcile "
      "with each run's strict_v1 accuracy")
shutil.rmtree(ROOT / ".dryrun_scratch", ignore_errors=True)
