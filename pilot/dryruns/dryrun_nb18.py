"""Dry-run pilot/18_mathverify_audit.ipynb locally, with Colab/Drive stubbed.

Notebook 18 is pure text -- no GPU, no model, no images, and it never touches
the gated FERMAT dataset. So everything except the Colab environment itself
runs for real here against the two n=300 CSVs.

    python pilot/dryruns/dryrun_nb18.py
"""

import json
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "18_mathverify_audit.ipynb"
CSVS = [
    "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv",
    "pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv",
]

for name in CSVS:
    if not (ROOT / "results" / name).exists():
        sys.exit(f"SKIP: results/{name} not present (download from Drive)")

sys.path.insert(0, str(ROOT))

import pilot.mathverify  # noqa: E402

if not pilot.mathverify.math_verify_available():
    sys.exit("SKIP: math-verify not installed (pip install math-verify)")

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

DRIVE = ROOT / ".dryrun_scratch18" / "drive"
(DRIVE / "results").mkdir(parents=True, exist_ok=True)
for name in CSVS:
    shutil.copy(ROOT / "results" / name, DRIVE / "results" / name)

SHELL_RE = re.compile(r"^\s*[!%]")


def prepare(src: str) -> str:
    lines = [f"print({json.dumps('[stub] ' + ln.strip())})" if SHELL_RE.match(ln)
             else ln for ln in src.splitlines()]
    out = "\n".join(lines)
    out = out.replace('"/content/drive/MyDrive/uncertainty-math-vlm"', f'"{DRIVE}"')
    out = re.sub(r'with open\(f"\{PROJECT_DIR\}/\.tokens\.json"\) as f:\n'
                 r'\s*HF_TOKEN = json\.load\(f\)\["HF_TOKEN"\]',
                 'HF_TOKEN = "stub-token-never-real"', out)
    out = out.replace('sys.path.insert(0, os.path.abspath("repo"))', "pass")
    out = re.sub(r"^for _name in \[m for m in sys\.modules.*?\n    del sys\.modules\[_name\]",
                 "pass", out, flags=re.M | re.S)
    return out


nb = json.loads(NB.read_text())
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells\n")

ns = {"__name__": "__main__"}
for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 58)
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

print("\n" + "=" * 72)
print("DRY RUN PASSED -- every cell executed against the real n=300 data")
print("=" * 72)

# --- post-conditions ------------------------------------------------------
assert ns["sanity"]["ok"].all(), "a Math-Verify sanity case regressed"
queues = ns["queues"]
assert set(queues) == {"Qwen2.5-VL-3B", "Pixtral-12B"}
for name, q in queues.items():
    assert len(q), f"{name}: an empty queue means the comparison is not running"
    assert set(q["direction"]) <= {"string_wrong_mv_right", "string_right_mv_wrong"}
# The decision this notebook supports: Math-Verify must NOT be adopted blindly.
qw = queues["Qwen2.5-VL-3B"]
audit = qw[qw.direction == "string_wrong_mv_right"]
assert 108 in set(audit["item"]), "the coefficient false positive must be queued"
print(f"post-conditions OK: "
      f"{ {k: len(v) for k, v in queues.items()} } disagreements queued")
shutil.rmtree(ROOT / ".dryrun_scratch18", ignore_errors=True)
