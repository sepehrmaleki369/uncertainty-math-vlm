"""Dry-run pilot/22_sample_representativeness.ipynb locally.

Only the CORPUS is stubbed -- 87% error, with the difficulty flags made
INDEPENDENT of has_error, mirroring the real dataset. `load_fermat_balanced`
runs for real against it, so the stratification this notebook is about is
actually exercised rather than faked.

The claim under test: the drawn sample matches the corpus on every observable
field except the deliberate error balance. If the selection ever started
looking at difficulty, the post-conditions here would fail.

    python pilot/dryruns/dryrun_nb22.py
"""

import json
import random
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NB = ROOT / "pilot" / "22_sample_representativeness.ipynb"
RESULTS = ROOT / "results"

CSVS = ["scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv",
        "pixtral_perception_full_n300_pixtral-12b_20260809T211028Z.csv"]
for name in CSVS:
    if not (RESULTS / name).exists():
        sys.exit(f"SKIP: results/{name} not present (download from Drive)")

cells = json.loads(NB.read_text())["cells"]


def src(i):
    return "".join(cells[i]["source"])


# --- stub Colab -----------------------------------------------------------
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda p: print(f"[stub] drive.mount({p})"))
google = types.ModuleType("google")
google.colab = colab
sys.modules.setdefault("google", google)
sys.modules["google.colab"] = colab


# --- stub corpus: 87% error, difficulty INDEPENDENT of has_error ---------
class DS(list):
    def __getitem__(self, k):
        return [r[k] for r in self] if isinstance(k, str) else list.__getitem__(self, k)

    @property
    def column_names(self):
        return list(self[0]) if self else []

    def shuffle(self, seed):
        r = list(self)
        random.Random(seed).shuffle(r)
        return DS(r)

    def select(self, idx):
        return DS([self[i] for i in idx])

    def remove_columns(self, cols):
        return DS([{k: v for k, v in r.items() if k not in cols} for r in self])


CORPUS = DS([
    {"image": None, "orig_q": "q" * (50 + i % 30), "pert_a": "a" * (100 + i % 200),
     "has_error": i % 100 < 87, "handwriting_style": i % 3 != 0,
     "image_quality": i % 4 != 0}
    for i in range(2200)])

import datasets  # noqa: E402
datasets.load_dataset = lambda *a, **k: CORPUS

import pilot.data  # noqa: E402  -- REAL load_fermat_balanced runs on the stub

# Stub huggingface_hub LAST: doing it earlier shadows the real module before
# `datasets` can import CommitInfo from it, and datasets fails to load at all.
hub = types.ModuleType("huggingface_hub")
hub.login = lambda token=None: print("[stub] huggingface_hub.login")
sys.modules["huggingface_hub"] = hub

WORKDIR = tempfile.mkdtemp(prefix="dryrun_nb22_")
SHELL = re.compile(r"^\s*[!%]")


def prepare(s):
    lines = [f"print({json.dumps('[stub] ' + ln.strip())})" if SHELL.match(ln) else ln
             for ln in s.splitlines()]
    out = "\n".join(lines)
    out = out.replace('"/content/drive/MyDrive/uncertainty-math-vlm"',
                      f'"{WORKDIR}"')
    out = re.sub(r'with open\(f"\{PROJECT_DIR\}/\.tokens\.json"\) as f:\n'
                 r'\s*HF_TOKEN = json\.load\(f\)\["HF_TOKEN"\]',
                 'HF_TOKEN = "stub"', out)
    out = out.replace('sys.path.insert(0, os.path.abspath("repo"))', "pass")
    out = re.sub(r"^for _name in \[m for m in sys\.modules.*?\n    del sys\.modules\[_name\]",
                 "pass", out, flags=re.M | re.S)
    out = out.replace('RESULTS_DIR = f"{PROJECT_DIR}/results"',
                      f'RESULTS_DIR = "{RESULTS}"')
    return out


code_cells = [(i, src(i)) for i, c in enumerate(cells) if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells\n")
ns = {"__name__": "__main__"}
for i, s in code_cells:
    print(f"----- cell {i} " + "-" * 56)
    try:
        exec(compile(prepare(s), f"<cell {i}>", "exec"), ns)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

print("\n" + "=" * 72)
print("DRY RUN PASSED")
print("=" * 72)

# --- post-conditions ------------------------------------------------------
import pandas as pd  # noqa: E402

comparison = ns["comparison"].set_index("field")
assert comparison.loc["frac_has_error", "delta"] < -0.30, comparison.to_string()
for field in ("frac_handwriting_style", "frac_image_quality",
              "median_answer_chars", "median_question_chars"):
    if field not in comparison.index:
        continue
    rel = abs(comparison.loc[field, "delta"]) / max(abs(comparison.loc[field, "corpus"]), 1)
    assert rel < 0.05, (field, comparison.loc[field].to_dict())
assert pd.isna(comparison.loc["n", "delta"]), "n must carry no delta"
assert len(ns["sample"]) == 300
print("post-conditions OK: only frac_has_error moves; every other field <5%")
shutil.rmtree(WORKDIR, ignore_errors=True)
