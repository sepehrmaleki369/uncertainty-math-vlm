"""Dry-run pilot/31_export_figure2_pages.ipynb. No network, no HF login.

Notebook 31 is the one Figure 2 step that needs gated dataset access. It has no
model and no GPU, so what has to be stubbed is Colab, the Hugging Face login and
the FERMAT loader. The corpus stub is built from the REAL run CSV, so the
question and reference-answer text the hash lookup sees is the genuine text that
produced the committed manifest, not invented fixtures.

The properties worth asserting:

  * every target page resolves by (question, reference answer) hash pair;
  * a drifted dataset FAILS instead of silently picking a page, including the
    case that matters here, where the question matches but the answer does not;
  * the pages land on Drive before the repo, since an ephemeral clone has
    already lost correctly-attached images once in this project;
  * the notebook's target items are exactly the builder's panel items.

    python pilot/dryruns/dryrun_nb31.py
"""
import hashlib
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "31_export_figure2_pages.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")

if not CSV.exists():
    sys.exit(f"SKIP: not present: {CSV}")

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

SCRATCH = ROOT / ".dryrun_scratch" / "nb31"
shutil.rmtree(SCRATCH, ignore_errors=True)
(SCRATCH / "drive").mkdir(parents=True, exist_ok=True)

# --- stub Colab, the HF login and PIL-backed images ------------------------
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda p: print(f"[stub] mount {p}"))
google = types.ModuleType("google")
google.colab = colab
sys.modules["google"] = google
sys.modules["google.colab"] = colab

# Patch login on the REAL module. Replacing huggingface_hub wholesale breaks
# `datasets`, which imports CommitInfo and friends from it at import time.
import huggingface_hub  # noqa: E402

huggingface_hub.login = lambda token=None, **kw: print("[stub] hf login")

# IPython is a Colab given, not a dependency of this repo's test environment.
_ipy = types.ModuleType("IPython")
_disp = types.ModuleType("IPython.display")
_disp.Image = lambda **kw: kw
_disp.display = lambda *a, **k: print("[stub] display")
_ipy.display = _disp
sys.modules["IPython"] = _ipy
sys.modules["IPython.display"] = _disp

# A token file so the notebook never calls getpass.
(SCRATCH / "drive").mkdir(parents=True, exist_ok=True)
(SCRATCH / "drive" / ".tokens.json").write_text(json.dumps({"HF_TOKEN": "hf_stub"}))


def sha(text):
    return hashlib.sha256(str(text if text is not None else "").encode("utf-8")).hexdigest()


run = pd.read_csv(CSV)


class FakeImage:
    """Stands in for a PIL image: records that it was saved, and where."""

    def __init__(self, tag):
        self.mode, self.size, self.tag = "RGB", (120, 160), tag

    def convert(self, mode):
        return self

    def save(self, path, quality=None):
        Path(path).write_bytes(b"\xff\xd8\xff" + str(self.tag).encode())


def make_corpus(break_item=None):
    """The real question/answer text, so the hash lookup is genuinely exercised.

    `break_item` corrupts one item's reference answer while leaving its question
    intact, which is the drift this project actually saw: `shuffle(seed=42)` is
    not stable across dataset revisions.
    """
    rows = []
    for i in run.index:
        pert = run.loc[i, "pert_a"]
        if break_item is not None and i == break_item:
            pert = str(pert) + "  DRIFTED"
        rows.append({"orig_q": run.loc[i, "orig_q"], "pert_a": pert,
                     "has_error": bool(run.loc[i, "has_error"]),
                     "image": FakeImage(i)})
    return rows


_corpus = {"rows": None}


def _stub_loader(**kw):
    return _corpus["rows"]

SHELL_RE = re.compile(r"^\s*[!%]")


def prepare(src: str) -> str:
    lines = [f"print({json.dumps('[stub] ' + ln.strip())})"
             if SHELL_RE.match(ln) else ln for ln in src.splitlines()]
    out = "\n".join(lines)
    # Cell 2 purges every `pilot.*` module, which is correct behaviour (a stale
    # clone once cost a Colab session) and which also deletes any stub applied
    # before it. Re-apply the loader stub at the point of import instead.
    out = out.replace("import pilot.data\n",
                      "import pilot.data\n"
                      "import __main__ as _dr\n"
                      "pilot.data.load_fermat_balanced = _dr._stub_loader\n")
    out = out.replace('PROJECT_DIR = "/content/drive/MyDrive/uncertainty-math-vlm"',
                      f'PROJECT_DIR = {str(SCRATCH / "drive")!r}')
    out = out.replace('REPO_DIR = os.path.abspath("nb31_repo")',
                      f'REPO_DIR = {str(SCRATCH / "repo")!r}')
    # The last cell shells out to the real builder, which would overwrite the
    # committed figure with stub images. Assert the coupling instead.
    out = out.replace('r = subprocess.run([sys.executable, "paper/build_four_quadrant.py"],\n'
                      '                   cwd=REPO_DIR, capture_output=True, text=True)',
                      'r = types.SimpleNamespace(stdout="[stub] builder not run", stderr="",\n'
                      '                          check_returncode=lambda: None)')
    out = out.replace('display(Image(filename=f"{REPO_DIR}/paper/figures/four_quadrant.png"))',
                      'print("[stub] would display the rebuilt figure")')
    return out


# The notebook copies the manifest out of its clone; point that at the checkout.
(SCRATCH / "repo" / "reference" / "wacv_evaluation_artifact").mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "reference" / "wacv_evaluation_artifact" / "fermat_n300_public_manifest.csv",
             SCRATCH / "repo" / "reference" / "wacv_evaluation_artifact")

nb = json.loads(NB.read_text())
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells; CPU-only page export (stubbed corpus)\n")

_corpus["rows"] = make_corpus()
ns = {"__name__": "__main__", "types": types}
for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 58)
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

# --- post-conditions -------------------------------------------------------
TARGETS = ns["TARGETS"]
drive_pages, repo_pages = ns["DRIVE_PAGES"], ns["REPO_PAGES"]

for item_id in TARGETS:
    name = f"item_{item_id}.jpg"
    assert (drive_pages / name).exists(), f"not on Drive: {name}"
    assert (repo_pages / name).exists(), f"not in repo: {name}"

# Each page must be the item it claims: FakeImage stamps its own run index.
for item_id in TARGETS:
    stamped = (drive_pages / f"item_{item_id}.jpg").read_bytes()[3:].decode()
    assert stamped == str(item_id), (
        f"item {item_id} got the page for run row {stamped}")

# The notebook and the builder must not drift apart.
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "b4q", ROOT / "paper" / "build_four_quadrant.py")
b4q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b4q)
assert [p[0] for p in b4q.PANELS] == TARGETS, (
    f"builder panels {[p[0] for p in b4q.PANELS]} != notebook targets {TARGETS}")

# ADVERSARIAL: a drifted reference answer must abort, not pick a page. The
# question still matches, which is the case a question-only lookup would miss.
_corpus["rows"] = make_corpus(break_item=TARGETS[0])
ns2 = {"__name__": "__main__", "types": types}
for i, src in code_cells[:4]:
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns2)
    except AssertionError as exc:
        assert "do not guess" in str(exc), exc
        print(f"\n[adversarial] drifted dataset correctly refused: {str(exc)[:90]}...")
        break
else:
    sys.exit("FAILED: a drifted dataset was accepted silently")

print("\n" + "=" * 74)
print("DRY RUN PASSED -- pages resolved by hash, Drive first, drift refused")
print("=" * 74)
print(f"targets        : {TARGETS}")
print(f"builder panels : {[p[0] for p in b4q.PANELS]}")

shutil.rmtree(SCRATCH, ignore_errors=True)
