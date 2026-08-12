"""Dry-run pilot/25_livemath_judge_diagnostic.ipynb, judge and GPU stubbed.

Notebook 25 is the EXPLORATORY diagnostic run after notebook 24's gate failed.
It has no gate by design, so the properties worth asserting are different:
the sample must be exactly 20/20 with both probes present, both prompts must
run, and the summary must contain NO accuracy figure.

The judge is stubbed to reproduce the real failure -- it rejects item 55's
sign flip but ACCEPTS item 273's repaired fraction, which is what actually
happened on the GPU.

    python pilot/dryruns/dryrun_nb25.py
"""
import argparse
import json
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "25_livemath_judge_diagnostic.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")

parser = argparse.ArgumentParser()
args = parser.parse_args()

if not CSV.exists():
    sys.exit(f"SKIP: not present: {CSV}")

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import pilot.canonicalize  # noqa: E402

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

# --- stub torch + transformers: no GPU, no 3B download --------------------
torch = types.ModuleType("torch")
torch.bfloat16 = "bfloat16"
torch.cuda = types.SimpleNamespace(is_available=lambda: True,
                                   get_device_name=lambda i: "stub-GPU")


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


torch.no_grad = _NoGrad
sys.modules["torch"] = torch

tf = types.ModuleType("transformers")


class _Tok:
    pad_token_id, eos_token_id = 0, 0

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def apply_chat_template(self, msgs, return_tensors=None, return_dict=None,
                            add_generation_prompt=None):
        assert return_dict, (
            "the notebook must pass return_dict=True -- the model card's own "
            "snippet does not, and then subscripts the returned TENSOR")
        # carry the prompt through so the stub judge can see it. Returns a
        # dict WITH .to(), because real transformers hands back a
        # BatchEncoding -- a plain dict would fail on .to(model.device) here
        # while working fine in Colab, i.e. a stub-only false alarm.
        return _Batch({"input_ids": _Ids(msgs[0]["content"]),
                       "attention_mask": _Ids("")})

    def decode(self, new, skip_special_tokens=True):
        return new


class _Batch(dict):
    def to(self, device):
        return self


class _Ids:
    def __init__(self, payload):
        self.payload = payload
        self.shape = (1, 0)

    def to(self, dev):
        return self

    def __getitem__(self, k):
        return self


class _Model:
    device = "cpu"

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None,
                 max_new_tokens=None, **k):
        assert max_new_tokens and max_new_tokens > 100, (
            "max_new_tokens must be set: the model writes an analysis before "
            "the boxed verdict, and the card's default of 20 truncates it")
        prompt = input_ids.payload
        # Reproduces the REAL observed behaviour: rejects item 55's sign flip,
        # ACCEPTS item 273's repaired fraction while narrating the arithmetic.
        if "1 + \\tan x \\tan y" in prompt:
            reply = r"Analysis: the examinee altered the sign. \boxed{no}"
        elif "\\frac{2}{4}" in prompt:
            reply = (r"Analysis: there are 3 favourable outcomes, so the "
                     r"correct answer is 3/4. \boxed{yes}")
        else:
            reply = (r"Analysis: matches. \boxed{yes}"
                     if hash(prompt) % 2 else r"Analysis: differs. \boxed{no}")
        return [_Payload(reply)]


class _Payload:
    def __init__(self, s):
        self.s = s

    def __getitem__(self, k):
        return self.s


tf.AutoTokenizer = _Tok
tf.AutoModelForCausalLM = _Model
sys.modules["transformers"] = tf

# --- scratch Drive ---------------------------------------------------------
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
    out = out.replace('"repo/reference/audit"', f'"{ROOT}/reference/audit"')
    return out


nb = json.loads(NB.read_text())
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells; EXPLORATORY diagnostic (no gate)\n")

ns = {"__name__": "__main__"}
raised = None
for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 60)
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns)
    except RuntimeError as e:
        raised = e
        print(f"RuntimeError (expected in fail-gate mode): {str(e)[:160]}")
        break
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

# --- post-conditions -------------------------------------------------------
import os as _os  # noqa: E402

assert raised is None, f"the diagnostic must NOT gate: {raised}"
sample, both = ns["sample"], ns["both"]
assert len(sample) == 40 and len(both) == 40
assert int(sample["has_error"].sum()) == 20, "20 has_error=1 required"
assert int((~sample["has_error"]).sum()) == 20, "20 clean required"
for probe in (55, 273):
    assert probe in both.index, f"probe {probe} missing"
assert sample["answer_type"].nunique() >= 6, "answer shapes must be spread"
assert int((sample["human_label"] != "").sum()) >= 25, (
    "human-labelled items should be preferred by the sampler")

# Both prompts really ran and are recorded separately.
for c in ("verdict_native", "verdict_fidelity", "raw_native", "raw_fidelity",
          "solving_native", "solving_fidelity", "answer_type", "human_label"):
    assert c in both.columns, c

# The observed failure must survive into the output.
assert both.loc[55, "verdict_fidelity"] == "incorrect"
assert both.loc[273, "verdict_fidelity"] == "correct", (
    "the stub reproduces the real 273 acceptance")
assert bool(both.loc[273, "solving_fidelity"]), (
    "the solving heuristic must catch a judge narrating the arithmetic")

csv_path, md_path = ns["CSV_PATH"], ns["MD_PATH"]
assert _os.path.basename(csv_path) == "livemath_judge_diagnostic40_20260812.csv"
assert _os.path.basename(md_path) == "livemath_judge_diagnostic40_summary_20260812.md"
assert _os.path.exists(csv_path) and _os.path.exists(md_path)
assert len(pd.read_csv(csv_path)) == 40

text = open(md_path).read()
low = text.lower()
assert "not a scoring run" in low
assert "gate failed" in low
for section in ("## Sample composition", "## Native vs fidelity prompt",
                "## Against determinate human labels", "## has_error=1 behaviour",
                "## Judge appears to solve", "## Parse failures"):
    assert section in text, section
# The one thing this file must never contain.
import re as _re  # noqa: E402
assert not _re.search(r"\baccuracy\b\s*[:=]", text, _re.I), (
    "the diagnostic summary must report NO accuracy figure")

print("\n" + "=" * 74)
print("DRY RUN PASSED -- 40 items, both prompts, no accuracy emitted")
print("=" * 74)
print(f"prompts disagree on "
      f"{int((both.verdict_native != both.verdict_fidelity).sum())} of 40; "
      f"solving flagged on "
      f"{int((both.solving_native | both.solving_fidelity).sum())}")

shutil.rmtree(ROOT / ".dryrun_scratch", ignore_errors=True)
