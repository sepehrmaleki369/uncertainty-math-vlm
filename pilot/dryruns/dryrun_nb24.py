"""Dry-run pilot/24_livemath_judge.ipynb locally, with Colab/GPU/LiveMath stubbed.

Notebook 24 runs `jnanliu/LiveMath-Judge` as a third, independent scorer on
all 300. The judge itself is a 3B model and never runs here -- it is replaced
by a scripted stub -- but everything around it does: the real n=300 CSV, the
real strict_v1 and strict_v2 rescoring, the real audit label loading, the real
gate logic and both real output writers.

Two modes, because the gate is the point of the notebook:

    python pilot/dryruns/dryrun_nb24.py            # judge rejects -> gate passes
    python pilot/dryruns/dryrun_nb24.py --fail-gate  # judge accepts -> must RAISE

The second is the one that matters. A gate that cannot stop the run is not a
gate, and notebooks 19 and 20 both shipped with a pre-flight that only printed.
"""

import argparse
import json
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "24_livemath_judge.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--fail-gate", action="store_true",
                    help="stub a judge that accepts the corrected errors")
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
        # A gated judge accepts everything; a good one rejects the two
        # silent-correction items and otherwise mirrors a plausible mix.
        if args.fail_gate:
            reply = r"Analysis: equivalent. \boxed{yes}"
        elif "1 + \\tan x \\tan y" in prompt or "\\frac{2}{4}" in prompt:
            reply = r"Analysis: the examinee altered the answer. \boxed{no}"
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
print(f"{len(code_cells)} code cells; mode="
      f"{'FAIL-GATE (must raise)' if args.fail_gate else 'gate passes'}\n")

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
if args.fail_gate:
    assert raised is not None, (
        "THE GATE DID NOT STOP THE RUN. A judge that accepted both silent "
        "corrections was allowed through to the 300-item run.")
    assert "GATE FAILED" in str(raised)
    assert "REPORTABLE RESULT" in str(raised), (
        "the failure message must say a failed gate is a result, not a bug")
    assert "judged" not in ns, "the 300-item run must not have started"
    print("\n" + "=" * 74)
    print("DRY RUN PASSED -- the gate stopped the run, as it must")
    print("=" * 74)
else:
    assert raised is None
    judged = ns["judged"]
    assert len(judged) == 300
    for c in ("question", "ground_truth_answer", "model_answer",
              "livemath_raw_output", "livemath_label", "parse_failed",
              "verdict", "has_error"):
        assert c in judged.columns, c
    assert ns["gate_fid"]["passed"] is True
    assert ns["gate_fid"]["fidelity_prompt"] is True
    assert ns["gate_nat"]["fidelity_prompt"] is False, (
        "the native prompt must be recorded separately and control nothing")
    import os as _os
    csv_path, md_path = ns["CSV_PATH"], ns["MD_PATH"]
    assert _os.path.exists(csv_path) and _os.path.exists(md_path)
    assert _os.path.basename(csv_path) == "livemath_judge_all300_20260812.csv"
    assert _os.path.basename(md_path) == "livemath_judge_all300_summary_20260812.md"
    head = pd.read_csv(csv_path)
    assert len(head) == 300 and "item_id" in head.columns
    text = open(md_path).read()
    for section in ("## Gate", "## Verdict counts", "## Accuracy",
                    "## Disagreement with the frozen rules", "## has_error split",
                    "## Against determinate human labels"):
        assert section in text, section
    assert text.index("## Gate") < text.index("## Accuracy"), (
        "the gate must be reported BEFORE any accuracy figure")
    assert "DO_NOT_QUOTE_ALONE" in text or "Do not quote the overall" in text
    d = ns["diag"]
    print("\n" + "=" * 74)
    print("DRY RUN PASSED -- gate, 300 items, both outputs written")
    print("=" * 74)
    print(f"counts {d['counts']}  acc(excl) {d['accuracy_excluding_unclear']:.1%}  "
          f"disagree v1 {d['disagree_strict_v1']}  v2 {d['disagree_strict_v2']}")
    print(f"human per-class: {d['human_per_class']}")

shutil.rmtree(ROOT / ".dryrun_scratch", ignore_errors=True)
