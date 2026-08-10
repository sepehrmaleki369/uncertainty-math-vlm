"""Dry-run pilot/20_verbalized_confidence.ipynb locally, with the GPU stubbed.

Real torch and real pilot code; only the Qwen model/processor and the gated
dataset are faked. Notebook 19 has ONE arm (transcription), unlike notebook
16, and its gate is driven by pre-registered thresholds, so this asserts the
things that would actually invalidate the experiment:

  * \\boxed{} compliance is computed from the samples, not assumed;
  * the checkpoint resumes without regenerating, and extends when PROCESS_N
    rises -- an n=300 GPU session is too expensive to restart;
  * each registered verdict is reachable, including signal_was_extractor,
    which is the outcome the run exists to be able to report.

    python pilot/dryruns/dryrun_nb19.py
"""

import json
import os
import random
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NB = ROOT / "pilot" / "20_verbalized_confidence.ipynb"

cells = json.loads(NB.read_text())["cells"]


def src(i):
    return "".join(cells[i]["source"])


def c(i, *reps):
    s = src(i)
    for a, b in reps:
        assert a in s, f"cell {i}: {a!r} not found"
        s = s.replace(a, b)
    return s


# --- fake the gated dataset ----------------------------------------------
class FakeImage:
    size = (800, 600)

    def save(self, p):
        open(p, "wb").write(b"PNG")


class DS:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        return [r[k] for r in self.rows] if isinstance(k, str) else self.rows[k]

    def __iter__(self):
        return iter(self.rows)

    @property
    def column_names(self):
        return list(self.rows[0]) if self.rows else []

    def shuffle(self, seed):
        r = list(self.rows)
        random.Random(seed).shuffle(r)
        return DS(r)

    def select(self, idx):
        return DS([self.rows[i] for i in idx])

    def remove_columns(self, cols):
        return DS([{k: v for k, v in r.items() if k not in cols} for r in self.rows])


import datasets  # noqa: E402
datasets.load_dataset = lambda name, split: DS([
    {"image": FakeImage(), "orig_q": f"Q{i}", "pert_a": "CONST",
     "has_error": i % 6 != 0, "handwriting_style": True, "image_quality": True}
    for i in range(1500)])

import torch  # noqa: E402  (real)

import pilot.canonicalize  # noqa: E402,F401
import pilot.data  # noqa: E402,F401
import pilot.entropy  # noqa: E402,F401
import pilot.parsing  # noqa: E402,F401
import pilot.plotting  # noqa: E402,F401
import pilot.prompts  # noqa: E402,F401
import pilot.rescore  # noqa: E402

CALLS = {"n": 0}
DECODES = {"n": 0}
# "conf"   -> emits a varying **Confidence:** field   (healthy)
# "noconf" -> omits it entirely                       (gated)
MODE = {"m": "conf"}


def _text():
    item = DECODES["n"]
    ans = "CONST" if item % 5 < 3 else "wrong-answer"
    if MODE["m"] == "noconf":
        return f"**Question:** q\n**Answer:** {ans}"
    # Vary the confidence so the AUROC is not degenerate on ties.
    conf = 90 if item % 5 < 3 else 40
    return f"**Question:** q\n**Answer:** {ans}\n**Confidence:** {conf}"


class Ids(list):
    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)

    def __getitem__(self, k):
        return Ids(list(self)) if isinstance(k, tuple) else list.__getitem__(self, k)


class BF(dict):
    def to(self, d):
        return self


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kw):
        # The shape the notebook builds must be the system+user shape.
        assert [m["role"] for m in messages] == ["system", "user"], messages
        assert any(c.get("type") == "image" for c in messages[1]["content"])
        text = "".join(c.get("text", "") for m in messages for c in m["content"])
        assert "Confidence" in text, "the confidence instruction never reached the prompt"
        return text

    def __call__(self, text=None, images=None, return_tensors=None, **kw):
        assert images, "no image passed to the processor"
        return BF(input_ids=Ids([[1, 2, 3]]))

    def batch_decode(self, t, **kw):
        out = [_text() for _ in range(len(t))]
        DECODES["n"] += 1          # one decode call per item
        return out


class FakeModel:
    dtype = "bfloat16"
    device = "cpu"

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def eval(self):
        return self

    def generate(self, **kw):
        CALLS["n"] += 1
        return Ids([[1, 2, 3, 4]] * kw.get("num_return_sequences", 1))


ft = types.ModuleType("transformers")
ft.AutoProcessor = FakeProcessor
ft.Qwen2_5_VLForConditionalGeneration = FakeModel
ft.BitsAndBytesConfig = lambda **k: None
sys.modules["transformers"] = ft


class _Bar:
    def __init__(self, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def update(self, n):
        pass


tq = types.ModuleType("tqdm.auto")
tq.tqdm = lambda *a, **k: _Bar()
sys.modules["tqdm.auto"] = tq
sys.modules["tqdm"] = types.ModuleType("tqdm")
sys.modules["tqdm"].auto = tq

WORKDIR = tempfile.mkdtemp(prefix="dryrun_nb19_")
ns = {"PROJECT_DIR": os.path.join(WORKDIR, "drive")}
os.makedirs(ns["PROJECT_DIR"], exist_ok=True)

# --- walk the notebook ----------------------------------------------------
exec(compile(src(3), "<model>", "exec"), ns)
assert ns["MODEL_ID"] == "Qwen/Qwen2.5-VL-3B-Instruct" and ns["QUANTIZED"] is False
print("OK  model load: same model as the reference run, bf16")

exec(compile(c(4, ("PROCESS_N = 300", "PROCESS_N = 6")), "<sample>", "exec"), ns)
assert len(ns["full_sample"]) == 300 and len(ns["sample"]) == 6
print("OK  sample: full 300 drawn, 6 processed (checkpoint carries over)")

exec(compile(src(5), "<adapter>", "exec"), ns)
print("OK  adapter + pre-flight: boxed instruction reaches the chat template")

ckdir = os.path.join(WORKDIR, "ck")
gen = c(6, ('f"{PROJECT_DIR}/checkpoints"', f'"{ckdir}"'))
exec(compile(gen, "<gen>", "exec"), ns)
assert len(ns["raw_results"]) == 6
assert len(ns["raw_results"][0]["transcription_samples_raw"]) == 5
assert "grading_samples_raw" not in ns["raw_results"][0], "this run has ONE arm"
print(f"OK  generation: {len(ns['raw_results'])} items x K=5, transcription only")

CALLS["n"] = 0
ns2 = dict(ns)
exec(compile(gen, "<resume>", "exec"), ns2)
assert CALLS["n"] == 0, "a resume must not regenerate -- an n=300 session is expensive"
print("OK  resume makes zero generate() calls")

ns3 = dict(ns)
ns3["PROCESS_N"] = 10
CALLS["n"] = 0
exec(compile(gen, "<extend>", "exec"), ns3)
assert len(ns3["raw_results"]) == 10 and CALLS["n"] > 0
print(f"OK  raising PROCESS_N 6->10 generated only the 4 new items ({CALLS['n']} calls)")

exec(compile(src(7), "<compare>", "exec"), ns)
scored = ns["scored_df"]
assert len(scored) == 6
assert (scored["n_confidence_parsed"] == 5).all(), "confidence must parse on every sample"
# Direction matters: higher confidence must mean LESS likely wrong, so the
# score fed to the AUROC is NEGATED. Getting this backwards inverts the result.
assert (scored["neg_confidence"] == -scored["mean_confidence"]).all()
assert scored["mean_confidence"].nunique() > 1, "a constant stub cannot test ranking"
print("OK  comparison cell: confidence parsed on every sample and negated correctly")

# The gate must actually fire when the model omits the field.
MODE["m"] = "noconf"
DECODES["n"] = 0
ns_g = dict(ns)
ns_g["raw_results"] = [
    {"item": r["item"], "transcription_samples_raw":
        ["**Question:** q\n**Answer:** CONST"] * 5,
     "quantized": False}
    for r in ns["raw_results"]]
import io as _io
import contextlib as _cl
_buf = _io.StringIO()
with _cl.redirect_stdout(_buf):
    exec(compile(src(7), "<gate-noconf>", "exec"), ns_g)
assert "GATED" in _buf.getvalue(), _buf.getvalue()
print("OK  gate fires when the model omits the Confidence field")

print("\n" + "=" * 72)
print("DRY RUN PASSED")
print("=" * 72)
import shutil  # noqa: E402
shutil.rmtree(WORKDIR, ignore_errors=True)
