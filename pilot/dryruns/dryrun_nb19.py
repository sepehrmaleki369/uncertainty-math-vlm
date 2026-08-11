"""Dry-run pilot/19_boxed_perception.ipynb locally, with the GPU stubbed.

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
NB = ROOT / "pilot" / "19_boxed_perception.ipynb"

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
# "boxed"  -> complies, unanimous per item, right on ~60%  (healthy)
# "plain"  -> ignores \boxed{}                             (gated)
# "noisy"  -> complies but every sample differs            (no signal)
MODE = {"m": "boxed"}


def _text():
    item = DECODES["n"]
    if MODE["m"] == "plain":
        return "**Question:** q\n**Answer:** CONST"
    if MODE["m"] == "noisy":
        DECODES["n"] += 1
        return f"**Question:** q\n**Answer:** \\boxed{{{DECODES['n'] * 7}}}"
    ans = "CONST" if item % 5 < 3 else "wrong-answer"
    return f"**Question:** q\n**Answer:** \\boxed{{{ans}}}"


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
        assert "\\boxed" in text, "the boxed instruction never reached the prompt"
        return text

    def __call__(self, text=None, images=None, return_tensors=None, **kw):
        assert images, "no image passed to the processor"
        return BF(input_ids=Ids([[1, 2, 3]]))

    def batch_decode(self, t, **kw):
        out = [_text() for _ in range(len(t))]
        if MODE["m"] == "boxed":
            DECODES["n"] += 1
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
assert ns["MODEL_ID"].startswith("Qwen/Qwen2.5-VL-") and ns["QUANTIZED"] is False
print(f"OK  model load: {ns['MODEL_ID']}, bf16")

exec(compile(c(4, ("PROCESS_N = 300", "PROCESS_N = 6")), "<sample>", "exec"), ns)
assert len(ns["full_sample"]) == 300 and len(ns["sample"]) == 6
print("OK  sample: full 300 drawn, 6 processed (checkpoint carries over)")

exec(compile(src(5), "<adapter>", "exec"), ns)
print("OK  adapter + pre-flight passes when the model complies")

# The whole point of the change: a non-compliant model must STOP the run here
# rather than print False and let 300 items be generated anyway.
MODE["m"] = "plain"
try:
    exec(compile(src(5), "<preflight-noncompliant>", "exec"), dict(ns))
except AssertionError as exc:
    assert "STOP HERE" in str(exc), str(exc)
    print("OK  pre-flight REFUSES a model that ignores \\boxed{} (this is the fix)")
else:
    raise SystemExit("pre-flight did not refuse a non-compliant model")
finally:
    MODE["m"] = "boxed"

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

exec(compile(src(7), "<gate>", "exec"), ns)
scored = ns["scored_df"]
assert len(scored) == 6
assert scored["frac_boxed"].mean() == 1.0, "compliance must be measured, not assumed"
assert (scored["n_distinct_tiers"] == 1).all(), "boxed extraction must be deterministic"
print("OK  gate: compliance 100%, every item on a single extractor tier")

# --- every registered verdict must be reachable ---------------------------
ci = lambda a, e: {"auroc": a, "excludes_chance": e}  # noqa: E731
cases = {
    "gated_low_compliance": (0.50, ci(0.90, True), 0.00),
    "manipulation_failed": (0.95, ci(0.90, True), 0.40),
    "signal_is_not_extractor": (0.95, ci(0.80, True), 0.02),
    "signal_was_extractor": (0.95, ci(0.55, False), 0.02),
    "inconclusive": (0.95, ci(0.72, True), 0.02),
}
for expected, args in cases.items():
    got = pilot.rescore.classify_boxed_result(*args)
    assert got == expected, f"{args} -> {got}, expected {expected}"
print(f"OK  all {len(cases)} registered verdicts reachable, "
      "including signal_was_extractor")

# --- the SAVE cell, which an earlier version of this dry run never executed --
# It referenced `df` (the gate builds `scored_df`) and still carried notebook
# 16's hardcoded "pixtral" filename, so it would have crashed and, had it not
# crashed, written a CSV labelled as a Pixtral run. Both were invisible here
# because the walk stopped at the gate.
save = c(8,
         ('subprocess.run(["git", "-C", "repo", *args], capture_output=True, text=True)',
          'FakeProc()'))
ns_s = dict(ns)
ns_s["PROJECT_DIR"] = os.path.join(WORKDIR, "drive")
ns_s["REPO_URL"] = "https://github.com/x/y.git"
ns_s["GH_TOKEN"] = ""
os.makedirs(os.path.join(WORKDIR, "repo", "results"), exist_ok=True)
_cwd = os.getcwd()
os.chdir(WORKDIR)
try:
    exec(compile("class FakeProc:\n    returncode = 0\n    stdout = stderr = ''\n"
                 + save, "<save>", "exec"), ns_s)
finally:
    os.chdir(_cwd)

written = ns_s["csv_name"]
assert "pixtral" not in written.lower(), f"mislabelled CSV: {written}"
assert ns_s["MODEL_ID"].split("/")[-1].lower() in written.lower(), written
assert written.startswith(ns_s["RUN_NAME"]), written
saved = os.path.join(WORKDIR, "drive", "results", written)
assert os.path.exists(saved), saved
import pandas as _pd
assert len(_pd.read_csv(saved)) == len(ns["scored_df"])
print(f"OK  save cell writes {written}")

# The 7B swap must change the filename, or two runs collide on Drive.
ns_7 = dict(ns_s)
ns_7["MODEL_ID"] = ("Qwen/Qwen2.5-VL-3B-Instruct"
                    if "7B" in ns_s["MODEL_ID"] else "Qwen/Qwen2.5-VL-7B-Instruct")
os.chdir(WORKDIR)
try:
    exec(compile("class FakeProc:\n    returncode = 0\n    stdout = stderr = ''\n"
                 + save, "<save7b>", "exec"), ns_7)
finally:
    os.chdir(_cwd)
assert ns_7["csv_name"] != written
print(f"OK  swapping MODEL_ID changes the filename to {ns_7['csv_name']}")

print("\n" + "=" * 72)
print("DRY RUN PASSED")
print("=" * 72)
import shutil  # noqa: E402
shutil.rmtree(WORKDIR, ignore_errors=True)
