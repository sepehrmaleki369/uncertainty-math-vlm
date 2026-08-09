"""Dry-run for pilot/16_pixtral_perception.ipynb.

Real torch and real pilot code; only the Pixtral model/processor are stubbed.
The processor stub asserts the message shape the PUBLISHED chat template
actually requires -- system as a plain string, image as a payload-free chunk
-- because getting that wrong is what crashed InternVL3 on its first run.

Drives the gate cell down BOTH branches: a passing model and an
InternVL3-like degenerate one.
"""
import io, json, os, random, sys, types
WORKDIR = os.environ["WORKDIR"]; os.makedirs(WORKDIR, exist_ok=True); os.chdir(WORKDIR)
REPO = "/Users/sepehrmaleki/Documents/spring 2026/uncertainty-math-vlm"
sys.path.insert(0, REPO)

nb = json.load(open(os.path.join(REPO, "pilot/16_pixtral_perception.ipynb")))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
def src(i): return "".join(cells[i]["source"])
def c(i, *reps):
    s = src(i)
    for a, b in reps:
        assert a in s, f"cell {i}: {a!r} not found"
        s = s.replace(a, b)
    return s

class FakeImage:
    size = (800, 600)
    def save(self, p): open(p, "wb").write(b"PNG")
class DS:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, k): return [r[k] for r in self.rows] if isinstance(k, str) else self.rows[k]
    def __iter__(self): return iter(self.rows)
    @property
    def column_names(self): return list(self.rows[0]) if self.rows else []
    def shuffle(self, seed):
        r = list(self.rows); random.Random(seed).shuffle(r); return DS(r)
    def select(self, idx): return DS([self.rows[i] for i in idx])
    def remove_columns(self, cols): return DS([{k:v for k,v in r.items() if k not in cols} for r in self.rows])
import datasets
datasets.load_dataset = lambda name, split: DS([
    {"image": FakeImage(), "orig_q": f"Q{i}", "pert_a": "CONST",
     "has_error": i % 6 != 0, "handwriting_style": "cursive", "image_quality": "clear"}
    for i in range(1500)])

import torch  # real
import pilot.data, pilot.prompts, pilot.parsing, pilot.entropy, pilot.canonicalize, pilot.plotting

CALLS = {"n": 0}
MODE = {"degenerate": False}
_i = {"n": 0}

DECODES = {"n": 0}

def _text():
    _i["n"] += 1
    if MODE["degenerate"]:                      # InternVL3-like: every sample differs
        return f"**Question:** frag {_i['n']}\n**Answer:** {_i['n'] * 7}"
    # Healthy model: unanimous per item, right on ~60% of them. Each item costs
    # two batch_decode calls (one per arm), so item index = DECODES // 2.
    item = DECODES["n"] // 2
    ans = "CONST" if item % 5 < 3 else "wrong-answer"
    return f"**Question:** q\n**Answer:** {ans}"

class Ids(list):
    @property
    def shape(self): return (len(self), len(self[0]) if self else 0)
    def __getitem__(self, k): return Ids(list(self)) if isinstance(k, tuple) else list.__getitem__(self, k)
class BF(dict):
    def to(self, d): return self

class FakeProcessor:
    class _Tok: padding_side = "right"
    tokenizer = _Tok()
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        # exactly what the published Pixtral template requires
        assert messages[0]["role"] == "system", "system must be first"
        assert isinstance(messages[0]["content"], str), \
            "Pixtral concatenates the system message as a string; a list would break"
        assert messages[1]["role"] == "user"
        chunks = messages[1]["content"]
        assert any(x.get("type") == "image" for x in chunks)
        assert all(set(x) <= {"type", "text"} for x in chunks), \
            "image chunk must carry no payload -- the template emits a bare [IMG]"
        assert any(x.get("type") == "text" for x in chunks)
        return "PROMPT"
    def __call__(self, text=None, images=None, return_tensors=None, **kw):
        assert images and not isinstance(images, str), "image must be passed to the processor"
        b = BF(); b["input_ids"] = Ids([[1, 2, 3]]); return b
    def batch_decode(self, t, **kw):
        DECODES["n"] += 1
        return [_text() for _ in range(len(t))]

class FakeModel:
    device = "cpu"
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def generate(self, **kw):
        CALLS["n"] += 1
        return Ids([[1,2,3,4] for _ in range(kw.get("num_return_sequences", 1))])

ft = types.ModuleType("transformers")
ft.AutoProcessor = FakeProcessor
ft.LlavaForConditionalGeneration = FakeModel
ft.BitsAndBytesConfig = lambda **k: None
class _PTB: pass
ft.PreTrainedTokenizerBase = _PTB
sys.modules["transformers"] = ft

tq = types.ModuleType("tqdm.auto")
class T:
    def __init__(self, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def update(self, n): pass
tq.tqdm = T
sys.modules["tqdm.auto"] = tq; sys.modules["tqdm"] = types.ModuleType("tqdm")

ns = {"PROJECT_DIR": os.path.join(WORKDIR, "drive"),
      "DRIVE_MODEL_CACHE": os.path.join(WORKDIR, "drive", "mc"),
      "GH_TOKEN": "FAKE", "REPO_URL": "https://github.com/x/y.git"}
os.makedirs(ns["DRIVE_MODEL_CACHE"], exist_ok=True)

exec(compile(src(2), "<model>", "exec"), ns)
assert ns["MODEL_ID"] == "mistral-community/pixtral-12b" and ns["QUANTIZED"] is False
assert ns["processor"].tokenizer.padding_side == "left"
print("OK  model load (padding_side set for batched generate)")

exec(compile(c(3, ("PROCESS_N = 50", "PROCESS_N = 6")), "<sample>", "exec"), ns)
assert len(ns["full_sample"]) == 300 and len(ns["sample"]) == 6
print(f"OK  sample: drew {len(ns['full_sample'])}, processing {len(ns['sample'])} "
      "(full draw so the checkpoint carries over)")

exec(compile(src(4), "<adapter>", "exec"), ns)
print("OK  adapter: message shape matches the published chat template")

ckdir = os.path.join(WORKDIR, "ck"); os.makedirs(ckdir, exist_ok=True)
gen = c(5, ('f"{PROJECT_DIR}/checkpoints"', f'"{ckdir}"'))
exec(compile(gen, "<gen>", "exec"), ns)
assert len(ns["raw_results"]) == 6 and len(ns["raw_results"][0]["transcription_samples_raw"]) == 5
print(f"OK  generation: {len(ns['raw_results'])} items x K=5 both arms")

CALLS["n"] = 0
ns2 = dict(ns); exec(compile(gen, "<resume>", "exec"), ns2)
assert CALLS["n"] == 0
print("OK  resume makes zero generate() calls")

# raising PROCESS_N must extend, not restart -- the whole point of the design
ns3 = dict(ns); ns3["PROCESS_N"] = 10
CALLS["n"] = 0
exec(compile(gen, "<extend>", "exec"), ns3)
assert len(ns3["raw_results"]) == 10 and CALLS["n"] > 0
print(f"OK  raising PROCESS_N 6->10 generated only the 4 new items "
      f"({CALLS['n']} calls, not {10*2})")

import io as _io, contextlib
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src(6), "<gate-pass>", "exec"), ns)
out = buf.getvalue()
assert "GATE PASSED" in out, out[-700:]
assert "AUROC" not in out.replace("No AUROC printed", "")
print(f"OK  gate PASS branch (acc {ns['acc']:.0%}, maxent {ns['maxent']:.0%}, no AUROC printed)")

MODE["degenerate"] = True; _i["n"] = 0
for f in os.listdir(ckdir): os.remove(os.path.join(ckdir, f))
nsd = dict(ns); exec(compile(gen, "<gen-degen>", "exec"), nsd)
buf2 = _io.StringIO()
with contextlib.redirect_stdout(buf2):
    exec(compile(src(6), "<gate-fail>", "exec"), nsd)
out2 = buf2.getvalue()
assert "GATE FAILED" in out2, out2[-700:]
assert "Do NOT scale to 300" in out2
print(f"OK  gate FAIL branch fires on an InternVL3-like model "
      f"(acc {nsd['acc']:.0%}, maxent {nsd['maxent']:.0%})")

real_sub = sys.modules.get("subprocess")
fs = types.ModuleType("subprocess")
class CP:
    def __init__(s): s.returncode, s.stdout, s.stderr = 0, "", ""
fs.run = lambda *a, **k: CP()
sys.modules["subprocess"] = fs
os.makedirs(os.path.join(WORKDIR, "repo"), exist_ok=True)
exec(compile(src(7), "<save>", "exec"), ns)
assert os.path.exists(f"{WORKDIR}/repo/results/{ns['csv_name']}")
assert "gate" in ns["csv_name"], "gate runs must be named distinctly from full runs"
print(f"OK  save: {ns['csv_name']}")
sys.modules["subprocess"] = real_sub
print("\nALL CELLS DRY-RAN CLEANLY (both gate branches, and PROCESS_N extension)")
