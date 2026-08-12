"""Dry-run pilot/27_open_judges_all300.ipynb with both judges stubbed.

Notebook 27 runs LiveMath-Judge and Omni-Judge over all 300 as an EXPLORATORY
diagnostic. Both had already failed a safety check, so the properties worth
asserting are not about verdict quality:

  * **both judges must see the IDENTICAL answer string per item.** This is the
    invariant the whole notebook rests on -- one refactor away from each judge
    recomputing its own majority answer, which would produce a
    plausible-looking comparison table of two different experiments. The stubs
    record what they were shown and the answers are compared element by
    element.
  * a parse failure must land as NA, never as disagreement.
  * the checkpoint must refuse to resume across a different model, and must
    actually resume.
  * the summary must contain no accuracy figure.
  * **the save cell must run**, with the exact requested filename and column
    order. Notebooks 19 and 20 both shipped a broken final cell because their
    dry runs stopped before it.

    python pilot/dryruns/dryrun_nb27.py
"""
import json
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "27_open_judges_all300.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")

if not CSV.exists():
    sys.exit(f"SKIP: not present: {CSV}")

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import pilot.canonicalize  # noqa: E402

# --- what the stubs saw, so identical-inputs can be asserted ---------------
SEEN = {"livemath": [], "omni": []}
CALLS = {"livemath": 0, "omni": 0}

# --- stub the Colab environment -------------------------------------------
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda p: print(f"[stub] drive.mount({p})"))
google = types.ModuleType("google")
google.colab = colab
sys.modules.setdefault("google", google)
sys.modules["google.colab"] = colab

hub = types.ModuleType("huggingface_hub")
hub.login = lambda token=None: print("[stub] huggingface_hub.login")
hub.hf_hub_download = lambda repo, name, token=None: "<stub tokenizer.json>"
sys.modules["huggingface_hub"] = hub

# --- stub torch: no GPU, no downloads --------------------------------------
torch = types.ModuleType("torch")
torch.bfloat16 = "bfloat16"
torch.cuda = types.SimpleNamespace(
    is_available=lambda: True, get_device_name=lambda i: "stub-GPU",
    empty_cache=lambda: None, memory_reserved=lambda: 0)


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


torch.no_grad = _NoGrad
sys.modules["torch"] = torch

tf = types.ModuleType("transformers")


class _Tokens:
    """Stands in for a tensor slice; decodes back to the reply string."""

    def __init__(self, s):
        self.s = s

    def cpu(self):
        return self

    def tolist(self):
        return self.s


class _Payload:
    def __init__(self, s):
        self.s = s

    def __getitem__(self, k):
        return _Tokens(self.s)


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


def _decode(x, skip_special_tokens=True):
    return getattr(x, "s", x)


class _LiveMathTok:
    pad_token_id, eos_token_id = 0, 0

    def apply_chat_template(self, msgs, return_tensors=None, return_dict=None,
                            add_generation_prompt=None):
        assert return_dict, (
            "the notebook must pass return_dict=True -- the model card's own "
            "snippet does not, and then subscripts the returned TENSOR")
        return _Batch({"input_ids": _Ids(msgs[0]["content"]),
                       "attention_mask": _Ids("")})

    decode = staticmethod(_decode)


class _OmniTok:
    """Carries the custom methods the real OmniJudgeTokenizer supplies."""

    pad_token_id, eos_token_id = 0, 0

    def __call__(self, ctx, return_tensors=None):
        return _Batch({"input_ids": _Ids(ctx), "attention_mask": _Ids("")})

    def get_context(self, question, gold, answer):
        return f"Q::{question}\nGOLD::{gold}\nANS::{answer}"

    def parse_response(self, text):
        # The REAL one returns all-None on this model's output, which is why
        # `parse_omni_text` exists. Reproduce that, so the fallback is what is
        # actually exercised here rather than a happy path Colab never takes.
        return {"answer": None, "judgement": None, "justification": None}

    def convert_tokens_to_ids(self, tok):
        return 128009

    decode = staticmethod(_decode)


class _PlainTok:
    def __init__(self, tokenizer_file=None):
        pass

    def __call__(self, s):
        return {"input_ids": _Tokens(s)}

    decode = staticmethod(_decode)


class _AutoTok:
    @classmethod
    def from_pretrained(cls, mid, trust_remote_code=False, token=None):
        return _OmniTok() if "Omni" in mid else _LiveMathTok()


_SIGN_FLIP = r"1 + \tan x \tan y"


class _Model:
    def __init__(self, mid):
        self.mid = mid
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, mid, **k):
        return cls(mid)

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None,
                 max_new_tokens=None, **k):
        prompt = input_ids.payload
        if "Omni" in self.mid:
            return [_Payload(self._omni(prompt))]
        assert max_new_tokens and max_new_tokens > 100, (
            "max_new_tokens must be set: the model writes an analysis before "
            "the boxed verdict and the card's default of 20 truncates it")
        return [_Payload(self._livemath(prompt))]

    def _livemath(self, prompt):
        n = CALLS["livemath"]
        CALLS["livemath"] += 1
        SEEN["livemath"].append(prompt.split("Examinee's Answer:", 1)[-1]
                                .rsplit("\n\nAnalysis:", 1)[0].strip())
        if n == 7:                       # exercise the NA / parse-failure path
            return "Analysis: I am unable to reach a verdict at all."
        if _SIGN_FLIP in prompt:         # item 55: rejects the sign flip
            return r"Analysis: the examinee altered the sign. \boxed{no}"
        return (r"Analysis: matches. \boxed{yes}" if n % 4
                else r"Analysis: differs. \boxed{no}")

    def _omni(self, ctx):
        n = CALLS["omni"]
        CALLS["omni"] += 1
        SEEN["omni"].append(ctx.split("ANS::", 1)[-1].strip())
        if n == 11:
            return "## Student's Final Answer\nnothing parseable here"
        if r"\frac{2}{4}" in ctx:        # item 273: the invented reference
            return ("## Student's Final Answer\n}\n## Equivalence Judgment\n"
                    "FALSE\n## Justification\nThe student's answer of 2/4 is "
                    "incorrect. The reference answer is 3/4.")
        if _SIGN_FLIP in ctx:
            return ("## Equivalence Judgment\nFALSE\n## Justification\n"
                    "The sign in the denominator differs.")
        return ("## Equivalence Judgment\n" + ("TRUE" if n % 3 else "FALSE") +
                "\n## Justification\nCompared against the reference.")


tf.AutoTokenizer = _AutoTok
tf.AutoModelForCausalLM = _Model
tf.PreTrainedTokenizerFast = _PlainTok
sys.modules["transformers"] = tf

# --- scratch Drive ---------------------------------------------------------
# Confined to a notebook-27 subdirectory, and this is NOT tidiness. Parts of
# `.dryrun_scratch/` are TRACKED IN GIT (committed by accident in 7c1512c),
# so wiping the shared path deletes tracked files and dirties the working
# tree on every dry run. Only this subtree is created and removed here.
SCRATCH = ROOT / ".dryrun_scratch" / "nb27"
DRIVE = str(SCRATCH / "drive")
shutil.rmtree(SCRATCH, ignore_errors=True)
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
print(f"{len(code_cells)} code cells; EXPLORATORY all-300 diagnostic\n")

ns = {"__name__": "__main__"}
for i, src in code_cells:
    print(f"----- cell {i} " + "-" * 60)
    try:
        exec(compile(prepare(src), f"<cell {i}>", "exec"), ns)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f"\nFAILED at cell {i}")

# --- post-conditions -------------------------------------------------------
import os as _os  # noqa: E402

frame, diag = ns["frame"], ns["diag"]
answers = ns["answers"]

assert len(frame) == 300, f"expected 300 rows, got {len(frame)}"

# THE LOAD-BEARING INVARIANT: both judges saw the same answer, item by item.
assert len(SEEN["livemath"]) == len(SEEN["omni"]) == 300
mismatched = [k for k in range(300) if SEEN["livemath"][k] != SEEN["omni"][k]]
assert not mismatched, (
    f"the two judges were shown DIFFERENT answers on {len(mismatched)} items "
    f"(first: {mismatched[:5]}) -- every comparison in the notebook would be "
    "between two different experiments")
assert SEEN["livemath"] == [str(answers[i]).strip() for i in ns["run"].index], (
    "the judges were not fed the shared `answers` series")

# Requested schema, exactly, in order, extras strictly after.
import pilot.judge as J  # noqa: E402

assert list(frame.columns[:len(J.OPEN_JUDGE_COLUMNS)]) == list(J.OPEN_JUDGE_COLUMNS)
assert list(frame.columns[len(J.OPEN_JUDGE_COLUMNS):]) == list(J.OPEN_JUDGE_EXTRA_COLUMNS)

# A parse failure is NA, never disagreement.
assert diag["livemath"]["parse_failed"] >= 1 and diag["omni"]["parse_failed"] >= 1, (
    "the stub must exercise the parse-failure path on both judges")
for tag in ("livemath", "omni"):
    unread = frame.index[frame[f"{tag}_verdict"] == "unclear"]
    for i in unread:
        assert pd.isna(frame.loc[i, f"{tag}_agrees_strict_v1"]), (
            f"{tag} item {i}: an unreadable verdict was recorded as "
            "disagreement, which asserts something we did not observe")
        assert pd.isna(frame.loc[i, "judges_agree"])

# The trivial baseline must be present and must actually bound the reading.
for tag in ("livemath", "omni"):
    for rule in ("strict_v1", "strict_v2"):
        d = diag[f"{tag}_vs_{rule}"]
        assert 0.0 <= d["always_yes_agreement"] <= 1.0
        assert d["agree"] + d["disagree"] == d["n_determinate"]
assert diag["judge_vs_judge"]["n_both_determinate"] <= 300
assert "kappa" in diag["judge_vs_judge"]

# The item-273 signature fired, and only where it should.
assert 273 in frame.index
assert frame.loc[273, "omni_invented_reference"], (
    "the invented-reference flag must catch the 3/4 the judge cited but was "
    "never shown")
assert "3/4" in frame.loc[273, "omni_invented_tokens"]
assert diag["omni_invented_reference_total"] >= 1

# Gate reproduction is reported, not enforced.
rep = diag["gate_reproduction"]
assert set(rep) == {"livemath", "omni"}
assert rep["omni"][273]["observed"] == "incorrect"
assert rep["livemath"][55]["observed"] == "incorrect"

# The save cell really ran, under the requested names.
csv_path, md_path = ns["CSV_PATH"], ns["MD_PATH"]
assert _os.path.basename(csv_path) == "open_judges_all300_diagnostic_20260812.csv"
assert _os.path.basename(md_path) == \
    "open_judges_all300_diagnostic_summary_20260812.md"
assert _os.path.exists(csv_path) and _os.path.exists(md_path)
written = pd.read_csv(csv_path)
assert len(written) == 300
assert list(written.columns[:len(J.OPEN_JUDGE_COLUMNS)]) == list(J.OPEN_JUDGE_COLUMNS)

text = open(md_path).read()
low = text.lower()
assert "not a scoring run" in low
assert "corrected accuracy" in low
assert "always-yes baseline" in text
assert not re.search(r"\baccuracy\b\s*[:=]", text, re.I), (
    "the summary must never emit an accuracy figure")
for section in ("## Verdict counts", "## Judge vs judge",
                "## Against the frozen rules", "## Split by `has_error`",
                "## Does the judge re-derive the answer instead of comparing?",
                "## Gate items, re-derived", "## Examples: the judges disagree",
                "## Examples: both judges agree but differ from `strict_v1`",
                "## Fixed sample for manual rationale inspection"):
    assert section in text, section

# The checkpoint refuses to resume across a different model...
before = dict(CALLS)
try:
    ns["load_ckpt"]("livemath", "some/other-model", "fidelity")
except RuntimeError as e:
    assert "Resuming would mix two judges" in str(e)
else:
    sys.exit("FAILED: a checkpoint from a different model was accepted")

# ...and does resume, doing no work the second time.
again = ns["run_judge"]("livemath", ns["LM_ID"], "fidelity", ns["livemath_judge"])
assert len(again) == 300
assert CALLS == before, (
    f"resume re-ran the judge: {CALLS} vs {before} -- a disconnect at item 280 "
    "would cost the whole run")

# The 3B must be released before the 8B loads, or a 16 GB runtime OOMs on the
# second model. `free_gpu` therefore takes NO arguments: binding the model to
# a parameter would make `del` drop only that local name while the caller's
# global kept it alive, printing a reassuring message and freeing nothing.
import inspect  # noqa: E402

assert not inspect.signature(ns["free_gpu"]).parameters, (
    "free_gpu must take no arguments -- passing the model in frees nothing")
for cell_i, name in ((4, "lm_model"), (5, "om_model")):
    src = "".join(nb["cells"][cell_i]["source"])
    assert src.index(f"del {name}") < src.rindex("free_gpu()"), (
        f"cell {cell_i} calls free_gpu() before deleting {name}, so the "
        "collection runs while the global still holds the model")

print("\n" + "=" * 74)
print("DRY RUN PASSED -- 300 items x 2 judges, identical inputs, no accuracy")
print("=" * 74)
jj = diag["judge_vs_judge"]
print(f"judges agree {jj['agree']}/{jj['n_both_determinate']} "
      f"({jj['rate']:.1%}), chance {jj['chance_rate']:.1%}, "
      f"kappa {jj['kappa']:.3f}")
print(f"parse failures: livemath {diag['livemath']['parse_failed']}, "
      f"omni {diag['omni']['parse_failed']}")
print(f"invented-reference flag: {diag['omni_invented_reference_total']} items")
print(f"rationale sample: {ns['rationales']}")

shutil.rmtree(SCRATCH, ignore_errors=True)
