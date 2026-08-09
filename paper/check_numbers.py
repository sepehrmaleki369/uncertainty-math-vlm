"""Check every number in paper/main.tex against the frozen snapshots.

The paper hand-types ~40 figures across its tables and prose. Any one of
them can be mistyped, or can silently go stale when a snapshot is
regenerated -- and a wrong number in a submitted paper is not recoverable
the way a wrong number in a notebook is.

This asserts the claims the paper actually makes, sourced from
reference/*.json (which are themselves recomputed from raw model output
and asserted by pilot/tests/). Run it after any edit to main.tex:

    python paper/check_numbers.py

Two complementary checks:

  1. Every figure the snapshots imply must APPEAR in the paper. Catches a
     stale paper after a snapshot is regenerated.
  2. Every three-decimal number in the paper body must BE a snapshot value
     (or an explicitly allowed constant). Catches a typo.

Check 2 exists because check 1 alone is not enough, which I found by
testing it: corrupting "AUROC 0.835" to "0.853" in the prose still passed,
since 0.835 also appears in Table 1 and the substring was therefore still
present somewhere. A whitelist catches that; an existence test cannot.

Neither catches a correct number placed in the wrong sentence. This is a
guard against drift and typos, not a substitute for proofreading.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
REF = ROOT / "reference"


def snap(name):
    return json.loads((REF / name).read_text())


def f3(x):
    return f"{x:.3f}"


def build_expectations():
    """(label, list-of-strings-that-must-appear) sourced from the snapshots."""
    exp = []

    q3b = snap("n300_balanced_20260802.json")
    p = q3b["perception"]
    s = p["sensitivity"]
    ab = q3b["abstention"]

    exp.append(("perception headline",
                [f3(p["auroc"]), f3(p["ci_low"]), f3(p["ci_high"])]))
    for cut, n in [("excl_parse_failures", 255), ("excl_max_entropy", 239),
                   ("excl_both", 206)]:
        d = s[cut]
        exp.append((f"sensitivity {cut}",
                    [f3(d["auroc"]), f3(d["ci_low"]), f3(d["ci_high"]), str(d["n_items"])]))
    exp.append(("abstention",
                [str(ab["n_flagged_wrong"]), str(ab["n_flagged"]),
                 f"{ab['precision'] * 100:.1f}", f"{ab['recall'] * 100:.1f}"]))
    exp.append(("AURC", [f3(p["aurc"]), f3(p["aurc_baseline"])]))

    q7 = snap("qwen7b_matched_n650_20260806.json")["reasoning"]
    q7_300 = snap("qwen7b_fermat_n300_20260805.json")["reasoning"]
    exp.append(("pooled reasoning (n=300, balanced)",
                [f3(q7_300["pooled"]["auroc"]), f3(q7_300["pooled"]["ci_low"]),
                 f3(q7_300["pooled"]["ci_high"])]))
    for key, label in [("error_stratum", "error stratum"), ("clean_stratum", "clean stratum")]:
        d = q7[key]
        exp.append((f"7B {label}",
                    [f3(d["auroc"]), f3(d["ci_low"]), f3(d["ci_high"]), str(d["n_error"])]))

    for f, label, n in [("qwen3b_stratum_powered_n800_20260806.json", "3B", 650),
                        ("qwen7b_matched_n650_20260806.json", "7B", 648),
                        ("llava_stratum_powered_n550_20260807.json", "LLaVA", 400)]:
        d = snap(f)["reasoning"]["error_stratum"]
        exp.append((f"family table {label}",
                    [f3(d["auroc"]), f3(d["ci_low"]), f3(d["ci_high"]),
                     str(d["n_error"]), str(n)]))

    iv = snap("internvl3_fermat_n300_20260807.json")["perception"]
    exp.append(("InternVL3 raw",
                [f3(iv["auroc"]), f3(iv["ci_low"]), f3(iv["ci_high"]),
                 f"{iv['accuracy'] * 100:.1f}"]))
    ivc = iv["sensitivity"]["excl_max_entropy"]
    exp.append(("InternVL3 after control",
                [f3(ivc["auroc"]), f3(ivc["ci_low"]), f3(ivc["ci_high"])]))

    sm = snap("scratchmath_gated_n100_20260808.json")
    ne = sm["extra"]["non_engagement"]
    exp.append(("ScratchMath",
                [f"{sm['grading']['accuracy'] * 100:.1f}",
                 str(ne["n_cannot_read"]), str(ne["n_samples"]),
                 str(ne["n_cannot_read_still_says_error"])]))
    return exp


# Constants that legitimately appear and are not snapshot values:
# pre-registered thresholds, chance, the sampling temperature, K.
ALLOWED_CONSTANTS = {
    "0.70", "0.700", "0.50", "0.500", "0.7", "0.5",
}


def allowed_values():
    """Every three-decimal string any snapshot could justify."""
    ok = set(ALLOWED_CONSTANTS)
    for path in sorted(REF.glob("*.json")):
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                ok.add(f"{float(o):.3f}")
                ok.add(f"{float(o) * 100:.1f}")
                ok.add(f"{abs(float(o)):.3f}")
        walk(json.loads(path.read_text()))
    return ok


def check_no_invented_numbers(text):
    """No three-decimal number in the body may be absent from the snapshots."""
    body = text.split("\\begin{document}", 1)[-1]
    body = re.sub(r"%.*", "", body)          # drop comments
    found = set(re.findall(r"(?<![\d.])(\d\.\d{3})(?![\d])", body))
    ok = allowed_values()
    invented = sorted(n for n in found if n not in ok)
    print(f"  {'no invented 3-decimal figures':38s} "
          f"{'ok (' + str(len(found)) + ' checked)' if not invented else 'INVENTED: ' + ', '.join(invented)}")
    return invented


def main():
    if not TEX.exists():
        print(f"missing {TEX}")
        return 1
    text = TEX.read_text()
    # Strip TeX spacing/markup that would break a literal match.
    flat = re.sub(r"[\s~]+", " ", text)

    bad = []
    for label, needles in build_expectations():
        missing = [n for n in needles if n not in flat]
        status = "ok" if not missing else "MISSING " + ", ".join(missing)
        print(f"  {label:38s} {status}")
        if missing:
            bad.append((label, missing))

    invented = check_no_invented_numbers(text)
    if invented:
        bad.append(("invented figures", invented))

    print()
    if bad:
        print(f"FAIL: {len(bad)} claim(s) in main.tex do not match reference/*.json")
        print("Either the paper has a typo, or a snapshot changed and the paper is stale.")
        return 1
    print("All checked figures in main.tex match the frozen snapshots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
