"""Dry-run pilot/30_wacv_evaluation_artifact.ipynb. No network, no HF login.

Notebook 30 builds the WACV artifact: the binary-stratification null, the
manifests, and the blinded second-rater packet. It is CPU-only by design, so
unlike the other dry runs there is no model to stub -- what has to be stubbed
is the *scale* (the full null grid is ~3 minutes) and the network.

The properties worth asserting are release properties:

  * the notebook runs end to end with no Hugging Face login and no network,
    and still refuses to claim the sample was reconstructed;
  * the public manifest contains no gated text, secret or local path;
  * the audit long form keeps 312 rows over 234 items with 78 overlaps;
  * every second-rater answer ships blank and the gate prints PENDING;
  * no model, GPU or API is touched.

    python pilot/dryruns/dryrun_nb30.py
"""
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "pilot" / "30_wacv_evaluation_artifact.ipynb"
CSV = (ROOT / "results"
       / "scaleup_n300_bal50_qwen25-vl-3b-instruct_20260802T163202Z.csv")

if not CSV.exists():
    sys.exit(f"SKIP: not present: {CSV}")

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

# --- no network. The notebook must still complete and still fail closed. ---
import pilot.wacv_artifact as W  # noqa: E402

_real_revision = W.dataset_revision


def _offline_revision(offline_ok: bool = True) -> dict:
    return {"dataset_id": W.DATASET_ID, "revision": None, "gated": "unknown",
            "source": "unavailable: DryRunOffline"}


W.dataset_revision = _offline_revision

SCRATCH = ROOT / ".dryrun_scratch" / "nb30"
shutil.rmtree(SCRATCH, ignore_errors=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

SHELL_RE = re.compile(r"^\s*[!%]")


def prepare(src: str) -> str:
    lines = [f"print({json.dumps('[stub] ' + ln.strip())})"
             if SHELL_RE.match(ln) else ln for ln in src.splitlines()]
    out = "\n".join(lines)
    # Write into scratch, never over the committed artifact.
    out = out.replace('ARTIFACT_DIR = "reference/wacv_evaluation_artifact"',
                      f'ARTIFACT_DIR = "{SCRATCH}"')
    # The full grid is ~3 minutes; the dry run checks plumbing, not precision.
    out = out.replace("GRID_KW = dict(n_items=650, n_sims=2000, base_seed=20260814)",
                      "GRID_KW = dict(n_items=120, n_sims=12, base_seed=20260814)")
    out = out.replace('grid = sn.null_grid(**GRID_KW)',
                      'grid = sn.null_grid(**GRID_KW, p_values=(0.5, 0.8), '
                      'k_values=(3, 5))')
    out = out.replace('null = sn.simulate_stratum(p, 5, 1, n_items=n, n_sims=2000, seed=20260814)',
                      'null = sn.simulate_stratum(p, 5, 1, n_items=n, n_sims=60, seed=20260814)')
    out = out.replace('cfg["smoke"] = False', 'cfg["smoke"] = True')
    out = out.replace('"smoke": False', '"smoke": True')
    return out


nb = json.loads(NB.read_text())
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells; CPU-only artifact build (smoke sizes)\n")

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
import pilot.second_rater as SR  # noqa: E402

alignment, public, private = ns["alignment"], ns["public"], ns["private"]
audit_long, queue, template = ns["audit_long"], ns["queue"], ns["template"]

# Fails closed with no network and no login.
assert alignment["alignment_status"] == W.PENDING, alignment
assert alignment["verified"] is False
assert public["alignment_status"].eq(W.PENDING).all()

# Release safety.
W.assert_public_manifest_is_safe(public, private)
assert list(public.columns) == list(W.PUBLIC_COLUMNS)
blob = public.astype(str).to_csv(index=False)
for pat in ("/content/drive", "MyDrive", "hf_", "/Users/"):
    assert pat not in blob, pat
assert "orig_q" not in public.columns and "pert_a" not in public.columns

# Audit invariants survive the round trip through the written file.
inv = W.audit_invariants(audit_long)
assert (inv["n_rows"], inv["n_unique_items"], inv["n_overlap_rows"]) == (312, 234, 78)
assert inv["hard_contradictions"] == [108, 149, 222, 230]

# Second-rater packet ships blank, and the gate holds.
assert SR.all_answers_blank(template)
assert not (set(SR.BLINDED_FIELDS) & set(template.columns))
core = set(queue.loc[queue["queue"] == "core", "item_id"])
chal = set(queue.loc[queue["queue"] == "challenge", "item_id"])
assert len(core) == SR.N_CORE and chal == set(SR.CHALLENGE_ITEMS)
assert not (core & chal)
gate = SR.agreement_or_pending(str(SCRATCH))
assert gate["state"] == "PENDING"
assert "SECOND-RATER STATUS: PENDING" in gate["message"]

# Files written, and the private ones are named so they cannot be released by
# accident.
written = sorted(os.listdir(SCRATCH))
for required in ("null_grid.csv", "null_grid_config.json",
                 "fermat_n300_public_manifest.csv",
                 "fermat_n300_private_manifest.csv", "audit_labels_long.csv",
                 "second_rater_queue.csv", "second_rater_template_blank.csv",
                 "second_rater_instructions.md", "provenance.json"):
    assert required in written, f"missing artifact: {required}"
assert "agreement_summary.json" not in written, (
    "agreement must not exist before a real second rater completes")
assert any("PRIVATE" in f for f in written), (
    "the blinding key must be named so it cannot be released by accident")

# CPU-only: nothing imported a model.
assert "torch" not in sys.modules and "transformers" not in sys.modules

print("\n" + "=" * 74)
print("DRY RUN PASSED -- offline, fails closed, no gated text, rater gate held")
print("=" * 74)
print(f"artifacts written: {len(written)}")
print(f"alignment status : {alignment['alignment_status']}")
print(f"second rater     : {gate['state']}")

shutil.rmtree(SCRATCH, ignore_errors=True)
