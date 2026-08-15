"""The releasable WACV evaluation artifact: manifests, hashes, audit long form.

Offline and CPU-only. Reads the frozen run CSV and the audit CSVs, writes
manifests. It runs no model, touches no scorer rule, and changes nothing that
is already reported.

**Two manifests, and the split is a licensing decision, not a convenience.**
FERMAT is a gated dataset. The PUBLIC manifest therefore carries only what can
be redistributed without the dataset: stable item ids, hashes, the pinned
revision, and the run's own derived numbers. The PRIVATE manifest carries the
gated text and the model generations derived from gated pages, and is excluded
from release. `assert_public_manifest_is_safe` is the check, not the intention.

**Reconstruction fails closed.** Materialising the exact 300 rows requires
downloading a gated dataset, which needs Hugging Face authentication. Without
it this module records `alignment_status = "PENDING_AUTHENTICATED_RUN"` and
refuses to claim the sample is verified. It never guesses a revision and never
accepts a partial or order-insensitive match, because a manifest that claims
verification it did not perform is worse than no manifest.
"""

import hashlib
import json
import os
from typing import Optional, Sequence

import pandas as pd

from . import audit_diagnostics as ad
from . import prompts

#: The dataset the frozen run was drawn from, and the selection that drew it.
DATASET_ID = "ai4bharat/FERMAT"
DATASET_SPLIT = "train"
SELECTION = {"loader": "pilot.data.load_fermat_balanced", "n": 300,
             "seed": 42, "target_error_frac": 0.5,
             "note": ("shuffle(seed) is NOT stable across dataset revisions, "
                      "which is why per-item content hashes are published "
                      "alongside the ids; an index alone does not reproduce")}

#: Generation settings behind the frozen run.
RUN_PROTOCOL = {"k_transcription": 5, "k_grading": 5, "temperature": 0.7,
                "frozen_scorer": "strict_v1",
                "scorer_note": ("stored columns are authoritative; any local "
                                "rescore is a separately named diagnostic")}

PENDING = "PENDING_AUTHENTICATED_RUN"

#: Columns the public manifest may contain. Anything else is a leak until
#: proven otherwise, so the allow-list is explicit rather than a deny-list.
PUBLIC_COLUMNS = (
    "item_id", "has_error", "handwriting_style", "image_quality",
    "question_sha256", "reference_answer_sha256", "item_content_sha256",
    "perception_entropy", "reasoning_entropy", "transcription_correct",
    "grading_correct", "n_transcription_parse_failures",
    "n_grading_parse_failures", "k_transcription", "temperature", "model_id",
    "transcription_prompt_sha256", "grading_prompt_sha256",
    "dataset_id", "dataset_revision", "dataset_split", "selection_seed",
    "source_row_index", "alignment_status", "in_audit_union",
    "audit_sources", "audit_selection_reason",
)


def sha256_text(text) -> str:
    return hashlib.sha256(str(text if text is not None else "").encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_hashes() -> dict:
    """Hash the EXACT prompt strings, copied not paraphrased.

    Hashing the literal in `pilot/prompts.py` means a later edit to the wording
    changes the hash and the manifest stops matching, which is the point: a
    reproduction is only meaningful against the prompt actually used.
    """
    named = {
        "transcription_system": prompts.TRANSCRIPTION_SYSTEM_PROMPT,
        "transcription_user": prompts.TRANSCRIPTION_USER_PROMPT,
        "grading_system": prompts.GRADING_SYSTEM_PROMPT,
        "grading_user": prompts.GRADING_USER_PROMPT,
    }
    return {k: {"sha256": sha256_text(v), "n_chars": len(v)}
            for k, v in named.items()}


def dataset_revision(offline_ok: bool = True) -> dict:
    """The pinned Hub revision. Reachable WITHOUT auth even though data is not.

    Metadata and data have different access levels here: `dataset_info`
    succeeds anonymously and returns the commit sha, while `load_dataset`
    raises because the dataset is gated. So the revision CAN be pinned
    offline; only the row-level alignment needs authentication.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(DATASET_ID, timeout=30)
        return {"dataset_id": DATASET_ID, "revision": info.sha,
                "gated": str(getattr(info, "gated", "unknown")),
                "source": "huggingface_hub.dataset_info (anonymous)"}
    except Exception as e:                     # network-free environments
        if not offline_ok:
            raise
        return {"dataset_id": DATASET_ID, "revision": None,
                "gated": "unknown", "source": f"unavailable: {type(e).__name__}"}


def frozen_run_hashes(run_csv: str, run: pd.DataFrame) -> dict:
    """File-level and per-item content hashes for the frozen run."""
    per_item = [sha256_text(f"{run.loc[i, 'orig_q']}\x1f{run.loc[i, 'pert_a']}"
                            f"\x1f{int(bool(run.loc[i, 'has_error']))}")
                for i in run.index]
    return {"run_csv": os.path.basename(run_csv),
            "run_csv_sha256": sha256_file(run_csv),
            "n_rows": len(run), "per_item_sha256": per_item}


def verify_reconstruction(run: pd.DataFrame, source: Optional[pd.DataFrame] = None,
                          revision: Optional[str] = None) -> dict:
    """Fail closed unless all 300 rows align EXACTLY against a pinned revision.

    `source` is the downloaded dataset; without it (no Hugging Face auth) this
    reports PENDING and verifies nothing. Alignment requires every one of
    `orig_q`, `pert_a` and `has_error` to match on every row -- not a subset,
    not order-insensitively -- because a partial match is exactly what a
    drifted revision produces.
    """
    if source is None or revision is None:
        return {"alignment_status": PENDING, "verified": False,
                "n_aligned": 0, "n_expected": len(run),
                "reason": ("FERMAT is gated: dataset_info is anonymous but "
                           "load_dataset requires authentication. Run this "
                           "stage in an authenticated session to complete it."),
                "revision": revision}
    # `source` must be the sample reconstructed by the declared selection
    # procedure, in its final order. Looking each run row up anywhere in the
    # full dataset would only prove that the content exists; it would not prove
    # that seed=42 and the balanced sampler reproduce this evaluation subset.
    src_records = source.to_dict("records")
    run_records = run.to_dict("records")
    rows = []
    n_ok = 0
    for pos, (got, expected) in enumerate(zip(src_records, run_records)):
        got_hash = sha256_text(
            f"{got['orig_q']}\x1f{got['pert_a']}\x1f{int(bool(got['has_error']))}")
        expected_hash = sha256_text(
            f"{expected['orig_q']}\x1f{expected['pert_a']}"
            f"\x1f{int(bool(expected['has_error']))}")
        if got_hash == expected_hash:
            n_ok += 1
            rows.append(int(got.get("source_row_index", pos)))
        else:
            rows.append(None)
    ok = len(source) == len(run) and n_ok == len(run)
    return {"alignment_status": "VERIFIED" if ok else "FAILED",
            "verified": bool(ok), "n_aligned": int(n_ok),
            "n_expected": len(run), "source_row_index": rows,
            "revision": revision,
            "reason": "" if ok else
                      f"only {n_ok}/{len(run)} rows aligned; refusing to claim "
                      "reconstruction. Do not guess the revision."}


def reconstruct_balanced_source(dataset, n: int = 300, seed: int = 42,
                                target_error_frac: float = 0.5) -> pd.DataFrame:
    """Rebuild the frozen balanced sample while preserving original row ids.

    This deliberately mirrors :func:`pilot.data.load_fermat_balanced`: shuffle
    the full pinned dataset, take the requested number from each truth pool,
    concatenate, then shuffle the selected rows again with the same seed. The
    extra ``source_row_index`` column is metadata only and lets an authorized
    user locate the gated row without publishing its contents.
    """
    if not 0.0 < target_error_frac < 1.0:
        raise ValueError("target_error_frac must be in (0, 1)")
    required = {"orig_q", "pert_a", "has_error"}
    columns = set(getattr(dataset, "column_names", []))
    if not required <= columns:
        raise ValueError(f"source dataset missing fields: {sorted(required - columns)}")

    tagged = dataset.add_column("source_row_index", list(range(len(dataset))))
    shuffled = tagged.shuffle(seed=seed)
    error_idx, clean_idx = [], []
    for i, has_error in enumerate(shuffled["has_error"]):
        (error_idx if bool(has_error) else clean_idx).append(i)

    want_error = round(n * target_error_frac)
    want_clean = n - want_error
    scale = min(
        1.0,
        len(error_idx) / want_error if want_error else 1.0,
        len(clean_idx) / want_clean if want_clean else 1.0,
    )
    take_error = int(want_error * scale)
    take_clean = int(want_clean * scale)
    if take_error + take_clean != n:
        raise ValueError(
            f"pinned dataset cannot supply n={n} at fraction={target_error_frac}; "
            f"would return {take_error + take_clean}")

    selected = error_idx[:take_error] + clean_idx[:take_clean]
    sample = shuffled.select(selected).shuffle(seed=seed)
    keep = ["orig_q", "pert_a", "has_error", "source_row_index"]
    return sample.select_columns(keep).to_pandas()


# --- audit long form -------------------------------------------------------

def audit_labels_long(audit_dir: str = "reference/audit") -> pd.DataFrame:
    """One row per item PER AUDIT PASS. Overlaps are preserved, never merged.

    Collapsing the 312 rows to the 234-item union would destroy the record
    that 78 of them were coded twice, and with it the only intra-rater
    reliability this project has.
    """
    rows = []
    for set_name, files in ad.SETS.items():
        for fname in files:
            d = pd.read_csv(os.path.join(audit_dir, fname))
            key = "item_id" if "item_id" in d.columns else "item"
            for _, r in d.iterrows():
                rows.append({
                    "item_id": int(r[key]),
                    "audit_set": set_name,
                    "source_file": fname,
                    "final_label": r["final_label"],
                    "truth": ad.TRUTH_MAP[r["final_label"]],
                    "confidence": r.get("confidence", ""),
                    "note": " ".join(str(r.get("note", "")).split()),
                    "coded_explicitly": r.get("coded_explicitly", ""),
                    "extraction_status": r.get("extraction_status", ""),
                })
    out = pd.DataFrame(rows).sort_values(["item_id", "audit_set"])
    return out.reset_index(drop=True)


def audit_invariants(long: pd.DataFrame) -> dict:
    """The five facts that must stay true of the audit, as a checkable dict."""
    dup = long.groupby("item_id").size()
    contradictions = []
    for item, g in long.groupby("item_id"):
        det = {t for t in g["truth"] if t in ("correct", "wrong")}
        if len(det) > 1:
            contradictions.append(int(item))
    return {
        "n_rows": len(long),
        "n_unique_items": int(long["item_id"].nunique()),
        "n_overlap_rows": int(len(long) - long["item_id"].nunique()),
        "n_items_coded_twice": int((dup > 1).sum()),
        "hard_contradictions": sorted(contradictions),
        "is_population_estimate": False,
        "coverage_note": ("three TARGETED sets, two of them one-directional by "
                          "construction; rates here are not estimates for all "
                          "300 items and must not be quoted as such"),
    }


# --- manifests -------------------------------------------------------------

def build_manifests(run: pd.DataFrame, run_csv: str,
                    audit_long: pd.DataFrame,
                    revision_info: dict,
                    alignment: dict) -> tuple:
    """(public, private). The public frame carries no gated text by construction."""
    hashes = frozen_run_hashes(run_csv, run)
    ph = prompt_hashes()
    src_idx = alignment.get("source_row_index") or [None] * len(run)
    by_item = audit_long.groupby("item_id")["audit_set"].apply(
        lambda s: "|".join(sorted(set(s)))).to_dict()

    pub, priv = [], []
    for n, i in enumerate(run.index):
        item = int(i)
        pub.append({
            "item_id": item,
            "has_error": bool(run.loc[i, "has_error"]),
            "handwriting_style": bool(run.loc[i, "handwriting_style"]),
            "image_quality": bool(run.loc[i, "image_quality"]),
            "question_sha256": sha256_text(run.loc[i, "orig_q"]),
            "reference_answer_sha256": sha256_text(run.loc[i, "pert_a"]),
            "item_content_sha256": hashes["per_item_sha256"][n],
            "perception_entropy": float(run.loc[i, "perception_entropy"]),
            "reasoning_entropy": float(run.loc[i, "reasoning_entropy"]),
            "transcription_correct": bool(run.loc[i, "transcription_correct"]),
            "grading_correct": bool(run.loc[i, "grading_correct"]),
            "n_transcription_parse_failures": int(run.loc[i, "n_transcription_parse_failures"]),
            "n_grading_parse_failures": int(run.loc[i, "n_grading_parse_failures"]),
            "k_transcription": int(run.loc[i, "k_transcription"]),
            "temperature": float(RUN_PROTOCOL["temperature"]),
            "model_id": str(run.loc[i, "model_id"]),
            "transcription_prompt_sha256": ph["transcription_user"]["sha256"],
            "grading_prompt_sha256": ph["grading_user"]["sha256"],
            "dataset_id": DATASET_ID,
            "dataset_revision": revision_info.get("revision") or "",
            "dataset_split": DATASET_SPLIT,
            "selection_seed": SELECTION["seed"],
            "source_row_index": "" if src_idx[n] is None else int(src_idx[n]),
            "alignment_status": alignment["alignment_status"],
            "in_audit_union": item in by_item,
            "audit_sources": by_item.get(item, ""),
            "audit_selection_reason": ("targeted audit set; NOT an IID sample "
                                       "of the 300" if item in by_item else ""),
        })
        priv.append({
            "item_id": item,
            "orig_q": run.loc[i, "orig_q"],
            "pert_a": run.loc[i, "pert_a"],
            "all_transcription_samples_raw": run.loc[i, "all_transcription_samples_raw"],
            "all_grading_samples_raw": run.loc[i, "all_grading_samples_raw"],
            "temp0_transcription_raw": run.loc[i, "temp0_transcription_raw"],
        })
    return pd.DataFrame(pub)[list(PUBLIC_COLUMNS)], pd.DataFrame(priv)


#: Substrings that must never appear in a public artifact.
_LEAK_PATTERNS = ("/content/drive", "MyDrive", "hf_", "ghp_", "sk-",
                  ".tokens.json", "/Users/")


def assert_public_manifest_is_safe(public: pd.DataFrame,
                                   private: pd.DataFrame) -> dict:
    """Refuse to release gated text, secrets, or local paths. Checked, not promised."""
    extra = [c for c in public.columns if c not in PUBLIC_COLUMNS]
    if extra:
        raise AssertionError(f"public manifest has non-allow-listed columns: {extra}")
    gated = {"orig_q", "pert_a", "all_transcription_samples_raw",
             "all_grading_samples_raw", "temp0_transcription_raw", "image"}
    if gated & set(public.columns):
        raise AssertionError(f"gated content in public manifest: {gated & set(public.columns)}")
    blob = public.astype(str).to_csv(index=False)
    hits = [p for p in _LEAK_PATTERNS if p in blob]
    if hits:
        raise AssertionError(f"public manifest contains {hits}")
    # The private frame is the one that legitimately holds gated text; assert
    # it actually does, or the split is cosmetic.
    if not {"orig_q", "pert_a"} <= set(private.columns):
        raise AssertionError("private manifest is missing the gated text it exists to hold")
    return {"public_columns": len(public.columns), "leaks": [], "safe": True}


def write_artifact(out_dir: str, public: pd.DataFrame, private: pd.DataFrame,
                   audit_long: pd.DataFrame, provenance: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "fermat_n300_public_manifest.csv": public,
        "fermat_n300_private_manifest.csv": private,
        "audit_labels_long.csv": audit_long,
    }
    written = {}
    for name, frame in paths.items():
        p = os.path.join(out_dir, name)
        frame.to_csv(p, index=False)
        written[p] = sha256_file(p)
    prov = os.path.join(out_dir, "provenance.json")
    with open(prov, "w") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True, default=str)
    written[prov] = sha256_file(prov)
    return written
