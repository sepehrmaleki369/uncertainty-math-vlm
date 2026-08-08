"""Case snapshots: what the model saw, what it produced, and why it was wrong.

A metrics snapshot (pilot.snapshot) freezes the numbers. This module freezes
the *evidence behind* the numbers -- the individual items a reader needs to
see to believe a claim, and that a paper needs for its qualitative figures.

The split matters because they fail differently. A metrics snapshot goes
stale when code changes; a case snapshot goes stale when the CSV is lost,
which is likelier here since ``results/`` is untracked and arrives from
Drive by hand. Several findings in this project rest on a handful of
specific items -- the three unanimous misgraded cases, InternVL3's
max-entropy degeneration, ScratchMath's "cannot read it, still says error"
-- and re-deriving those from scratch each time is exactly how the
label-noise misreading survived as long as it did.

A case bundle is a directory holding:

    case.json    every field below, machine-readable
    image.png    the actual handwritten input, when available

and carries, per case:
  - the ground truth (question, reference answer, has_error)
  - every raw model sample, verbatim and unabridged
  - what each sample parsed to, so a parse failure is visible as a parse
    failure rather than as model disagreement
  - the derived entropy and correctness
  - ``category``, the reason this case was selected (low-entropy-correct,
    high-entropy-wrong, ...) so a reader can tell a cherry-picked example
    from a systematically drawn one
  - ``note``, a short human explanation -- deliberately a free-text field
    written by a person, not generated, because the interesting part of
    every case in this project so far has been something no metric caught.

Images are handled separately from text by design: FERMAT is a gated
dataset, so its images can only be fetched in an authenticated session
(Colab), while the text bundles build offline from the CSVs alone. Build
the bundles first, attach images later with attach_image(); a bundle
without an image is still useful and is never treated as an error.
"""

import ast
import io
import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

import pilot.canonicalize
import pilot.entropy
import pilot.parsing

# The categories worth freezing. Each names a distinct thing a reader might
# doubt, which is why "low entropy + correct" is here alongside the failures:
# without it, the failure cases have nothing to be contrasted against.
CASE_CATEGORIES = (
    "low_entropy_correct",       # the signal working: confident and right
    "high_entropy_wrong",        # the signal working: uncertain and wrong
    "high_entropy_correct",      # false alarm -- uncertain but right anyway
    "low_entropy_wrong",         # the dangerous one: confidently wrong
    "max_entropy_degenerate",    # InternVL3: 5 samples, 5 incompatible fragments
    "non_engagement_says_error",  # ScratchMath: "cannot read it" -> Error: 1
    "confidently_wrong_grading",  # grading unanimously wrong on a real error
)


def _parse_samples(raw, arm: str):
    """Return (samples, parsed) for one item, for whichever arm."""
    samples = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
    if arm == "grading":
        parsed = [pilot.parsing.parse_grading(s) for s in samples]
    else:
        parsed = [pilot.parsing.parse_transcription(s) for s in samples]
    return samples, parsed


def build_case(
    row: pd.Series,
    category: str,
    arm: str = "grading",
    note: str = "",
    run_label: str = "",
) -> dict:
    """One case bundle as a dict, recomputed from the row's raw samples.

    Recomputed rather than copied from the stored columns for the same
    reason snapshot_metrics recomputes: a bundle that merely echoed the CSV
    could not be used to check the CSV.
    """
    if category not in CASE_CATEGORIES:
        raise ValueError(
            f"unknown case category {category!r}; expected one of {list(CASE_CATEGORIES)}"
        )

    raw_col = "all_grading_samples_raw" if arm == "grading" else "all_transcription_samples_raw"
    samples, parsed = _parse_samples(row[raw_col], arm)

    if arm == "grading":
        labels = [None if d is None else str(d) for d in parsed]
    else:
        labels = [pilot.canonicalize.canonical_answer_label(p) for p in parsed]

    entropy = pilot.entropy.cluster_entropy(labels)
    majority, majority_count = pilot.entropy.majority_cluster(labels)

    case = {
        "run_label": run_label,
        "category": category,
        "arm": arm,
        "note": note,
        "model_id": row.get("model_id"),
        "ground_truth": {
            "question": row.get("question", row.get("orig_q")),
            "reference_answer": row.get("answer", row.get("pert_a")),
            "student_answer": row.get("student_answer"),
            "has_error": (int(row["has_error"]) if "has_error" in row else None),
        },
        "model_output": {
            "k": len(samples),
            "samples_raw": [str(s) for s in samples],
            "parsed": [None if p is None else str(p) for p in parsed],
            "cluster_labels": [str(x) for x in labels],
            "n_parse_failures": sum(1 for p in parsed if p is None),
        },
        "derived": {
            "entropy": float(entropy),
            "max_possible_entropy": float(math.log(len(samples))) if samples else 0.0,
            "at_max_entropy": bool(
                samples and math.isclose(entropy, math.log(len(samples)))
            ),
            "majority_label": str(majority),
            "majority_count": int(majority_count),
            "n_distinct_labels": len(set(labels)),
            "correct": (
                bool(row["grading_correct"]) if arm == "grading" and "grading_correct" in row
                else (bool(row["transcription_correct"])
                      if "transcription_correct" in row else None)
            ),
        },
    }
    return case


def write_case(case: dict, out_dir: str | Path, case_id: str) -> Path:
    """Write one bundle to out_dir/case_id/case.json and return its directory."""
    d = Path(out_dir) / case_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "case.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    return d


MAX_IMAGE_DIM = 1600


def attach_image(
    case_dir: str | Path,
    image,
    stem: str = "image",
    max_dim: int = MAX_IMAGE_DIM,
    jpeg_quality: int = 88,
) -> Path:
    """Save a PIL image next to an already-written bundle, downscaled to fit.

    Separate from write_case because FERMAT is gated: the text bundles build
    offline, the images need an authenticated session. Records the stored
    filename into case.json so a bundle is self-describing either way.

    Downscaled because these are tracked in git and the sources are large --
    FERMAT pages run 2-6 MB each, which would dwarf the rest of the repo for
    a handful of cases. 1600px on the long edge stays legible for reading
    handwriting, and the ORIGINAL dimensions are recorded so it is obvious
    the stored file is a reduction; the full-resolution image is always
    re-derivable from the dataset via the recorded ground truth.

    Format is chosen by encoding both ways and keeping the smaller file,
    rather than by a rule of thumb about line art vs photographs. On the
    real cases JPEG happens to win both kinds -- 24 KB -> 16 KB for
    ScratchMath's sparse stylus strokes, and 1.6 MB -> 264 KB for FERMAT's
    photographed pages -- but the margins differ by two orders of
    magnitude, and the intuition that PNG should win on line art was simply
    wrong here. Measuring costs one extra in-memory encode and removes the
    guess.
    """
    d = Path(case_dir)

    original = (image.width, image.height)
    stored = image
    if max_dim and max(original) > max_dim:
        scale = max_dim / max(original)
        stored = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        )

    png_buf = io.BytesIO()
    stored.save(png_buf, format="PNG", optimize=True)

    jpeg_buf = io.BytesIO()
    jpeg_ok = True
    try:
        rgb = stored.convert("RGB") if stored.mode not in ("RGB", "L") else stored
        rgb.save(jpeg_buf, format="JPEG", quality=jpeg_quality, optimize=True)
    except OSError:
        jpeg_ok = False

    if jpeg_ok and jpeg_buf.tell() < png_buf.tell():
        path = d / f"{stem}.jpg"
        path.write_bytes(jpeg_buf.getvalue())
    else:
        path = d / f"{stem}.png"
        path.write_bytes(png_buf.getvalue())

    # Never leave a stale file from the other encoding behind.
    for other in (d / f"{stem}.png", d / f"{stem}.jpg"):
        if other != path and other.exists():
            other.unlink()

    meta_path = d / "case.json"
    if meta_path.exists():
        case = json.loads(meta_path.read_text())
        case["image"] = {
            "filename": path.name,
            "width": stored.width,
            "height": stored.height,
            "original_width": original[0],
            "original_height": original[1],
            "downscaled": stored.size != original,
            "bytes": path.stat().st_size,
        }
        meta_path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    return path


def select_cases(
    df: pd.DataFrame,
    category: str,
    arm: str = "grading",
    n: int = 3,
    entropy_col: Optional[str] = None,
    correct_col: Optional[str] = None,
    predicate: Optional[Callable[[pd.Series], bool]] = None,
) -> pd.DataFrame:
    """Draw candidate rows for a category, deterministically.

    Selection is by rank on entropy (not random) so the same CSV always
    yields the same cases, and so "high entropy" means the actual extreme
    rather than an arbitrary member of a broad band. ``predicate`` covers
    the categories that are defined by sample *text* rather than by a
    number -- non-engagement in particular.
    """
    entropy_col = entropy_col or (
        "reasoning_entropy" if arm == "grading" else "perception_entropy"
    )
    correct_col = correct_col or (
        "grading_correct" if arm == "grading" else "transcription_correct"
    )
    correct = df[correct_col].astype(bool)

    if predicate is not None:
        sub = df[df.apply(predicate, axis=1)]
    elif category == "low_entropy_correct":
        sub = df[correct].nsmallest(n, entropy_col)
    elif category == "high_entropy_wrong":
        sub = df[~correct].nlargest(n, entropy_col)
    elif category == "high_entropy_correct":
        sub = df[correct].nlargest(n, entropy_col)
    elif category in ("low_entropy_wrong", "confidently_wrong_grading"):
        sub = df[~correct].nsmallest(n, entropy_col)
    elif category == "max_entropy_degenerate":
        k = len(ast.literal_eval(
            df.iloc[0]["all_transcription_samples_raw" if arm == "transcription"
                       else "all_grading_samples_raw"]))
        at_max = df[entropy_col].apply(lambda e: math.isclose(float(e), math.log(k)))
        sub = df[at_max & ~correct]
    else:
        raise ValueError(f"category {category!r} needs an explicit predicate")

    return sub.head(n)


def write_case_index(out_dir: str | Path) -> Path:
    """Write an index of every bundle under out_dir, for browsing.

    Exists so the collection is readable without opening each case.json --
    the manual failure audit is a reading task, and a directory of opaque
    ids is a bad interface for it.
    """
    root = Path(out_dir)
    entries = []
    for meta in sorted(root.glob("*/case.json")):
        case = json.loads(meta.read_text())
        entries.append({
            "case_id": meta.parent.name,
            "run_label": case.get("run_label"),
            "category": case.get("category"),
            "arm": case.get("arm"),
            "model_id": case.get("model_id"),
            "entropy": case["derived"]["entropy"],
            "correct": case["derived"]["correct"],
            "has_image": "image" in case,
            "note": case.get("note", "")[:200],
        })
    index_path = root / "index.json"
    index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    return index_path
