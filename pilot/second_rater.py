"""Blinded second-rater packet, and a gate that refuses to invent agreement.

Offline. Builds a deterministic queue over the 234-item audit union, a BLANK
annotation template, and rater instructions. It computes nothing about
agreement until a real human returns a completed file.

**No label in this module is ever synthesised, inferred, copied from the first
rater, or requested from a model.** `agreement_or_pending` prints
`SECOND-RATER STATUS: PENDING` and returns without computing anything until a
completed file exists and validates. That refusal is the feature: an agreement
number derived from anything other than a second human reading the page would
be a fabricated result, and it would be indistinguishable from a real one in
the output.

**What the core can and cannot estimate.** The 120-item agreement core is
drawn from the AUDIT UNION, which is itself three targeted sets. It therefore
estimates agreement for that design and nothing wider: not the 300-item run,
not FERMAT. The four known intra-rater contradictions are held in a separate
challenge set precisely so they cannot inflate or deflate the representative
figure.
"""

import hashlib
import json
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "second_rater_v1_20260814"

#: Sizes fixed before any second-rater label exists, so the design cannot be
#: adjusted after seeing agreement.
N_CORE = 120
CHALLENGE_ITEMS = (108, 149, 222, 230)

MODEL_CORRECTNESS = ("correct", "wrong", "indeterminate")
REFERENCE_FIDELITY = ("faithful", "unfaithful", "indeterminate")
FAILURE_CATEGORY = ("notation_misread", "copied_wrong_line", "hallucination",
                    "extraction_issue", "reference_issue",
                    "ambiguous_multianswer", "other", "not_applicable")
CONFIDENCE = ("high", "medium", "low")

#: The rater fills these. They ship blank and are validated on return.
ANSWER_FIELDS = ("model_correctness", "reference_fidelity", "failure_category",
                 "confidence", "evidence_note", "rater_pseudonym")

#: Never shown during coding. Each would leak the answer or the design.
BLINDED_FIELDS = ("final_label", "truth", "note", "audit_set",
                  "strict_v1_correct", "strict_v2_correct", "perception_entropy",
                  "has_error", "is_challenge", "queue")


def _entropy_bin(h: float) -> str:
    if h <= 0.0:
        return "H=0"
    if h < 0.7:
        return "H_low"
    if h < 1.4:
        return "H_mid"
    return "H_high"


def build_queue(public: pd.DataFrame, audit_long: pd.DataFrame,
                run: pd.DataFrame, n_core: int = N_CORE,
                seed: int = 20260814) -> pd.DataFrame:
    """Deterministic queue over the audit union: core, challenge, extension.

    The core is stratified across the frozen scorer verdict, `has_error`, an
    entropy bin and the first-audit source, and is drawn BEFORE any
    second-rater label exists. Challenge items are removed first so they can
    never enter the representative sample.
    """
    rng = np.random.default_rng(seed)
    union = sorted(audit_long["item_id"].unique())
    src = (audit_long.groupby("item_id")["audit_set"]
           .apply(lambda s: "|".join(sorted(set(s)))).to_dict())
    pub = public.set_index("item_id")

    rows = []
    for i in union:
        rows.append({
            "item_id": int(i),
            "stratum": "|".join([
                "v1_correct" if bool(pub.loc[i, "transcription_correct"]) else "v1_wrong",
                "err" if bool(pub.loc[i, "has_error"]) else "clean",
                _entropy_bin(float(pub.loc[i, "perception_entropy"])),
                src.get(i, "?"),
            ]),
            "first_audit_sources": src.get(i, ""),
        })
    frame = pd.DataFrame(rows)

    challenge = [i for i in CHALLENGE_ITEMS if i in set(frame["item_id"])]
    pool = frame[~frame["item_id"].isin(challenge)].copy()

    # Proportional allocation across strata, largest remainder, then a seeded
    # draw inside each stratum. Deterministic and documented, so the selection
    # probability is reconstructable rather than folklore.
    sizes = pool.groupby("stratum").size()
    want = (sizes / sizes.sum() * n_core)
    take = np.floor(want).astype(int)
    remainder = want.sub(take)
    # DETERMINISTIC TIE-BREAK, and it is not cosmetic. 48 strata floor to 102
    # seats, so 18 are handed out by largest remainder -- and the remainder
    # takes only 13 distinct values, with the cut falling inside an EIGHT-WAY
    # tie. `sort_values` defaults to quicksort, which is not stable, so the
    # order within a tie group depended on the pandas version: the same seed
    # produced a different core on Colab than locally, and the queue file
    # hash differed. Sorting by (-remainder, stratum name) makes the
    # allocation a function of the data alone.
    for s in sorted(remainder.index, key=lambda st: (-float(remainder[st]), str(st))):
        if take.sum() >= n_core:
            break
        take[s] += 1
    core = []
    for stratum, n_take in take.items():
        ids = sorted(pool.loc[pool["stratum"] == stratum, "item_id"])
        rng.shuffle(ids)
        core.extend(ids[:int(n_take)])
    core = sorted(core)

    frame["queue"] = np.where(frame["item_id"].isin(challenge), "challenge",
                              np.where(frame["item_id"].isin(core), "core",
                                       "extension"))
    frame["selection_prob"] = [
        "" if q != "core" else
        round(float(take.get(st, 0)) / max(1, int((pool["stratum"] == st).sum())), 6)
        for q, st in zip(frame["queue"], frame["stratum"])]

    # A randomised review id, so display order carries no information about
    # item id, first-audit source, or which items are challenge cases.
    order = list(frame.index)
    rng.shuffle(order)
    frame["review_order"] = pd.Series(range(len(order)), index=order)
    frame["review_id"] = [f"R{n:04d}" for n in frame["review_order"]]
    frame["protocol_version"] = PROTOCOL_VERSION
    frame["seed"] = seed
    return frame.sort_values("review_order").reset_index(drop=True)


def queue_summary(queue: pd.DataFrame) -> str:
    c = queue["queue"].value_counts()
    core, chal = set(queue.loc[queue["queue"] == "core", "item_id"]), \
        set(queue.loc[queue["queue"] == "challenge", "item_id"])
    assert not (core & chal), "core and challenge must be disjoint"
    return (f"queue over {len(queue)} audited items: "
            f"core {int(c.get('core', 0))}, challenge {int(c.get('challenge', 0))}, "
            f"extension {int(c.get('extension', 0))}\n"
            f"  core is stratified over {queue.loc[queue['queue']=='core','stratum'].nunique()} strata; "
            f"challenge = {sorted(chal)}\n"
            "  the core estimates agreement for the AUDITED UNION only, not "
            "the 300-item run and not FERMAT")


def blank_template(queue: pd.DataFrame) -> pd.DataFrame:
    """The file the rater fills. Every answer column ships EMPTY."""
    out = pd.DataFrame({
        "review_id": queue["review_id"],
        "protocol_version": PROTOCOL_VERSION,
    })
    for f in ANSWER_FIELDS:
        out[f] = ""
    out["timestamp"] = ""
    return out


def all_answers_blank(template: pd.DataFrame) -> bool:
    return all((template[f].astype(str).str.strip() == "").all()
               for f in ANSWER_FIELDS)


def rater_instructions() -> str:
    return f"""# Second-rater instructions ({PROTOCOL_VERSION})

You are the **second** reader. Someone has already coded these pages; you will
not see their labels, and that is deliberate. If you could see them you would
anchor on them, and the agreement number would measure suggestion rather than
reliability.

## What you are shown

The handwritten page, the model's transcription, and the dataset's reference
answer. Nothing else. In particular you are NOT shown the automatic
correct/wrong verdict, the first rater's label or note, the entropy, or whether
the page carries a deliberately injected error.

**You must open the image.** Several of these cannot be decided from the text
alone; that is the point of a second read.

## The fields

**`model_correctness`** -- did the model read the page correctly?
  * `correct` -- the model's answer matches what is written on the page.
  * `wrong` -- it does not.
  * `indeterminate` -- you cannot tell from the page. Use this freely; it is a
    real answer, not a failure to decide.

**`reference_fidelity`** -- does the dataset's reference answer match the page?
  * `faithful` / `unfaithful` / `indeterminate`. The reference is sometimes
    wrong or truncated, and that is worth recording separately from the model.

**`failure_category`** -- only if `model_correctness = wrong`, else
`not_applicable`:
  * `notation_misread` -- misread a symbol or digit on the page.
  * `copied_wrong_line` -- copied a real line of the page, the wrong one.
  * `hallucination` -- produced content not on the page at all.
  * `extraction_issue` -- the model's answer looks right but only part of it
    was captured, or the comparison used a fragment.
  * `reference_issue` -- the reference answer, not the model, is at fault.
  * `ambiguous_multianswer` -- several answers on the page, unclear which counts.
  * `other` -- with a note.

**`confidence`** -- `high`, `medium`, `low`.

**`evidence_note`** -- REQUIRED. One line naming what on the page decided it.

**`rater_pseudonym`** -- any stable string that is not your name.

## Two rules that matter for the analysis

1. `extraction_issue` and `indeterminate` are **not** ways of saying the model
   was wrong. They will never be converted into "wrong" downstream.
2. Do not go back and revise earlier rows after forming a theory. If you change
   your mind about a criterion, say so in the note and keep coding forward.

Leave a row blank rather than guessing. A blank row is analysable; a guessed
one is not.
"""


def write_packet(out_dir: str, queue: pd.DataFrame,
                 template: pd.DataFrame) -> dict:
    """Queue, blank template, blinding key, and instructions.

    The key mapping review ids to item ids is written SEPARATELY so the rater's
    working copy carries no item identity.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not all_answers_blank(template):
        raise AssertionError("refusing to write a template with answers in it")
    files = {
        "second_rater_queue.csv": queue,
        "second_rater_template_blank.csv": template,
        "second_rater_blinding_key_PRIVATE.csv": queue[["review_id", "item_id"]],
    }
    out = {}
    for name, frame in files.items():
        p = os.path.join(out_dir, name)
        frame.to_csv(p, index=False)
        out[p] = _sha(p)
    ip = os.path.join(out_dir, "second_rater_instructions.md")
    with open(ip, "w") as fh:
        fh.write(rater_instructions())
    out[ip] = _sha(ip)
    return out


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_col(d: pd.DataFrame, field: str) -> pd.Series:
    """Normalise a returned column to stripped strings, blanks as "".

    `astype(str)` is NOT enough: on an all-empty column pandas round-trips to
    float64 NaN, and `astype(str)` then leaves float nan in place rather than
    producing the string "nan", so downstream string calls raise. Caught by the
    all-blank-return test, which is exactly the file a rater would hand back if
    they opened the template and saved it without coding anything.
    """
    if field not in d.columns:
        return pd.Series([], dtype=str)
    return d[field].astype("string").fillna("").str.strip()


def validate_completed(path: str, queue: pd.DataFrame) -> dict:
    """Check a returned file before any agreement is computed."""
    d = pd.read_csv(path)
    problems = []
    if "review_id" not in d.columns:
        problems.append("no review_id column")
        return {"ok": False, "problems": problems}
    if d["review_id"].duplicated().any():
        problems.append("duplicate review_id rows")
    unknown = set(d["review_id"]) - set(queue["review_id"])
    if unknown:
        problems.append(f"review_ids not in the queue: {sorted(unknown)[:5]}")
    if "protocol_version" in d.columns:
        bad = set(d["protocol_version"].astype(str)) - {PROTOCOL_VERSION}
        if bad:
            problems.append(f"wrong protocol_version: {sorted(bad)}")
    for field, allowed in (("model_correctness", MODEL_CORRECTNESS),
                           ("reference_fidelity", REFERENCE_FIDELITY),
                           ("failure_category", FAILURE_CATEGORY),
                           ("confidence", CONFIDENCE)):
        if field not in d.columns:
            problems.append(f"missing field {field}")
            continue
        vals = {v for v in _clean_col(d, field) if v and v.lower() != "nan"}
        bad = vals - set(allowed)
        if bad:
            problems.append(f"{field}: values outside the vocabulary {sorted(bad)}")
    mc = _clean_col(d, "model_correctness")
    n_filled = int(mc.ne("").sum()) if len(mc) else 0
    if n_filled == 0:
        problems.append("no rows carry a model_correctness answer")
    return {"ok": not problems, "problems": problems, "n_rows": len(d),
            "n_answered": n_filled}


def agreement_or_pending(artifact_dir: str,
                         completed_filename: str = "second_rater_completed.csv") -> dict:
    """THE GATE. Refuses to compute agreement until a real human file exists.

    Returns rather than raising, so a notebook can run end to end and still
    report honestly that the protocol is unfinished.
    """
    path = os.path.join(artifact_dir, completed_filename)
    if not os.path.exists(path):
        return {"state": "PENDING", "path": path, "message": (
            "SECOND-RATER STATUS: PENDING\n"
            "  No completed second-rater file at "
            f"{path}.\n"
            "  Agreement, kappa and adjudication are NOT computed, and no\n"
            "  inter-rater claim may be made. This is the designed behaviour:\n"
            "  the only way to fill it is a second human reading the pages.")}
    return {"state": "FILE_PRESENT", "path": path, "message": (
        "A completed file exists; validate it with `validate_completed` and "
        "only then compute agreement.")}
