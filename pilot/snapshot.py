"""Compute every headline metric for a run and dump it as a comparable snapshot.

Why this exists: the results CSVs live in ``results/``, which is untracked and
arrives from Drive by hand. If that file is lost or a future run replaces it,
the numbers we reported are gone with it. A snapshot is a small tracked JSON
holding every figure, so a later run can be diffed against this one without
needing the original CSV at all.

Usage::

    python -m pilot.snapshot results/scaleup_n300....csv reference/n300.json
    python -m pilot.snapshot --compare reference/n300.json reference/new.json
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pilot.plotting import (
    aurc,
    classify_stratum_result,
    auroc_sensitivity,
    bootstrap_auroc_ci,
    evaluate_conformal_abstention,
    majority_class_baseline,
    risk_coverage_curve,
)

# Metrics whose movement between runs is worth flagging loudly in a comparison.
HEADLINE_KEYS = (
    "perception.auroc",
    "perception.aurc",
    "reasoning.auroc",
    "grading.accuracy",
    "abstention.precision",
)


def snapshot_metrics(
    df: pd.DataFrame,
    label: str = "",
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Every reported figure for one run, as a plain nested dict.

    Deliberately recomputed from the raw columns rather than copied from a
    report, so a snapshot can never drift from what the data actually says.
    """
    gt = df["has_error"].astype(bool)
    k = int(df["k_transcription"].iloc[0]) if "k_transcription" in df else 5

    def ci(frame, entropy_col, correct_col):
        r = bootstrap_auroc_ci(frame, entropy_col, correct_col, n_boot=n_boot, seed=seed)
        return {kk: r[kk] for kk in
                ("auroc", "ci_low", "ci_high", "n_items", "n_error", "excludes_chance")}

    at_max = np.isclose(df["perception_entropy"].astype(float), math.log(k))
    flagged = df[at_max]
    n_flag = int(len(flagged))
    n_flag_wrong = int((~flagged["transcription_correct"].astype(bool)).sum())
    total_errors = int((~df["transcription_correct"].astype(bool)).sum())

    sens = auroc_sensitivity(
        df, "perception_entropy", "transcription_correct",
        "n_transcription_parse_failures", k=k, n_boot=n_boot, seed=seed,
    )

    out = {
        "label": label,
        "n_items": int(len(df)),
        "model_id": str(df["model_id"].iloc[0]) if "model_id" in df else None,
        "k_transcription": k,
        "k_grading": int(df["k_grading"].iloc[0]) if "k_grading" in df else None,
        "frac_has_error": float(gt.mean()),
        # Recorded because a run scored without SymPy's LaTeX parser is not
        # comparable to one scored with it; see canonicalize.latex_parser_available.
        "latex_parser_available": (
            bool(df["latex_parser_available"].iloc[0])
            if "latex_parser_available" in df else None
        ),
        "sympy_version": (
            str(df["sympy_version"].iloc[0]) if "sympy_version" in df else None
        ),
        "perception": {
            **ci(df, "perception_entropy", "transcription_correct"),
            "accuracy": float(df["transcription_correct"].astype(bool).mean()),
            "aurc": aurc(df, "perception_entropy", "transcription_correct")["aurc"],
            "aurc_baseline": aurc(df, "perception_entropy", "transcription_correct")["baseline_aurc"],
            "sensitivity": {
                cut: (None if sens[cut] is None else
                      {kk: sens[cut][kk] for kk in ("auroc", "ci_low", "ci_high", "n_items")})
                for cut in ("full", "excl_parse_failures", "excl_max_entropy", "excl_both")
            },
            "robust": bool(sens["robust"]),
        },
        "reasoning": {
            **ci(df, "reasoning_entropy", "grading_correct"),
            "aurc": aurc(df, "reasoning_entropy", "grading_correct")["aurc"],
            "aurc_baseline": aurc(df, "reasoning_entropy", "grading_correct")["baseline_aurc"],
            "error_stratum": ci(df[gt], "reasoning_entropy", "grading_correct"),
            "clean_stratum": ci(df[~gt], "reasoning_entropy", "grading_correct"),
        },
        "grading": {
            "accuracy": float(df["grading_correct"].astype(bool).mean()),
            "accuracy_on_error_items": float(df.loc[gt, "grading_correct"].astype(bool).mean()),
            "accuracy_on_clean_items": float(df.loc[~gt, "grading_correct"].astype(bool).mean()),
            "baseline_accuracy": majority_class_baseline(df, "has_error")["baseline_accuracy"],
        },
        "abstention": {
            "n_flagged": n_flag,
            "n_flagged_wrong": n_flag_wrong,
            "precision": n_flag_wrong / n_flag if n_flag else float("nan"),
            "recall": n_flag_wrong / total_errors if total_errors else float("nan"),
        },
        "risk_coverage": risk_coverage_curve(
            df, "perception_entropy", "transcription_correct"
        ).to_dict(orient="records"),
        "conformal": {
            f"target_{t:.2f}": evaluate_conformal_abstention(
                df, "perception_entropy", "transcription_correct",
                target_risk=t, n_splits=1000, seed=seed,
            )
            for t in (0.30, 0.40, 0.50)
        },
        "integrity": {
            "transcription_parse_failure_rate": float(
                df["n_transcription_parse_failures"].sum() / (len(df) * k)
            ),
            "grading_parse_failure_rate": float(
                df["n_grading_parse_failures"].sum()
                / (len(df) * (df["k_grading"].iloc[0] if "k_grading" in df else 5))
            ),
        },
    }
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[path] = float(value)
    return flat


def compare_snapshots(old: dict, new: dict, tol: float = 1e-6) -> dict:
    """Numeric differences between two snapshots, headline metrics called out.

    Returns changed/added/removed keys rather than a verdict: whether a change
    is a problem depends on whether the two runs were meant to be identical
    (a code refactor) or deliberately different (a new model), and only the
    caller knows which.
    """
    a, b = _flatten(old), _flatten(new)
    changed = {
        key: {"old": a[key], "new": b[key], "delta": b[key] - a[key]}
        for key in sorted(set(a) & set(b))
        if not (math.isnan(a[key]) and math.isnan(b[key])) and abs(b[key] - a[key]) > tol
    }
    return {
        "changed": changed,
        "headline_changed": {k: v for k, v in changed.items() if k in HEADLINE_KEYS},
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
    }



def snapshot_grading_metrics(
    df: pd.DataFrame,
    label: str = "",
    n_boot: int = 10000,
    seed: int = 0,
    extra: Optional[dict] = None,
) -> dict:
    """Snapshot for a grading-arm-only run (no perception columns).

    snapshot_metrics above assumes a full FERMAT-shaped CSV with
    perception_entropy/transcription_correct. Most runs in this project are
    grading-only extensions, and one (ScratchMath) has no has_error=0 items
    at all, so it needs a shape that:

      - omits the perception arm rather than emitting NaNs for it;
      - reports the pooled AUROC but flags it, because on a non-50/50
        sample pooling mechanically favours the larger stratum and has
        repeatedly looked like a result when it was not (see the n=500/n=800
        CSVs, which read 0.61 pooled while the strata run in opposite
        directions);
      - records an explicit verdict per stratum via classify_stratum_result,
        including "not_measured" when a stratum is absent entirely -- which
        is a different statement from "underpowered" and must not collapse
        into it;
      - records the entropy value distribution, because a degenerate one
        (InternVL3: 67% pinned at max; ScratchMath: 70% at zero) invalidates
        the AUROC regardless of what the CI says.

    ``extra`` is merged in verbatim for run-specific evidence that does not
    generalise -- e.g. ScratchMath's non-engagement rate.
    """
    gt = df["has_error"].astype(bool)
    k = int(df["k_grading"].iloc[0]) if "k_grading" in df else 5

    def ci(frame):
        if len(frame) == 0:
            return None
        r = bootstrap_auroc_ci(frame, "reasoning_entropy", "grading_correct",
                               n_boot=n_boot, seed=seed)
        return {kk: r[kk] for kk in
                ("auroc", "ci_low", "ci_high", "n_items", "n_error", "excludes_chance")}

    error_ci = ci(df[gt])
    clean_ci = ci(df[~gt])
    pooled_ci = ci(df)

    entropy_counts = (
        df["reasoning_entropy"].round(3).value_counts().sort_index()
    )
    max_share = float(entropy_counts.max() / len(df)) if len(df) else float("nan")

    out = {
        "label": label,
        "arm": "grading_only",
        "n_items": int(len(df)),
        "model_id": str(df["model_id"].iloc[0]) if "model_id" in df else None,
        "quantized": (bool(df["quantized"].iloc[0]) if "quantized" in df else None),
        "k_grading": k,
        "composition": {
            "n_has_error": int(gt.sum()),
            "n_clean": int((~gt).sum()),
            "frac_has_error": float(gt.mean()),
            "is_balanced": bool(abs(gt.mean() - 0.5) < 0.01),
        },
        "grading": {
            "accuracy": float(df["grading_correct"].astype(bool).mean()),
            "baseline_accuracy": majority_class_baseline(df, "has_error")["baseline_accuracy"],
            "accuracy_on_error_items": (
                float(df.loc[gt, "grading_correct"].astype(bool).mean()) if gt.any() else None
            ),
            "accuracy_on_clean_items": (
                float(df.loc[~gt, "grading_correct"].astype(bool).mean())
                if (~gt).any() else None
            ),
        },
        "reasoning": {
            "pooled": pooled_ci,
            # Load-bearing caveat, not decoration: read the strata, not this.
            "pooled_is_misleading_unless_balanced": not bool(abs(gt.mean() - 0.5) < 0.01),
            "error_stratum": error_ci,
            "error_stratum_verdict": classify_stratum_result(error_ci),
            "clean_stratum": clean_ci,
            "clean_stratum_verdict": classify_stratum_result(clean_ci),
        },
        "entropy_distribution": {
            "n_distinct_values": int(len(entropy_counts)),
            "counts": {str(v): int(c) for v, c in entropy_counts.items()},
            "largest_single_value_share": max_share,
            # Mirrors the InternVL3 / ScratchMath diagnosis: past ~2/3 on one
            # value there is little left for a ranking metric to order.
            "is_degenerate": bool(max_share >= 0.60),
        },
        "integrity": {
            "grading_parse_failure_rate": float(
                df["n_grading_parse_failures"].sum() / (len(df) * k)
            ) if "n_grading_parse_failures" in df else None,
        },
    }
    if extra:
        out["extra"] = extra
    return out


def write_grading_snapshot(
    csv_path: str | Path,
    out_path: str | Path,
    label: str = "",
    extra: Optional[dict] = None,
) -> dict:
    df = pd.read_csv(csv_path)
    snap = snapshot_grading_metrics(df, label=label or Path(csv_path).stem, extra=extra)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(snap, indent=2) + "\n")
    return snap


def write_snapshot(csv_path: str | Path, out_path: str | Path, label: str = "") -> dict:
    df = pd.read_csv(csv_path)
    snap = snapshot_metrics(df, label=label or Path(csv_path).stem)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(snap, indent=2) + "\n")
    return snap


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("csv", nargs="?")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    if args.compare:
        old = json.loads(Path(args.compare[0]).read_text())
        new = json.loads(Path(args.compare[1]).read_text())
        diff = compare_snapshots(old, new)
        if not diff["changed"]:
            print("identical on every numeric field")
            return
        print(f"{len(diff['changed'])} numeric fields differ "
              f"({len(diff['headline_changed'])} headline):")
        for key, d in diff["changed"].items():
            mark = " <-- HEADLINE" if key in diff["headline_changed"] else ""
            print(f"  {key:52s} {d['old']:9.4f} -> {d['new']:9.4f}  "
                  f"({d['delta']:+.4f}){mark}")
        return

    if not args.csv or not args.out:
        parser.error("give a CSV and an output path, or use --compare")
    snap = write_snapshot(args.csv, args.out, args.label)
    print(f"wrote {args.out}")
    print(f"  perception AUROC {snap['perception']['auroc']:.3f} "
          f"[{snap['perception']['ci_low']:.3f}, {snap['perception']['ci_high']:.3f}]")
    print(f"  reasoning  AUROC {snap['reasoning']['auroc']:.3f} "
          f"[{snap['reasoning']['ci_low']:.3f}, {snap['reasoning']['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
