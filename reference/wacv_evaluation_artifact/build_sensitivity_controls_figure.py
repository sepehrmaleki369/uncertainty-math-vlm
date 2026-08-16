"""Build the controls companion to real_parameter_sensitivity_full.png.

The figure the notebook drew plots UNCONTROLLED AUROC against temperature, and
it rises. Read alone it argues for hotter sampling. It does not survive the
artifact controls, and the paper reports no temperature effect, so an artifact
folder containing only the rising curve would tell a reader the opposite of the
finding it belongs to.

This draws both panels on ONE shared y-axis, which is the whole point: the
collapse is only legible if the two panels are measured against the same scale.

    python reference/wacv_evaluation_artifact/build_sensitivity_controls_figure.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SNAP = os.path.join(ROOT, "reference", "parameter_sensitivity_20260816.json")
OUT = os.path.join(HERE, "real_parameter_sensitivity_controls.png")

TEMPS = (0.3, 0.7, 1.0)
# Categorical slots 1-3 in fixed order, never cycled. Validated for CVD
# separation and the normal-vision floor before use.
SERIES = ((3, "#2a78d6"), (5, "#eb6834"), (10, "#1baf7a"))
PANELS = (("full", "uncontrolled"),
          ("excl_both", "after removing parse failures\nand ceiling-entropy items"))

INK, MUTED = "#1a1a19", "#6b6b66"


def main() -> str:
    with open(SNAP) as fh:
        cond = json.load(fh)["conditions"]

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, (cut, title) in zip(axes, PANELS):
        ends = []
        for k, color in SERIES:
            ys = [cond[f"T{t}_K{k}"][cut]["auroc"] for t in TEMPS]
            lo = [ys[i] - cond[f"T{t}_K{k}"][cut]["ci_low"] for i, t in enumerate(TEMPS)]
            hi = [cond[f"T{t}_K{k}"][cut]["ci_high"] - ys[i] for i, t in enumerate(TEMPS)]
            ax.errorbar(TEMPS, ys, yerr=[lo, hi], color=color, lw=2, marker="o",
                        markersize=8, capsize=4, elinewidth=1.4, zorder=3,
                        markeredgecolor="white", markeredgewidth=1.4)
            ends.append((ys[-1], k))

        # Direct labels: identity is never colour-alone, and they discharge the
        # validator's contrast warning on the aqua series. Stagger any pair
        # closer than MIN_SEP -- at T=1.0 uncontrolled, K=5 and K=3 sit 0.014
        # apart and the two labels overlapped in the first render.
        MIN_SEP = 0.022
        placed = []
        for y, k in sorted(ends):
            if placed and y - placed[-1] < MIN_SEP:
                y = placed[-1] + MIN_SEP
            placed.append(y)
            ax.annotate(f"K={k}", xy=(TEMPS[-1], y), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=10,
                        color=MUTED)

        ax.axhline(0.5, ls="--", lw=1, color="#b8b8b2", zorder=1)
        ax.annotate("chance", xy=(0.3, 0.5), xytext=(0, 5),
                    textcoords="offset points", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=11, color=INK, pad=10)
        ax.set_xlabel("sampling temperature", fontsize=10, color=MUTED)
        ax.set_xticks(TEMPS)
        ax.set_xlim(0.18, 1.16)
        ax.grid(axis="y", color="#e8e8e4", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d5d5d0")
        ax.tick_params(colors=MUTED, labelsize=9)

    axes[0].set_ylabel("AUROC against the frozen scorer", fontsize=10, color=MUTED)
    # Explicit proxy handles. Passing labels positionally let matplotlib bind
    # them to whatever artists it had collected, including the chance line, so
    # the first render's legend was shifted by one and named the wrong colours.
    axes[0].legend(handles=[Line2D([], [], color=c, lw=2, marker="o",
                                   markersize=7, label=f"K={k}")
                            for k, c in SERIES],
                   loc="lower right", frameon=False, fontsize=9,
                   labelcolor=MUTED)
    fig.suptitle("Sampling temperature: the trend does not survive the controls",
                 fontsize=13, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)

    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    for cut, label in PANELS:
        vals = [f"T={t} K=5 {cond[f'T{t}_K5'][cut]['auroc']:.3f}" for t in TEMPS]
        print(f"  {label.splitlines()[0]:22s} {'  '.join(vals)}")
    return OUT


if __name__ == "__main__":
    main()
