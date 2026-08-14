"""Build Figure 2, the four quadrants of self-agreement x scorer verdict.

Offline. Writes `paper/figures/four_quadrant.png` from four FERMAT pages in
`paper/figures/pages/`. No dataset download, no GPU, no network.

Until 2026-08-15 this figure had no builder in the repository, so its burned-in
panel labels could not be regenerated or checked against the audit. Two of the
four original panels turned out to contradict their own labels once checked; see
`paper/figure2_panel_verification.md` for the full record.

Every panel is now a human-verified case with a single consistent audit verdict.
Each panel's entropy and scorer verdict are asserted against the committed
public manifest, so a label cannot drift away from the item it describes. The
manifest carries no gated text, only ids, hashes and derived numbers, which is
why it is the reference here rather than the run CSV.

The pages themselves are gated FERMAT content and are NOT committed; see
`paper/figures/pages/README.md`. Export them once with
`pilot/31_export_figure2_pages.ipynb`, then:

    python paper/build_four_quadrant.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(ROOT, "paper", "figures", "pages")
OUT = os.path.join(ROOT, "paper", "figures", "four_quadrant.png")
MANIFEST = os.path.join(ROOT, "reference", "wacv_evaluation_artifact",
                        "fermat_n300_public_manifest.csv")

# (item id, headline, sub-line, expected entropy, expected scorer verdict).
#
# Laid out as a contingency table: the LEFT column is low entropy (the five
# samples agree) and the RIGHT column high entropy (they disagree); the TOP row
# is what the scorer passed and the BOTTOM row what it failed. The left column
# is then exactly the set the method is blind to, which the caption and
# \S"Failure Analysis" both refer to as a column rather than as a diagonal.
#
# All four are human-coded, each carries ONE verdict across every audit pass it
# appears in, AND each was re-checked against the raw samples rather than against
# the extracted span alone. That last step rejected three otherwise-obvious
# candidates:
#
#   176  reads as a clean unanimous failure; two passes both recorded "Model
#        correct" and the reference span had swallowed the whole display line.
#   271  is coded `true_wrong` ("misread the final result"), but all five
#        samples wrote 2.25 correctly. The span `25` is the decimal-split
#        extractor bug truncating `2.25`, so the coding is mistaken.
#   166  is `extraction_issue` in the census and `true_wrong` in the v2 pass, so
#        a panel asserting either would take a side in an open disagreement.
#
# Note what survived: at exactly H = 0 this run has three scored-wrong items
# (176, 218, 271) and every one is a scoring artifact, so there is no genuine
# unanimous model failure to show. The bottom-left panel is therefore the
# lowest-entropy REAL failure, a four-of-five agreement at H = 0.50.
PANELS = (
    (229, "low entropy, scored correct",
     "item 229,  H = 0.00,  audit: correct", 0.0, True),
    (92, "high entropy, scored correct",
     "item 92,  H = 1.61,  audit: correct", 1.6094379124341003, True),
    (160, "low entropy, scored wrong",
     "item 160,  H = 0.50,  audit: model wrong", 0.5004024235381879, False),
    (251, "high entropy, scored wrong",
     "item 251,  H = 1.33,  audit: model wrong", 1.3321790402101223, False),
)

# The figure is included at \linewidth in a single column, about 3.25in, so
# every font is scaled by 3.25/FIG_W in the printed PDF. At a 12in canvas a 13pt
# title printed at 3.5pt and the sub-line at 2.6pt, which is unreadable. Sizing
# the canvas near the column width keeps the labels legible: at 6.5in the scale
# is 0.5, so 14pt and 11pt print at 7pt and 5.5pt.
FIG_W, FIG_H = 6.5, 5.6
TITLE_PT, SUB_PT = 14, 11


def main() -> str:
    man = pd.read_csv(MANIFEST).set_index("item_id")
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))

    for ax, (item, headline, provenance, exp_h, exp_ok) in zip(
            axes.ravel(), PANELS):
        row = man.loc[item]
        assert abs(float(row["perception_entropy"]) - exp_h) < 1e-6, (
            f"item {item}: manifest entropy {row['perception_entropy']} "
            f"!= labelled {exp_h}")
        assert bool(row["transcription_correct"]) is exp_ok, (
            f"item {item}: manifest verdict {row['transcription_correct']} "
            f"!= labelled {exp_ok}")

        img = os.path.join(PAGES, f"item_{item}.jpg")
        assert os.path.exists(img), (
            f"missing page for item {item}: {os.path.relpath(img, ROOT)}\n"
            "FERMAT is gated and these pages are not committed. Export them "
            "with pilot/31_export_figure2_pages.ipynb.")

        ax.imshow(mpimg.imread(img))
        ax.set_xticks([])
        ax.set_yticks([])
        # The pages have different aspect ratios, so each axes shrinks to a
        # different height and centres in its cell, leaving the two rows of
        # headings at different heights. Pin every panel to the top of its cell.
        ax.set_anchor("N")
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb")
            s.set_linewidth(0.8)

        # imshow shrinks the axes box to the image aspect, so a title and a
        # transAxes label both anchor to the same edge and overlap. Offset both
        # in points from that edge instead.
        ax.set_title(headline, fontsize=TITLE_PT, pad=16, color="#111111")
        ax.annotate(provenance, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=SUB_PT, color="#555555")

    fig.subplots_adjust(left=0.012, right=0.988, top=0.945, bottom=0.012,
                        wspace=0.05, hspace=0.14)
    # 220 dpi over a 6.5in canvas is ~440 effective dpi once scaled into the
    # 3.25in column. Higher only inflates the upload.
    fig.savefig(OUT, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    for item, headline, provenance, _, _ in PANELS:
        print(f"  item {item:>3}  {headline:<30}  {provenance}")
    return OUT


if __name__ == "__main__":
    main()
