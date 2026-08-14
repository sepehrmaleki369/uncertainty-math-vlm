# Figure 2 source pages

The four handwritten FERMAT pages that Figure 2 is built from:
`item_229.jpg`, `item_92.jpg`, `item_160.jpg`, `item_251.jpg`.

**These are gated dataset content and are deliberately not committed.**
FERMAT is `gated=auto`, so the pages are excluded here for the same reason the
run CSVs were untracked in `9b22af6`. The `.jpg` files in this directory are
gitignored; this README is not.

## Getting them

Run `pilot/31_export_figure2_pages.ipynb` in Colab once. No GPU, no model, no
inference: it needs a Hugging Face login and nothing else. It writes the pages to
Drive first and then into the repo clone, and rebuilds the figure so the result
is visible before you leave the session. Copy the Drive folder
`uncertainty-math-vlm/figure2_pages/` into this directory to build locally.

Each page is resolved by matching `sha256(orig_q)` **and** `sha256(pert_a)`
against `reference/wacv_evaluation_artifact/fermat_n300_public_manifest.csv`.
Question text alone is not sufficient: only 246 of the 300 questions are
distinct, and an earlier attempt to identify a Figure 2 panel that way returned
two candidates. If the Hub copy has drifted the notebook aborts rather than
guessing a page.

## Consequence for reproducibility

`paper/figures/four_quadrant.png` is committed and does contain these pages, as
any paper figure of a dataset must. What is not committed is the unrendered
source, so a reader without FERMAT access gets the figure but cannot rebuild it.
That is the intended trade-off; `paper/figure2_panel_verification.md` records
every panel's provenance in text so the figure's claims stay checkable without
the images.
