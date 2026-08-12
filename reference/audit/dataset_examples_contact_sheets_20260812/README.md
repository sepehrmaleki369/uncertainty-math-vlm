# Example contact sheets — TEXT ONLY

Open `index.html`. Every caption field the image sheet would carry is already present: item id, `has_error`, both rule verdicts, answer type, entropy, the truth and model spans, both labels, any human label, and a one-line note.

**Images are not attached, and cannot be offline.** FERMAT is a gated dataset and its `image` column is never persisted to the run CSV, so no crop exists locally. **Image export requires an authenticated Colab/Drive session** — reuse notebook 23's contact-sheet machinery, keyed on `manifest.csv`'s `item_id`. That is an asset export, not a re-analysis: no number here changes when the tiles arrive.

20 items: 10 `has_error=1`, 10 clean, 10 scored correct by `strict_v1`. Selection is seeded and round-robin over answer types.
