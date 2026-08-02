# uncertainty-math-vlm

Pilot testing whether entropy over a VLM's own repeated samples predicts when
it's wrong, on FERMAT (handwritten math) with Qwen2.5-VL-3B. Two arms:
**perception** (transcription instability) and **reasoning** (grading
instability). See `README.md` for setup/layout basics — this file is
working context for future sessions, not a repeat of that.

## Current findings — read `report/report.tex` for full detail

Don't re-derive findings from git history; the report is the maintained,
compiled source of truth (`report/report.pdf` is the rendered version, kept
in sync with `report/report.tex`). Status as of the last update:

- **Perception entropy: working, AUROC 0.79.** Two prior approaches failed
  first (exact-string clustering on the full derivation — too strict, 83%
  pinned at max entropy; naive NLI/embedding clustering — over-merged, 95%
  implausible correctness). Fix: extract just the final answer span
  (`pilot/canonicalize.py:extract_final_answer`) before clustering, not the
  whole derivation.
- **Reasoning entropy, digit-only: working but weak** (AUROC 0.62 at K=5,
  0.665 at K=15 — real but close to a structural ceiling for a binary signal).
- **Reasoning entropy, text-clustering: works at K=5 (AUROC 0.70), fails at
  K=15 (AUROC 0.52, worse than digit-only).** Root cause: NLI + union-find
  transitive cascade merging — see `pilot/semantic.py:nli_cluster_labels`'s
  docstring for the mechanism and a real example. **Do not use
  `nli_cluster_labels`/`semantic_cluster_entropy` at K > 5 without either
  re-verifying for that specific use case or replacing the union-find step
  with a stricter merge criterion** (e.g. minimum intra-cluster pairwise
  agreement fraction, not any single transitive chain).

## Repo-specific conventions

- **Notebooks**: use the `NotebookEdit` tool, not `Edit` — `pilot.ipynb` and
  `pilot/02_reasoning_text_entropy.ipynb` will reject direct `Edit` calls.
  For multi-cell inserts, a small Python script (`json.load` → mutate
  `cells` → `json.dump`) is more reliable than chained `NotebookEdit` insert
  calls — see git history for examples.
- **`pilot.ipynb` diffs against a clean tree are usually not yours.** The
  user runs it live (Colab or VS Code's Jupyter connection), and execution
  outputs sync back to the file on disk. Check `git diff --stat` and scan
  for leaked tokens before committing, but don't strip legitimate output
  diffs or revert cells the user added themselves.
- **Dry-run notebook cell logic with stubs before handoff.** Extract the
  cell source, monkeypatch GPU-dependent calls (`generate()`, image
  processing) with fakes, and exec it locally to catch bugs before the user
  burns Colab time on it. This caught real bugs multiple times (checkpoint
  threshold gaps, `re.sub` backslash-escaping corrupting injected stubs).
- **Every real bug found on actual pilot data gets a regression test that
  reproduces it**, not just a fix. Search `pilot/tests/` for docstrings
  citing specific dates/numbers (e.g. "found on the 2026-08-02 K=15 run") —
  these are real failure modes, not synthetic edge cases.
- **Tokens (HF/GitHub): never hardcode, even temporarily.** The notebooks
  use `getpass()`-based entry specifically so the tracked file never
  contains a real secret. If a real token ever appears in a diff, chat, or
  notebook output, treat it as compromised — tell the user to revoke it
  immediately, don't just quietly fix the code.
- **`results/` is intentionally untracked in git** (not gitignored, just
  never `git add`ed) — CSVs come from Google Drive, downloaded manually by
  the user into `results/` when needed for local analysis. Large/expensive
  reproducible outputs (`figures/` at repo root) genuinely are gitignored;
  `report/figures/` is not (anchored `/figures/` pattern in `.gitignore` —
  don't widen that pattern back to bare `figures/`, it'll swallow both).
- **Verify surprising metric improvements against real data before
  reporting them as findings**, especially from NLI/embedding-based
  clustering — it has two confirmed failure modes on this project
  (lexical-overlap over-merging, transitive cascade merging). A plausibility
  check against other known numbers, plus spot-checking individual
  clustering decisions, catches both.

## Testing

```
source .venv/bin/activate
pytest pilot/tests/ -v
```

Local, no GPU needed. All `pilot/` modules are unit-tested with synthetic
and (for regression tests) real-data-derived fixtures — no network, no
model downloads required to run the suite.
