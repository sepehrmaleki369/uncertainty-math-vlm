# uncertainty-math-vlm

Pilot: checking whether perception entropy (instability transcribing handwritten
math) and reasoning entropy (instability grading an answer for errors) carry real
signal in a vision-language model (Qwen2.5-VL) on the FERMAT dataset. A fast,
scoped-down first pass — not the full study.

## Layout

- `pilot/` — the pilot's code. `data.py`, `prompts.py`, `parsing.py`, `entropy.py`
  need no GPU and are unit-tested locally. `plotting.py` is a Step 4 scaffold.
  `pilot.ipynb` is the Colab notebook that adds the GPU-dependent parts (model
  loading, sampling) and imports everything else from this package.
- `results/` — CSVs pushed back from Colab runs (created by the notebook's save
  cell; not present until the first run). Code and results live in this one repo.

## Local setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running tests

```
pytest pilot/tests/ -v
```

## One-time setup before running `pilot/pilot.ipynb`

1. Hugging Face account with access to `ai4bharat/FERMAT` accepted, and an HF
   access token ready.
2. This repo is already pushed to GitHub (`REPO_URL` in the notebook's auth cell
   points at it) — the notebook clones it and `pip install -e .`s it each session,
   so local edits are picked up automatically, and results CSVs get committed and
   pushed back into `results/` at the end of each run.
3. **Auth note:** `google.colab.userdata.get()` (the usual Colab-secrets approach)
   only works when running in the native Colab browser UI. If you're connecting
   from VS Code's Jupyter interface instead, the notebook's auth cell has you
   paste `HF_TOKEN` (and `GH_TOKEN`, if the repo is private) directly as string
   literals. **Fill these in only in your live session — never commit this file
   back to GitHub with a real token in it**, since that puts the secret in git
   history permanently. The tracked copy in this repo keeps a placeholder.

## Status

Step 1 (local modules + tests) and Step 2 (`pilot.ipynb`) are done. Step 3 (running
the notebook in Colab) and Step 4 (pulling results back, plotting, writing up) are
next.
