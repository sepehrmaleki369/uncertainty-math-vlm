"""Offline smoke test for Notebook 32's non-GPU scoring and fail-closed gate."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pilot.parameter_sensitivity as ps


rows = []
for temperature in ps.TEMPERATURES:
    for k in ps.K_VALUES:
        for item_id in range(40):
            wrong = item_id % 2 == 0
            rows.append({
                "item_id": item_id,
                "temperature": temperature,
                "k": k,
                "perception_entropy": float(wrong) + (item_id % 3) / 10,
                "transcription_correct": not wrong,
                "n_transcription_parse_failures": 0,
            })

grid = pd.DataFrame(rows)
summary = ps.summarize_grid(grid, n_items=40, n_boot=100)
assert len(summary) == 9
assert not ps.reviewer_gate(summary, n_items=40)["paper_eligible"]

try:
    ps.validate_complete_grid(grid.iloc[:-1], n_items=40)
except ValueError as exc:
    assert "incomplete" in str(exc)
else:
    raise AssertionError("an incomplete grid was accepted")

print("OK nb32 offline logic; a smoke run cannot pass the paper gate")
