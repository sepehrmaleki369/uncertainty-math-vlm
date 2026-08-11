"""The coded 31-item failure audit: what kind of wrong the remaining errors are.

Answers the reviewer question "are these real model mistakes or parser
artifacts?" with counts rather than an assertion. The headline is that most
are NOT model mistakes.

PROVENANCE, and it bounds what this can claim: the labels were assigned by
reading the four contact sheets through the Drive OCR, not by a human looking
at the images. 22 of 31 are high confidence (decidable from the labels alone),
7 medium, 2 undecidable. A human pass would firm up the medium ones --
`notation_misread` especially, where eyes beat OCR.

The sample is PURPOSIVE, not random: it spans five strata and one of them
selects for scoring artifacts. So the 58% figure is not a population rate and
the tests below refuse to let it read as one.
"""

from pathlib import Path

import pandas as pd
import pytest

SHEET = (Path(__file__).resolve().parents[2] / "reference" / "audit"
         / "coding_sheet_31_qwen_20260811.csv")


@pytest.fixture(scope="module")
def sheet():
    if not SHEET.exists():
        pytest.skip(f"{SHEET} not present")
    return pd.read_csv(SHEET)


def test_every_item_is_coded_and_carries_its_provenance(sheet):
    assert len(sheet) == 31
    assert sheet["final_label"].notna().all()
    assert sheet["coder_confidence"].isin({"high", "medium", "low"}).all()
    assert sheet["coded_by"].str.contains("OCR").all(), (
        "provenance must stay on the sheet -- these are not human-verified labels")


def test_most_remaining_errors_are_scoring_not_the_model(sheet):
    """The finding. Even after four scoring rules, the largest category in the
    audited sample is the scoring machinery rather than a misread page."""
    counts = sheet["final_label"].value_counts()
    assert counts["extraction_issue"] == 18
    assert counts["notation_misread"] == 7
    assert counts["copied_wrong_line"] == 4
    assert counts.get("needs_visual", 0) == 2
    assert counts.get("hallucination", 0) == 0
    assert counts.idxmax() == "extraction_issue"


def test_hallucination_is_zero_under_human_coding_too(sheet):
    """Worth stating on its own: the model never invents content unrelated to
    the page. That is the failure a reviewer most fears from a VLM asked to
    transcribe, and it does not occur -- not in the pre-sorter, not here."""
    assert (sheet["final_label"] == "hallucination").sum() == 0


def test_the_58_percent_is_not_a_population_rate(sheet):
    """The sample deliberately oversamples likely artifacts -- a whole stratum
    selects for them -- so quoting 58% as "half of all model errors are
    scoring" would overstate it. Excluding that stratum still leaves ~52%,
    and the text-only pre-sorter over the full 104 says >=15%, so the
    defensible claim is a RANGE."""
    assert (sheet["stratum"] == "likely_scoring_artifact").sum() >= 4

    excl = sheet[sheet["stratum"] != "likely_scoring_artifact"]
    share_all = (sheet["final_label"] == "extraction_issue").mean()
    share_excl = (excl["final_label"] == "extraction_issue").mean()
    assert share_all == pytest.approx(0.581, abs=0.01)
    assert share_excl == pytest.approx(0.519, abs=0.01)
    # The two must differ, or the stratification did nothing and the caveat
    # would be pointless.
    assert share_all > share_excl


def test_the_presorter_deferred_rather_than_guessed_wrong(sheet):
    """42% agreement sounds poor until the direction is checked: 14 of the 18
    disagreements are `needs_visual` resolving to a real label, i.e. it
    declined to guess. Only 4 were confidently wrong. That is the designed
    behaviour and the reason proposed_label and final_label are kept apart."""
    agree = (sheet["final_label"] == sheet["proposed_label"]).mean()
    assert agree == pytest.approx(0.42, abs=0.03)

    deferred = ((sheet["proposed_label"] == "needs_visual")
                & (sheet["final_label"] != "needs_visual")).sum()
    actively_wrong = ((sheet["proposed_label"] != "needs_visual")
                      & (sheet["final_label"] != sheet["proposed_label"])).sum()
    assert deferred == 14
    assert actively_wrong == 4
    assert deferred > 3 * actively_wrong


def test_item_55_is_coded_as_copied_wrong_line_not_auto_correction(sheet):
    """Retracts a claim this project made twice. The page's derivation ends
    1 - tan x tan y while its Answer line carries FERMAT's injected 1 + ...,
    and the model transcribed the derivation. It copied a real line of the
    page, the wrong one -- not a prior overriding the image. The
    auto-correction mechanism therefore rests on item 273 alone."""
    row = sheet[sheet["item"] == 55].iloc[0]
    assert row["final_label"] == "copied_wrong_line"
    assert "derivation" in str(row["notes"]).lower()
