from pilot.prompts import (
    GRADING_SYSTEM_PROMPT,
    GRADING_USER_PROMPT,
    TRANSCRIPTION_SYSTEM_PROMPT,
    TRANSCRIPTION_USER_PROMPT,
    build_grading_messages,
    build_messages,
    build_transcription_messages,
)


def test_all_constants_nonempty_strings():
    for prompt in (
        GRADING_SYSTEM_PROMPT,
        GRADING_USER_PROMPT,
        TRANSCRIPTION_SYSTEM_PROMPT,
        TRANSCRIPTION_USER_PROMPT,
    ):
        assert isinstance(prompt, str)
        assert len(prompt) > 0


def test_transcription_user_prompt_contains_markers():
    assert "**Question:**" in TRANSCRIPTION_USER_PROMPT
    assert "**Answer:**" in TRANSCRIPTION_USER_PROMPT


def test_grading_user_prompt_contains_markers():
    assert "**Reasoning:**" in GRADING_USER_PROMPT
    assert "**Error:**" in GRADING_USER_PROMPT


def test_build_messages_shape_and_order():
    messages = build_messages("sys text", "user text", image="dummy_image")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "sys text"}]
    assert messages[1]["content"] == [
        {"type": "image", "image": "dummy_image"},
        {"type": "text", "text": "user text"},
    ]


def test_build_transcription_messages_uses_transcription_prompts():
    messages = build_transcription_messages(None)
    assert messages[0]["content"][0]["text"] == TRANSCRIPTION_SYSTEM_PROMPT
    text_entries = [c for c in messages[1]["content"] if c["type"] == "text"]
    image_entries = [c for c in messages[1]["content"] if c["type"] == "image"]
    assert len(text_entries) == 1
    assert text_entries[0]["text"] == TRANSCRIPTION_USER_PROMPT
    assert len(image_entries) == 1


def test_build_grading_messages_uses_grading_prompts():
    messages = build_grading_messages(None)
    assert messages[0]["content"][0]["text"] == GRADING_SYSTEM_PROMPT
    text_entries = [c for c in messages[1]["content"] if c["type"] == "text"]
    assert text_entries[0]["text"] == GRADING_USER_PROMPT
