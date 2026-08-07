"""Prompt text for the transcription and grading tasks.

The four constants below are the exact wording validated by the FERMAT dataset
authors. Do not paraphrase or shorten them -- every result so far is measured
against these, and changing them silently would break comparability with the
2026-08-02 reference run.

The GRADING_*_VARIANT prompts at the bottom are OURS, not FERMAT's, and are
clearly separated for that reason. They exist to test one specific finding: on
a balanced set the model answers "there is a mistake" for 93% of items and
grades at 51.7% against a 0.500 baseline. That could be a genuine capability
ceiling, or it could be an artifact of how the question is posed. These
variants distinguish the two.
"""

from typing import Any

GRADING_SYSTEM_PROMPT = (
    "You are a math assistant responsible for evaluating handwritten math problems "
    "at the middle or high school level. Your task is to analyze the given Image, "
    "which contains the handwritten math Question-Answer pair, and determine whether "
    "the solution contains any errors. Follow the specific instructions provided "
    "carefully and output your response strictly in the requested format. Ensure "
    "that your evaluation is precise, accurate, and based solely on the information "
    "visible in the Image."
)

GRADING_USER_PROMPT = (
    "The image provided contains a handwritten math problem consisting of both a "
    "Question and an Answer at a middle or high school level. Your task is to "
    "analyze the Answer to determine whether there is any error. Begin by providing "
    "a brief reasoning for your analysis, explaining where and why you believe an "
    "error is present or absent in the Answer. If the problem is multiple-choice "
    "(MCQ), judge the presence or absence of error based only on the explanation "
    "given in the Answer, not the option selected by the student.\n"
    "\n"
    "After the reasoning, provide a binary output indicating whether an error "
    "exists (1 for error, 0 for no error).\n"
    "\n"
    "Please follow the exact format below without adding any extra information:\n"
    "\n"
    "**Reasoning:** <Brief Explanation of Error Presence or Absence>\n"
    "\n"
    "**Error:** <0 or 1>"
)

TRANSCRIPTION_SYSTEM_PROMPT = (
    "You are a math assistant specializing in extracting mathematical question and "
    "answer content from handwritten images of math problems by middle or high "
    "school students. Your task is to analyze the given Image, which contains the "
    "handwritten math Question-Answer pair, and convert it into LaTeX format. Follow "
    "the specific instructions provided carefully and output your response strictly "
    "in the requested format. Ensure that your evaluation is precise, accurate, and "
    "based solely on the information visible in the image."
)

TRANSCRIPTION_USER_PROMPT = (
    "The image provided contains a handwritten math problem with both a Question "
    "and an Answer at a middle or high school level. Your task is to explicitly "
    "perform OCR on the handwritten text and extract the content in LaTeX format. "
    "Return only the extracted content, formatted in LaTeX, exactly as it appears "
    "in the image, in the **Question:** and **Answer:** fields.\n"
    "\n"
    "Ensure that no extra information is added that is not in the image.\n"
    "\n"
    "Please return the LaTeX output as follows:\n"
    "**Question:**<Extracted Question text in LaTeX>\n"
    "**Answer:**<Extracted Answer text in LaTeX>"
)


def build_messages(system_prompt: str, user_prompt: str, image: Any) -> list[dict]:
    """Build Qwen2.5-VL chat-template-shaped messages for a system+user prompt pair."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def build_transcription_messages(image: Any) -> list[dict]:
    return build_messages(TRANSCRIPTION_SYSTEM_PROMPT, TRANSCRIPTION_USER_PROMPT, image)


def build_grading_messages(image: Any) -> list[dict]:
    return build_messages(GRADING_SYSTEM_PROMPT, GRADING_USER_PROMPT, image)


# --- Grading prompt variants (ours, not FERMAT's) -------------------------
#
# Each keeps the **Reasoning:** / **Error:** output format so existing parsing
# applies unchanged, which is what makes the comparison to the baseline clean:
# only the question changes, not the scoring path.

GRADING_USER_PROMPT_RESTATE = (
    "The image provided contains a handwritten math problem consisting of both a "
    "Question and an Answer at a middle or high school level. Your task is to "
    "analyze the Answer to determine whether there is any error.\n"
    "\n"
    "First, transcribe the final result the student arrived at, exactly as written. "
    "Then check that result by working the problem yourself. Only then decide "
    "whether the Answer contains an error. If the problem is multiple-choice "
    "(MCQ), judge the presence or absence of error based only on the explanation "
    "given in the Answer, not the option selected by the student.\n"
    "\n"
    "After the reasoning, provide a binary output indicating whether an error "
    "exists (1 for error, 0 for no error).\n"
    "\n"
    "Please follow the exact format below without adding any extra information:\n"
    "\n"
    "**Reasoning:** <Transcribe the student's final result, then verify it>\n"
    "\n"
    "**Error:** <0 or 1>"
)

GRADING_USER_PROMPT_BALANCED = (
    "The image provided contains a handwritten math problem consisting of both a "
    "Question and an Answer at a middle or high school level. Your task is to "
    "analyze the Answer to determine whether there is any error.\n"
    "\n"
    "Important: in this set, roughly half of the answers are completely correct "
    "and roughly half contain an error. Do not assume an error is present. A "
    "correct answer is just as likely as an incorrect one, so report an error "
    "only when you can point to a specific step that is actually wrong.\n"
    "\n"
    "Begin by providing a brief reasoning for your analysis. If the problem is "
    "multiple-choice (MCQ), judge the presence or absence of error based only on "
    "the explanation given in the Answer, not the option selected by the student.\n"
    "\n"
    "After the reasoning, provide a binary output indicating whether an error "
    "exists (1 for error, 0 for no error).\n"
    "\n"
    "Please follow the exact format below without adding any extra information:\n"
    "\n"
    "**Reasoning:** <Brief Explanation of Error Presence or Absence>\n"
    "\n"
    "**Error:** <0 or 1>"
)

GRADING_USER_PROMPT_COMMIT = (
    "The image provided contains a handwritten math problem consisting of both a "
    "Question and an Answer at a middle or high school level. Your task is to "
    "analyze the Answer to determine whether there is any error.\n"
    "\n"
    "Some images may be partly hard to read. If so, do not default to reporting "
    "no error just because part of the image is unclear -- base your judgement on "
    "whatever content you can make out, and commit to your best assessment rather "
    "than treating illegibility itself as evidence the Answer is correct.\n"
    "\n"
    "Begin by providing a brief reasoning for your analysis, explaining where and "
    "why you believe an error is present or absent in the Answer. If the problem "
    "is multiple-choice (MCQ), judge the presence or absence of error based only "
    "on the explanation given in the Answer, not the option selected by the "
    "student.\n"
    "\n"
    "After the reasoning, provide a binary output indicating whether an error "
    "exists (1 for error, 0 for no error).\n"
    "\n"
    "Please follow the exact format below without adding any extra information:\n"
    "\n"
    "**Reasoning:** <Brief Explanation of Error Presence or Absence>\n"
    "\n"
    "**Error:** <0 or 1>"
)

GRADING_USER_PROMPT_CONFIDENCE = (
    "The image provided contains a handwritten math problem consisting of both a "
    "Question and an Answer at a middle or high school level. Your task is to "
    "analyze the Answer to determine whether there is any error. Begin by providing "
    "a brief reasoning for your analysis, explaining where and why you believe an "
    "error is present or absent in the Answer. If the problem is multiple-choice "
    "(MCQ), judge the presence or absence of error based only on the explanation "
    "given in the Answer, not the option selected by the student.\n"
    "\n"
    "After the reasoning, provide a binary output indicating whether an error "
    "exists (1 for error, 0 for no error), followed by how confident you are in "
    "that judgement as a number from 0 to 100.\n"
    "\n"
    "Please follow the exact format below without adding any extra information:\n"
    "\n"
    "**Reasoning:** <Brief Explanation of Error Presence or Absence>\n"
    "\n"
    "**Error:** <0 or 1>\n"
    "\n"
    "**Confidence:** <0-100>"
)

# Keyed so a notebook can loop over them and record which one produced a row.
GRADING_VARIANTS = {
    "baseline": GRADING_USER_PROMPT,
    "restate": GRADING_USER_PROMPT_RESTATE,
    "balanced": GRADING_USER_PROMPT_BALANCED,
    "confidence": GRADING_USER_PROMPT_CONFIDENCE,
    "commit": GRADING_USER_PROMPT_COMMIT,
}


def build_grading_messages_variant(image: Any, variant: str) -> list[dict]:
    """Grading messages for a named prompt variant.

    Raises on an unknown name rather than falling back to the baseline: a typo
    that silently scored the baseline four times would look like a null result.
    """
    if variant not in GRADING_VARIANTS:
        raise KeyError(
            f"unknown grading variant {variant!r}; expected one of "
            f"{sorted(GRADING_VARIANTS)}"
        )
    return build_messages(GRADING_SYSTEM_PROMPT, GRADING_VARIANTS[variant], image)
