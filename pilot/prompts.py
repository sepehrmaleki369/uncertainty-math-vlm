"""Exact prompt text validated by the FERMAT dataset authors. Do not paraphrase or shorten."""

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
