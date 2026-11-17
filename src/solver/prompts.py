"""Prompt templates. Versioned: any change must bump the version (it is part of the cache key).

The prompts were written before looking at any validation/test outputs and are
not tuned on test data.
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."  # model-card default

SOLVER_PROMPT_VERSION = "solver-v1"
SOLVER_USER_TEMPLATE = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

POT_PROMPT_VERSION = "pot-v1"
POT_USER_TEMPLATE = (
    "Solve the following math word problem by writing a short Python program.\n"
    "Rules:\n"
    "- Use only numbers, variables and the operators + - * / // % ** and round().\n"
    "- Do not use loops, functions, input() or imports.\n"
    "- Store the final numeric result in a variable named answer.\n"
    "Output only the code inside a ```python code block.\n\n"
    "Problem: {question}"
)


def solver_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SOLVER_USER_TEMPLATE.format(question=question.strip())}]


def pot_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": POT_USER_TEMPLATE.format(question=question.strip())}]


def extract_code_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, flags=re.DOTALL | re.IGNORECASE)  # unterminated block
    return m.group(1).strip() if m else None
