"""Extract a final numeric answer from model output (no reference needed)."""
from __future__ import annotations

import re

from src.data.answers import canonical_str, parse_answer_number

_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def _last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    while idx != -1:
        j = text.find("{", idx)
        if j == -1:
            return None
        depth, k = 0, j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    return text[j + 1 : k]
            k += 1
        idx = text.rfind("\\boxed", 0, idx)  # unbalanced: try an earlier one
    return None


def _clean_box(s: str) -> str:
    s = re.sub(r"\\(?:text|mathrm|textbf)\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    s = s.replace("\\$", "$").replace("\\%", "%").replace("\\,", "").replace("\\!", "").replace("{,}", ",")
    s = s.replace("−", "-").strip()
    if "=" in s:
        s = s.split("=")[-1]
    return s.strip()


def _parse_box_content(s: str) -> str | None:
    s = _clean_box(s)
    v = parse_answer_number(s)
    if v is not None:
        return canonical_str(v)
    # e.g. "18 dollars", "$18 per week": accept iff exactly one number is present
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?", s)
    if len(nums) == 1:
        v = parse_answer_number(nums[0])
        if v is not None:
            return canonical_str(v)
    return None


def extract_final_answer(text: str) -> tuple[str | None, str]:
    """Return (canonical_answer | None, method).

    Order: last \\boxed{...}; then an explicit 'final answer is X' /
    'answer: X' / '#### X' phrase. No 'last number in text' fallback: an
    answer the model did not mark is reported as not extractable.
    """
    box = _last_boxed(text)
    if box is not None:
        ans = _parse_box_content(box)
        return (ans, "boxed") if ans is not None else (None, "boxed_unparseable")
    pats = [
        rf"####\s*({_NUM})",
        rf"final answer\s*(?:is|:)?\s*[:=]?\s*\**\s*({_NUM})",
        rf"(?:the\s+)?answer\s*(?:is|:)\s*\**\s*({_NUM})",
    ]
    for p in pats:
        ms = list(re.finditer(p, text, flags=re.IGNORECASE))
        if ms:
            v = parse_answer_number(ms[-1].group(1).replace(" ", ""))
            if v is not None:
                return canonical_str(v), "phrase"
    return None, "none"
