"""DEMO solver for previewing the web UI without loading Qwen.

It returns FIXED canned text that ignores the question. The UI shows a clear
"demo mode" banner whenever this solver is used. Never use it for results.
"""
from __future__ import annotations


class _Gen:
    def __init__(self, text: str):
        self.text, self.finished = text, True
        self.input_tokens, self.output_tokens, self.latency_s = 60, 120, 0.05


_SOLUTION_WRONG = (
    "Sara has 5 boxes and each box has 12 pencils.\n\n"
    r"Total pencils: \(5 \times 12 = 60\)." "\n\n"
    r"She gives away 17 pencils: \(60 - 17 = 42\)." "\n\n"
    r"So she has \(\boxed{42}\) pencils."
)
_SOLUTION_RIGHT = (
    "First, find how many pencils are in the boxes.\n\n"
    r"\[ 5 \times 12 = 60 \]" "\n\n"
    r"Then subtract the pencils she gives away: \(60 - 17 = 43\)." "\n\n"
    r"The final answer is \(\boxed{43}\)."
)
_POT = "```python\nboxes = 5\nper_box = 12\ngiven = 17\nanswer = boxes * per_box - given\n```"


class DemoSolver:
    def generate(self, message_lists, *, params, max_new_tokens, num_return_sequences=1, seed=None):
        user = message_lists[0][-1]["content"]
        if "Python program" in user:
            text = _POT
        elif not params.get("do_sample"):
            text = _SOLUTION_WRONG          # first (greedy) attempt has an arithmetic slip on purpose
        else:
            text = _SOLUTION_RIGHT
        return [[_Gen(text) for _ in range(num_return_sequences)]]

    def info(self) -> dict:
        return {"model_id": "DEMO (canned text, no model)", "revision": None, "device": "cpu", "dtype": None}
