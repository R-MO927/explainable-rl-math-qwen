"""Find explicit arithmetic statements ("12 * 3 = 36") in a solution and check them.

A passed check only shows that the *written* computations are internally
consistent. It does not show that the solution models the problem correctly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from .safe_eval import safe_eval_expr

_ARITH_TAIL = re.compile(r"[-(]*\s*\d[\d.\s+\-*/()]*$")
_ARITH_HEAD = re.compile(r"^\s*-?\s*\(?\s*\d[\d.\s+\-*/()]*")
_OPS = re.compile(r"[\d)]\s*[+\-*/]\s*[\d(]")


def _latex_to_plain(text: str) -> str:
    t = text
    for _ in range(3):  # nested \frac
        t = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\text\s*\{[^{}]*\}", " ", t)
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("·", "*")
    t = t.replace("\\div", "/").replace("÷", "/").replace("−", "-")
    t = re.sub(r"\\[()\[\]]", " ", t)        # \( \) \[ \]
    t = t.replace("\\$", " ").replace("$", " ").replace("\\%", "%").replace("\\,", "").replace("\\!", "")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,000 -> 1000
    t = re.sub(r"(?<=\d)\s*[xX]\s*(?=\(?\d)", " * ", t)  # 3 x 5 -> 3 * 5
    t = re.sub(r"(?<=\d)\s*%", "", t)  # 20% kept as 20 (ambiguous; see notes)
    return t


def _balance(expr: str) -> str:
    expr = expr.strip()
    while expr.count("(") > expr.count(")") and expr.startswith("("):
        expr = expr[1:].strip()
    while expr.count(")") > expr.count("(") and expr.endswith(")"):
        expr = expr[:-1].strip()
    return expr.rstrip(".").strip()


def _decimals(s: str) -> int:
    m = re.search(r"\.(\d+)\s*$", s)
    return len(m.group(1)) if m else 0


def values_match(computed: Fraction, claimed_text: str, claimed: Fraction) -> bool:
    """Exact match, or match after rounding to the claimed number's decimals
    (allows '10/3 = 3.33'). Integers must match exactly."""
    if computed == claimed:
        return True
    d = _decimals(claimed_text)
    if d == 0:
        return False
    return abs(computed - claimed) <= Fraction(1, 2 * 10**d) + Fraction(1, 10**12)


@dataclass
class ArithmeticReport:
    checked: int = 0
    errors: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.checked == 0:
            return "none_found"
        return "error_found" if self.errors else "all_consistent"


def check_arithmetic(solution: str, max_details: int = 20) -> ArithmeticReport:
    rep = ArithmeticReport()
    plain = _latex_to_plain(solution)
    for line in plain.splitlines():
        parts = line.split("=")
        for lhs_raw, rhs_raw in zip(parts, parts[1:]):
            m_l = _ARITH_TAIL.search(lhs_raw)
            m_r = _ARITH_HEAD.match(rhs_raw)
            if not m_l or not m_r:
                continue
            lhs, rhs = _balance(m_l.group(0)), _balance(m_r.group(0))
            if not _OPS.search(lhs):
                continue  # "x = 5" or "Total = 12" is not a computation
            lv = safe_eval_expr(lhs)
            rv = safe_eval_expr(rhs)
            if lv is None or rv is None:
                continue
            ok = values_match(lv, rhs, rv) if not _OPS.search(rhs) else lv == rv
            rep.checked += 1
            if not ok:
                rep.errors += 1
            if len(rep.details) < max_details:
                rep.details.append({"lhs": lhs, "rhs": rhs, "lhs_value": str(lv), "consistent": ok})
    return rep
