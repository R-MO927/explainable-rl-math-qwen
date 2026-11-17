"""Exact numeric parsing / normalization of final answers.

All comparisons are done on `fractions.Fraction` values, never on floats,
so "0.30" == "0.3" == "3/10" exactly and 0.1 + 0.2 issues cannot occur.

Canonical string form (`canonical_str`):
    integer          -> "1080", "-3", "0"
    non-integer      -> reduced fraction "p/q" with q > 0, e.g. "7/2", "-1/3"

Two parsers are provided:
    parse_reference_number(s)  strict: for GSM8K reference answers after "####".
    parse_answer_number(s)     lenient: also strips a leading currency sign,
                               a trailing "%" or unit-less trailing period;
                               intended for model outputs (phase 2).
Both return None (never raise) on input that is not an unambiguous number.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction

FINAL_ANSWER_MARKER = "####"

# Integer with optional, *correctly grouped* thousands separators, optional decimals.
_GROUPED = r"\d{1,3}(?:,\d{3})+"
_PLAIN = r"\d+"
_DECIMAL_NUMBER = re.compile(rf"^(?P<sign>[+-]?)(?P<int>{_GROUPED}|{_PLAIN})?(?:\.(?P<frac>\d+))?$")
_FRACTION = re.compile(r"^(?P<sign>[+-]?)(?P<num>\d+)\s*/\s*(?P<den>\d+)$")

_UNICODE_MINUS = {"−": "-", "–": "-", "—": "-", "﹣": "-", "－": "-"}


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for k, v in _UNICODE_MINUS.items():
        s = s.replace(k, v)
    return s.strip()


def _parse_core(s: str) -> Fraction | None:
    m = _FRACTION.match(s)
    if m:
        den = int(m.group("den"))
        if den == 0:
            return None
        val = Fraction(int(m.group("num")), den)
        return -val if m.group("sign") == "-" else val
    m = _DECIMAL_NUMBER.match(s)
    if not m or (m.group("int") is None and m.group("frac") is None):
        return None
    int_part = (m.group("int") or "0").replace(",", "")
    frac_part = m.group("frac")
    text = int_part + ("." + frac_part if frac_part is not None else "")
    try:
        val = Fraction(Decimal(text))
    except (InvalidOperation, ValueError):
        return None
    return -val if m.group("sign") == "-" else val


def parse_reference_number(s: str) -> Fraction | None:
    """Strict parser for GSM8K reference final answers."""
    if not isinstance(s, str):
        return None
    return _parse_core(_clean(s))


def parse_answer_number(s: str) -> Fraction | None:
    """Lenient parser for a candidate answer string (e.g. model output).

    Accepts the strict forms plus: leading '$' (after an optional sign),
    a trailing '%' (the number itself is kept: "25%" -> 25), and a trailing
    sentence period. Anything with other letters/units returns None.
    """
    if not isinstance(s, str):
        return None
    t = _clean(s)
    if t.endswith(".") and not t.endswith(".."):
        t = t[:-1].strip()  # trailing sentence period: "42." -> "42"
    t = t.removesuffix("%").strip()
    t = re.sub(r"^([+-]?)\s*\$\s*", r"\1", t)
    return _parse_core(t)


def canonical_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def answers_equal(a: str, b: str) -> bool:
    """Exact numeric equality of two answer strings (lenient parsing)."""
    fa, fb = parse_answer_number(a), parse_answer_number(b)
    return fa is not None and fb is not None and fa == fb


def split_reference_solution(answer_field: str) -> tuple[str, str]:
    """Split a GSM8K `answer` field into (worked_solution, final_answer_text).

    Raises ValueError unless the marker occurs exactly once and the final
    answer text is non-empty.
    """
    count = answer_field.count(FINAL_ANSWER_MARKER)
    if count != 1:
        raise ValueError(f"expected exactly one '{FINAL_ANSWER_MARKER}', found {count}")
    body, final = answer_field.split(FINAL_ANSWER_MARKER)
    final = final.strip()
    if not final:
        raise ValueError("empty final answer after marker")
    return body.rstrip(), final
