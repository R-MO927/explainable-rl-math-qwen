from fractions import Fraction

import pytest

from src.data.answers import (
    answers_equal,
    canonical_str,
    parse_answer_number,
    parse_reference_number,
    split_reference_solution,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("18", "18"),
        (" 1,080", "1080"),            # GSM8K style leading space + grouping
        ("109,200,000", "109200000"),
        ("-10", "-10"),
        ("-3", "-3"),
        ("0", "0"),
        ("-0", "0"),
        ("007", "7"),
        ("0.5", "1/2"),
        ("1.50", "3/2"),
        ("-2.25", "-9/4"),
        (".5", "1/2"),
        ("3/4", "3/4"),
        ("6/4", "3/2"),
        ("-1/3", "-1/3"),
        ("−5", "-5"),             # unicode minus sign
        ("1,234.5", "2469/2"),
    ],
)
def test_reference_number_canonical(text, expected):
    v = parse_reference_number(text)
    assert v is not None
    assert canonical_str(v) == expected


@pytest.mark.parametrize(
    "text",
    ["", "abc", "1,23", "12,3456", "1.2.3", "5/0", "1e3", "$5", "5%", "five", "--5", "1,000,00"],
)
def test_reference_number_rejects_ambiguous_or_non_numeric(text):
    # strict parser: malformed grouping, units, scientific notation are rejected (None, no exception)
    assert parse_reference_number(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [("$5", Fraction(5)), ("-$1,200", Fraction(-1200)), ("25%", Fraction(25)), ("42.", Fraction(42)),
     ("3.5.", Fraction(7, 2)), ("$ 0.75", Fraction(3, 4))],
)
def test_lenient_answer_parser(text, expected):
    assert parse_answer_number(text) == expected


def test_lenient_parser_still_rejects_units_and_words():
    assert parse_answer_number("5 apples") is None
    assert parse_answer_number("about 5") is None


def test_exact_equality_without_float_error():
    assert answers_equal("0.30", "0.3")
    assert answers_equal("3/10", "0.3")
    assert answers_equal("1,000", "1000")
    assert not answers_equal("0.1", "0.10000000000000001")  # floats would call these equal
    assert not answers_equal("abc", "abc")                   # unparseable never counts as correct


def test_split_reference_solution():
    body, final = split_reference_solution("He has 2+3=<<2+3=5>>5 apples.\n#### 5")
    assert final == "5"
    assert body.endswith("5 apples.")
    with pytest.raises(ValueError):
        split_reference_solution("no marker here")
    with pytest.raises(ValueError):
        split_reference_solution("a #### 1 #### 2")
    with pytest.raises(ValueError):
        split_reference_solution("work\n####   ")
