from fractions import Fraction

import pytest

from src.solver.extract import extract_final_answer
from src.verifier.arithmetic import check_arithmetic
from src.verifier.safe_eval import safe_eval_expr, safe_run_program


# ---------------------------------------------------------------- safe eval
def test_safe_eval_exact_arithmetic():
    assert safe_eval_expr("3*(4+2)/4") == Fraction(9, 2)
    assert safe_eval_expr("0.1+0.2") == Fraction(3, 10)  # exact, not 0.30000000000000004
    assert safe_eval_expr("-2**3") == -8
    assert safe_eval_expr("7//2") == 3 and safe_eval_expr("7%4") == 3


@pytest.mark.parametrize(
    "expr",
    ['__import__("os")', "open('x')", "(1).__class__", "[1,2]", "'a'*3", "2**100", "10**10**10",
     "1/0", "x+1", "lambda: 1", "True+1", "", "1e400", "9" * 500],
)
def test_safe_eval_rejects_unsafe_or_invalid(expr):
    assert safe_eval_expr(expr) is None


def test_safe_program_runs_assignments_only():
    code = "import math\nprice = 12\nqty = 5\nanswer = price * qty - math.ceil(7/2)\nprint(answer)\n"
    assert safe_run_program(code) == (Fraction(56), "ok")
    assert safe_run_program("a = 2\nb = a * 3") == (Fraction(6), "ok_last_assignment")
    total = safe_run_program("t = 10\nt += 5\nanswer = round(t / 4, 2)")
    assert total == (Fraction(375, 100), "ok")
    # final bare expression (common model idiom) is evaluated with the same whitelist
    assert safe_run_program("a = 140\nb = a + 40\ntotal = a + b\ntotal") == (Fraction(320), "ok_last_assignment")
    assert safe_run_program("x = 2\n__import__('os')")[0] is None
    assert safe_run_program("from sympy import symbols\nanswer = 1")[1].startswith("unsupported_statement")


@pytest.mark.parametrize(
    "code, status_prefix",
    [("import os", "unsupported_statement"), ("for i in range(3):\n    x = i", "unsupported_statement"),
     ("x = [1]", "eval_error"), ("x = (1).__class__", "eval_error"), ("def f():\n    return 1", "unsupported_statement"),
     ("x = open('f')", "eval_error"), ("x = ", "syntax_error"), ("", "empty"), ("print(1)", "no_assignment"),
     ("y = 1\nopen('f')", "eval_error"), ("y = 1\n[1, 2]", "eval_error")],
)
def test_safe_program_rejects(code, status_prefix):
    value, status = safe_run_program(code)
    assert value is None and status.startswith(status_prefix)


# ---------------------------------------------------------------- arithmetic check
def test_arithmetic_detects_consistent_and_wrong_steps():
    text = (
        "Milk costs $3 \\times 5 = $15.\n"
        "Then 15 + 6 = 22 in total.\n"          # wrong
        "Each share is \\( \\frac{10}{3} = 3.33 \\)\n"  # rounded, accepted
        "Total = 40\n"                           # no computation -> not checked
        "We have 1,200 - 200 = 1,000 left.\n"
        "So 2 x 4 = 8 apples."
    )
    rep = check_arithmetic(text)
    lhs = {d["lhs"]: d["consistent"] for d in rep.details}
    assert lhs == {"3 * 5": True, "15 + 6": False, "((10)/(3))": True, "1200 - 200": True, "2 * 4": True}
    assert rep.checked == 5 and rep.errors == 1 and rep.status == "error_found"


def test_arithmetic_none_found_is_not_proof():
    rep = check_arithmetic("She has some apples, so the answer is 7.")
    assert rep.checked == 0 and rep.status == "none_found"


def test_arithmetic_rounding_rule_is_strict_for_integers():
    assert check_arithmetic("10 / 3 = 3").errors == 1
    assert check_arithmetic("10 / 4 = 2.5").errors == 0
    assert check_arithmetic("10 / 3 = 3.4").errors == 1


# ---------------------------------------------------------------- extraction
@pytest.mark.parametrize(
    "text, expected",
    [
        (r"so \boxed{1,080}", ("1080", "boxed")),
        (r"\boxed{\$18.50}", ("37/2", "boxed")),
        (r"\boxed{18 \text{ apples}}", ("18", "boxed")),
        (r"\boxed{x = -3}", ("-3", "boxed")),
        (r"\boxed{\frac{7}{2}}", ("7/2", "boxed")),
        (r"first \boxed{5} then corrected \boxed{6}", ("6", "boxed")),
        (r"\boxed{3 and 4}", (None, "boxed_unparseable")),
        ("The final answer is 42.", ("42", "phrase")),
        ("Answer: $1,250", ("1250", "phrase")),
        ("She has 3 apples and 4 pears.", (None, "none")),  # no last-number fallback
    ],
)
def test_extract_final_answer(text, expected):
    assert extract_final_answer(text) == expected
