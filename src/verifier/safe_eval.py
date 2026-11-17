"""Safe arithmetic evaluation with exact Fractions (no eval/exec).

Only a whitelisted subset of Python's AST is interpreted:
numbers, + - * / // % **, unary +/-, parentheses, and - for programs -
simple `name = expr` assignments and a few pure functions.
Anything else (attribute access except math.floor/ceil, imports other than
`math`, loops, comprehensions, strings, ...) is rejected.
"""
from __future__ import annotations

import ast
import math
from decimal import Decimal
from fractions import Fraction

MAX_EXPR_CHARS = 400
MAX_PROGRAM_CHARS = 3000
MAX_STATEMENTS = 40
MAX_ABS_VALUE = Fraction(10) ** 15
MAX_EXPONENT = 12


class UnsafeOrInvalid(Exception):
    pass


def _num(v) -> Fraction:
    if isinstance(v, bool):
        raise UnsafeOrInvalid("bool")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise UnsafeOrInvalid("non-finite")
        return Fraction(Decimal(repr(v)))
    raise UnsafeOrInvalid(f"constant {type(v).__name__}")


def _check(v: Fraction) -> Fraction:
    if abs(v) > MAX_ABS_VALUE:
        raise UnsafeOrInvalid("value too large")
    return v


def _round(x: Fraction, nd: Fraction | None = None) -> Fraction:
    if nd is None:
        return Fraction(round(x))
    if nd.denominator != 1 or not (0 <= nd <= 10):
        raise UnsafeOrInvalid("bad round digits")
    return Fraction(round(x, int(nd)))


# name -> (min_args, max_args, fn)
_FUNCS = {
    "abs": (1, 1, lambda a: abs(a[0])),
    "round": (1, 2, lambda a: _round(*a)),
    "min": (1, 10, lambda a: min(a)),
    "max": (1, 10, lambda a: max(a)),
    "int": (1, 1, lambda a: Fraction(int(a[0]))),
    "float": (1, 1, lambda a: a[0]),
    "floor": (1, 1, lambda a: Fraction(math.floor(a[0]))),
    "ceil": (1, 1, lambda a: Fraction(math.ceil(a[0]))),
}


def _call(fname: str, args: list[Fraction]) -> Fraction:
    lo, hi, fn = _FUNCS[fname]
    if not lo <= len(args) <= hi:
        raise UnsafeOrInvalid(f"{fname} arity")
    return fn(args)


def _eval(node: ast.AST, env: dict[str, Fraction]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return _num(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand, env)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return _check(a + b)
        if isinstance(op, ast.Sub):
            return _check(a - b)
        if isinstance(op, ast.Mult):
            return _check(a * b)
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if b == 0:
                raise UnsafeOrInvalid("division by zero")
            if isinstance(op, ast.Div):
                return _check(a / b)
            if isinstance(op, ast.FloorDiv):
                return Fraction(a // b)
            return _check(a % b)
        if isinstance(op, ast.Pow):
            if b.denominator != 1 or abs(b) > MAX_EXPONENT or (a == 0 and b < 0):
                raise UnsafeOrInvalid("unsupported exponent")
            if abs(a) > 10**6:
                raise UnsafeOrInvalid("base too large")
            return _check(a ** int(b))
        raise UnsafeOrInvalid(f"operator {type(op).__name__}")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise UnsafeOrInvalid(f"unknown name {node.id}")
    if isinstance(node, ast.Call) and not node.keywords:
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "math" and node.func.attr in ("floor", "ceil")):
            fname = node.func.attr
        if fname in _FUNCS:
            return _check(_call(fname, [_eval(a, env) for a in node.args]))
        raise UnsafeOrInvalid(f"call {ast.dump(node.func)[:40]}")
    raise UnsafeOrInvalid(f"node {type(node).__name__}")


def safe_eval_expr(expr: str) -> Fraction | None:
    """Evaluate a pure arithmetic expression; None if invalid/unsafe."""
    if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_EXPR_CHARS:
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return _eval(tree, {})
    except (SyntaxError, UnsafeOrInvalid, ValueError, OverflowError, ZeroDivisionError, RecursionError):
        return None


def safe_run_program(code: str) -> tuple[Fraction | None, str]:
    """Run a tiny straight-line program of numeric assignments.

    Returns (value, status). The value is the variable `answer` if assigned,
    otherwise the value of the last assignment or bare expression statement
    (e.g. a final `total` line). status is "ok" or a short error tag.
    """
    if not isinstance(code, str) or not code.strip():
        return None, "empty"
    if len(code) > MAX_PROGRAM_CHARS:
        return None, "too_long"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return None, "syntax_error"
    if len(tree.body) > MAX_STATEMENTS:
        return None, "too_many_statements"
    env: dict[str, Fraction] = {}
    last: Fraction | None = None
    try:
        for st in tree.body:
            if isinstance(st, ast.Import) and all(a.name == "math" and a.asname is None for a in st.names):
                continue
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name) \
                    and st.value.func.id == "print":
                continue  # print(...) is ignored, never executed
            if isinstance(st, ast.Expr):
                # bare expression such as a final `total` line: evaluated with the same whitelist
                last = _eval(st.value, env)
                continue
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                v = _eval(st.value, env)
                env[st.targets[0].id] = v
                last = v
                continue
            if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id in env:
                v = _eval(ast.BinOp(left=ast.Name(id=st.target.id, ctx=ast.Load()), op=st.op, right=st.value), env)
                env[st.target.id] = v
                last = v
                continue
            return None, f"unsupported_statement:{type(st).__name__}"
    except (UnsafeOrInvalid, ValueError, OverflowError, ZeroDivisionError, RecursionError) as exc:
        return None, f"eval_error:{str(exc)[:40]}"
    if "answer" in env:
        return env["answer"], "ok"
    if last is not None:
        return last, "ok_last_assignment"
    return None, "no_assignment"
