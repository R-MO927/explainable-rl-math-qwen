"""Aggregate metrics and uncertainty estimates."""
from __future__ import annotations

import math
import random
from statistics import mean


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(logs: list[dict]) -> dict:
    n = len(logs)
    if n == 0:
        return {"n": 0}
    k = sum(bool(l["correctness"]) for l in logs)
    answered = [l for l in logs if l["final_answer"] is not None]
    k_ans = sum(bool(l["correctness"]) for l in answered)
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "accuracy": k / n,
        "accuracy_ci95_wilson": [lo, hi],
        "n_correct": k,
        "coverage": len(answered) / n,
        "accuracy_on_answered": (k_ans / len(answered)) if answered else None,
        "no_answer_rate": 1 - len(answered) / n,
        "budget_exhausted_rate": mean(l["termination_reason"] == "accepted_budget_exhausted" for l in logs),
        "mean_attempts": mean(l["attempts"] for l in logs),
        "mean_verifier_calls": mean(l["verifier_calls"] for l in logs),
        "mean_pot_calls": mean(l["pot_calls"] for l in logs),
        "mean_model_calls": mean(l["model_calls"] for l in logs),
        "mean_input_tokens": mean(l["input_tokens"] for l in logs),
        "mean_output_tokens": mean(l["output_tokens"] for l in logs),
        "mean_latency_s": mean(l["latency_s"] for l in logs),
        "mean_return": mean(l["return"] for l in logs) if logs[0].get("return") is not None else None,
        "first_attempt_unextractable_rate": mean(l["generated_answers"][0] is None for l in logs),
    }


def paired_bootstrap_diff(a: list[bool], b: list[bool], iters: int = 10000, seed: int = 0) -> dict:
    """Accuracy(b) - Accuracy(a) on the same problems, with a percentile bootstrap CI."""
    assert len(a) == len(b)
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return {"diff": (sum(b) - sum(a)) / n, "ci95": [diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]]}


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on discordant pairs."""
    n01 = sum((not x) and y for x, y in zip(a, b))  # a wrong, b right
    n10 = sum(x and (not y) for x, y in zip(a, b))
    m = n01 + n10
    if m == 0:
        return {"b_only_correct": 0, "a_only_correct": 0, "p_value": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(m, i) for i in range(k + 1)) / 2**m
    return {"b_only_correct": n01, "a_only_correct": n10, "p_value": min(1.0, 2 * p)}
