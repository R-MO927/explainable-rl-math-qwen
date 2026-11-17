"""Unified Python entry point (ready to be wrapped by a web API later).

    from src.pipeline import solve_problem
    result = solve_problem("Tom has 3 apples ...", method="rl", budget=4, seed=0)

No reference answer is needed or accepted. Generation is LIVE (not cached).
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from src.config import ENV_CONFIG
from src.coordinator.fixed_policy import FixedPolicy, MaxBudgetPolicy, SingleAttemptPolicy
from src.coordinator.q_learning import QLearningCoordinator
from src.environment.sources import LiveSource
from src.evaluation.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "artifacts" / "policies" / "coordinator.json"
METHODS = ("single", "fixed", "rl", "max_budget")

_SOLVER = None
_POLICIES: dict[str, object] = {}


def get_solver():
    """Loads Qwen once per process and reuses it."""
    global _SOLVER
    if _SOLVER is None:
        from src.solver.qwen_solver import QwenSolver
        _SOLVER = QwenSolver()
    return _SOLVER


def _policy(method: str, policy_path: Path):
    if method == "single":
        return SingleAttemptPolicy()
    if method == "fixed":
        return FixedPolicy()
    if method == "max_budget":
        return MaxBudgetPolicy()
    if method == "rl":
        key = str(policy_path)
        if key not in _POLICIES:
            _POLICIES[key] = QLearningCoordinator.load(policy_path)
        return _POLICIES[key]
    raise ValueError(f"method must be one of {METHODS}")


def solve_problem(question: str, method: str = "rl", budget: int = 4, seed: int = 0,
                  policy_path: str | Path = DEFAULT_POLICY, solver=None) -> dict:
    """Solve one new question.

    budget = maximum number of solver attempts (1..4; the RL policy was trained with 4,
    smaller budgets are enforced by the action mask). Returns a JSON-serializable dict.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not 1 <= budget <= ENV_CONFIG.max_attempts:
        raise ValueError(f"budget must be in 1..{ENV_CONFIG.max_attempts}")
    solver = solver or get_solver()
    policy = _policy(method, Path(policy_path))
    env_cfg = replace(ENV_CONFIG, max_attempts=budget)
    source = LiveSource(solver, question, seed=seed, max_attempts=budget)
    t0 = time.perf_counter()
    log = run_episode(policy, source, env_cfg=env_cfg, reward_cfg=None, reference=None,
                      method=method, seed=seed, problem_id=None, with_explanations=True)
    wall = time.perf_counter() - t0
    return {
        "question": question,
        "method": method,
        "budget": budget,
        "seed": seed,
        "final_answer": log["final_answer"],
        "termination_reason": log["termination_reason"],
        "attempt_answers": log["generated_answers"],
        "attempt_texts": [source.texts[f"attempt_{i}"] for i in range(log["attempts"])],
        "pot_program_output": source.texts.get("pot"),
        "verifier_outputs": log["verifier_outputs"],
        "actions": log["actions"],
        "explanations": log["decisions"],
        "usage": {k: log[k] for k in ("attempts", "model_calls", "verifier_calls", "input_tokens", "output_tokens")},
        "latency_s": {"generation": log["latency_s"], "wall_clock": round(wall, 3), "kind": "live_generation"},
        "model": solver.info(),
        "note": "No reference answer is used. Explanations describe the coordinator's decision, not Qwen's reasoning.",
    }
