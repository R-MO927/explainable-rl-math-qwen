"""Reward design. Kept separate from the environment and the verifier:
only this module (training/evaluation code) ever sees the reference answer.

Episode return  G = terminal - sum(step costs)
  terminal = +1   final answer numerically equal to the reference
             -1   wrong final answer
             -1   no answer (ACCEPT with no extractable candidate) -> abstaining
                  is never better than answering, so it cannot be exploited
  step cost = 0.05 per REGENERATE (one extra solver generation, <= 512 tokens)
              0.03 per VERIFY_MORE (one extra, shorter PoT generation, <= 320 tokens)
Units: "reward points"; 1 correct answer = +1. The first attempt is free
(it is required by every method), so A/B/C differ only in extra costs.
The cost values are a priori design choices (roughly proportional to the
generation length budget), not tuned on validation or test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from src.data.answers import parse_answer_number, parse_reference_number

from .coordination_env import ACCEPT, REGENERATE, VERIFY_MORE


@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    incorrect: float = -1.0
    no_answer: float = -1.0
    regenerate_cost: float = 0.05
    verify_cost: float = 0.03

    def to_dict(self) -> dict:
        return asdict(self)


def step_cost(action: int, cfg: RewardConfig) -> float:
    if action == REGENERATE:
        return -cfg.regenerate_cost
    if action == VERIFY_MORE:
        return -cfg.verify_cost
    if action == ACCEPT:
        return 0.0
    raise ValueError(action)


def is_correct(final_answer: str | None, reference_normalized: str) -> bool:
    if final_answer is None:
        return False
    a, r = parse_answer_number(final_answer), parse_reference_number(reference_normalized)
    return a is not None and r is not None and a == r


def terminal_reward(final_answer: str | None, reference_normalized: str, cfg: RewardConfig) -> float:
    if final_answer is None:
        return cfg.no_answer
    return cfg.correct if is_correct(final_answer, reference_normalized) else cfg.incorrect
