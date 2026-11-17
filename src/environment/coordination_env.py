"""Coordination environment: one episode = one problem.

The environment never receives the reference answer. Rewards are computed
outside it (src/environment/reward.py, used by src/evaluation/runner.py).

Flow:
  reset(source)  -> the first solver attempt (attempt 0) is always generated
                    and basic-checked; an observation is returned.
  step(action)   -> ACCEPT ends the episode with the current candidate answer;
                    REGENERATE adds one solver attempt (masked at max_attempts);
                    VERIFY_MORE runs the program-of-thought check (once per
                    episode, masked afterwards or when unavailable).

Candidate answer = plurality vote over extractable attempts so far; ties are
broken by PoT agreement, then arithmetic status, then earliest attempt.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from src.verifier.verifier import BasicCheck, PotCheck, basic_check, compare_pot, pot_check

ACCEPT, REGENERATE, VERIFY_MORE = 0, 1, 2
ACTION_NAMES = ("ACCEPT", "REGENERATE", "VERIFY_MORE")
N_ACTIONS = 3
_ARITH_RANK = {"all_consistent": 0, "none_found": 1, "error_found": 2}


@dataclass
class GenRecord:
    text: str
    finished: bool
    input_tokens: int
    output_tokens: int
    latency_s: float
    from_cache: bool


class AttemptSource(Protocol):
    """Provides solver attempts and the PoT program for ONE question. No reference answer."""

    max_available_attempts: int

    def attempt(self, i: int) -> GenRecord: ...

    def pot(self) -> GenRecord | None: ...


@dataclass
class EnvConfig:
    max_attempts: int = 4
    allow_verify_more: bool = True

    @property
    def max_steps(self) -> int:  # (max_attempts-1) regenerations + 1 verify + 1 accept
        return self.max_attempts + 1


@dataclass
class Usage:
    model_calls: int = 0
    verifier_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached_generations: int = 0
    live_generations: int = 0

    def add(self, g: GenRecord) -> None:
        self.model_calls += 1
        self.input_tokens += g.input_tokens
        self.output_tokens += g.output_tokens
        self.latency_s += g.latency_s
        if g.from_cache:
            self.cached_generations += 1
        else:
            self.live_generations += 1


@dataclass
class EnvState:
    attempts: list[GenRecord] = field(default_factory=list)
    checks: list[BasicCheck] = field(default_factory=list)
    pot: PotCheck | None = None
    pot_used: bool = False
    steps: int = 0
    done: bool = False
    usage: Usage = field(default_factory=Usage)


class CoordinationEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.source: AttemptSource | None = None
        self.s = EnvState()

    # ----------------------------------------------------------- core API
    def reset(self, source: AttemptSource) -> dict:
        self.source = source
        self.s = EnvState()
        self._new_attempt()
        return self.observation()

    def valid_actions(self) -> list[bool]:
        s = self.s
        if s.done:
            return [False, False, False]
        can_regen = len(s.attempts) < min(self.cfg.max_attempts, self.source.max_available_attempts)
        can_verify = self.cfg.allow_verify_more and not s.pot_used and self._pot_available()
        return [True, can_regen, can_verify]

    def step(self, action: int) -> tuple[dict, bool, dict]:
        s = self.s
        if s.done:
            raise RuntimeError("episode already finished")
        mask = self.valid_actions()
        if not (0 <= action < N_ACTIONS) or not mask[action]:
            raise ValueError(f"invalid action {action} with mask {mask}")
        s.steps += 1
        info: dict = {"action": ACTION_NAMES[action]}
        if action == REGENERATE:
            self._new_attempt()
        elif action == VERIFY_MORE:
            g = self.source.pot()
            s.usage.add(g)
            s.usage.verifier_calls += 1
            s.pot = pot_check(g.text)
            s.pot_used = True
            info["pot"] = s.pot.to_dict()
        else:  # ACCEPT
            s.done = True
        if not s.done and s.steps >= self.cfg.max_steps:  # safety net, unreachable with valid masks
            s.done = True
            info["forced_termination"] = "step_limit"
        return self.observation(), s.done, info

    # ----------------------------------------------------------- helpers
    def _pot_available(self) -> bool:
        return getattr(self.source, "pot_available", True)

    def _new_attempt(self) -> None:
        i = len(self.s.attempts)
        g = self.source.attempt(i)
        self.s.attempts.append(g)
        self.s.usage.add(g)
        self.s.checks.append(basic_check(g.text, g.finished))
        self.s.usage.verifier_calls += 1

    def candidate(self) -> tuple[str | None, int]:
        """(answer, support). Plurality vote with documented tie-breaks."""
        answers = [c.answer for c in self.s.checks if c.answer is not None]
        if not answers:
            return None, 0
        counts = Counter(answers)
        top = max(counts.values())
        tied = [a for a in counts if counts[a] == top]

        def rank(a: str):
            pot_agree = 0 if compare_pot(self.s.pot, a) == "agree" else 1
            arith = min(_ARITH_RANK[c.arith_status] for c in self.s.checks if c.answer == a)
            first = min(i for i, c in enumerate(self.s.checks) if c.answer == a)
            return (pot_agree, arith, first)

        best = min(tied, key=rank)
        return best, top

    def observation(self) -> dict:
        """Only information available at run time. No reference answer."""
        s = self.s
        cand, support = self.candidate()
        cand_checks = [c for c in s.checks if cand is not None and c.answer == cand]
        cand_arith = min((c.arith_status for c in cand_checks), key=_ARITH_RANK.get) if cand_checks else "none_found"
        n_extract = sum(c.has_answer for c in s.checks)
        mask = self.valid_actions()
        return {
            "n_attempts": len(s.attempts),
            "remaining_attempts": self.cfg.max_attempts - len(s.attempts),
            "candidate_answer": cand,
            "support": support,
            "n_extractable": n_extract,
            "n_distinct_answers": len({c.answer for c in s.checks if c.answer is not None}),
            "candidate_arith_status": cand_arith,
            "latest_has_answer": s.checks[-1].has_answer,
            "latest_truncated": s.checks[-1].truncated,
            "latest_arith_status": s.checks[-1].arith_status,
            "pot_status": compare_pot(s.pot, cand) if s.pot_used else "not_run",
            "verify_used": s.pot_used,
            "steps": s.steps,
            "action_mask": mask,
        }

    def final_answer(self) -> str | None:
        return self.candidate()[0]
