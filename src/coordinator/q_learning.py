"""Tabular Q-learning coordinator (method C).

Q(s,a) <- Q(s,a) + lr * (r + gamma * max_{a' valid} Q(s',a') - Q(s,a)),
lr = alpha (constant) or 1/n(s,a) (visit-count schedule).
Exploration (epsilon-greedy over *valid* actions) is used only during training.
Inference is greedy over valid actions. If the state was never visited in
training, a documented fallback (the fixed rule, method B) is used and flagged.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from src.environment.coordination_env import ACTION_NAMES, N_ACTIONS

from .features import key_to_str, state_key, str_to_key
from .fixed_policy import FixedPolicy


@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    episodes: int = 20000
    seed: int = 0
    init_q: float = 0.0
    # "constant": step size alpha. "visit": step size 1/n(s,a) (sample-average estimate; with
    # gamma=1 and short episodes this averages observed returns and removes the oscillation
    # seen with a constant step size). Added after observing seed instability on validation.
    alpha_mode: str = "constant"


class QLearningCoordinator:
    name = "rl_qlearning"

    def __init__(self, config: QConfig | None = None):
        self.cfg = config or QConfig()
        self.q: dict[tuple, list[float]] = {}
        self.visits: dict[tuple, list[int]] = {}
        self.fallback = FixedPolicy()

    # ------------------------------------------------------------ table access
    def _row(self, key: tuple) -> list[float]:
        if key not in self.q:
            self.q[key] = [self.cfg.init_q] * N_ACTIONS
            self.visits[key] = [0] * N_ACTIONS
        return self.q[key]

    def q_values(self, obs: dict) -> list[float] | None:
        key = state_key(obs)
        return list(self.q[key]) if key in self.q else None

    def is_known(self, obs: dict) -> bool:
        key = state_key(obs)
        if key not in self.visits:
            return False
        mask = obs["action_mask"]
        return all(self.visits[key][a] > 0 for a in range(N_ACTIONS) if mask[a])

    @staticmethod
    def _masked_argmax(values: list[float], mask: list[bool]) -> int:
        best, best_v = None, None
        for a in range(N_ACTIONS):  # ties -> lowest index (ACCEPT first): deterministic
            if mask[a] and (best_v is None or values[a] > best_v):
                best, best_v = a, values[a]
        return best

    # ------------------------------------------------------------ acting
    def act(self, obs: dict, explore: bool = False, rng: random.Random | None = None, epsilon: float = 0.0):
        mask = obs["action_mask"]
        key = state_key(obs)
        if explore:
            row = self._row(key)
            if rng.random() < epsilon:
                return rng.choice([a for a in range(N_ACTIONS) if mask[a]]), {"explored": True}
            return self._masked_argmax(row, mask), {"explored": False}
        if not self.is_known(obs):
            a, info = self.fallback.act(obs)
            return a, {"fallback": True, "fallback_rule": info["rule"]}
        return self._masked_argmax(self.q[key], mask), {"fallback": False}

    def update(self, obs: dict, action: int, reward: float, next_obs: dict | None, done: bool) -> float:
        key = state_key(obs)
        row = self._row(key)
        self.visits[key][action] += 1
        target = reward
        if not done:
            nrow = self._row(state_key(next_obs))
            nmask = next_obs["action_mask"]
            target += self.cfg.gamma * max(nrow[a] for a in range(N_ACTIONS) if nmask[a])
        td = target - row[action]
        if self.cfg.alpha_mode == "visit":
            lr = 1.0 / self.visits[key][action]
        elif self.cfg.alpha_mode == "constant":
            lr = self.cfg.alpha
        else:
            raise ValueError(f"unknown alpha_mode {self.cfg.alpha_mode}")
        row[action] += lr * td
        return td

    # ------------------------------------------------------------ persistence
    def to_dict(self, extra: dict | None = None) -> dict:
        return {
            "format": "tabular-q-v1",
            "actions": list(ACTION_NAMES),
            "config": asdict(self.cfg),
            "q": {key_to_str(k): v for k, v in sorted(self.q.items())},
            "visits": {key_to_str(k): v for k, v in sorted(self.visits.items())},
            "extra": extra or {},
        }

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(extra), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningCoordinator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "tabular-q-v1" or d.get("actions") != list(ACTION_NAMES):
            raise ValueError(f"unsupported policy file {path}")
        obj = cls(QConfig(**d["config"]))
        obj.q = {str_to_key(k): list(v) for k, v in d["q"].items()}
        obj.visits = {str_to_key(k): list(v) for k, v in d["visits"].items()}
        return obj
