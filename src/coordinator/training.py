"""Q-learning training loop over a set of attempt sources (one per problem)."""
from __future__ import annotations

import random
from statistics import mean
from typing import Callable

from src.environment.coordination_env import CoordinationEnv, EnvConfig
from src.environment.reward import RewardConfig, is_correct, step_cost, terminal_reward

from .q_learning import QConfig, QLearningCoordinator


def train_q_learning(
    problems: list[tuple[str, object, str]],  # (problem_id, source, reference_normalized)
    *,
    q_cfg: QConfig,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    log_every: int = 500,
    eval_fn: Callable[[QLearningCoordinator], dict] | None = None,
) -> tuple[QLearningCoordinator, list[dict]]:
    """Episodes sample problems uniformly with a seeded RNG. Epsilon decays
    linearly from epsilon_start to epsilon_end over the episodes.
    Returns the agent and a learning curve (moving averages over `log_every`)."""
    rng = random.Random(q_cfg.seed)
    agent = QLearningCoordinator(q_cfg)
    env = CoordinationEnv(env_cfg)
    curve, window = [], []
    for ep in range(q_cfg.episodes):
        frac = ep / max(1, q_cfg.episodes - 1)
        eps = q_cfg.epsilon_start + (q_cfg.epsilon_end - q_cfg.epsilon_start) * frac
        _pid, source, ref = problems[rng.randrange(len(problems))]
        obs = env.reset(source)
        done, ret, n_actions = False, 0.0, 0
        while not done:
            a, _ = agent.act(obs, explore=True, rng=rng, epsilon=eps)
            nobs, done, _ = env.step(a)
            r = step_cost(a, reward_cfg)
            if done:
                r += terminal_reward(env.final_answer(), ref, reward_cfg)
            agent.update(obs, a, r, nobs, done)
            ret += r
            n_actions += 1
            obs = nobs
        window.append((ret, is_correct(env.final_answer(), ref), len(env.s.attempts), int(env.s.pot_used)))
        if (ep + 1) % log_every == 0 or ep + 1 == q_cfg.episodes:
            point = {
                "episode": ep + 1,
                "epsilon": round(eps, 4),
                "train_return_ma": mean(w[0] for w in window),
                "train_accuracy_ma": mean(w[1] for w in window),
                "train_attempts_ma": mean(w[2] for w in window),
                "train_pot_rate_ma": mean(w[3] for w in window),
                "n_states": len(agent.q),
            }
            if eval_fn is not None:
                point.update(eval_fn(agent))
            curve.append(point)
            window = []
    return agent, curve
