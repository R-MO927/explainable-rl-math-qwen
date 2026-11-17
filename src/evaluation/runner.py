"""Run one episode with any policy and produce a structured log record.

The reference answer is passed only here (for reward/correctness), never to the
environment, the source or the policy.
"""
from __future__ import annotations

from src.coordinator.q_learning import QLearningCoordinator
from src.environment.coordination_env import ACCEPT, ACTION_NAMES, CoordinationEnv, EnvConfig
from src.environment.reward import RewardConfig, is_correct, step_cost, terminal_reward
from src.explainability.explain import explain_decision


def _public_obs(obs: dict) -> dict:
    return {k: v for k, v in obs.items()}


def run_episode(policy, source, *, env_cfg: EnvConfig, reward_cfg: RewardConfig | None = None,
                reference: str | None = None, method: str | None = None, seed: int | None = None,
                problem_id: str | None = None, with_explanations: bool = True) -> dict:
    env = CoordinationEnv(env_cfg)
    obs = env.reset(source)
    observations, actions, rewards, decisions = [_public_obs(obs)], [], [], []
    termination = None
    done = False
    while not done:
        mask = obs["action_mask"]
        only_accept = mask == [True, False, False]
        a, info = policy.act(obs)
        if with_explanations:
            if isinstance(policy, QLearningCoordinator):
                qv = policy.q_values(obs)
                from src.coordinator.features import state_key
                vis = policy.visits.get(state_key(obs))
                decisions.append(explain_decision(obs, a, policy.name, qv, vis, info))
            else:
                decisions.append(explain_decision(obs, a, policy.name, None, None, info))
        obs, done, step_info = env.step(a)
        actions.append(ACTION_NAMES[a])
        if reward_cfg is not None:
            rewards.append(step_cost(a, reward_cfg))
        observations.append(_public_obs(obs))
        if "forced_termination" in step_info:
            termination = step_info["forced_termination"]
        elif a == ACCEPT:
            if env.final_answer() is None:
                termination = "accepted_no_answer"
            elif only_accept:
                termination = "accepted_budget_exhausted"
            else:
                termination = "accepted"
    final = env.final_answer()
    correct = None
    if reference is not None:
        correct = is_correct(final, reference)
        if reward_cfg is not None:
            rewards[-1] += terminal_reward(final, reference, reward_cfg)
    u = env.s.usage
    return {
        "problem_id": problem_id or getattr(source, "problem_id", None),
        "method": method or getattr(policy, "name", type(policy).__name__),
        "seed": seed,
        "attempts": len(env.s.attempts),
        "generated_answers": [c.answer for c in env.s.checks],
        "verifier_outputs": {"basic": [c.to_dict() for c in env.s.checks],
                             "pot": env.s.pot.to_dict() if env.s.pot else None},
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "return": sum(rewards) if rewards else None,
        "final_answer": final,
        "correctness": correct,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "model_calls": u.model_calls,
        "verifier_calls": u.verifier_calls,
        "pot_calls": int(env.s.pot_used),
        "latency_s": round(u.latency_s, 4),
        "latency_kind": "amortized_batched_generation" if u.live_generations == 0 else "live_generation",
        "cached_generations": u.cached_generations,
        "live_generations": u.live_generations,
        "termination_reason": termination,
        "decisions": decisions,
    }
