"""Template-based explanations of coordinator decisions.

Explanations are generated from the recorded numbers (state, mask, Q-values,
visit counts) with fixed templates - never by a language model - and describe
the *coordinator's* decision only, not Qwen's internal reasoning.
Q-values are estimates of expected return under the designed reward
(+1 correct, -1 wrong/no answer, minus action costs); they are NOT
probabilities that the answer is correct.
"""
from __future__ import annotations

from src.coordinator.features import describe_state, state_key
from src.environment.coordination_env import ACTION_NAMES, N_ACTIONS

Q_NOTE = "Q = expected return under the designed reward, not a probability of correctness."


def explain_decision(obs: dict, action: int, policy_name: str, q_values: list[float] | None = None,
                     visits: list[int] | None = None, info: dict | None = None) -> dict:
    info = info or {}
    mask = obs["action_mask"]
    key = state_key(obs)
    desc = describe_state(key)
    avail = [ACTION_NAMES[a] for a in range(N_ACTIONS) if mask[a]]
    signals = (f"{desc['n_attempts']} attempt(s); candidate {obs['candidate_answer']!s} supported by "
               f"{desc['support']}; arithmetic {desc['candidate_arithmetic']}; PoT {desc['pot_check']}")
    out = {
        "policy": policy_name,
        "state_key": list(key),
        "state": desc,
        "available_actions": avail,
        "chosen_action": ACTION_NAMES[action],
        "q_values": None,
        "visits": visits,
        "fallback": bool(info.get("fallback")),
        "rule": info.get("rule") or info.get("fallback_rule"),
        "candidate": obs.get("candidate_answer"),
    }
    if q_values is not None:
        out["q_values"] = {ACTION_NAMES[a]: (round(q_values[a], 4) if mask[a] else None) for a in range(N_ACTIONS)}
    if info.get("fallback"):
        out["text"] = (f"State {list(key)} was not (fully) visited during training, so the documented fallback "
                       f"(fixed rule: {info.get('fallback_rule')}) chose {ACTION_NAMES[action]}. Signals: {signals}.")
    elif q_values is not None:
        others = [(q_values[a], ACTION_NAMES[a]) for a in range(N_ACTIONS) if mask[a] and a != action]
        if others:
            q2, name2 = max(others)
            out["margin_over_next_best"] = round(q_values[action] - q2, 4)
            out["text"] = (f"Chose {ACTION_NAMES[action]} (Q={q_values[action]:+.3f}) over {name2} (Q={q2:+.3f}), "
                           f"margin {q_values[action] - q2:+.3f}. Signals: {signals}. {Q_NOTE}")
        else:
            out["text"] = f"Only {ACTION_NAMES[action]} was available (budget exhausted). Signals: {signals}."
    else:
        out["text"] = f"{policy_name} chose {ACTION_NAMES[action]} by rule '{info.get('rule')}'. Signals: {signals}."
    return out
