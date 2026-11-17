"""Behavioral tests of the environment, reward, Q-learning, persistence and XAI,
using MOCK attempt sources. These tests say nothing about Qwen's quality."""
import inspect
import random

import pytest

from src.coordinator.features import state_key
from src.coordinator.fixed_policy import FixedPolicy, MaxBudgetPolicy, SingleAttemptPolicy
from src.coordinator.q_learning import QConfig, QLearningCoordinator
from src.coordinator.training import train_q_learning
from src.environment.coordination_env import (ACCEPT, REGENERATE, VERIFY_MORE, CoordinationEnv, EnvConfig,
                                              GenRecord)
from src.environment.reward import RewardConfig, is_correct, terminal_reward
from src.evaluation.metrics import mcnemar_exact, summarize, wilson_ci
from src.evaluation.runner import run_episode
from src.verifier import verifier as verifier_module


class MockSource:
    def __init__(self, texts, pot_text="```python\nanswer = 2\n```", pid="p", finished=True):
        self.texts, self.pot_text, self.problem_id = texts, pot_text, pid
        self.max_available_attempts = len(texts)
        self.pot_available = pot_text is not None
        self.finished = finished

    def attempt(self, i):
        return GenRecord(self.texts[i], self.finished, 10, 20, 0.1, from_cache=True)

    def pot(self):
        return GenRecord(self.pot_text, True, 8, 12, 0.05, from_cache=True)


WRONG = "3 + 4 = 8 so \\boxed{1}"     # arithmetic error, answer 1
RIGHT = "1 + 1 = 2 so \\boxed{2}"     # consistent, answer 2
NONE = "I am not sure."


def test_reset_generates_first_attempt_and_masks():
    env = CoordinationEnv(EnvConfig(max_attempts=2))
    obs = env.reset(MockSource([WRONG, RIGHT]))
    assert obs["n_attempts"] == 1 and obs["candidate_answer"] == "1"
    assert obs["candidate_arith_status"] == "error_found"
    assert obs["action_mask"] == [True, True, True]
    obs, done, _ = env.step(REGENERATE)
    assert obs["n_attempts"] == 2 and not done
    assert obs["action_mask"] == [True, False, True]       # max attempts reached
    obs, done, info = env.step(VERIFY_MORE)
    assert info["pot"]["pot_answer"] == "2" and obs["pot_status"] == "agree"
    assert obs["action_mask"] == [True, False, False]      # PoT is deterministic: once only
    with pytest.raises(ValueError):
        env.step(VERIFY_MORE)
    obs, done, _ = env.step(ACCEPT)
    assert done and env.final_answer() == "2"              # tie 1 vs 2 broken by PoT agreement
    with pytest.raises(RuntimeError):
        env.step(ACCEPT)


def test_verify_masked_when_pot_unavailable():
    env = CoordinationEnv(EnvConfig(max_attempts=3))
    obs = env.reset(MockSource([RIGHT], pot_text=None))
    assert obs["action_mask"] == [True, False, False]


def test_plurality_vote_and_tiebreaks():
    env = CoordinationEnv(EnvConfig(max_attempts=4))
    env.reset(MockSource([WRONG, RIGHT, RIGHT, WRONG], pot_text="```python\nanswer = 7\n```"))
    env.step(REGENERATE)
    assert env.candidate() == ("2", 1)   # tie: consistent arithmetic beats error
    env.step(REGENERATE)
    assert env.candidate() == ("2", 2)
    obs, _, _ = env.step(VERIFY_MORE)
    assert obs["pot_status"] == "disagree"


def test_observation_and_verifier_never_see_reference():
    # Verifier API has no reference parameter at all.
    for fn in (verifier_module.basic_check, verifier_module.pot_check):
        assert not any("ref" in p for p in inspect.signature(fn).parameters)
    # Same episode with two different references: identical observations and actions.
    logs = [run_episode(FixedPolicy(), MockSource([WRONG, RIGHT, RIGHT]), env_cfg=EnvConfig(max_attempts=3),
                        reward_cfg=RewardConfig(), reference=ref) for ref in ("2", "999")]
    assert logs[0]["observations"] == logs[1]["observations"]
    assert logs[0]["actions"] == logs[1]["actions"]
    assert logs[0]["correctness"] is True and logs[1]["correctness"] is False
    for o in logs[0]["observations"]:
        assert "999" not in map(str, o.values()) and not any("ref" in k for k in o)


def test_reward_values_and_no_abstention_exploit():
    cfg = RewardConfig()
    assert terminal_reward("2", "2", cfg) == 1.0
    assert terminal_reward("3", "2", cfg) == -1.0
    assert terminal_reward(None, "2", cfg) == -1.0          # abstaining is not better than a wrong answer
    assert is_correct("2.0", "2") and not is_correct(None, "2")
    log = run_episode(MaxBudgetPolicy(), MockSource([RIGHT, RIGHT, RIGHT]), env_cfg=EnvConfig(max_attempts=3),
                      reward_cfg=cfg, reference="2")
    assert log["actions"] == ["REGENERATE", "REGENERATE", "VERIFY_MORE", "ACCEPT"]
    assert log["return"] == pytest.approx(1 - 2 * 0.05 - 0.03)
    assert log["termination_reason"] == "accepted_budget_exhausted"
    none_log = run_episode(SingleAttemptPolicy(), MockSource([NONE]), env_cfg=EnvConfig(), reward_cfg=cfg,
                           reference="2")
    assert none_log["final_answer"] is None and none_log["return"] == -1.0
    assert none_log["termination_reason"] == "accepted_no_answer"


def test_usage_accounting():
    log = run_episode(FixedPolicy(), MockSource([WRONG, RIGHT, RIGHT]), env_cfg=EnvConfig(max_attempts=3),
                      reward_cfg=RewardConfig(), reference="2")
    # attempt0 answers 1 -> rule 2 VERIFY (PoT=2 disagrees) -> rule 4 REGENERATE (attempt1 answers 2; tie 1 vs 2
    # broken by PoT agreement -> candidate 2) -> rule 3 ACCEPT
    assert log["actions"] == ["VERIFY_MORE", "REGENERATE", "ACCEPT"]
    assert log["final_answer"] == "2" and log["correctness"] is True
    assert log["model_calls"] == log["attempts"] + log["pot_calls"]
    assert log["verifier_calls"] == log["attempts"] + log["pot_calls"]
    assert log["input_tokens"] == 10 * log["attempts"] + 8 * log["pot_calls"]
    assert log["latency_kind"] == "amortized_batched_generation"


def test_q_update_math_and_masked_bootstrap():
    agent = QLearningCoordinator(QConfig(alpha=0.5, gamma=1.0))
    env = CoordinationEnv(EnvConfig(max_attempts=2))
    obs = env.reset(MockSource([WRONG, RIGHT]))
    nobs, done, _ = env.step(REGENERATE)
    nkey = state_key(nobs)
    agent._row(nkey)
    agent.q[nkey] = [0.4, 100.0, 0.2]  # REGENERATE value must be ignored: masked in next state
    td = agent.update(obs, REGENERATE, -0.05, nobs, done=False)
    assert td == pytest.approx(-0.05 + 0.4)
    assert agent.q[state_key(obs)][REGENERATE] == pytest.approx(0.5 * 0.35)
    agent.update(nobs, ACCEPT, 1.0, None, done=True)          # terminal: no bootstrap
    assert agent.q[nkey][ACCEPT] == pytest.approx(0.4 + 0.5 * (1.0 - 0.4))


def test_visit_count_step_size_is_sample_average():
    agent = QLearningCoordinator(QConfig(alpha_mode="visit"))
    env = CoordinationEnv(EnvConfig(max_attempts=2))
    obs = env.reset(MockSource([RIGHT, RIGHT]))
    for r in (1.0, -1.0, 1.0, 1.0):  # terminal rewards -> Q equals their running mean
        agent.update(obs, ACCEPT, r, None, done=True)
    assert agent.q[state_key(obs)][ACCEPT] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        QLearningCoordinator(QConfig(alpha_mode="bogus")).update(obs, ACCEPT, 1.0, None, done=True)


def test_save_load_roundtrip_and_fallback(tmp_path):
    agent = QLearningCoordinator(QConfig(seed=3))
    env = CoordinationEnv(EnvConfig(max_attempts=2))
    obs = env.reset(MockSource([RIGHT, RIGHT]))
    for a in (ACCEPT, REGENERATE, VERIFY_MORE):
        agent._row(state_key(obs))
        agent.visits[state_key(obs)][a] += 1
    agent.q[state_key(obs)] = [0.1, 0.7, 0.3]
    p = tmp_path / "policy.json"
    agent.save(p, extra={"note": "x"})
    loaded = QLearningCoordinator.load(p)
    assert loaded.q == agent.q and loaded.visits == agent.visits and loaded.cfg == agent.cfg
    assert loaded.act(obs)[0] == REGENERATE and loaded.act(obs)[1]["fallback"] is False
    env2 = CoordinationEnv(EnvConfig(max_attempts=2))
    unseen = env2.reset(MockSource([NONE, NONE]))
    a, info = loaded.act(unseen)
    assert info["fallback"] is True and a == FixedPolicy().act(unseen)[0]


def test_end_to_end_learning_on_mock_problems():
    """Attempt 0 is always wrong with a visible arithmetic error; later attempts are right.
    Q-learning should learn to avoid accepting the first attempt."""
    probs = [(f"p{i}", MockSource([WRONG, RIGHT, RIGHT, RIGHT], pot_text=None, pid=f"p{i}"), "2") for i in range(5)]
    agent, curve = train_q_learning(probs, q_cfg=QConfig(episodes=3000, alpha=0.1, seed=0),
                                    env_cfg=EnvConfig(max_attempts=4), reward_cfg=RewardConfig(), log_every=1000)
    log = run_episode(agent, probs[0][1], env_cfg=EnvConfig(max_attempts=4), reward_cfg=RewardConfig(), reference="2")
    assert log["correctness"] is True and log["actions"][0] == "REGENERATE"
    assert curve[-1]["train_accuracy_ma"] > curve[0]["train_accuracy_ma"]
    d = log["decisions"][0]
    assert d["chosen_action"] == "REGENERATE" and d["q_values"]["REGENERATE"] > d["q_values"]["ACCEPT"]
    assert "not a probability" in d["text"]
    # single-attempt baseline on the same mock data is always wrong
    a_log = run_episode(SingleAttemptPolicy(), probs[0][1], env_cfg=EnvConfig(), reward_cfg=RewardConfig(),
                        reference="2")
    assert a_log["correctness"] is False


def test_metrics():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    logs = [
        {"correctness": True, "final_answer": "1", "termination_reason": "accepted", "attempts": 1,
         "verifier_calls": 1, "pot_calls": 0, "model_calls": 1, "input_tokens": 5, "output_tokens": 5,
         "latency_s": 1.0, "return": 1.0, "generated_answers": ["1"]},
        {"correctness": False, "final_answer": None, "termination_reason": "accepted_no_answer", "attempts": 2,
         "verifier_calls": 2, "pot_calls": 0, "model_calls": 2, "input_tokens": 5, "output_tokens": 5,
         "latency_s": 1.0, "return": -1.05, "generated_answers": [None, None]},
    ]
    s = summarize(logs)
    assert s["accuracy"] == 0.5 and s["coverage"] == 0.5 and s["accuracy_on_answered"] == 1.0
    assert s["no_answer_rate"] == 0.5 and s["first_attempt_unextractable_rate"] == 0.5
    m = mcnemar_exact([True, False, False, False], [True, True, True, False])
    assert m["b_only_correct"] == 2 and m["a_only_correct"] == 0 and m["p_value"] == pytest.approx(0.5)
