"""Method B: hand-written, documented coordination rule (no learning).

Written a priori (before any validation/test results) and never tuned:
  1. if the candidate is supported by >= 2 attempts           -> ACCEPT
  2. if a candidate exists and PoT has not been run (allowed)  -> VERIFY_MORE
  3. if PoT agrees with the candidate                          -> ACCEPT
  4. if another attempt is allowed                             -> REGENERATE
  5. otherwise                                                 -> ACCEPT (budget exhausted)
It uses exactly the same tools and limits as the RL coordinator.
"""
from __future__ import annotations

from src.environment.coordination_env import ACCEPT, REGENERATE, VERIFY_MORE


class FixedPolicy:
    name = "fixed_rule"

    def act(self, obs: dict) -> tuple[int, dict]:
        mask = obs["action_mask"]
        if obs["support"] >= 2:
            return ACCEPT, {"rule": "1: candidate supported by >= 2 attempts"}
        if obs["candidate_answer"] is not None and mask[VERIFY_MORE]:
            return VERIFY_MORE, {"rule": "2: verify candidate with PoT"}
        if obs["pot_status"] == "agree":
            return ACCEPT, {"rule": "3: PoT agrees"}
        if mask[REGENERATE]:
            return REGENERATE, {"rule": "4: regenerate"}
        return ACCEPT, {"rule": "5: budget exhausted"}


class SingleAttemptPolicy:
    """Method A: accept the first attempt (no verifier use beyond answer extraction)."""

    name = "single_attempt"

    def act(self, obs: dict) -> tuple[int, dict]:
        return ACCEPT, {"rule": "single attempt"}


class MaxBudgetPolicy:
    """Extra reference (not one of A/B/C): always spend the full budget
    (all attempts + PoT), then accept the plurality answer (self-consistency)."""

    name = "max_budget"

    def act(self, obs: dict) -> tuple[int, dict]:
        mask = obs["action_mask"]
        if mask[REGENERATE]:
            return REGENERATE, {"rule": "use all attempts"}
        if mask[VERIFY_MORE]:
            return VERIFY_MORE, {"rule": "use PoT"}
        return ACCEPT, {"rule": "budget exhausted"}
