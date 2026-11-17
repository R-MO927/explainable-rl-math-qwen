"""Discretization of the observation into a tabular state.

state = (n_attempts, support, disagreement, cand_arith, pot)
  n_attempts   1..max_attempts           attempts generated so far
  support      0,1,2,3   (3 = "3+")      attempts agreeing with the candidate (0 = no answer)
  disagreement 0/1                       more than one distinct extractable answer
  cand_arith   0 consistent, 1 none found, 2 error found   (best attempt supporting the candidate)
  pot          0 not run, 1 agree, 2 disagree, 3 failed / no candidate
Upper bound 4*4*2*3*4 = 384 states for max_attempts=4 (many are unreachable).
Remaining budget is implied by n_attempts (fixed max) and pot != 0.
"""
from __future__ import annotations

ARITH_CODE = {"all_consistent": 0, "none_found": 1, "error_found": 2}
POT_CODE = {"not_run": 0, "agree": 1, "disagree": 2, "failed": 3, "no_candidate": 3}
FEATURE_NAMES = ("n_attempts", "support", "disagreement", "cand_arith", "pot")


def state_key(obs: dict) -> tuple[int, int, int, int, int]:
    return (
        int(obs["n_attempts"]),
        min(int(obs["support"]), 3),
        int(obs["n_distinct_answers"] > 1),
        ARITH_CODE[obs["candidate_arith_status"]],
        POT_CODE[obs["pot_status"]],
    )


def key_to_str(key: tuple) -> str:
    return ",".join(map(str, key))


def str_to_key(s: str) -> tuple:
    return tuple(int(x) for x in s.split(","))


def describe_state(key: tuple) -> dict:
    n, sup, dis, ar, pot = key
    return {
        "n_attempts": n,
        "support": f"{sup}+" if sup == 3 else sup,
        "disagreement": bool(dis),
        "candidate_arithmetic": {0: "consistent", 1: "none found", 2: "error found"}[ar],
        "pot_check": {0: "not run", 1: "agrees", 2: "disagrees", 3: "failed/no candidate"}[pot],
    }
