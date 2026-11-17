"""Attempt sources: cached replay (pre-generated) and live generation.

Neither source has access to reference answers.

Cached replay: attempts were generated in advance by scripts/generate_attempts.py
(attempt 0 greedy, attempts 1..K-1 sampled i.i.d. from the same prompt). Since
sampled attempts are independent draws, consuming them in order is equivalent in
distribution to generating them on demand. Latency recorded at generation time
(amortized batched GPU time) is reported, never the cache read time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.solver.prompts import POT_PROMPT_VERSION, SOLVER_PROMPT_VERSION

from .coordination_env import GenRecord


def generation_cache_key(question: str, *, model_id: str, revision: str, prompt_version: str, params: dict,
                         max_new_tokens: int, attempt_index: int, seed: int | None) -> str:
    payload = {"q": question, "model": model_id, "rev": revision, "prompt": prompt_version, "params": params,
               "max_new": max_new_tokens, "attempt": attempt_index, "seed": seed}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class CachedSource:
    def __init__(self, record: dict):
        self.record = record
        self.problem_id = record["problem_id"]
        self.max_available_attempts = len(record["attempts"])
        self.pot_available = record.get("pot") is not None

    @staticmethod
    def _rec(d: dict) -> GenRecord:
        return GenRecord(text=d["text"], finished=d["finished"], input_tokens=d["input_tokens"],
                         output_tokens=d["output_tokens"], latency_s=d["latency_s"], from_cache=True)

    def attempt(self, i: int) -> GenRecord:
        return self._rec(self.record["attempts"][i])

    def pot(self) -> GenRecord | None:
        return self._rec(self.record["pot"]) if self.pot_available else None


class LiveSource:
    """Generates on demand with the loaded solver (used by solve_problem / CLI)."""

    def __init__(self, solver, question: str, seed: int, max_attempts: int):
        from src.solver.qwen_solver import (GREEDY_PARAMS, MAX_NEW_TOKENS_POT, MAX_NEW_TOKENS_SOLVER,
                                            SAMPLE_PARAMS, derive_seed)
        from src.solver.prompts import pot_messages, solver_messages
        self._solver, self._q, self._seed = solver, question, seed
        self._cfg = (GREEDY_PARAMS, SAMPLE_PARAMS, MAX_NEW_TOKENS_SOLVER, MAX_NEW_TOKENS_POT, derive_seed,
                     solver_messages, pot_messages)
        self.max_available_attempts = max_attempts
        self.pot_available = True
        self.texts: dict[str, str] = {}

    def _gen(self, msgs, params, max_new, seed) -> GenRecord:
        g = self._solver.generate([msgs], params=params, max_new_tokens=max_new, seed=seed)[0][0]
        return GenRecord(g.text, g.finished, g.input_tokens, g.output_tokens, g.latency_s, from_cache=False)

    def attempt(self, i: int) -> GenRecord:
        greedy, sample, max_sol, _, derive, solver_messages, _ = self._cfg
        if i == 0:
            rec = self._gen(solver_messages(self._q), greedy, max_sol, None)
        else:
            rec = self._gen(solver_messages(self._q), sample, max_sol, derive("live", self._seed, self._q, i))
        self.texts[f"attempt_{i}"] = rec.text
        return rec

    def pot(self) -> GenRecord:
        greedy, _, _, max_pot, _, _, pot_messages = self._cfg
        rec = self._gen(pot_messages(self._q), greedy, max_pot, None)
        self.texts["pot"] = rec.text
        return rec


def load_cache(path: str | Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["problem_id"]] = r
    return out


PROMPT_VERSIONS = {"solver": SOLVER_PROMPT_VERSION, "pot": POT_PROMPT_VERSION}
