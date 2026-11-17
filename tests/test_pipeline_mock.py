"""Pipeline / CLI-level behaviour with a MOCK solver (no Qwen needed)."""
import json
import sys
from pathlib import Path

import pytest

from src.solver.prompts import extract_code_block
from src.verifier.verifier import compare_pot, pot_check

ROOT = Path(__file__).resolve().parents[1]


def test_extract_code_block_variants():
    assert extract_code_block("x\n```python\na = 1\nanswer = a + 1\n```\nmore") == "a = 1\nanswer = a + 1"
    assert extract_code_block("```\nanswer = 3\n```") == "answer = 3"
    assert extract_code_block("```python\nanswer = 4\n") == "answer = 4"   # truncated block still usable
    assert extract_code_block("no code") is None


def test_pot_check_and_compare():
    p = pot_check("```python\nprice = 0.1\nanswer = price * 3\n```")
    assert p.status == "ok" and p.pot_answer == "3/10"
    assert compare_pot(p, "0.3") == "agree" and compare_pot(p, "3") == "disagree"
    bad = pot_check("```python\nimport os\nanswer = 1\n```")
    assert bad.pot_answer is None and compare_pot(bad, "1") == "failed"
    assert compare_pot(None, "1") == "not_run"


class FakeGen:
    def __init__(self, text, finished=True):
        self.text, self.finished, self.input_tokens, self.output_tokens, self.latency_s = text, finished, 30, 40, 0.2


class FakeSolver:
    """Returns a wrong first attempt, then correct ones; PoT agrees with the correct answer."""

    def __init__(self):
        self.calls = []

    def generate(self, message_lists, *, params, max_new_tokens, num_return_sequences=1, seed=None):
        user = message_lists[0][-1]["content"]
        self.calls.append(("pot" if "Python program" in user else "solve", params["do_sample"], seed))
        if "Python program" in user:
            text = "```python\nanswer = 6 * 4\n```"
        elif not params["do_sample"]:
            text = "6 * 4 = 25, so \\boxed{25}"
        else:
            text = "6 * 4 = 24, so \\boxed{24}"
        return [[FakeGen(text)]]

    def info(self):
        return {"model_id": "fake"}


@pytest.mark.parametrize("method", ["single", "fixed", "max_budget"])
def test_solve_problem_structured_output(method):
    from src.pipeline import solve_problem
    fs = FakeSolver()
    r = solve_problem("Six boxes hold 4 pens each. How many pens?", method=method, budget=3, seed=1, solver=fs)
    json.dumps(r)  # must be JSON-serializable for a future web API
    assert r["usage"]["model_calls"] == len(fs.calls)
    assert r["latency_s"]["kind"] == "live_generation"
    assert len(r["attempt_texts"]) == r["usage"]["attempts"] <= 3
    if method == "single":
        assert r["final_answer"] == "25" and r["actions"] == ["ACCEPT"]
    else:
        assert r["final_answer"] == "24"
    # sampled attempts get distinct, seed-derived generator seeds
    seeds = [c[2] for c in fs.calls if c[0] == "solve" and c[1]]
    assert len(seeds) == len(set(seeds))


def test_solve_problem_validates_input():
    from src.pipeline import solve_problem
    with pytest.raises(ValueError):
        solve_problem("   ", solver=FakeSolver())
    with pytest.raises(ValueError):
        solve_problem("q", budget=9, solver=FakeSolver())
    with pytest.raises(ValueError):
        solve_problem("q", method="magic", solver=FakeSolver())


def test_training_script_refuses_test_split():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_coordinator
    with pytest.raises(SystemExit):
        train_coordinator.load_problems("test_small")
