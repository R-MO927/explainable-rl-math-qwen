"""Checks on the real processed GSM8K files (skipped if data not prepared yet)."""
import json
from pathlib import Path

import pytest

from src.data.answers import parse_reference_number, canonical_str
from src.data.gsm8k import loose_key

PROC = Path(__file__).resolve().parents[1] / "data" / "processed" / "gsm8k"
pytestmark = pytest.mark.skipif(not (PROC / "split_manifest.json").exists(), reason="run scripts/prepare_data.py first")


def load(name):
    with open(PROC / f"{name}.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_real_sizes():
    sizes = {n: len(load(n)) for n in ["train", "validation", "test", "train_small", "validation_small", "test_small"]}
    assert sizes["test"] == 1319
    assert sizes["train"] + sizes["validation"] == 7473  # nothing excluded in this source version
    assert (sizes["train_small"], sizes["validation_small"], sizes["test_small"]) == (500, 100, 200)


def test_real_no_text_overlap_between_splits():
    k = {n: {loose_key(r["question"]) for r in load(n)} for n in ["train", "validation", "test"]}
    assert not k["train"] & k["validation"]
    assert not k["train"] & k["test"]
    assert not k["validation"] & k["test"]


def test_real_normalized_answers_roundtrip():
    for n in ["train", "validation", "test"]:
        for r in load(n):
            v = parse_reference_number(r["reference_final_answer"])
            assert canonical_str(v) == r["normalized_final_answer"]
            assert r["reference_solution"].strip().endswith(r["reference_final_answer"])


def test_real_known_examples():
    test = {r["id"]: r for r in load("test")}
    # Representative real cases: thousands separator and negative answers.
    assert test["gsm8k-test-00146"]["reference_final_answer"] == "2,125"
    assert test["gsm8k-test-00146"]["normalized_final_answer"] == "2125"
    assert test["gsm8k-test-00489"]["normalized_final_answer"] == "-10"
    assert test["gsm8k-test-00611"]["normalized_final_answer"] == "1450000"
