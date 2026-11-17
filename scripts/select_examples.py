"""Pick real success/failure examples from evaluation logs (deterministic: first ids per category).

Writes results/examples/examples_<split>.json and .md
Usage: python scripts/select_examples.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.sources import load_cache  # noqa: E402

CATEGORIES = {
    "rl_fixed_it": "A wrong, C (RL) right: the coordinator spent extra actions and recovered",
    "rl_accepted_wrong": "A wrong, C wrong after accepting the first attempt (no warning signal or Q favoured ACCEPT)",
    "fixed_right_rl_wrong": "B right, C wrong: the cheaper RL policy stopped too early",
    "rl_right_fixed_wrong": "C right, B wrong",
    "all_wrong_after_full_budget": "max-budget reference also wrong: none of the tools helped",
}


def load_logs(split: str, method: str) -> dict:
    with open(ROOT / "results" / "eval" / split / f"logs_{method}.jsonl", encoding="utf-8") as fh:
        return {r["problem_id"]: r for r in map(json.loads, fh)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-category", type=int, default=2)
    args = ap.parse_args(argv)
    A, B, C, M = (load_logs(args.split, m) for m in ("A_single_attempt", "B_fixed_rule", "C_rl", "ref_max_budget"))
    data = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/gsm8k" / f"{args.split}.jsonl",
                                                               encoding="utf-8")}
    cache = load_cache(ROOT / "artifacts/cache/test.jsonl")
    pick = {k: [] for k in CATEGORIES}
    for pid in sorted(C):
        a, b, c, m = A[pid]["correctness"], B[pid]["correctness"], C[pid]["correctness"], M[pid]["correctness"]
        cond = {
            "rl_fixed_it": (not a) and c and len(C[pid]["actions"]) > 1,
            "rl_accepted_wrong": (not a) and (not c) and C[pid]["actions"] == ["ACCEPT"],
            "fixed_right_rl_wrong": b and not c,
            "rl_right_fixed_wrong": c and not b,
            "all_wrong_after_full_budget": not m,
        }
        for k, ok in cond.items():
            if ok and len(pick[k]) < args.per_category:
                pick[k].append(pid)
    out, md = {}, [f"# Real examples from `{args.split}` (selected deterministically: first ids per category)\n"]
    for k, ids in pick.items():
        md.append(f"\n## {k}\n_{CATEGORIES[k]}_\n")
        out[k] = []
        for pid in ids:
            c = C[pid]
            ex = {
                "problem_id": pid,
                "question": data[pid]["question"],
                "reference_answer": data[pid]["normalized_final_answer"],
                "A": {"answer": A[pid]["final_answer"], "correct": A[pid]["correctness"]},
                "B": {"answer": B[pid]["final_answer"], "correct": B[pid]["correctness"], "actions": B[pid]["actions"]},
                "C": {"answer": c["final_answer"], "correct": c["correctness"], "actions": c["actions"],
                      "attempt_answers": c["generated_answers"],
                      "explanations": [d["text"] for d in c["decisions"]]},
                "first_attempt_excerpt": cache[pid]["attempts"][0]["text"][-500:],
            }
            out[k].append(ex)
            md.append(f"\n### {pid}\n**Question:** {ex['question']}\n\n**Reference:** {ex['reference_answer']} | "
                      f"A: {ex['A']['answer']} ({'✓' if ex['A']['correct'] else '✗'}) | "
                      f"B: {ex['B']['answer']} ({'✓' if ex['B']['correct'] else '✗'}) via {' → '.join(ex['B']['actions'])} | "
                      f"C: {ex['C']['answer']} ({'✓' if ex['C']['correct'] else '✗'}) via {' → '.join(ex['C']['actions'])}\n\n"
                      f"C attempt answers: {ex['C']['attempt_answers']}\n\nC decisions:\n" +
                      "".join(f"- {t}\n" for t in ex["C"]["explanations"]))
    dest = ROOT / "results" / "examples"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"examples_{args.split}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (dest / f"examples_{args.split}.md").write_text("".join(md), encoding="utf-8")
    print({k: len(v) for k, v in pick.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
