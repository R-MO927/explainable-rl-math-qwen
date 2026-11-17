"""Command-line interface: solve a new question (no reference answer needed).

Examples:
  python scripts/solve.py "A shop sells pens at $3 each. How much do 7 pens cost?"
  python scripts/solve.py --method fixed --budget 3 "..."
  python scripts/solve.py --json "..."          # full structured output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import METHODS, solve_problem  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--method", choices=METHODS, default="rl")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the full JSON result")
    args = ap.parse_args(argv)
    r = solve_problem(args.question, method=args.method, budget=args.budget, seed=args.seed)
    if args.json:
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0
    print(f"Final answer : {r['final_answer']}  ({r['termination_reason']})")
    print(f"Attempts     : {r['attempt_answers']}")
    print(f"Actions      : {' -> '.join(r['actions'])}")
    print(f"Usage        : {r['usage']}, live generation {r['latency_s']['generation']:.1f}s")
    print("Decisions:")
    for d in r["explanations"]:
        print("  -", d["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
