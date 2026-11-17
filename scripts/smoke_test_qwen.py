"""Smoke test on a few TRAIN problems: does generation + answer extraction work?

Uses only train_small (never validation/test). Writes results/smoke/smoke_train.json.
Correctness against the reference is computed here only to sanity-check the
extraction pipeline.

Usage: python scripts/smoke_test_qwen.py --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.reward import is_correct  # noqa: E402
from src.solver.prompts import pot_messages, solver_messages  # noqa: E402
from src.solver.qwen_solver import (GREEDY_PARAMS, MAX_NEW_TOKENS_POT, MAX_NEW_TOKENS_SOLVER,  # noqa: E402
                                    SAMPLE_PARAMS, QwenSolver)
from src.verifier.verifier import basic_check, pot_check  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args(argv)
    recs = [json.loads(l) for l in open(ROOT / "data/processed/gsm8k/train_small.jsonl", encoding="utf-8")][: args.n]

    t0 = time.perf_counter()
    solver = QwenSolver()
    load_s = time.perf_counter() - t0
    import torch
    print("[smoke] model loaded in %.1fs" % load_s, solver.info())

    rows = []
    for b in range(0, len(recs), args.batch_size):
        chunk = recs[b : b + args.batch_size]
        msgs = [solver_messages(r["question"]) for r in chunk]
        t = time.perf_counter()
        greedy = solver.generate(msgs, params=GREEDY_PARAMS, max_new_tokens=MAX_NEW_TOKENS_SOLVER)
        t_g = time.perf_counter() - t
        t = time.perf_counter()
        sampled = solver.generate(msgs, params=SAMPLE_PARAMS, max_new_tokens=MAX_NEW_TOKENS_SOLVER,
                                  num_return_sequences=2, seed=7)
        t_s = time.perf_counter() - t
        pots = solver.generate([pot_messages(r["question"]) for r in chunk], params=GREEDY_PARAMS,
                               max_new_tokens=MAX_NEW_TOKENS_POT)
        print(f"[smoke] batch {b // args.batch_size}: greedy {t_g:.1f}s, 2x sampled {t_s:.1f}s")
        for j, r in enumerate(chunk):
            g = greedy[j][0]
            bc = basic_check(g.text, g.finished)
            pc = pot_check(pots[j][0].text)
            s_answers = [basic_check(s.text, s.finished).answer for s in sampled[j]]
            rows.append({
                "id": r["id"], "reference": r["normalized_final_answer"],
                "greedy_answer": bc.answer, "extraction": bc.extraction_method, "greedy_correct": is_correct(bc.answer, r["normalized_final_answer"]),
                "truncated": bc.truncated, "arith": [bc.arith_checked, bc.arith_errors],
                "sampled_answers": s_answers, "sampled_distinct_from_greedy": [a != bc.answer for a in s_answers],
                "pot_answer": pc.pot_answer, "pot_status": pc.status,
                "pot_correct": is_correct(pc.pot_answer, r["normalized_final_answer"]),
                "out_tokens": g.output_tokens, "greedy_text": g.text, "pot_text": pots[j][0].text,
            })
    n = len(rows)
    summary = {
        "n": n,
        "greedy_accuracy": sum(r["greedy_correct"] for r in rows) / n,
        "greedy_extractable": sum(r["greedy_answer"] is not None for r in rows) / n,
        "greedy_truncated": sum(r["truncated"] for r in rows) / n,
        "pot_ok": sum(r["pot_status"] == "ok" for r in rows) / n,
        "pot_accuracy": sum(r["pot_correct"] for r in rows) / n,
        "sampled_text_differs_from_greedy_rate": sum(sum(r["sampled_distinct_from_greedy"]) for r in rows) / (2 * n),
        "mean_out_tokens": sum(r["out_tokens"] for r in rows) / n,
        "max_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "model": solver.info(),
        "note": "train_small only; sanity check of the pipeline, not an evaluation.",
    }
    out = ROOT / "results" / "smoke" / "smoke_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in rows[:20]:
        print(f"{r['id']} ref={r['reference']:>8} greedy={r['greedy_answer']!s:>8} ({r['extraction']}) "
              f"sampled={r['sampled_answers']} pot={r['pot_answer']!s:>8} [{r['pot_status']}] arith={r['arith']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
