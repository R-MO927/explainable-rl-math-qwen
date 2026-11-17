"""Pre-generate solver attempts and PoT programs for a data split (cache for replay).

Only problem ids and question texts are passed to the model; reference answers
are never read by this script.

Per problem:
  attempt 0          greedy               (MAX_NEW_TOKENS_SOLVER)
  attempts 1..K-1    sampled i.i.d.       (official generation_config values)
  pot                greedy PoT program   (MAX_NEW_TOKENS_POT)
Every generation has a cache key = sha256(question, model, revision, prompt version,
decoding params, max_new_tokens, attempt index, seed). Existing entries with the
same keys are reused (resumable).

Usage:
  python scripts/generate_attempts.py --split train_small --k 4
  python scripts/generate_attempts.py --split validation_small --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.sources import generation_cache_key  # noqa: E402
from src.solver.prompts import POT_PROMPT_VERSION, SOLVER_PROMPT_VERSION, pot_messages, solver_messages  # noqa: E402
from src.solver.qwen_solver import (GREEDY_PARAMS, MAX_NEW_TOKENS_POT, MAX_NEW_TOKENS_SOLVER, MODEL_ID,  # noqa: E402
                                    MODEL_REVISION, SAMPLE_PARAMS, QwenSolver, derive_seed)

DATA = ROOT / "data" / "processed" / "gsm8k"
CACHE = ROOT / "artifacts" / "cache"


def load_questions(split: str) -> list[tuple[str, str]]:
    out = []
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append((r["id"], r["question"]))  # reference fields deliberately dropped here
    return out


def keys_for(q: str, k: int, seed_sample: int) -> dict:
    common = dict(model_id=MODEL_ID, revision=MODEL_REVISION)
    keys = {"attempt_0": generation_cache_key(q, prompt_version=SOLVER_PROMPT_VERSION, params=GREEDY_PARAMS,
                                              max_new_tokens=MAX_NEW_TOKENS_SOLVER, attempt_index=0, seed=None,
                                              **common)}
    for i in range(1, k):
        keys[f"attempt_{i}"] = generation_cache_key(q, prompt_version=SOLVER_PROMPT_VERSION, params=SAMPLE_PARAMS,
                                                    max_new_tokens=MAX_NEW_TOKENS_SOLVER, attempt_index=i,
                                                    seed=seed_sample, **common)
    keys["pot"] = generation_cache_key(q, prompt_version=POT_PROMPT_VERSION, params=GREEDY_PARAMS,
                                       max_new_tokens=MAX_NEW_TOKENS_POT, attempt_index=0, seed=None, **common)
    return keys


def gen_dict(g, *, index, mode, params, key, batch_size, max_new) -> dict:
    return {"index": index, "mode": mode, "params": params, "max_new_tokens": max_new, "cache_key": key,
            "text": g.text, "finished": g.finished, "input_tokens": g.input_tokens, "output_tokens": g.output_tokens,
            "latency_s": round(g.latency_s, 5), "batch_size": batch_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=4, help="attempts per problem (1 greedy + k-1 sampled)")
    ap.add_argument("--batch-size", type=int, default=16, help="problems per generation batch")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    problems = load_questions(args.split)[: args.limit]
    out = args.out or CACHE / f"{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                done[r["problem_id"]] = r
    todo = [p for p in problems if p[0] not in done]
    print(f"[gen] split={args.split} problems={len(problems)} cached={len(problems) - len(todo)} todo={len(todo)}")
    if not todo:
        return 0

    solver = QwenSolver()
    info = solver.info()
    print("[gen] model:", info)
    todo.sort(key=lambda p: len(p[1]))  # less padding
    t_start = time.perf_counter()
    n_done = 0
    with open(out, "a", encoding="utf-8", newline="\n") as fh:
        for b in range(0, len(todo), args.batch_size):
            chunk = todo[b : b + args.batch_size]
            ids = [p[0] for p in chunk]
            seed_sample = derive_seed("sample", args.seed, ids)
            msgs = [solver_messages(q) for _, q in chunk]
            greedy = solver.generate(msgs, params=GREEDY_PARAMS, max_new_tokens=MAX_NEW_TOKENS_SOLVER)
            sampled = solver.generate(msgs, params=SAMPLE_PARAMS, max_new_tokens=MAX_NEW_TOKENS_SOLVER,
                                      num_return_sequences=args.k - 1, seed=seed_sample) if args.k > 1 else None
            pots = solver.generate([pot_messages(q) for _, q in chunk], params=GREEDY_PARAMS,
                                   max_new_tokens=MAX_NEW_TOKENS_POT)
            for j, (pid, q) in enumerate(chunk):
                keys = keys_for(q, args.k, seed_sample)
                attempts = [gen_dict(greedy[j][0], index=0, mode="greedy", params=GREEDY_PARAMS,
                                     key=keys["attempt_0"], batch_size=len(chunk), max_new=MAX_NEW_TOKENS_SOLVER)]
                for i in range(1, args.k):
                    attempts.append(gen_dict(sampled[j][i - 1], index=i, mode="sample", params=SAMPLE_PARAMS,
                                             key=keys[f"attempt_{i}"], batch_size=len(chunk) * (args.k - 1),
                                             max_new=MAX_NEW_TOKENS_SOLVER))
                rec = {
                    "problem_id": pid,
                    "split": args.split,
                    "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": info["dtype"],
                              "transformers": info["transformers"], "torch": info["torch"]},
                    "prompt_versions": {"solver": SOLVER_PROMPT_VERSION, "pot": POT_PROMPT_VERSION},
                    "sample_seed": seed_sample,
                    "attempts": attempts,
                    "pot": gen_dict(pots[j][0], index=0, mode="greedy", params=GREEDY_PARAMS, key=keys["pot"],
                                    batch_size=len(chunk), max_new=MAX_NEW_TOKENS_POT),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += len(chunk)
            el = time.perf_counter() - t_start
            print(f"[gen] {n_done}/{len(todo)} problems  {el:.0f}s  ({el / n_done:.2f}s/problem)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
