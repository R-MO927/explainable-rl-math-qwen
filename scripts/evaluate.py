"""Final comparison of A / B / C on a held-out split (default: test_small).

All settings must be frozen before running this (policy selected on validation).
  A  single attempt (greedy attempt 0, answer extraction only)
  B  fixed documented coordination rule (same tools/limits as C)
  C  RL coordinator: deployed policy (artifacts/policies/coordinator.json) and,
     for variance, every final training seed
  ref_max_budget  extra reference: always spend the full budget (self-consistency + PoT)

Outputs (results/eval/<split>/):
  logs_<method>.jsonl     one structured record per problem
  metrics.json / metrics.csv
  comparisons.json        paired bootstrap CI + exact McNemar
Usage:
  python scripts/evaluate.py --split test_small
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ENV_CONFIG, REWARD_CONFIG  # noqa: E402
from src.coordinator.fixed_policy import FixedPolicy, MaxBudgetPolicy, SingleAttemptPolicy  # noqa: E402
from src.coordinator.q_learning import QLearningCoordinator  # noqa: E402
from src.environment.sources import CachedSource, load_cache  # noqa: E402
from src.evaluation.metrics import mcnemar_exact, paired_bootstrap_diff, summarize  # noqa: E402
from src.evaluation.runner import run_episode  # noqa: E402

DATA = ROOT / "data" / "processed" / "gsm8k"
CACHE = ROOT / "artifacts" / "cache"
POL = ROOT / "artifacts" / "policies"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_small")
    ap.add_argument("--cache", default=None,
                    help="cache file name (default: same as split). test_small is a subset of test, so "
                         "--cache test reuses exactly the same generations for the subset.")
    args = ap.parse_args(argv)
    out = ROOT / "results" / "eval" / args.split
    out.mkdir(parents=True, exist_ok=True)

    refs = {json.loads(l)["id"]: json.loads(l)["normalized_final_answer"]
            for l in open(DATA / f"{args.split}.jsonl", encoding="utf-8")}
    cache = load_cache(CACHE / f"{args.cache or args.split}.jsonl")
    missing = sorted(set(refs) - set(cache))
    if missing:
        raise SystemExit(f"{len(missing)} problems of {args.split} are not generated yet; run generate_attempts.py")
    pids = sorted(refs)

    policies = [("A_single_attempt", SingleAttemptPolicy(), None), ("B_fixed_rule", FixedPolicy(), None),
                ("C_rl", QLearningCoordinator.load(POL / "coordinator.json"), None),
                ("ref_max_budget", MaxBudgetPolicy(), None)]
    for p in sorted(POL.glob("coordinator_seed*.json")):
        seed = int(p.stem.replace("coordinator_seed", ""))
        policies.append((f"C_rl_seed{seed}", QLearningCoordinator.load(p), seed))

    results, correct = {}, {}
    for name, pol, seed in policies:
        logs = [run_episode(pol, CachedSource(cache[pid]), env_cfg=ENV_CONFIG, reward_cfg=REWARD_CONFIG,
                            reference=refs[pid], method=name, seed=seed, problem_id=pid) for pid in pids]
        with open(out / f"logs_{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for l in logs:
                fh.write(json.dumps(l, ensure_ascii=False) + "\n")
        s = summarize(logs)
        s["fallback_decision_rate"] = (mean(d["fallback"] for l in logs for d in l["decisions"])
                                       if name.startswith("C_rl") else None)
        s["action_counts"] = {a: sum(l["actions"].count(a) for l in logs) for a in ("ACCEPT", "REGENERATE", "VERIFY_MORE")}
        results[name] = s
        correct[name] = [bool(l["correctness"]) for l in logs]
        print(f"{name:18s} acc={s['accuracy']:.3f} [{s['accuracy_ci95_wilson'][0]:.3f},{s['accuracy_ci95_wilson'][1]:.3f}]"
              f" attempts={s['mean_attempts']:.2f} pot={s['mean_pot_calls']:.2f} calls={s['mean_model_calls']:.2f}"
              f" out_tok={s['mean_output_tokens']:.0f} return={s['mean_return']:+.3f}")

    seed_names = [n for n in results if n.startswith("C_rl_seed")]
    if seed_names:
        results["C_rl_across_seeds"] = {
            "n_seeds": len(seed_names),
            "accuracy_mean": mean(results[n]["accuracy"] for n in seed_names),
            "accuracy_std": pstdev(results[n]["accuracy"] for n in seed_names),
            "mean_model_calls_mean": mean(results[n]["mean_model_calls"] for n in seed_names),
            "mean_return_mean": mean(results[n]["mean_return"] for n in seed_names),
        }
    comps = {}
    for a, b in [("A_single_attempt", "B_fixed_rule"), ("A_single_attempt", "C_rl"), ("B_fixed_rule", "C_rl"),
                 ("C_rl", "ref_max_budget"), ("B_fixed_rule", "ref_max_budget")]:
        comps[f"{b} - {a}"] = {"accuracy_diff_bootstrap": paired_bootstrap_diff(correct[a], correct[b]),
                               "mcnemar_exact": mcnemar_exact(correct[a], correct[b]),
                               "model_calls_diff": results[b]["mean_model_calls"] - results[a]["mean_model_calls"]}
    meta = {"split": args.split, "cache": args.cache or args.split, "n_problems": len(pids), "env": ENV_CONFIG.__dict__, "reward": REWARD_CONFIG.to_dict(),
            "deployed_policy_meta": json.loads((POL / "coordinator.json").read_text(encoding="utf-8"))["extra"].get("grid_row"),
            "latency_note": "latency_s = amortized batched GPU generation time recorded when the cache was built "
                            "(batch wall time / sequences in batch); cache read time is never counted.",
            "generation_mode": "cached replay of pre-generated attempts"}
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": results, "comparisons": comps}, indent=1),
                                      encoding="utf-8")
    (out / "comparisons.json").write_text(json.dumps(comps, indent=1), encoding="utf-8")
    cols = ["n", "accuracy", "accuracy_ci95_wilson", "coverage", "accuracy_on_answered", "no_answer_rate",
            "budget_exhausted_rate", "mean_attempts", "mean_verifier_calls", "mean_pot_calls", "mean_model_calls",
            "mean_input_tokens", "mean_output_tokens", "mean_latency_s", "mean_return",
            "first_attempt_unextractable_rate", "fallback_decision_rate"]
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + cols)
        for name, s in results.items():
            if "n" in s:
                w.writerow([name] + [json.dumps(s.get(c)) if isinstance(s.get(c), list) else s.get(c) for c in cols])
    for k, v in comps.items():
        d = v["accuracy_diff_bootstrap"]
        print(f"{k:34s} diff={d['diff']:+.3f} CI95=[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] "
              f"McNemar p={v['mcnemar_exact']['p_value']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
