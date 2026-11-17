"""Train the tabular Q-learning coordinator on TRAIN caches; select settings on VALIDATION only.

1. Grid search: every config is trained with `--grid-seeds` seeds on train and
   evaluated greedily on validation; the config with the highest mean
   validation return (the objective the reward defines) is selected.
2. Final training: the selected config is trained with `--final-seeds` seeds.
   Each policy is saved; the seed with the best validation return becomes
   artifacts/policies/coordinator.json (used by evaluation and solve_problem).
Test data is never loaded here.

Usage:
  python scripts/train_coordinator.py --train-split train_small --val-split validation_small
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ENV_CONFIG, REWARD_CONFIG  # noqa: E402
from src.coordinator.fixed_policy import FixedPolicy, MaxBudgetPolicy, SingleAttemptPolicy  # noqa: E402
from src.coordinator.q_learning import QConfig  # noqa: E402
from src.coordinator.training import train_q_learning  # noqa: E402
from src.environment.sources import CachedSource, load_cache  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.evaluation.runner import run_episode  # noqa: E402

DATA = ROOT / "data" / "processed" / "gsm8k"
CACHE = ROOT / "artifacts" / "cache"
POL = ROOT / "artifacts" / "policies"
RES = ROOT / "results" / "training"

GRID = {"alpha": [0.05, 0.1, 0.2], "episodes": [10000, 30000, 100000], "epsilon_end": [0.05, 0.2]}
# Second grid block (added after the first run showed large seed-to-seed instability on validation):
# visit-count step size 1/n(s,a); alpha is unused there.
GRID_VISIT = {"episodes": [10000, 30000, 100000], "epsilon_end": [0.05, 0.2]}


def grid_configs() -> list[dict]:
    rows = [{"alpha_mode": "constant", "alpha": a, "episodes": e, "epsilon_end": x}
            for a, e, x in itertools.product(GRID["alpha"], GRID["episodes"], GRID["epsilon_end"])]
    rows += [{"alpha_mode": "visit", "alpha": None, "episodes": e, "epsilon_end": x}
             for e, x in itertools.product(GRID_VISIT["episodes"], GRID_VISIT["epsilon_end"])]
    return rows


def qcfg(row: dict, seed: int) -> QConfig:
    return QConfig(alpha=row["alpha"] if row["alpha"] is not None else 0.0, episodes=row["episodes"],
                   epsilon_end=row["epsilon_end"], seed=seed, alpha_mode=row["alpha_mode"])


def load_problems(split: str) -> list[tuple[str, CachedSource, str]]:
    if split.startswith("test"):
        raise SystemExit("refusing to use test data for training/selection")
    refs = {}
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            refs[r["id"]] = r["normalized_final_answer"]
    cache = load_cache(CACHE / f"{split}.jsonl")
    missing = [pid for pid in refs if pid not in cache]
    if missing:
        print(f"[warn] {len(missing)} {split} problems not in cache; using the {len(cache)} cached ones")
    return [(pid, CachedSource(cache[pid]), refs[pid]) for pid in sorted(cache) if pid in refs]


def evaluate(policy, problems, method, seed=None, explanations=False) -> tuple[dict, list[dict]]:
    logs = [run_episode(policy, src, env_cfg=ENV_CONFIG, reward_cfg=REWARD_CONFIG, reference=ref, method=method,
                        seed=seed, problem_id=pid, with_explanations=explanations) for pid, src, ref in problems]
    return summarize(logs), logs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-split", default="train_small")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--grid-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=5)
    args = ap.parse_args(argv)
    RES.mkdir(parents=True, exist_ok=True)
    POL.mkdir(parents=True, exist_ok=True)

    train = load_problems(args.train_split)
    val = load_problems(args.val_split)
    print(f"[train] train problems={len(train)} validation problems={len(val)}")

    # ---------------- reference policies on validation (for context only)
    refs = {}
    for pol, name in [(SingleAttemptPolicy(), "A_single_attempt"), (FixedPolicy(), "B_fixed_rule"),
                      (MaxBudgetPolicy(), "ref_max_budget")]:
        s, logs = evaluate(pol, val, name)
        refs[name] = s
        with open(RES / f"val_logs_{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for l in logs:
                fh.write(json.dumps(l, ensure_ascii=False) + "\n")
        print(f"[val] {name:18s} acc={s['accuracy']:.3f} return={s['mean_return']:+.3f} "
              f"attempts={s['mean_attempts']:.2f} pot={s['mean_pot_calls']:.2f}")

    # ---------------- grid search on validation
    rows = []
    t0 = time.perf_counter()
    for g in grid_configs():
        rets, accs, atts = [], [], []
        for seed in range(args.grid_seeds):
            agent, _ = train_q_learning(train, q_cfg=qcfg(g, 1000 + seed), env_cfg=ENV_CONFIG,
                                        reward_cfg=REWARD_CONFIG, log_every=g["episodes"])
            s, _ = evaluate(agent, val, "C_rl")
            rets.append(s["mean_return"])
            accs.append(s["accuracy"])
            atts.append(s["mean_attempts"])
        row = {**g, "val_return_mean": mean(rets), "val_return_std": pstdev(rets), "val_return_min": min(rets),
               "val_accuracy_mean": mean(accs), "val_attempts_mean": mean(atts)}
        rows.append(row)
        print(f"[grid] mode={g['alpha_mode']} a={g['alpha']} ep={g['episodes']} eps_end={g['epsilon_end']}: "
              f"return={row['val_return_mean']:+.4f} sd={row['val_return_std']:.4f} acc={row['val_accuracy_mean']:.3f} "
              f"attempts={row['val_attempts_mean']:.2f}")
    with open(RES / "grid_search.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: (round(r["val_return_mean"], 6), r["val_accuracy_mean"], -r["episodes"]))
    print(f"[grid] selected on validation: {best}  ({time.perf_counter() - t0:.0f}s)")

    # ---------------- final training with several seeds
    finals = []
    for seed in range(args.final_seeds):
        cfg = qcfg(best, seed)

        def eval_fn(agent):
            s, _ = evaluate(agent, val, "C_rl")
            return {"val_return": s["mean_return"], "val_accuracy": s["accuracy"], "val_attempts": s["mean_attempts"]}

        agent, curve = train_q_learning(train, q_cfg=cfg, env_cfg=ENV_CONFIG, reward_cfg=REWARD_CONFIG,
                                        log_every=max(1, cfg.episodes // 20), eval_fn=eval_fn)
        s_val, val_logs = evaluate(agent, val, f"C_rl_seed{seed}", seed=seed, explanations=True)
        s_tr, tr_logs = evaluate(agent, train, f"C_rl_seed{seed}", seed=seed)
        meta = {"train_split": args.train_split, "val_split": args.val_split, "selected_by": "validation mean return",
                "grid_row": best, "env": ENV_CONFIG.__dict__, "reward": REWARD_CONFIG.to_dict(),
                "validation_summary": s_val, "train_summary": s_tr}
        agent.save(POL / f"coordinator_seed{seed}.json", extra=meta)
        (RES / f"curve_seed{seed}.json").write_text(json.dumps(curve, indent=1), encoding="utf-8")
        for name, logs in (("val", val_logs), ("train", tr_logs)):
            with open(RES / f"{name}_logs_C_rl_seed{seed}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
                for l in logs:
                    fh.write(json.dumps(l, ensure_ascii=False) + "\n")
        finals.append({"seed": seed, "val": s_val, "train": s_tr, "n_states": len(agent.q)})
        print(f"[final] seed={seed} val acc={s_val['accuracy']:.3f} return={s_val['mean_return']:+.4f} "
              f"attempts={s_val['mean_attempts']:.2f} pot={s_val['mean_pot_calls']:.2f} "
              f"fallback_decisions={sum(d['fallback'] for l in val_logs for d in l['decisions'])}")
    chosen = max(finals, key=lambda f: (f["val"]["mean_return"], f["val"]["accuracy"], -f["seed"]))
    (POL / "coordinator.json").write_text((POL / f"coordinator_seed{chosen['seed']}.json").read_text(encoding="utf-8"),
                                          encoding="utf-8")
    summary = {"reference_policies_on_validation": refs, "grid": rows, "selected_config": best,
               "final_seeds": finals, "deployed_seed": chosen["seed"],
               "deployed_selection": "highest validation mean return among final seeds"}
    (RES / "training_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[done] deployed policy = seed {chosen['seed']} -> artifacts/policies/coordinator.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
