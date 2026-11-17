"""Static figures from real result files (no synthetic data).

  results/figures/accuracy_vs_cost_<split>.png   accuracy (Wilson 95% CI) vs mean model calls
  results/figures/learning_curves.png            train/validation return during Q-learning
  results/figures/actions_<split>.png            action counts per method
  results/figures/policy_qvalues.png             Q-values of the deployed policy, most visited states
Numbers are also available as CSV/JSON in results/ (table view).

Usage: python scripts/make_plots.py --split test_small
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coordinator.features import describe_state, str_to_key  # noqa: E402

FIG = ROOT / "results" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"A_single_attempt": "#2a78d6", "B_fixed_rule": "#eb6834", "C_rl": "#1baf7a", "ref_max_budget": "#8a8984"}
MARKERS = {"A_single_attempt": "o", "B_fixed_rule": "s", "C_rl": "D", "ref_max_budget": "^"}
# label offset (points) and alignment per method, chosen to avoid the CI bars of neighbours
LABEL_POS = {"A_single_attempt": (10, -22, "left"), "B_fixed_rule": (-12, 0, "right"),
             "C_rl": (12, 0, "left"), "ref_max_budget": (12, 0, "left")}
LABELS = {"A_single_attempt": "A: single attempt", "B_fixed_rule": "B: fixed rule", "C_rl": "C: RL (deployed)",
          "ref_max_budget": "ref: max budget"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig(w=7.5, h=4.5):
    f, ax = plt.subplots(figsize=(w, h), dpi=150)
    f.patch.set_facecolor(SURFACE)
    return f, ax


def accuracy_vs_cost(split: str) -> None:
    m = json.loads((ROOT / "results" / "eval" / split / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    f, ax = fig()
    for name, s in m.items():
        if name.startswith("C_rl_seed"):
            ax.plot(s["mean_model_calls"], s["accuracy"], "D", ms=5, color=COLORS["C_rl"], alpha=0.35, zorder=2)
    for name in COLORS:
        s = m[name]
        lo, hi = s["accuracy_ci95_wilson"]
        ax.errorbar(s["mean_model_calls"], s["accuracy"], yerr=[[s["accuracy"] - lo], [hi - s["accuracy"]]],
                    fmt=MARKERS[name], ms=9, color=COLORS[name], ecolor=COLORS[name], elinewidth=2, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        dx, dy, ha = LABEL_POS[name]
        ax.annotate(f"{LABELS[name]}  {s['accuracy']:.1%}", (s["mean_model_calls"], s["accuracy"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=INK, fontsize=9,
                    zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
    style(ax, f"Accuracy vs cost on {split} (n={m['A_single_attempt']['n']}; bars = Wilson 95% CI;"
              " faint = RL seeds)", "mean model calls per problem (solver attempts + PoT)", "accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    xs = [m[n]["mean_model_calls"] for n in COLORS]
    ax.set_xlim(0.8, max(xs) + 1.2)
    f.tight_layout()
    f.savefig(FIG / f"accuracy_vs_cost_{split}.png")
    plt.close(f)


def learning_curves() -> None:
    curves = sorted((ROOT / "results" / "training").glob("curve_seed*.json"))
    if not curves:
        return
    f, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, sharey=True)
    f.patch.set_facecolor(SURFACE)
    for i, p in enumerate(curves):
        c = json.loads(p.read_text(encoding="utf-8"))
        ep = [r["episode"] for r in c]
        lab = "per seed" if i == 0 else None
        axes[0].plot(ep, [r["train_return_ma"] for r in c], color="#2a78d6", lw=2, alpha=0.55, label=lab)
        axes[1].plot(ep, [r["val_return"] for r in c], color="#1baf7a", lw=2, alpha=0.55, label=lab)
    tr = json.loads((ROOT / "results" / "training" / "training_summary.json").read_text(encoding="utf-8"))
    ref = tr["reference_policies_on_validation"]
    for name, ls in (("B_fixed_rule", "--"), ("A_single_attempt", ":")):
        axes[1].axhline(ref[name]["mean_return"], color=INK2, lw=1.2, ls=ls)
        axes[1].annotate(LABELS[name], (ep[-1], ref[name]["mean_return"]), xytext=(-4, 4),
                         textcoords="offset points", ha="right", color=INK2, fontsize=8)
    style(axes[0], "Training return (moving avg, with exploration)", "episode", "mean return per episode")
    style(axes[1], "Validation return (greedy policy)", "episode", "")
    f.tight_layout()
    f.savefig(FIG / "learning_curves.png")
    plt.close(f)


def actions(split: str) -> None:
    m = json.loads((ROOT / "results" / "eval" / split / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    names = list(COLORS)
    acts = ("REGENERATE", "VERIFY_MORE")
    f, ax = fig(7.5, 3.6)
    n = m[names[0]]["n"]
    width = 0.36
    for j, a in enumerate(acts):
        vals = [m[k]["action_counts"][a] / n for k in names]
        xs = [i + (j - 0.5) * (width + 0.02) for i in range(len(names))]
        ax.bar(xs, vals, width=width, color=["#6250d6", "#eda100"][j], label=a, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        color=INK)
    ax.set_xticks(range(len(names)), [LABELS[k] for k in names])
    style(ax, f"Extra actions per problem on {split}", "", "mean count per problem")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    f.tight_layout()
    f.savefig(FIG / f"actions_{split}.png")
    plt.close(f)


def policy_q() -> None:
    p = json.loads((ROOT / "artifacts" / "policies" / "coordinator.json").read_text(encoding="utf-8"))
    rows = sorted(p["q"], key=lambda k: -sum(p["visits"][k]))[:14]
    f, ax = fig(9, 5.8)
    names = p["actions"]
    colors = ["#2a78d6", "#6250d6", "#eda100"]
    for i, k in enumerate(rows):
        q, v = p["q"][k], p["visits"][k]
        for j in range(3):
            if v[j] > 0:
                ax.barh(i + (j - 1) * 0.27, q[j], height=0.25, color=colors[j], zorder=3,
                        label=names[j] if i == 0 else None)
    labels = []
    for k in rows:
        d = describe_state(str_to_key(k))
        labels.append(f"n={d['n_attempts']} sup={d['support']} dis={int(d['disagreement'])} "
                      f"arith={d['candidate_arithmetic']} pot={d['pot_check']}")
    ax.set_yticks(range(len(rows)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    style(ax, "Deployed policy: Q-values, 14 most visited states\n(actions shown only if tried in training)",
          "Q (expected return under the designed reward; not a probability)", "")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower right")
    f.tight_layout()
    f.savefig(FIG / "policy_qvalues.png")
    plt.close(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_small")
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    learning_curves()
    policy_q()
    if (ROOT / "results" / "eval" / args.split / "metrics.json").exists():
        accuracy_vs_cost(args.split)
        actions(args.split)
    print("figures in", FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
