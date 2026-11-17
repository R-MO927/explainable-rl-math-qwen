"""Assemble the Hugging Face Space folder (build/hf_space) with only what the web app needs.

    .venv\\Scripts\\python.exe scripts\\build_hf_space.py

The folder is then pushed to the Space's own git repository. Nothing here changes results:
the Space reads the same saved metrics, figures and trained policy as the local app.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf_space"

COPY = [
    "src",
    "web",
    "scripts/run_web.py",
    "artifacts/policies/coordinator.json",
    "results/eval/test/metrics.json",
    "results/eval/test_small/metrics.json",
    "results/figures/accuracy_vs_cost_test.png",
    "results/figures/accuracy_vs_cost_test_small.png",
    "results/figures/actions_test.png",
    "results/figures/actions_test_small.png",
    "results/figures/learning_curves.png",
    "results/figures/policy_qvalues.png",
]
SPACE_FILES = ["Dockerfile", "README.md", "requirements.txt"]
GITATTRIBUTES = "*.png filter=lfs diff=lfs merge=lfs -text\n*.woff2 filter=lfs diff=lfs merge=lfs -text\n"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "demo_solver.py.bak")


def main():
    if OUT.exists():
        # keep the Space's .git folder (if this folder is already a clone of the Space)
        for child in OUT.iterdir():
            if child.name == ".git":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    for rel in COPY:
        src, dst = ROOT / rel, OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
    for name in SPACE_FILES:
        shutil.copy2(ROOT / "deploy" / "hf_space" / name, OUT / name)
    (OUT / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8", newline="\n")
    n = sum(1 for p in OUT.rglob("*") if p.is_file() and ".git" not in p.parts)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"[ok] {OUT}  ({n} files, {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
