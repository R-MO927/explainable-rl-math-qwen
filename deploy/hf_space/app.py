"""Hugging Face Space entry point (Gradio SDK, ZeroGPU hardware).

Runs the same web app as scripts/run_web.py. ZeroGPU assigns a GPU only while a
function decorated with @spaces.GPU runs, so each solve request goes through
`gpu_solve`. The model weights are loaded once at startup (pinned revision).
"""
import os

os.environ["HF_HUB_OFFLINE"] = "0"          # first start downloads the pinned model revision
os.environ.setdefault("MM_SITE", "space")
os.environ.setdefault("MM_BUSY_WAIT_S", "240")

import spaces  # noqa: E402  (must be imported before torch on ZeroGPU)

from src.pipeline import get_solver, solve_problem  # noqa: E402
from web.server import build_server  # noqa: E402

SOLVER = get_solver()


@spaces.GPU(duration=120)
def gpu_solve(question, method="rl", budget=4, seed=0, solver=None):
    return solve_problem(question, method=method, budget=budget, seed=seed, solver=SOLVER)


def zerogpu_ready():
    """ZeroGPU normally runs its startup step when Gradio's server starts. This app serves the page
    with its own small HTTP server, so it runs the same step itself (spaces.zero.lifecycle API, read from spaces 0.52.1):
    it registers the @spaces.GPU function and packs the loaded weights for the GPU workers."""
    try:
        from spaces.zero.lifecycle import zerogpu_mode_enabled, zerogpu_startup
    except ImportError:          # local runs without the real `spaces` package
        return
    if zerogpu_mode_enabled():
        zerogpu_startup()
        print("ZeroGPU startup done", flush=True)


if __name__ == "__main__":
    zerogpu_ready()
    port = int(os.environ.get("PORT", "7860"))
    httpd, _ = build_server(port=port, host="0.0.0.0", solver_factory=lambda: SOLVER, solve_fn=gpu_solve)
    print(f"MathMates Space listening on 0.0.0.0:{port}", flush=True)
    httpd.serve_forever()
