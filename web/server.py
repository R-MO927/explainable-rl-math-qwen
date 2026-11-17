"""Local web server for the project (standard library only).

    .venv\\Scripts\\python.exe scripts\\run_web.py            # real Qwen on the GPU
    .venv\\Scripts\\python.exe scripts\\run_web.py --demo     # canned demo text, no model (UI preview only)

Endpoints
    GET  /                    the single-page app (web/static)
    GET  /api/health          model status
    POST /api/solve           {"question", "method", "budget"} -> solve_problem() result + display extras
    GET  /api/results         real metrics read from results/eval/*/metrics.json (nothing recomputed)
    GET  /figures/<name>.png  figures from results/figures (whitelisted names only)

Binds to 127.0.0.1 only. One GPU request runs at a time; a second request waits
up to BUSY_WAIT_S and then gets HTTP 429.
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.verifier.arithmetic import check_arithmetic

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
FIGURES = ROOT / "results" / "figures"
EVAL = ROOT / "results" / "eval"

WEB_METHODS = ("single", "fixed", "rl", "max_budget")
MAX_QUESTION_CHARS = 1000
MAX_BODY_BYTES = 16_000
BUSY_WAIT_S = 2.0
FIGURE_NAMES = {"accuracy_vs_cost_test.png", "accuracy_vs_cost_test_small.png", "actions_test.png",
                "actions_test_small.png", "learning_curves.png", "policy_qvalues.png"}
RESULT_KEYS = ("A_single_attempt", "B_fixed_rule", "C_rl", "ref_max_budget")
METRIC_FIELDS = ("n", "accuracy", "accuracy_ci95_wilson", "n_correct", "coverage", "no_answer_rate",
                 "mean_attempts", "mean_pot_calls", "mean_model_calls", "mean_output_tokens", "mean_return",
                 "fallback_decision_rate")


class ModelHolder:
    """Loads the solver in a background thread so the page is usable immediately."""

    def __init__(self, solver_factory, demo: bool):
        self.factory, self.demo = solver_factory, demo
        self.solver, self.error, self.loading = None, None, False
        self.gpu_lock = threading.Lock()
        self._load_lock = threading.Lock()

    def start_loading(self):
        with self._load_lock:
            if self.solver is not None or self.loading:
                return
            self.loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.solver = self.factory()
        except Exception as e:  # reported via /api/health, never hidden
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.loading = False

    def status(self) -> dict:
        state = "ready" if self.solver is not None else ("error" if self.error else "loading")
        info = None
        if self.solver is not None:
            try:
                info = self.solver.info()
            except Exception:
                info = None
        return {"status": state, "demo": self.demo, "error": self.error, "model": info}


def enrich(result: dict) -> dict:
    """Adds display-only fields derived from the solver texts. Nothing here changes the answer."""
    answers = result["attempt_answers"]
    final = result["final_answer"]
    shown = next((i for i, a in enumerate(answers) if final is not None and a == final), 0 if answers else None)
    result["display"] = {
        "shown_attempt": shown,
        "arithmetic": [check_arithmetic(t, max_details=30).details for t in result["attempt_texts"]],
    }
    return result


def load_results() -> dict:
    out = {}
    for split in ("test", "test_small"):
        p = EVAL / split / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        out[split] = {
            "n_problems": m["meta"]["n_problems"],
            "generation_mode": m["meta"]["generation_mode"],
            "reward": m["meta"]["reward"],
            "methods": {k: {f: m["metrics"][k].get(f) for f in METRIC_FIELDS} for k in RESULT_KEYS},
            "seeds": m["metrics"].get("C_rl_across_seeds"),
            "seed_runs": {k: {"accuracy": v["accuracy"], "mean_model_calls": v["mean_model_calls"]}
                          for k, v in m["metrics"].items() if k.startswith("C_rl_seed")},
            "comparisons": {k: {"diff": c["accuracy_diff_bootstrap"]["diff"],
                                "ci95": c["accuracy_diff_bootstrap"]["ci95"],
                                "p_value": c["mcnemar_exact"]["p_value"]}
                            for k, c in m["comparisons"].items()},
        }
    return out


def make_handler(holder: ModelHolder, solve_fn):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MathMates/1.0"

        def log_message(self, fmt, *args):  # quieter console: one line per API call
            if self.path.startswith("/api/"):
                print(f"[web] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

        # ---------- helpers ----------
        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _file(self, path: Path, cache: bool = False):
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
                ctype += "; charset=utf-8"
            self._send(200, path.read_bytes(), ctype, cache)

        # ---------- routes ----------
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/health":
                return self._json(200, holder.status())
            if path == "/api/results":
                return self._json(200, load_results())
            if path.startswith("/figures/"):
                name = path[len("/figures/"):]
                if name in FIGURE_NAMES and (FIGURES / name).is_file():
                    return self._file(FIGURES / name)
                return self._json(404, {"error": "not_found"})
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC / rel).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                return self._json(404, {"error": "not_found"})
            return self._file(target, cache="/vendor/" in path)

        def do_POST(self):
            if urlparse(self.path).path != "/api/solve":
                return self._json(404, {"error": "not_found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if n <= 0 or n > MAX_BODY_BYTES:
                return self._json(413 if n > MAX_BODY_BYTES else 400, {"error": "bad_body"})
            try:
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                question = str(req.get("question", "")).strip()
                method = str(req.get("method", "rl"))
                budget = int(req.get("budget", 4))
            except (ValueError, AttributeError, UnicodeDecodeError):
                return self._json(400, {"error": "bad_json"})
            if not question:
                return self._json(400, {"error": "empty_question"})
            if len(question) > MAX_QUESTION_CHARS:
                return self._json(400, {"error": "question_too_long", "max_chars": MAX_QUESTION_CHARS})
            if method not in WEB_METHODS:
                return self._json(400, {"error": "bad_method", "allowed": list(WEB_METHODS)})
            if not 1 <= budget <= 4:
                return self._json(400, {"error": "bad_budget"})
            if holder.solver is None:
                holder.start_loading()
                return self._json(503, {**holder.status(), "load_error": holder.error, "error": "model_not_ready"})
            if not holder.gpu_lock.acquire(timeout=BUSY_WAIT_S):
                return self._json(429, {"error": "busy"})
            try:
                t0 = time.perf_counter()
                result = solve_fn(question, method=method, budget=budget, seed=0, solver=holder.solver)
                result = enrich(result)
                result["demo"] = holder.demo
                result["server_time_s"] = round(time.perf_counter() - t0, 2)
            except ValueError as e:
                return self._json(400, {"error": "invalid", "detail": str(e)})
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {"error": "solver_failed", "detail": f"{type(e).__name__}: {e}"})
            finally:
                holder.gpu_lock.release()
            return self._json(200, result)

    return Handler


def build_server(port: int = 8000, demo: bool = False, solver_factory=None, solve_fn=None,
                 host: str = "127.0.0.1", preload: bool = True) -> tuple[ThreadingHTTPServer, ModelHolder]:
    if solve_fn is None:
        from src.pipeline import solve_problem as solve_fn
    if solver_factory is None:
        if demo:
            from web.demo_solver import DemoSolver as solver_factory
        else:
            from src.pipeline import get_solver as solver_factory
    holder = ModelHolder(solver_factory, demo)
    if preload:
        holder.start_loading()
    httpd = ThreadingHTTPServer((host, port), make_handler(holder, solve_fn))
    httpd.daemon_threads = True
    return httpd, holder
