"""Web server behaviour with the DEMO solver (no Qwen, no GPU)."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from web.server import build_server


@pytest.fixture(scope="module")
def base_url():
    httpd, holder = build_server(port=0, demo=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    t0 = time.time()
    while holder.solver is None and time.time() - t0 < 10:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def _post(url, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_static_and_health(base_url):
    s, ctype, body = _get(base_url + "/")
    assert s == 200 and "text/html" in ctype and b"MathMates" in body
    for f in ("app.js", "styles.css", "i18n.js", "mascots.js", "background.js", "vendor/katex/katex.min.js"):
        assert _get(f"{base_url}/{f}")[0] == 200, f
    s, _, body = _get(base_url + "/api/health")
    h = json.loads(body)
    assert s == 200 and h["status"] == "ready" and h["demo"] is True


def test_path_traversal_and_unknown_figures_are_blocked(base_url):
    assert _get(base_url + "/../src/config.py")[0] == 404
    assert _get(base_url + "/%2e%2e/src/config.py")[0] == 404
    assert _get(base_url + "/vendor/../../server.py")[0] == 404
    assert _get(base_url + "/figures/../metrics.json")[0] == 404
    assert _get(base_url + "/figures/not_a_figure.png")[0] == 404


def test_results_endpoint_reads_saved_metrics(base_url):
    s, _, body = _get(base_url + "/api/results")
    r = json.loads(body)
    saved = json.load(open("results/eval/test/metrics.json", encoding="utf-8"))
    assert s == 200
    for k in ("A_single_attempt", "B_fixed_rule", "C_rl"):
        assert r["test"]["methods"][k]["accuracy"] == saved["metrics"][k]["accuracy"]
    assert r["test"]["n_problems"] == 1319


@pytest.mark.parametrize("method", ["single", "fixed", "rl", "max_budget"])
def test_solve_returns_answer_and_display_fields(base_url, method):
    s, r = _post(base_url + "/api/solve", {"question": "Sara has 5 boxes...", "method": method, "budget": 4})
    assert s == 200 and r["demo"] is True
    assert r["actions"][-1] == "ACCEPT" and len(r["explanations"]) == len(r["actions"])
    shown = r["display"]["shown_attempt"]
    assert r["attempt_answers"][shown] == r["final_answer"]
    assert len(r["display"]["arithmetic"]) == len(r["attempt_texts"])
    for d in r["explanations"]:
        assert "candidate" in d and "rule" in d


def test_solve_flags_the_arithmetic_slip(base_url):
    s, r = _post(base_url + "/api/solve", {"question": "q", "method": "single", "budget": 1})
    assert s == 200 and r["final_answer"] == "42"          # demo greedy attempt is wrong on purpose
    details = r["display"]["arithmetic"][0]
    assert any(not d["consistent"] and d["lhs_value"] == "43" for d in details)


@pytest.mark.parametrize("body,code,err", [
    ({"question": "", "method": "rl"}, 400, "empty_question"),
    ({"question": "x" * 1001}, 400, "question_too_long"),
    ({"question": "q", "method": "nope"}, 400, "bad_method"),
    ({"question": "q", "budget": 0}, 400, "bad_budget"),
    ({"question": "q", "budget": 9}, 400, "bad_budget"),
])
def test_solve_rejects_bad_input(base_url, body, code, err):
    s, r = _post(base_url + "/api/solve", body)
    assert s == code and r["error"] == err


def test_solve_rejects_malformed_json(base_url):
    s, r = _post(base_url + "/api/solve", None, raw=b"{not json")
    assert s == 400 and r["error"] == "bad_json"


def test_busy_returns_429(base_url, monkeypatch):
    import web.server as ws
    monkeypatch.setattr(ws, "BUSY_WAIT_S", 0.05)
    # a separate server whose GPU lock is already held
    httpd, holder = build_server(port=0, demo=True, preload=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    while holder.solver is None:
        time.sleep(0.02)
    holder.gpu_lock.acquire()
    try:
        s, r = _post(f"http://127.0.0.1:{httpd.server_address[1]}/api/solve", {"question": "q"})
        assert s == 429 and r["error"] == "busy"
    finally:
        holder.gpu_lock.release()
        httpd.shutdown()
        httpd.server_close()
