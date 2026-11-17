"""Start the local web app.

    .venv\\Scripts\\python.exe scripts\\run_web.py              # real model (GPU), http://127.0.0.1:8000
    .venv\\Scripts\\python.exe scripts\\run_web.py --demo       # UI preview with canned text, no model
    .venv\\Scripts\\python.exe scripts\\run_web.py --port 8080 --no-browser
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the model is already in the local Hugging Face cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default=os.environ.get("MM_HOST", "127.0.0.1"),
                    help="127.0.0.1 = this computer only; 0.0.0.0 = inside the Hugging Face Space container")
    ap.add_argument("--demo", action="store_true", help="canned text, no model (UI preview only)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from web.server import build_server
    httpd, _ = build_server(port=args.port, demo=args.demo, host=args.host)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"MathMates is running at {url}  ({'DEMO mode' if args.demo else 'loading Qwen in the background'})")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
