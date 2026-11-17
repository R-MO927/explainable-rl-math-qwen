"""Download the official GSM8K files from github.com/openai/grade-school-math.

The raw files are fetched from a *pinned commit* (not the moving `master`
branch) so that every re-run downloads byte-identical content. The script
verifies content (UTF-8, one JSON object per line, required keys, record
count) rather than trusting the HTTP status code alone, and writes a
manifest with source URLs, timestamps, SHA-256 hashes and record counts.

Raw files are stored unmodified in data/raw/gsm8k/.

Usage:
    python scripts/download_data.py            # download (skip if hashes match)
    python scripts/download_data.py --force    # re-download
    python scripts/download_data.py --commit <sha>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "gsm8k"
MANIFEST_PATH = RAW_DIR / "download_manifest.json"

REPO = "openai/grade-school-math"
REPO_URL = f"https://github.com/{REPO}"
# Commit of `master` verified with `git ls-remote` on 2026-11-17.
DEFAULT_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"

DATA_FILES = {
    "train.jsonl": "grade_school_math/data/train.jsonl",
    "test.jsonl": "grade_school_math/data/test.jsonl",
}
# Downloaded only so the license / usage terms are documented from the source.
DOC_FILES = {
    "LICENSE": "LICENSE",
    "README.md": "README.md",
}
EXPECTED_KEYS = {"question", "answer"}
USER_AGENT = "gsm8k-data-downloader/1.0"


def raw_url(commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        body = resp.read()
        declared = resp.headers.get("Content-Length")
    if status != 200:
        raise RuntimeError(f"HTTP {status} for {url}")
    if declared is not None and int(declared) != len(body):
        raise RuntimeError(
            f"Truncated download for {url}: got {len(body)} bytes, "
            f"Content-Length={declared}"
        )
    return body


def resolve_branch_head(branch: str = "master") -> str | None:
    """Return the current commit of `branch` via the GitHub API (informational)."""
    url = f"https://api.github.com/repos/{REPO}/commits/{branch}"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["sha"]
    except Exception as exc:  # network/API rate limit: not fatal, pinned commit is used
        print(f"[warn] could not resolve {branch} head: {exc}")
        return None


def validate_jsonl(data: bytes, name: str) -> dict:
    """Content-level checks. Raises ValueError on any structural problem."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: not valid UTF-8: {exc}") from exc
    if text.startswith("﻿"):
        raise ValueError(f"{name}: unexpected BOM")
    lines = text.split("\n")
    trailing_empty = 0
    while lines and lines[-1] == "":
        lines.pop()
        trailing_empty += 1
    n = 0
    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            raise ValueError(f"{name}: blank line inside file at line {i}")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}: invalid JSON at line {i}: {exc}") from exc
        if not isinstance(obj, dict) or not EXPECTED_KEYS.issubset(obj):
            raise ValueError(f"{name}: line {i} missing keys {EXPECTED_KEYS - set(obj)}")
        n += 1
    if n == 0:
        raise ValueError(f"{name}: no records")
    return {
        "records": n,
        "lines_incl_trailing_newline": n + trailing_empty,
        "uses_crlf": "\r\n" in text,
    }


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_manifest() -> dict | None:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return None


def already_downloaded(manifest: dict | None, commit: str) -> bool:
    if not manifest or manifest.get("commit") != commit:
        return False
    for name, info in {**manifest["data_files"], **manifest["doc_files"]}.items():
        p = RAW_DIR / name
        if not p.exists() or sha256_bytes(p.read_bytes()) != info["sha256"]:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    if not args.force and already_downloaded(manifest, args.commit):
        print(f"[ok] raw files already present and hashes match manifest (commit {args.commit[:10]})")
        return 0

    head = resolve_branch_head("master")
    entries: dict[str, dict] = {}
    doc_entries: dict[str, dict] = {}

    for name, repo_path in DATA_FILES.items():
        url = raw_url(args.commit, repo_path)
        print(f"[get] {url}")
        body = fetch(url)
        stats = validate_jsonl(body, name)
        atomic_write(RAW_DIR / name, body)
        entries[name] = {
            "repo_path": repo_path,
            "source_url": url,
            "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(body),
            "sha256": sha256_bytes(body),
            **stats,
        }
        print(f"      {stats['records']} records, sha256={entries[name]['sha256'][:16]}...")

    for name, repo_path in DOC_FILES.items():
        url = raw_url(args.commit, repo_path)
        print(f"[get] {url}")
        body = fetch(url)
        body.decode("utf-8")  # must be text
        atomic_write(RAW_DIR / name, body)
        doc_entries[name] = {
            "repo_path": repo_path,
            "source_url": url,
            "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(body),
            "sha256": sha256_bytes(body),
        }

    license_text = (RAW_DIR / "LICENSE").read_text(encoding="utf-8")
    new_manifest = {
        "dataset": "GSM8K (Grade School Math 8K)",
        "repository": REPO_URL,
        "commit": args.commit,
        "master_head_at_download": head,
        "pinned_commit_equals_master_head": (head == args.commit) if head else None,
        "license_file_first_line": license_text.strip().splitlines()[0],
        "data_files": entries,
        "doc_files": doc_entries,
        "note": "Raw files are stored byte-for-byte as served; no modification.",
    }
    atomic_write(MANIFEST_PATH, (json.dumps(new_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"[ok] manifest written to {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
