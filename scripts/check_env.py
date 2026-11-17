"""Record the execution environment to artifacts/environment.json (no installs, read-only).

Usage: python scripts/check_env.py
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception:
        return None


def total_ram_gb() -> float | None:
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return {"total_gb": round(m.ullTotalPhys / 2**30, 1), "available_gb_at_check": round(m.ullAvailPhys / 2**30, 1)}
    return None


def main() -> int:
    du = shutil.disk_usage(ROOT)
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    smi = run(["nvidia-smi"]) or ""
    cuda = next((ln.split("CUDA Version:")[1].strip(" |") for ln in smi.splitlines() if "CUDA Version:" in ln), None)
    pkgs = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    env = {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "in_venv": sys.prefix != sys.base_prefix,
        "ram": total_ram_gb(),
        "disk_free_gb": round(du.free / 2**30, 1),
        "gpu": gpu,
        "cuda_driver_max_version": cuda,
        "git": run(["git", "--version"]),
        "key_packages": {k: pkgs.get(k) for k in ["pytest", "ipykernel", "torch", "transformers", "numpy"]},
    }
    out = ROOT / "artifacts" / "environment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
