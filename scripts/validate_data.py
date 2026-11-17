"""Independent validation of the raw and processed GSM8K data.

Checks are written to avoid re-using the code under test where possible
(e.g. final answers are re-extracted with a separate, simpler method).
Exits with code 1 if any check fails. Writes data/processed/gsm8k/validation_result.json.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --skip-determinism
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gsm8k"
PROC = ROOT / "data" / "processed" / "gsm8k"
FULL = ("train", "validation", "test")
SMALL = {"train_small": "train", "validation_small": "validation", "test_small": "test"}
EXPECTED_SMALL = {"train_small": 500, "validation_small": 100, "test_small": 200}
REQUIRED_FIELDS = [
    "id", "source_split", "source_line", "split", "question", "reference_solution",
    "reference_final_answer", "normalized_final_answer", "question_sha256",
]

results: list[dict] = []


def check(name: str, ok: bool, detail: object = None) -> None:
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail is not None and not ok else ""))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[^0-9a-z]+", "", s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args(argv)

    # ---------------- raw files
    dm = json.loads((RAW / "download_manifest.json").read_text(encoding="utf-8"))
    raw = {}
    for split in ("train", "test"):
        p = RAW / f"{split}.jsonl"
        info = dm["data_files"][f"{split}.jsonl"]
        check(f"raw {split}: sha256 matches download manifest", sha(p) == info["sha256"])
        raw[split] = load(p)
        check(f"raw {split}: record count == manifest ({info['records']})", len(raw[split]) == info["records"])
    check("raw: official sizes 7473 / 1319", (len(raw["train"]), len(raw["test"])) == (7473, 1319),
          (len(raw["train"]), len(raw["test"])))
    check("raw: license file is MIT", dm["license_file_first_line"].strip() == "MIT License")

    # ---------------- processed files
    sm = json.loads((PROC / "split_manifest.json").read_text(encoding="utf-8"))
    rep = json.loads((PROC / "data_report.json").read_text(encoding="utf-8"))
    data = {}
    for name in (*FULL, *SMALL):
        p = PROC / f"{name}.jsonl"
        data[name] = load(p)
        check(f"{name}: file sha256 matches split_manifest", sha(p) == sm["files_sha256"][f"{name}.jsonl"])
        check(f"{name}: ids match split_manifest (order incl.)", [r["id"] for r in data[name]] == sm["ids"][name])
        ids_txt = (PROC / "split_ids" / f"{name}.txt").read_text(encoding="utf-8").split()
        check(f"{name}: split_ids/{name}.txt matches", ids_txt == sm["ids"][name])
        check(f"{name}: ids unique", len({r["id"] for r in data[name]}) == len(data[name]))
        check(f"{name}: all required fields present, no extras",
              all(list(r) == REQUIRED_FIELDS for r in data[name]))
        expected_split = SMALL.get(name, name)
        check(f"{name}: split field == '{expected_split}'", all(r["split"] == expected_split for r in data[name]))

    # ---------------- per-record content vs raw (independent re-extraction)
    raw_by_id = {f"gsm8k-{s}-{i:05d}": (s, i, obj) for s in raw for i, obj in enumerate(raw[s])}
    bad = []
    for name in FULL:
        for r in data[name]:
            s, i, obj = raw_by_id[r["id"]]
            if r["source_split"] != s or r["source_line"] != i + 1:
                bad.append((r["id"], "source mismatch"))
            if r["question"] != obj["question"] or r["reference_solution"] != obj["answer"]:
                bad.append((r["id"], "text not verbatim"))
            final = obj["answer"].rsplit("####", 1)[1].strip()
            if r["reference_final_answer"] != final:
                bad.append((r["id"], "final answer text"))
            # GSM8K finals are integers with optional ',' grouping and '-': independent integer parse
            if not re.fullmatch(r"-?\d{1,3}(,\d{3})*|-?\d+", final):
                bad.append((r["id"], f"unexpected final format {final!r}"))
            elif str(int(final.replace(",", ""))) != r["normalized_final_answer"]:
                bad.append((r["id"], "normalized answer mismatch"))
            if r["question_sha256"] != hashlib.sha256(
                re.sub(r"\s+", " ", unicodedata.normalize("NFKC", r["question"]).casefold()
                       .replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')).strip()
                .encode("utf-8")).hexdigest():
                bad.append((r["id"], "question_sha256"))
            if not r["question"].strip() or not r["reference_solution"].strip():
                bad.append((r["id"], "empty field"))
    check("records: verbatim text, correct source, final answer & normalization re-derived independently",
          not bad, bad[:10])

    # ---------------- split integrity
    ids = {n: {r["id"] for r in data[n]} for n in FULL}
    check("splits: train/validation/test ids pairwise disjoint",
          not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]))
    keys = {n: {norm_key(r["question"]) for r in data[n]} for n in FULL}
    ov = {"train&val": len(keys["train"] & keys["validation"]), "train&test": len(keys["train"] & keys["test"]),
          "val&test": len(keys["validation"] & keys["test"])}
    check("splits: no overlap by normalized question text (punctuation/space/case-insensitive)",
          not any(ov.values()), ov)
    check("test: equals the full official test set, unchanged",
          sorted(ids["test"]) == [f"gsm8k-test-{i:05d}" for i in range(len(raw["test"]))])
    excluded = {e["id"] for e in rep["excluded_from_train_pool"]["records"]}
    invalid = {e["id"] for e in rep["invalid_records"]["train"]}
    all_train_ids = {f"gsm8k-train-{i:05d}" for i in range(len(raw["train"]))}
    check("train+validation+excluded+invalid == all official train ids (nothing dropped silently)",
          (ids["train"] | ids["validation"] | excluded | invalid) == all_train_ids
          and len(ids["train"]) + len(ids["validation"]) + len(excluded) + len(invalid) == len(all_train_ids))
    check("validation size == round(10% of train pool)",
          len(ids["validation"]) == round((len(ids["train"]) + len(ids["validation"])) * 0.10))
    for name, parent in SMALL.items():
        sid = {r["id"] for r in data[name]}
        check(f"{name}: size == {EXPECTED_SMALL[name]}", len(sid) == EXPECTED_SMALL[name], len(sid))
        check(f"{name}: subset of {parent}", sid <= ids[parent])
        parent_rows = {r["id"]: r for r in data[parent]}
        check(f"{name}: rows identical to {parent} rows", all(r == parent_rows[r["id"]] for r in data[name]))

    # ---------------- determinism
    if not args.skip_determinism:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "--out-dir", td],
                           check=True, capture_output=True)
            diffs = [f.name for f in sorted(PROC.glob("*.jsonl")) if sha(f) != sha(Path(td) / f.name)]
            sm2 = json.loads((Path(td) / "split_manifest.json").read_text(encoding="utf-8"))
            check("determinism: re-running prepare_data.py gives byte-identical jsonl files and ids",
                  not diffs and sm2["ids"] == sm["ids"], diffs)

    n_fail = sum(not r["passed"] for r in results)
    out = {"passed": n_fail == 0, "n_checks": len(results), "n_failed": n_fail, "checks": results}
    (PROC / "validation_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
