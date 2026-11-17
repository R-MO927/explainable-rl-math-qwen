"""Prepare processed GSM8K splits from the unmodified raw files.

Reads  data/raw/gsm8k/{train,test}.jsonl (verified against download_manifest.json)
Writes data/processed/gsm8k/
    train.jsonl, validation.jsonl, test.jsonl                 full splits
    train_small.jsonl, validation_small.jsonl, test_small.jsonl
    split_ids/<split>.txt                                     one id per line
    split_manifest.json                                       seed, sizes, ids, file hashes
    data_report.json                                          counts, checks, exclusions, overlaps

Deterministic: same raw files + same config => byte-identical outputs.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --out-dir <dir>   # e.g. for a reproducibility check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.gsm8k import (  # noqa: E402
    PROCESSED_FIELDS,
    SCHEMA_VERSION,
    build_splits,
    near_duplicates,
    parse_raw_record,
    seeded_sample,
)

RAW_DIR = ROOT / "data" / "raw" / "gsm8k"
DEFAULT_OUT = ROOT / "data" / "processed" / "gsm8k"

CONFIG = {
    "seed": 42,
    "validation_fraction_of_train_pool": 0.10,
    "small_sizes": {"train_small": 500, "validation_small": 100, "test_small": 200},
    "near_duplicate_jaccard_threshold": 0.8,
    "near_duplicate_shingle_words": 3,
    "near_duplicate_index_words": 5,
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps({k: r[k] for k in PROCESSED_FIELDS}, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_raw(split: str, manifest: dict) -> tuple[list[dict], list[dict]]:
    path = RAW_DIR / f"{split}.jsonl"
    expected = manifest["data_files"][f"{split}.jsonl"]["sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"[error] {path} sha256 {actual} != manifest {expected}; re-run download_data.py")
    records, errors = [], []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"id": f"gsm8k-{split}-{i:05d}", "reason": f"invalid_json: {exc}"})
                continue
            res = parse_raw_record(obj, split, i)
            if res.error:
                errors.append({"id": f"gsm8k-{split}-{i:05d}", "reason": res.error})
            else:
                records.append(res.record)
    return records, errors


def answer_format_stats(records: list[dict]) -> dict:
    c = Counter()
    for r in records:
        t = r["reference_final_answer"]
        if "," in t:
            c["has_thousands_separator"] += 1
        if t.startswith("-"):
            c["negative"] += 1
        if "." in t:
            c["decimal_point"] += 1
        if "/" in t:
            c["fraction"] += 1
        if "/" not in r["normalized_final_answer"]:
            c["integer_value"] += 1
        else:
            c["non_integer_value"] += 1
    return dict(sorted(c.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    out: Path = args.out_dir
    cfg = CONFIG

    manifest = json.loads((RAW_DIR / "download_manifest.json").read_text(encoding="utf-8"))
    raw_train, err_train = load_raw("train", manifest)
    raw_test, err_test = load_raw("test", manifest)
    if err_test:
        # The official test set must never be altered silently. Stop and report.
        raise SystemExit(f"[error] {len(err_test)} invalid official test records: {err_test[:5]}")

    sr = build_splits(raw_train, raw_test, seed=cfg["seed"], val_fraction=cfg["validation_fraction_of_train_pool"])

    splits = {"train": sr.train, "validation": sr.validation, "test": sr.test}
    small_src = {"train_small": "train", "validation_small": "validation", "test_small": "test"}
    small = {}
    for name, src in small_src.items():
        by_id = {r["id"]: r for r in splits[src]}
        ids = seeded_sample(list(by_id), cfg["small_sizes"][name], cfg["seed"], name)
        small[name] = [by_id[i] for i in ids]

    # ---- write data files
    all_sets = {**splits, **small}
    file_hashes = {}
    for name, recs in all_sets.items():
        p = out / f"{name}.jsonl"
        write_jsonl(p, recs)
        file_hashes[f"{name}.jsonl"] = sha256_file(p)
        ids_path = out / "split_ids" / f"{name}.txt"
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("".join(r["id"] + "\n" for r in recs), encoding="utf-8", newline="\n")

    # ---- overlap / near-duplicate analysis (report only)
    thr = cfg["near_duplicate_jaccard_threshold"]
    nd_kwargs = dict(threshold=thr, index_n=cfg["near_duplicate_index_words"], sim_n=cfg["near_duplicate_shingle_words"])
    near_test_dev = near_duplicates(sr.test, sr.train + sr.validation, **nd_kwargs)
    near_val_train = near_duplicates(sr.validation, sr.train, **nd_kwargs)
    tmpl_test_dev = near_duplicates(sr.test, sr.train + sr.validation, mask_numbers=True, **nd_kwargs)
    tmpl_val_train = near_duplicates(sr.validation, sr.train, mask_numbers=True, **nd_kwargs)

    def sha_set(recs):
        return {r["question_sha256"] for r in recs}

    exact_overlap = {
        "train_validation": len(sha_set(sr.train) & sha_set(sr.validation)),
        "train_test": len(sha_set(sr.train) & sha_set(sr.test)),
        "validation_test": len(sha_set(sr.validation) & sha_set(sr.test)),
    }

    excluded_by_reason = Counter(e["reason"] for e in sr.excluded)
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "fields": list(PROCESSED_FIELDS),
        "source": {
            "repository": manifest["repository"],
            "commit": manifest["commit"],
            "raw_sha256": {k: v["sha256"] for k, v in manifest["data_files"].items()},
        },
        "config": cfg,
        "id_scheme": "gsm8k-<source_split>-<0-based line index in raw file at pinned commit, 5 digits>",
        "split_method": (
            "test = official test.jsonl unchanged. Official train minus excluded records "
            "(duplicates of test questions; repeated questions within train) forms the pool; "
            "validation = seeded random sample of round(pool * validation_fraction); train = rest. "
            "Small subsets = seeded random samples of each full split. "
            "Sub-seeds = first 8 bytes of sha256('<seed>:<name>')."
        ),
        "sizes": {k: len(v) for k, v in all_sets.items()},
        "files_sha256": file_hashes,
        "ids": {k: [r["id"] for r in v] for k, v in all_sets.items()},
    }
    write_json(out / "split_manifest.json", split_manifest)

    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "source": split_manifest["source"],
        "raw_counts": {
            "train": manifest["data_files"]["train.jsonl"]["records"],
            "test": manifest["data_files"]["test.jsonl"]["records"],
        },
        "parsed_valid_counts": {"train": len(raw_train), "test": len(raw_test)},
        "invalid_records": {"train": err_train, "test": err_test},
        "excluded_from_train_pool": {
            "total": len(sr.excluded),
            "by_reason": dict(excluded_by_reason),
            "records": sr.excluded,
        },
        "test_internal_duplicate_groups": sr.test_internal_duplicates,
        "processed_counts": split_manifest["sizes"],
        "answer_formats": {
            "train_raw": answer_format_stats(raw_train),
            "test_raw": answer_format_stats(raw_test),
        },
        "overlap_checks": {
            "exact_normalized_text_overlap_after_split": exact_overlap,
            "near_duplicates": {
                "method": f"word {cfg['near_duplicate_shingle_words']}-gram Jaccard >= {thr}, candidates share a word "
                f"{cfg['near_duplicate_index_words']}-gram. Report only, nothing removed.",
                "test_vs_train_and_validation": {"count": len(near_test_dev), "pairs": near_test_dev},
                "validation_vs_train": {"count": len(near_val_train), "pairs": near_val_train},
            },
            "shared_templates_numbers_masked": {
                "method": "same as near_duplicates but every integer token replaced by <num>; detects the same "
                "problem template with different numbers. These are different problems (different answers in "
                "most cases) and are NOT removed; reported for transparency about template-level similarity.",
                "test_vs_train_and_validation": {"count": len(tmpl_test_dev), "pairs": tmpl_test_dev},
                "validation_vs_train": {"count": len(tmpl_val_train), "pairs": tmpl_val_train},
            },
        },
        "notes": [
            "Questions and reference solutions are stored verbatim (no translation, no edits).",
            "No difficulty labels or categories are added; GSM8K does not provide them.",
            "RL trajectories / rewards are not part of GSM8K; they will be generated in phase 2.",
        ],
    }
    write_json(out / "data_report.json", report)

    print("[ok] processed splits written to", out)
    for k, v in split_manifest["sizes"].items():
        print(f"     {k:18s} {v}")
    print("     excluded from train pool:", dict(excluded_by_reason) or 0)
    print("     invalid raw records: train", len(err_train), "test", len(err_test))
    print("     test internal duplicate groups:", len(sr.test_internal_duplicates))
    print("     exact overlap after split:", exact_overlap)
    print(f"     near-dups (J>={thr}) test~dev: {len(near_test_dev)}  val~train: {len(near_val_train)}")
    print(f"     shared templates, numbers masked (J>={thr}) test~dev: {len(tmpl_test_dev)}  val~train: {len(tmpl_val_train)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
