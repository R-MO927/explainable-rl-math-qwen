"""GSM8K record parsing, text normalization, de-duplication and splitting.

Pure functions only (no I/O) so they can be unit-tested and reused.
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from .answers import canonical_str, parse_reference_number, split_reference_solution

SCHEMA_VERSION = "1.0"
PROCESSED_FIELDS = (
    "id",
    "source_split",
    "source_line",
    "split",
    "question",
    "reference_solution",
    "reference_final_answer",
    "normalized_final_answer",
    "question_sha256",
)


# ---------------------------------------------------------------- text keys
def normalize_text(s: str) -> str:
    """Normalization used for duplicate / overlap detection.

    NFKC, case-fold, unify quotes, collapse whitespace. Numbers are kept,
    because two problems that differ only in a number are different problems.
    """
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def loose_key(s: str) -> str:
    """Stricter overlap key: normalized text with all non-alphanumerics removed
    (catches duplicates differing only in punctuation/spacing)."""
    return re.sub(r"[^0-9a-z]+", "", normalize_text(s))


def text_sha256(s: str) -> str:
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()


def word_tokens(s: str, mask_numbers: bool = False) -> list[str]:
    toks = re.findall(r"[0-9a-z]+", normalize_text(s))
    if mask_numbers:
        toks = ["<num>" if t.isdigit() else t for t in toks]
    return toks


def shingles(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# ---------------------------------------------------------------- records
def make_id(source_split: str, index: int) -> str:
    """Stable id from the raw file split and 0-based line index (pinned commit)."""
    return f"gsm8k-{source_split}-{index:05d}"


@dataclass
class ParseResult:
    record: dict | None
    error: str | None = None


def parse_raw_record(obj: object, source_split: str, index: int) -> ParseResult:
    """Validate a raw JSON object and convert it to the processed schema.

    Never silently fixes data: any problem returns an error string instead.
    `split` is filled in later by the splitter.
    """
    if not isinstance(obj, dict):
        return ParseResult(None, "not_a_json_object")
    q, a = obj.get("question"), obj.get("answer")
    if not isinstance(q, str) or not isinstance(a, str):
        return ParseResult(None, "missing_or_non_string_field")
    if not q.strip():
        return ParseResult(None, "empty_question")
    if not a.strip():
        return ParseResult(None, "empty_answer")
    try:
        _, final_text = split_reference_solution(a)
    except ValueError as exc:
        return ParseResult(None, f"bad_final_answer_marker: {exc}")
    value = parse_reference_number(final_text)
    if value is None:
        return ParseResult(None, f"unparseable_final_answer: {final_text!r}")
    return ParseResult(
        {
            "id": make_id(source_split, index),
            "source_split": source_split,
            "source_line": index + 1,
            "split": None,
            "question": q,  # kept verbatim
            "reference_solution": a,  # kept verbatim, incl. <<calc>> annotations and '#### x'
            "reference_final_answer": final_text,
            "normalized_final_answer": canonical_str(value),
            "question_sha256": text_sha256(q),
        }
    )


# ---------------------------------------------------------------- splitting
def derived_seed(seed: int, name: str) -> int:
    """Deterministic sub-seed independent of Python's hash randomization."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def seeded_sample(ids: list[str], k: int, seed: int, name: str) -> list[str]:
    if k > len(ids):
        raise ValueError(f"cannot sample {k} from {len(ids)} ids for {name}")
    pool = sorted(ids)
    random.Random(derived_seed(seed, name)).shuffle(pool)
    return sorted(pool[:k])


@dataclass
class SplitResult:
    train: list[dict]
    validation: list[dict]
    test: list[dict]
    excluded: list[dict] = field(default_factory=list)  # {"id","reason","detail"}
    test_internal_duplicates: list[list[str]] = field(default_factory=list)


def build_splits(train_records: list[dict], test_records: list[dict], *, seed: int, val_fraction: float) -> SplitResult:
    """Create train / validation (from official train) and test (official test).

    Leakage handling (documented, never silent):
      1. Official test is kept unchanged.
      2. Official-train records whose loose text key equals any test question
         are excluded from train/validation (reason: duplicate_of_test).
      3. Exact duplicates *within* official train keep the first occurrence
         (lowest source line) and exclude the rest (reason: duplicate_within_train),
         so the same question cannot appear in both train and validation.
    """
    excluded: list[dict] = []
    test_keys: dict[str, str] = {}
    by_key_test: dict[str, list[str]] = defaultdict(list)
    for r in test_records:
        k = loose_key(r["question"])
        by_key_test[k].append(r["id"])
        test_keys.setdefault(k, r["id"])
    test_dups = [ids for ids in by_key_test.values() if len(ids) > 1]

    pool: list[dict] = []
    first_by_key: dict[str, dict] = {}
    for r in sorted(train_records, key=lambda x: x["source_line"]):
        k = loose_key(r["question"])
        if k in test_keys:
            excluded.append({"id": r["id"], "reason": "duplicate_of_test", "detail": {"test_id": test_keys[k]}})
            continue
        if k in first_by_key:
            kept = first_by_key[k]
            excluded.append(
                {
                    "id": r["id"],
                    "reason": "duplicate_within_train",
                    "detail": {
                        "kept_id": kept["id"],
                        "same_final_answer": kept["normalized_final_answer"] == r["normalized_final_answer"],
                    },
                }
            )
            continue
        first_by_key[k] = r
        pool.append(r)

    pool_ids = [r["id"] for r in pool]
    n_val = round(len(pool_ids) * val_fraction)
    val_ids = set(seeded_sample(pool_ids, n_val, seed, "validation"))

    def with_split(r: dict, split: str) -> dict:
        out = dict(r)
        out["split"] = split
        return out

    train = [with_split(r, "train") for r in pool if r["id"] not in val_ids]
    validation = [with_split(r, "validation") for r in pool if r["id"] in val_ids]
    test = [with_split(r, "test") for r in test_records]
    key = lambda r: r["id"]  # noqa: E731
    return SplitResult(sorted(train, key=key), sorted(validation, key=key), sorted(test, key=key), excluded, test_dups)


# ---------------------------------------------------------------- near duplicates
def near_duplicates(
    queries: list[dict],
    corpus: list[dict],
    *,
    threshold: float,
    index_n: int = 5,
    sim_n: int = 3,
    mask_numbers: bool = False,
) -> list[dict]:
    """Report query/corpus pairs with word-`sim_n`-gram Jaccard >= threshold.

    Candidates are pairs sharing at least one word-`index_n`-gram (inverted
    index), which keeps the search tractable. With `mask_numbers=True` every
    integer token becomes "<num>", which detects shared problem *templates*
    whose numbers differ. Report-only; nothing is removed.
    """
    index: dict[tuple, set[int]] = defaultdict(set)
    corpus_tok = [word_tokens(r["question"], mask_numbers) for r in corpus]
    corpus_sim = [shingles(t, sim_n) for t in corpus_tok]
    for i, toks in enumerate(corpus_tok):
        for g in shingles(toks, index_n):
            index[g].add(i)
    hits = []
    for q in queries:
        qt = word_tokens(q["question"], mask_numbers)
        qs = shingles(qt, sim_n)
        cands: set[int] = set()
        for g in shingles(qt, index_n):
            cands |= index.get(g, set())
        for i in cands:
            if corpus[i]["id"] == q["id"]:
                continue
            inter = len(qs & corpus_sim[i])
            union = len(qs | corpus_sim[i])
            j = inter / union if union else 0.0
            if j >= threshold:
                hits.append(
                    {
                        "query_id": q["id"],
                        "corpus_id": corpus[i]["id"],
                        "jaccard": round(j, 4),
                        "same_final_answer": q["normalized_final_answer"] == corpus[i]["normalized_final_answer"],
                    }
                )
    hits.sort(key=lambda h: (-h["jaccard"], h["query_id"], h["corpus_id"]))
    return hits
