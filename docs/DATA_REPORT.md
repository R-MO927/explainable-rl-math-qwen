# Data Preparation Report — GSM8K

**Project:** Explainable Multi-Agent Reinforcement Learning for Mathematical Reasoning Using Qwen-0.5B
**Phase:** 1 (data download and preparation only)
**Prepared on:** 2026-11-17

> All numbers in this report are copied from the generated files:
> `data/raw/gsm8k/download_manifest.json`, `data/processed/gsm8k/data_report.json`, `split_manifest.json` and `validation_result.json`.
> If you re-run the preparation, those files are the authoritative source.

---

## 1. Data source

| Item | Value |
|---|---|
| Official repository | https://github.com/openai/grade-school-math |
| Pinned commit | `3101c7d5072418e28b9008a6636bde82a006892c` (the head of `master` at download time, checked via the GitHub API) |
| Train file | `grade_school_math/data/train.jsonl` |
| Test file | `grade_school_math/data/test.jsonl` |
| Download time (UTC) | 2026-11-16 22:10:32 (train), 22:10:33 (test) |
| SHA-256 of train.jsonl | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` (4,166,206 bytes) |
| SHA-256 of test.jsonl | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` (749,738 bytes) |
| License | The repository `LICENSE` file: **MIT License**, "Copyright (c) 2021 OpenAI". Saved unchanged as `data/raw/gsm8k/LICENSE`. |
| Citation | The repository `README.md` asks to cite `cobbe2021gsm8k` (a copy is in `data/raw/gsm8k/README.md`). |

**Download method:** `raw.githubusercontent.com` URLs pinned to the commit, not to `master`, so every later download returns exactly the same content. An HTTP 200 is not treated as success on its own; the script also checks:
- size matches `Content-Length`;
- valid UTF-8;
- every line is a valid JSON object with `question` and `answer`;
- no blank lines inside the file;
- record count.

The file is then written atomically to `data/raw/gsm8k/` **without any modification**.

## 2. Raw record format

Each line of the raw file has two fields:

```json
{"question": "...problem text...", "answer": "...solution steps with <<3*5=15>> annotations...\n#### 2"}
```

- `answer` holds the written solution, with calculator annotations of the form `<<expr=result>>`, followed by the final answer after `####`.
- The source has no other fields (such as difficulty or category), **and we added none**.

## 3. Actual counts

| Stage | train | test |
|---|---|---|
| Raw records (source) | **7,473** | **1,319** |
| Valid records after checks | 7,473 | 1,319 |
| Corrupt records | 0 | 0 |
| Excluded records (duplicate/leak) | 0 | — |

**Processed splits:**

| File | Count | Source |
|---|---|---|
| `train.jsonl` | **6,726** | official train file |
| `validation.jsonl` | **747** | official train file (10%) |
| `test.jsonl` | **1,319** | the full official test file, unchanged |
| `train_small.jsonl` | **500** | fixed sample of `train` |
| `validation_small.jsonl` | **100** | fixed sample of `validation` |
| `test_small.jsonl` | **200** | fixed sample of `test` |

Check: 6,726 + 747 = 7,473, so no record was lost.

> The small sizes (500/100/200) are for initial runs and resource-limited experiments, **and are not enough to show a statistically significant improvement**. A 95% confidence interval for accuracy measured on 200 problems is roughly ±7 percentage points wide. The full files are available for larger runs.

## 4. Processed record format

```json
{
  "id": "gsm8k-test-00025",
  "source_split": "test",
  "source_line": 26,
  "split": "test",
  "question": "...original text, verbatim...",
  "reference_solution": "...original answer field, verbatim, including <<...>> and #### 2...",
  "reference_final_answer": "2",
  "normalized_final_answer": "2",
  "question_sha256": "de36e8d3..."
}
```

| Field | Meaning |
|---|---|
| `id` | Stable identifier: `gsm8k-<train|test>-<0-based line number in the raw file, 5 digits>`. Stable because the raw file is pinned to a commit. |
| `source_split` / `source_line` | Original file and 1-based line number, so every record can be traced to its source. |
| `split` | Split after processing: `train`, `validation` or `test`. |
| `question` / `reference_solution` | Original texts, **not translated or modified**. |
| `reference_final_answer` | The text after `####` as-is, with only surrounding whitespace removed (e.g. `"1,080"`). |
| `normalized_final_answer` | Canonical form of the numeric value (see section 5). |
| `question_sha256` | Hash of the normalized text, used in overlap checks. |

## 5. Answer extraction and normalization

Code: `src/data/answers.py`.

1. **Extraction:** `####` must appear **exactly once** in `answer`, and the text after it must be non-empty. Otherwise the record is logged as corrupt with a clear reason; it is never silently repaired.
2. **Strict numeric parsing** (`parse_reference_number`):
   - Unicode NFKC normalization; non-standard minus signs (such as `−`) become `-`.
   - Thousands separators are accepted **only when correctly grouped**: `1,080` and `109,200,000` are accepted; `1,23` and `12,3456` are rejected as ambiguous.
   - Signs, decimals and fractions `a/b` are accepted.
   - Scientific notation (`1e3`), units, words and division by zero are rejected. The function returns `None` instead of raising.
3. **Representation:** values become `fractions.Fraction` via `Decimal`, **never `float`**. So `0.30 = 0.3 = 3/10` exactly, and `0.1 ≠ 0.10000000000000001`.
4. **Canonical form:** integers are written as-is (`"1080"`, `"-3"`, `"0"`); non-integers as a reduced fraction (`"7/2"`).
5. **Lenient parser** (`parse_answer_number`), prepared for Phase 2 to read model answers. It additionally accepts a leading `$`, a trailing `%` (keeping the number itself) and a sentence-final period. It rejects any text with units or words, such as `"5 apples"`.

**What the source actually contains:**

| Format | train | test |
|---|---|---|
| Integers | 7,473 (all) | 1,319 (all) |
| With thousands separator | 79 | 14 |
| Negative | 3 | 2 |
| Decimals or fractions | 0 | 0 |

## 6. Split method and seed

- **Seed:** `42`. Deterministic sub-seeds are derived as `int(sha256("42:<name>")[:8])`, so nothing depends on Python's randomized `hash`.
- **test:** the full official test file, unchanged.
- **Training pool:** the official train file after removing:
  1. any record whose normalized question matches a test question (`duplicate_of_test`);
  2. any duplicate inside train itself, keeping the first occurrence (`duplicate_within_train`), so the same problem cannot land in both train and validation.

  **Actual result: no record was excluded**, because the source contains no such cases.
- **validation:** a fixed random sample of size `round(10% × pool)`, i.e. 747 records. The rest is **train**.
- **Small subsets:** a fixed random sample from each full split, then sorted by `id`.
- **Files** are sorted by `id`. The ids of each split are saved in:
  - `split_ids/<name>.txt`
  - `split_manifest.json` (together with the SHA-256 of each file).
- **Binding rule:** `test` and `test_small` problems are **not used** to choose features, rewards or settings. All tuning is done on `validation`.

## 7. Integrity and overlap checks

### 7.1 Exact overlap
Done on **normalized text**, not just on ids. Normalization:
- NFKC;
- case folding;
- unified quote marks;
- removal of all non-alphanumeric symbols.

**Numbers are kept**, because two problems that differ in one number are different problems.

| Pair | Exact match after normalization |
|---|---|
| train ∩ validation | 0 |
| train ∩ test | 0 |
| validation ∩ test | 0 |
| duplicates inside test | 0 |
| duplicates inside train | 0 |

### 7.2 Near-duplicate check
**Reporting only; it removes nothing.** Jaccard similarity on word 3-grams with a 0.8 threshold:

- test vs (train + validation): **0 pairs**.
- validation vs train: **0 pairs**.

To make sure the detector itself works and the "zero" is not a bug, it was also run at a lower threshold:
- at 0.5, only 3 pairs appeared, the highest at 0.586;
- comparing a record with itself gave 1.0.

### 7.3 Shared-template check
Also **reporting only**. Same check as above, but after replacing every number with `<num>`, to find problems built from the same template with different numbers:

- **test vs development data: 2 pairs**, both for the same test problem `gsm8k-test-00602`:
  - the problem: a plane covers 1200 miles in 3 hours;
  - it matches `gsm8k-train-01314` and `gsm8k-train-05162`: a train covering 270 or 360 miles;
  - the numbers differ **and the final answers differ** in both cases, so we do not treat this as answer leakage.
- **validation vs train: 1 pair** (`gsm8k-train-05419` and `gsm8k-train-04997`), with different answers.
- **Handling:** nothing was removed and test was not changed. This is documented for transparency; the likely effect is negligible (one problem out of 1,319 shares a template with training problems).

### 7.4 Independent validation (`scripts/validate_data.py`)
**Result: 58/58 checks passed.** Details in `data/processed/gsm8k/validation_result.json`. The checks cover:
- SHA-256 of the raw files matches the manifest, and the official counts 7,473 / 1,319 match.
- SHA-256 of every processed file, and its ids and their order, match `split_manifest.json` and `split_ids/`.
- For every record:
  - the text is copied verbatim from the raw file;
  - source and line number are correct;
  - **the answer is re-extracted and re-normalized independently** of the preparation code;
  - the text hash is correct.
- Splits are disjoint, both by id and by normalized text.
- test equals the full official test file.
- train + validation + excluded + corrupt = all original train records, i.e. nothing was dropped silently.
- Small-subset sizes are correct, they are subsets of their splits, and their rows are identical to the parent rows.
- **Determinism:** re-running `prepare_data.py` in a temporary folder produces **byte-identical** files.

### 7.5 pytest
**Result at the end of Phase 1: 49 tests passed** (the suite has since grown to 107 with Phase 2). Files:
- `tests/test_answers.py`: representative numeric cases:
  - thousands separators, negatives, `-0`, leading zeros;
  - decimals, fractions and the Unicode minus sign;
  - rejection of ambiguous formats;
  - exact equality without float errors;
  - rejection of a duplicated or missing `####`.
- `tests/test_gsm8k_splits.py`: synthetic data that **forces** the exclusion paths to run (a leak from test, duplicates inside train with the same or a different answer). The real source has no such cases, so these paths cannot be tested on it. Also covers:
  - determinism and independence from input order;
  - the effect of changing the seed;
  - template detection.
- `tests/test_processed_data.py`: checks on the real files, including actual examples:
  - `test-00146`: `"2,125"` becomes `2125`;
  - `test-00489`: answer `-10`;
  - `test-00611`: `1,450,000`.

## 8. Excluded records

**None.** No corrupt, duplicated or overlapping record was found in the pinned source version. If any appear in a later version, `data_report.json` lists them under `invalid_records` and `excluded_from_train_pool` with the reason and the id of the matching record.

## 9. File locations

```
data/raw/gsm8k/                    # raw copies, unmodified
  train.jsonl, test.jsonl, LICENSE, README.md, download_manifest.json
data/processed/gsm8k/
  train.jsonl, validation.jsonl, test.jsonl
  train_small.jsonl, validation_small.jsonl, test_small.jsonl
  split_ids/*.txt                  # ids of each split
  split_manifest.json              # seed, method, sizes, ids, SHA-256
  data_report.json                 # counts, checks, overlap, answer formats
  validation_result.json           # output of validate_data.py
src/data/answers.py, src/data/gsm8k.py   # extraction, normalization and split logic
scripts/download_data.py, prepare_data.py, validate_data.py, check_env.py
tests/                             # pytest tests
artifacts/environment.json         # recorded runtime environment
requirements.txt, requirements-lock.txt
```

## 10. Commands to re-prepare

Run from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe scripts\download_data.py        # skips if files are intact; --force to re-download
.\.venv\Scripts\python.exe scripts\prepare_data.py
.\.venv\Scripts\python.exe scripts\validate_data.py
.\.venv\Scripts\python.exe -m pytest -q
```

## 11. Environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro (10.0.26300) |
| CPU | Intel Core i9-13900HX |
| RAM | 15.7 GB total; **only about 2.2 GB was free at check time** because of other open programs |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6 GB VRAM, driver 560.94, supports up to CUDA 12.6 |
| Free disk | about 137 GB |
| Python | 3.12.10 in an isolated `.venv`. Python 3.14 is also installed, but 3.12 was chosen for PyTorch compatibility. |

Phase 1 used only the Python standard library plus pytest and ipykernel. **PyTorch, Transformers and the Qwen model were not downloaded in Phase 1** (they were added in Phase 2).

## 12. Note on RL data

GSM8K provides **only problems and reference solutions**. The data for the reinforcement-learning experiments (states, actions, rewards, solver attempts and verifier outputs) **is not part of GSM8K**; it is generated in Phase 2 by running the system (Solver + Verifier + Coordinator) on `train` and `validation`. The reference answer is used only inside the reward computation and the evaluation code, and never reaches the observation or the verifier.

## 13. Known limitations

- Qwen2.5 may have seen GSM8K problems during pre-training (contamination). Evaluation on test alone therefore does not prove generalization to completely new problems. This is discussed in the project report.
- The near-duplicate check relies on word overlap, so it cannot detect paraphrases that use entirely different words.
- All reference answers in this version are integers. Support for fractions and decimals is tested on synthetic cases and is aimed at model outputs.
