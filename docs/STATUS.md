# Project Status — 2026-11-17

## Completed and verified

| Item | Status | Evidence |
|---|---|---|
| Phase 1: download and prepare GSM8K | ✅ | `docs/DATA_REPORT.md`; `validate_data.py`: 58/58 |
| Download Qwen2.5-0.5B-Instruct (pinned revision, hash verified) | ✅ | `src/solver/qwen_solver.py` |
| PyTorch CUDA build on RTX 4050 | ✅ | `artifacts/environment.json`, `requirements.txt` |
| Smoke test on 20 train problems | ✅ | `results/smoke/smoke_train.json` |
| Solver, Verifier (safe AST evaluator, PoT), environment and reward | ✅ | `src/` |
| Attempt generation: train_small (500), validation (747), test (1,319) | ✅ real runs | `artifacts/cache/*.jsonl`, `results/logs/gen_*.log` |
| Q-learning training with settings chosen on validation only (30 configs × 3 seeds, then 5 final seeds) | ✅ | `results/training/`, `artifacts/policies/` |
| Policies frozen before testing (SHA-256 hashes) | ✅ | `results/eval/FROZEN_BEFORE_TEST.json` |
| Evaluation of A, B, C (plus the max-budget reference) on test_small and the full test set | ✅ | `results/eval/{test_small,test}/` (metrics.json, metrics.csv, comparisons.json, logs_*.jsonl) |
| Figures | ✅ | `results/figures/*.png` |
| Real success and failure examples | ✅ | `results/examples/examples_test.md` |
| XAI: Q-values, action mask, visit counts, template explanations, documented fallback | ✅ | `decisions` field in every log record |
| `solve_problem(question, method, budget, seed)` API | ✅ tested live | `src/pipeline.py` |
| Command-line interface | ✅ tested live | `scripts/solve.py` |
| Tests | ✅ 122 passed | `tests/`. **All use mock data** or the data files, **so they are not evidence of Qwen's quality** |
| Notebook `RL.ipynb` | ✅ runs end to end with no errors | shows results, figures and one live question |
| Project report and README | ✅ | `docs/PROJECT_REPORT.md`, `README.md` |
| Pre-web verification pass | ✅ | see below |
| Local web app (EN/AR, steps, KaTeX formulas, fruit visuals, XAI decisions, results page) | ✅ tested live on the GPU | `web/`, `scripts/run_web.py`, `tests/test_web_server.py` (15 tests, demo solver) |

## Pre-web verification (2026-11-17)

- All 14 frozen files still match their SHA-256 hashes.
- Re-running the evaluation from the cache reproduces every metric exactly, with identical answers and actions for every problem.
- Accuracy recomputed by independent code directly from the raw GSM8K file agrees with our grader on 1,319/1,319 problems.
- Live GPU run: 3 new (non-GSM8K) questions × 4 methods, all 12 answers correct; same seed gives the same result; budget limits respected.
- Invalid inputs (empty question, unknown method, budget outside 1–4) raise clear `ValueError`s. A non-math question returns `answer=None` with `accepted_no_answer`.
- A fresh `git clone` is byte-identical to the working folder (`.gitattributes` stores files without line-ending conversion), so all recorded hashes stay valid.

## Headline result (full test set, 1,319 problems)

| | A | B | C (deployed) |
|---|---|---|---|
| Accuracy | 48.7% | **55.3%** | 50.9% |
| Model calls / problem | 1.00 | 3.46 | 1.49 |

- **RL did not beat the fixed rule on accuracy.** It is more accurate than A at a much lower cost than B.
- **Training is unstable across seeds:** test accuracy ranged from 50.9% to 55.3%.

## Not done / known limitations

- **Repeating attempt generation with more than one seed:** not done; there was a single generation run. The reported variance covers coordinator training only.
- **DQN:** not used, because the tabular state (144 visited states) is enough. The current problem is instability, not state size.
- **Training the Solver itself:** not done; Qwen's weights stay frozen by design.
- **RL instability:** not solved. Suggested fixes are in section 9 of the report.
- **Possible exposure of Qwen to GSM8K during pre-training:** cannot be verified. The evaluation does not prove generalization to completely new problems.
- **Live latency depends on power mode:** about 4–7 s per attempt on AC power, 8–20 s on battery (Balanced mode).

## Web app

- **Online (public):** https://huggingface.co/spaces/Rahmamohamed5/mathmates on free ZeroGPU hardware. Verified 2026-11-17: the public URL solved a new question correctly (26) in about 4–5 s with methods C and B.
- `scripts/run_web.py` serves the page at http://127.0.0.1:8000 (local only). The model loads in the background (about 15 s).
- One GPU request at a time (a second one gets HTTP 429), 5-minute client timeout, loading animation with a timer, clear "no answer" display.
- Standard-library server instead of FastAPI: Smart App Control on this machine has blocked new native wheels before, and FastAPI depends on one (pydantic-core).
- Not deployed to the internet: the model needs the local GPU.
