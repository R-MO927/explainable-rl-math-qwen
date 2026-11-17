# Explainable Multi-Agent RL for Mathematical Reasoning (Qwen2.5-0.5B, GSM8K)

Three cooperating software roles solve grade-school math word problems:

| Role | Implementation | Learns? |
|---|---|---|
| **Solver** | `Qwen/Qwen2.5-0.5B-Instruct` (revision `7ae55760`), frozen weights. Attempt 0 is greedy; later attempts are sampled with the model's official generation config. | no |
| **Verifier** | Does not see the reference. It checks for an extractable answer, checks explicit arithmetic with a safe AST evaluator (no `eval`), and flags truncation. `VERIFY_MORE` runs a program-of-thought cross-check: Qwen writes a tiny program, a safe interpreter runs it, and the result is compared with the candidate. | no |
| **Coordinator** | Tabular Q-learning over the actions `ACCEPT` / `REGENERATE` / `VERIFY_MORE`. | **yes (RL)** |

Compared methods (same model, tools and limits):
- **A**: Qwen with one greedy attempt.
- **B**: Solver and Verifier with a fixed, documented rule.
- **C**: Solver and Verifier with the RL coordinator.

`ref_max_budget` is an extra reference that always uses the full budget.

The question is accuracy **versus cost**. We do not assume that RL helps.

## Results (GSM8K test, n = 1,319; cached replay; Wilson 95% CI)

| Method | Accuracy | Model calls / problem | Output tokens / problem |
|---|---|---|---|
| A: single attempt | 48.7% [46.1, 51.4] | 1.00 | 308 |
| B: fixed rule | **55.3%** [52.7, 58.0] | 3.46 | 915 |
| C: RL coordinator (deployed, seed 1) | 50.9% [48.2, 53.6] | 1.49 | 451 |
| C: mean of 5 seeds | 53.2% ± 1.9 | 3.09 | — |
| ref: full budget | 55.0% [52.3, 57.7] | 5.00 | 1,348 |

Paired differences:
- B − A: +6.6 points, CI [+4.9, +8.4].
- C − A: +2.1 points, CI [+1.3, +3.0].
- C − B: −4.5 points, CI [−6.1, −2.9].

The RL coordinator learned a *cheap* policy: it accepts the first attempt 82% of the time and always regenerates after an arithmetic error or a missing answer. It does **not** beat the fixed rule on accuracy, and training is unstable across seeds. See `docs/PROJECT_REPORT_AR.md` §7.

## Setup (Windows, tested)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# GPU build of PyTorch (CUDA 12.6 wheel):
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
```

## Reproduce

```powershell
$py = ".\.venv\Scripts\python.exe"
# Phase 1: data
& $py scripts\download_data.py; & $py scripts\prepare_data.py; & $py scripts\validate_data.py
# Phase 2
& $py scripts\smoke_test_qwen.py --n 20                     # train problems only
& $py scripts\generate_attempts.py --split train_small      # cache of solver attempts + PoT programs
& $py scripts\generate_attempts.py --split validation_small
& $py scripts\train_coordinator.py                          # grid search on validation, 5 seeds
& $py scripts\generate_attempts.py --split test_small       # only after settings are frozen
& $py scripts\evaluate.py --split test_small                # A / B / C comparison
& $py scripts\make_plots.py --split test_small
& $py -m pytest -q                                          # mock-based behaviour tests
```

## Try a new question (no reference answer needed)

```powershell
& $py scripts\solve.py "A baker makes 24 rolls and sells 3/4 of them. How many are left?"
& $py scripts\solve.py --method fixed --budget 3 --json "..."
```

From Python (this is the interface a future web API can wrap):

```python
from src.pipeline import solve_problem
r = solve_problem(question, method="rl", budget=4, seed=0)   # method: single | fixed | rl | max_budget
r["final_answer"], r["actions"], r["explanations"], r["usage"]
```

## Layout

```
configs/experiment.json      limits, reward, decoding, grid (documentation of what the code uses)
src/data/                    answer parsing (exact Fractions), GSM8K splits
src/solver/                  Qwen wrapper, prompts, answer extraction
src/verifier/                safe evaluator, arithmetic checker, verifier role
src/environment/             RL environment, reward (only place that sees references), attempt sources
src/coordinator/             features, fixed rule, Q-learning, training loop
src/explainability/          template explanations from Q-values
src/evaluation/              episode runner, metrics (Wilson CI, paired bootstrap, McNemar)
src/pipeline.py              solve_problem(...)
scripts/                     CLI entry points (see above)
artifacts/cache/             cached generations (keyed by question/model/prompt/decoding/seed)
artifacts/policies/          trained Q-tables (JSON)
results/                     logs, metrics (CSV/JSON), figures
docs/                        Arabic reports
```

Full write-up (Arabic): `docs/PROJECT_REPORT_AR.md`. Data report: `docs/DATA_REPORT_AR.md`.
Status of what is and is not done: `docs/STATUS_AR.md`.
