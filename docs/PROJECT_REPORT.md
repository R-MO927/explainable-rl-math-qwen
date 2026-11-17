# Project Report: Explainable Multi-Agent Reinforcement Learning for Mathematical Reasoning Using Qwen-0.5B

**Status:** Phase 2 is experimentally complete. The final comparison of systems A, B and C on test is done. The web interface has not been built yet (next phase).
**Date:** 2026-11-17
**Source of numbers:** every number in this report is copied from the actual result files:
- `results/eval/*/metrics.json`
- `results/training/training_summary.json`
- `results/logs/*`

---

## 1. Summary

We built a system that solves GSM8K problems with three software roles:
- **Solver:** Qwen2.5-0.5B-Instruct with frozen weights.
- **Verifier:** never sees the reference answer.
- **Coordinator:** learns with Q-learning when to accept a solution, when to ask for a new attempt, and when to ask for an extra check.

**Results on the full test set (1,319 problems):**

| System | Accuracy | 95% CI | Model calls / problem | Output tokens / problem |
|---|---|---|---|---|
| **A** — Qwen, single attempt | **48.7%** | [46.1, 51.4] | 1.00 | 308 |
| **B** — Solver + Verifier, fixed rule | **55.3%** | [52.7, 58.0] | 3.46 | 915 |
| **C** — Solver + Verifier, RL coordinator (deployed policy) | **50.9%** | [48.2, 53.6] | 1.49 | 451 |
| C — mean of 5 seeds | 53.2% ± 1.9 | — | 3.09 | — |
| Extra reference: always use the full budget | 55.0% | [52.3, 57.7] | 5.00 | 1,348 |

**Plain conclusions:**
1. **Coordinating solving and checking does improve accuracy.** The fixed rule B beats A by **6.6 points**, bootstrap CI [+4.9, +8.4], McNemar p < 10⁻¹². But this gain costs about 3.5× the model calls.
2. **The deployed RL coordinator did not beat the fixed rule on accuracy.** It is 4.5 points below B, CI [−6.1, −2.9]. In exchange it uses **only 43% of B's model calls**, and beats A by 2.1 points, CI [+1.3, +3.0], for 49% more calls.
3. **The RL coordinator learned what the reward pays for:** a balance between accuracy and cost. On validation its return (0.241) was higher than B's (0.211). On test the returns are close: −0.004 for C vs +0.004 for B. So **we cannot claim RL is "better"**. What we can say is that it reaches a different point on the accuracy–cost curve.
4. **Training is unstable across seeds:** with the same settings, test accuracy ranged from 50.9% to 55.3%, and cost from 1.4 to 4.5 calls. This is a fundamental limitation, documented in section 7.

## 2. Architecture

```
            question (no reference)
                   │
          ┌────────▼────────┐   attempt 0 (greedy) is always generated
          │  Solver (Qwen)  │◄──────────────── REGENERATE (sampled attempt)
          └────────┬────────┘
                   │ text
          ┌────────▼────────┐   basic check after every attempt
          │    Verifier     │◄──────────────── VERIFY_MORE (PoT, once)
          └────────┬────────┘
                   │ signals only (no reference)
          ┌────────▼────────┐
          │  Coordinator    │── ACCEPT ──► final answer = plurality vote
          │  (Q-learning)   │
          └─────────────────┘
   Reward (training/evaluation code only): uses the reference answer
```

- **The roles are software roles.** The "agents" are not independent language models. Qwen is used to generate solutions and to generate the PoT program the verifier asks for. The verifier itself is deterministic code.
- **Final answer:** plurality vote over the attempts that produced an extractable answer. Ties are broken in this order: PoT agreement, then arithmetic status, then the earliest attempt.

### 2.1 Solver (`src/solver/`)
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf…`, bf16 on an RTX 4050, frozen weights. The SHA-256 of the weights was checked against Hugging Face.
- **Prompt:** the official chat template and the default system message from the model card, asking: "Please reason step by step, and put your final answer within \boxed{}". Prompt version `solver-v1`.
- **Decoding:**

| Attempt | Settings | Source |
|---|---|---|
| Attempt 0 | greedy | — |
| Attempts 1–3 | sampling: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1 | official `generation_config.json` |

- **Limits:** 512 input tokens, 512 output tokens for a solution, 320 tokens for a PoT program. The longest actual prompt was 313 tokens.
- **Answer extraction:** the last `\boxed{}`. If none, an explicit phrase such as "final answer is". **We never take "the last number in the text"**; an unmarked answer counts as "not extractable".
- **Normalization:** all comparisons use exact `Fraction`s, not `float`.
- **Caveat:** we do not assume regeneration fixes mistakes. On train, the accuracy of sampled attempts (49–52%) is **lower** than greedy (59.6%).

### 2.2 Verifier (`src/verifier/`)
| Check | What it does | What it does **not** prove |
|---|---|---|
| Extractability | a numeric final answer exists | — |
| Explicit arithmetic | finds expressions like `12 × 3 = 36` (LaTeX supported) and evaluates them with a whitelisted safe AST evaluator, without `eval`, using exact Fractions | that the solution understood the problem. "No error" does not mean "correct" |
| Truncation | did generation stop at the length limit? | — |
| `VERIFY_MORE` = PoT | asks Qwen for a short program, runs it in a safe interpreter (numeric assignments only, no loops, no imports except `math`, no attributes), then compares the result with the candidate answer | agreement is correlated evidence (same model), not ground truth. PoT alone was 45% accurate on train |

- The verifier **never receives the reference answer**. This is tested explicitly: verifier functions have no reference parameter, and observations are identical when the reference is changed.
- The PoT check is deterministic (greedy), so it is **available once per episode** and then masked by the action mask. It is also masked when not available.

### 2.3 RL environment (`src/environment/`)
- **Episode:** one problem. Attempt 0 is always generated, then the coordinator decides.
- **Actions:** `ACCEPT`, `REGENERATE` (up to 4 attempts) and `VERIFY_MORE` (once). At most 5 steps.
- **At the limit:** only `ACCEPT` remains available, and the termination reason is logged as `accepted_budget_exhausted`. If there is no answer, `accepted_no_answer` is logged and the answer counts as wrong. **No case is dropped from evaluation.**
- **Observation:** runtime signals only: number of attempts, remaining budget, candidate answer and its support, number of distinct answers, arithmetic status, truncation, PoT result, and the action mask.
- **State discretization for the table** in `src/coordinator/features.py`:

| Feature | Values |
|---|---|
| `n_attempts` | 1–4 |
| `support` | 0, 1, 2, 3+ |
| `disagreement` | 0/1 |
| `cand_arith` | consistent / none found / error |
| `pot` | not run / agrees / disagrees / failed |

  The theoretical upper bound is 384 states; the deployed policy visited 144 of them.

### 2.4 Reward (`src/environment/reward.py`)
Return: G = terminal reward − sum of action costs.

| Event | Value |
|---|---|
| Correct answer | +1 |
| Wrong answer | −1 |
| No answer (abstain) | −1, so abstaining is not better than being wrong and cannot be exploited |
| `REGENERATE` | −0.05 (an extra generation of up to 512 tokens) |
| `VERIFY_MORE` | −0.03 (a shorter generation of up to 320 tokens) |

- The first attempt is free, because every system needs it.
- The costs are an up-front design decision **and were not tuned on validation or test**.
- The reference answer is used **only** inside the reward computation and the evaluation code.

## 3. Data

Full details in `docs/DATA_REPORT.md`.

| Use | File | Count |
|---|---|---|
| Coordinator training | `train_small` | 500 |
| Setting selection | `validation` (full) | 747 |
| Primary evaluation (pre-registered) | `test_small` | 200 |
| Extended evaluation | `test` (full) | 1,319 |

- The decision to use the full validation and test sets was made at 02:30, **after** measuring generation speed on train and **before** generating or evaluating anything on validation or test.
- `test_small` is a subset of `test` and is evaluated on exactly the same generated attempts.
- **The test example shown in the notebook** was used only to inspect the data format. Nothing from test was used to choose prompts, verifier rules, rewards or settings. The two code changes made after the first Qwen run were based on **train only**:
  - accepting a bare final expression as the last line of a PoT program;
  - protecting sampling against NaN values.

## 4. Training (`scripts/train_coordinator.py`)

- **Algorithm:** tabular Q-learning, γ=1, ε-greedy exploration during training **only**. ε decays linearly from 1.0. Inference with the final policy is greedy (no exploration).
- **Training data:** cached replay of `train_small` attempts; problems are sampled at random with a fixed seed.
- **Search grid:**

| Parameter | Values |
|---|---|
| constant α | {0.05, 0.1, 0.2} |
| episodes | {10k, 30k, 100k} |
| final ε | {0.05, 0.2} |
| α = 1/n (decays with visit count) | ×{10k, 30k, 100k} × {0.05, 0.2} |

  Each configuration is trained with 3 seeds and evaluated on validation.
- **Selection rule** (pre-registered): highest mean return on validation.
- **Selected configuration:** constant α=0.05, 100k episodes, final ε 0.2. Mean validation return 0.2073 ± 0.0051.
- **Transparency note:** the α=1/n option was added to the grid **after** the first run showed instability across seeds on validation. The first run is kept in `results/training/run1_constant_alpha_only/`. The new option was more stable (std 0.003–0.009) but had a slightly lower return (0.201). We kept the announced rule and did not change it after seeing the results.
- **Final training:** 5 seeds with the selected configuration. The deployed policy (`artifacts/policies/coordinator.json`) is **seed 1**, the highest return on validation.

**Validation results (747 problems):**

| Policy | Accuracy | Return | Calls / problem |
|---|---|---|---|
| A | 59.7% | 0.194 | 1.00 |
| B | 65.1% | 0.211 | 3.20 |
| max budget | 66.4% | 0.148 | 5.00 |
| C seed 0 | 66.4% | 0.175 | 4.44 |
| **C seed 1 (deployed)** | 62.9% | **0.241** | 1.40 |
| C seed 2 | 65.1% | 0.206 | 3.29 |
| C seed 3 | 65.3% | 0.148 | 4.50 |
| C seed 4 | 62.7% | 0.238 | 1.36 |

Figures: `results/figures/learning_curves.png` (including the noisy validation-return curve) and `results/figures/policy_qvalues.png`.

## 5. Evaluation (`scripts/evaluate.py`)

- **Fair comparison:** same problems, same model and revision, exactly the same generated attempts (cached replay), same verifier tools and the same limits (4 attempts and 1 PoT) for B and C.
- **The single-attempt comparison** (A vs B/C) **is not an equal-budget comparison.** That is why cost is always shown next to accuracy, and the "max budget" reference shows the accuracy ceiling when the full budget is used.

### 5.1 Full test set (n=1,319)

| Metric | A | B | C (deployed) | max budget |
|---|---|---|---|---|
| Accuracy | 48.7% | 55.3% | 50.9% | 55.0% |
| Coverage (an answer exists) | 93.4% | 99.4% | 99.4% | 99.4% |
| Accuracy on answered problems | 52.2% | 55.7% | 51.2% | 55.4% |
| "No answer" rate | 6.6% | 0.6% | 0.6% | 0.6% |
| Budget-exhausted rate | 0% | 32.5% | 7.7% | 99.4% |
| Mean solver attempts | 1.00 | 2.47 | 1.34 | 4.00 |
| Mean PoT checks | 0.00 | 0.99 | 0.15 | 1.00 |
| Mean verifier checks (basic + PoT) | 1.00 | 3.46 | 1.49 | 5.00 |
| Model calls | 1.00 | 3.46 | 1.49 | 5.00 |
| Input tokens | 105 | 430 | 171 | 584 |
| Output tokens | 308 | 915 | 451 | 1,348 |
| Generation time (s, amortized over the batch) | 0.36 | 1.01 | 0.50 | 1.48 |
| Mean return | −0.025 | +0.004 | −0.004 | −0.079 |
| Fallback decision rate | — | — | 0.9% | — |

**Paired comparisons** (bootstrap with 10,000 resamples, exact McNemar):

| Accuracy difference | Value | 95% CI | p |
|---|---|---|---|
| B − A | +6.6 | [+4.9, +8.4] | 5.6×10⁻¹³ |
| C − A | +2.1 | [+1.3, +3.0] | 2.5×10⁻⁷ |
| C − B | −4.5 | [−6.1, −2.9] | 7.5×10⁻⁸ |
| max budget − B | −0.3 | [−1.1, +0.5] | 0.61 |

**Variance of C across seeds:**

| Seed | Accuracy | Calls |
|---|---|---|
| 0 | 55.3% | 4.50 |
| 1 | 50.9% | 1.49 |
| 2 | 55.0% | 3.57 |
| 3 | 53.8% | 4.44 |
| 4 | 50.9% | 1.43 |

Mean **53.2% ± 1.9**.

### 5.2 test_small (n=200) — the pre-registered primary evaluation

| | A | B | C | max budget |
|---|---|---|---|---|
| Accuracy | 48.0% [41.2, 54.9] | 53.5% [46.6, 60.3] | 49.0% [42.2, 55.9] | 53.0% |
| Calls | 1.00 | 3.50 | 1.34 | 5.00 |

- B − A = +5.5, CI [+2.0, +9.5], p=0.007.
- C − A = +1.0, CI [0.0, +2.5], p=0.50.
- C − B = −4.5, CI [−8.0, −1.0], p=0.023.

Same direction as the full test set, but the confidence intervals on 200 problems are much wider (about ±7 points for absolute accuracy). **Differences of 1–2 points at this size are not meaningful.**

Figures:
- `results/figures/accuracy_vs_cost_test.png` and `accuracy_vs_cost_test_small.png`
- `results/figures/actions_test.png`

### 5.3 Latency measurement
- **Hardware:** RTX 4050 Laptop (6 GB), bf16, torch 2.9.1+cu126.
- **Method:** batch time on the GPU, with `cuda.synchronize`, divided by the number of sequences in the batch, measured when the cache was created. **Cache read time is never counted.**
- **Caveat:** about 176 validation problems were generated while another run was using the same GPU, so their time is inflated. Test was generated alone.
- **Live runs** (`solve_problem`) measure real time for a single question without batching: about 4–7 s per attempt on AC power.
- **Effect of power mode:** in a later check, on battery in Balanced mode and with no other process on the GPU, one attempt took 8–20 s. A `max_budget` request (5 calls) took 50–71 s. The web interface must show a loading state and enforce a timeout.

## 6. XAI — explaining coordinator decisions (`src/explainability/explain.py`)

For every decision we log:
- the observed state and its description;
- the available actions;
- the Q-value of each available action;
- the visit count of the state during training;
- the chosen action;
- the margin over the second-best action;
- a text generated **from a fixed template built on these numbers**, not by a language model.

**What the deployed policy actually learned on test:**
- It accepted the first attempt in **82%** of cases (1,081 of 1,319).
- Whenever the verifier found an **arithmetic error** in the first attempt, it chose `REGENERATE` (77 of 77).
- Whenever **no extractable answer** existed, it chose `REGENERATE` (87 of 87).

**A real example** (`gsm8k-test-00026`, reference 243), where A was wrong (233) and C fixed it:
1. `REGENERATE`, because Q=+0.086 beat VERIFY_MORE (Q=−0.067). Signal: arithmetic error in the first attempt.
2. `REGENERATE` again, because candidate 243 had the support of only one attempt.
3. `VERIFY_MORE` (Q=+0.308): candidate 233 was supported by two attempts but contained an arithmetic error.
4. `REGENERATE`, with a margin of +0.638 over ACCEPT: the PoT check **disagreed** with 233.
5. The budget was exhausted and the candidate became 243 (support 2, arithmetic consistent, PoT agrees), so `ACCEPT` was the only available action.

**Limits of the explanations:**
- Q-values are **estimates of expected return under the designed reward, not probabilities of correctness.**
- The explanations cover the coordinator's decision only **and do not reveal Qwen's internal reasoning**.
- If a state was not fully visited during training, the fixed rule is used as a **documented fallback**, and the explanation says so explicitly. This happened in 0.9% of decisions on test.

**Real success and failure examples:** `results/examples/examples_test.md`. Selection is deterministic (the first ids in each category). Categories:
- RL fixed A's mistake.
- RL accepted a wrong answer.
- B right, C wrong.
- C right, B wrong.
- All tools failed.

## 7. Limitations and caveats

1. **RL instability:** the same settings produced radically different policies across seeds; some are "economical" (about 1.4 calls) and some "expensive" (about 4.5 calls). The reason is that the Q-values of ACCEPT and REGENERATE are very close in the important states. Choosing the deployed seed on validation means choosing a point on the cost curve, not "the best policy" in an absolute sense.
2. **Validation/test gap:** A's accuracy on validation (59.7%, [56.2, 63.1]) is much higher than on test (48.7%, [46.1, 51.4]). Validation comes from the official GSM8K train file. **Our hypothesis** is that Qwen2.5 may have seen the train problems during pre-training, **and we cannot prove it**. This gap makes the first attempt look more reliable on validation than on test, which encourages an "accept early" policy. It is likely part of why C falls behind B on test.
3. **Possible contamination of test itself:** we cannot rule out that Qwen has seen GSM8K test problems. So **the evaluation does not prove generalization to completely new problems.**
4. **A single generation run:** attempts were generated once with one generation seed. The reported seed variance covers coordinator training only, not generation randomness.
5. **Cached replay:**
   - Sampled attempts were pre-generated as independent samples from the same prompt. This is distributionally equivalent to on-demand generation, but live mode (`solve_problem`) will not reproduce the same texts verbatim.
   - Even the greedy result can differ slightly between batch size 1 and larger batches because of padding and bf16 precision.
6. **Verifier limits:**
   - The arithmetic check cannot see comprehension mistakes.
   - PoT relies on the same model.
   - A live example: a new question whose answer is 10.5 was answered 11 by Qwen with "consistent" arithmetic, and the coordinator accepted it.
7. **Data size:** 500 training problems. The primary evaluation on 200 problems has wide intervals. The full test set is more precise but is still an estimate for the GSM8K distribution only.
8. **Technical incident:** a CUDA "illegal memory access" error happened once during sampling, while another run was using the same GPU. We added `remove_invalid_values` and `renormalize_logits`, which do not change the distribution when the logits are valid, plus automatic restart and resume. The error did not recur.
9. **Windows environment:** Smart App Control blocked numpy 2.5.5 and pandas 3.1.1. We pinned numpy 2.2.6 and dropped pandas.

## 8. Running and reproducing

See `README.md`. Main commands:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\generate_attempts.py --split train_small ; & $py scripts\generate_attempts.py --split validation --batch-size 32
& $py scripts\train_coordinator.py
& $py scripts\generate_attempts.py --split test --batch-size 32
& $py scripts\evaluate.py --split test ; & $py scripts\evaluate.py --split test_small --cache test
& $py scripts\make_plots.py --split test ; & $py scripts\select_examples.py --split test
& $py scripts\solve.py "A question in English..."
```

## 9. Suggested next steps (not done)

- Address the instability with one of:
  - averaging Q across several seeds (ensemble);
  - a selection rule that penalizes variance;
  - α=1/n with a pre-registered choice.
- Train the coordinator on data less likely to be contaminated, or calibrate the validation/test gap.
- Add state features such as solution length and truncation, or try DQN if the features grow (not needed yet).
- Web interface: `solve_problem()` is ready and returns serializable JSON.
