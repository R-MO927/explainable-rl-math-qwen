---
title: MathMates
emoji: 🧮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Explainable multi-agent RL math solver with Qwen2.5-0.5B
---

# MathMates — Explainable Multi-Agent RL for Mathematical Reasoning

A university project. Three roles solve grade-school math word problems:

- **Robo (Solver):** `Qwen/Qwen2.5-0.5B-Instruct`, frozen weights.
- **Hoot (Verifier):** checks the written arithmetic with a safe evaluator and can run a small program check. It never sees an answer key.
- **Captain Q (Coordinator):** tabular Q-learning; decides to accept, try again, or double-check. Every decision is explained with its Q-values.

The page is in English and Arabic. Questions should be in English.

This Space runs on free CPU hardware, so each try can take 20–60 seconds, and one question is solved at a time.
The results section shows the saved evaluation on the GSM8K test set (1,319 problems); it does not change with your question.

Code, data preparation, training and the full report: https://github.com/R-MO927/explainable-rl-math-qwen

Model: Qwen2.5-0.5B-Instruct (Apache-2.0). Data: GSM8K (MIT). KaTeX (MIT).
