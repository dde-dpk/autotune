#!/usr/bin/env bash
# Regenerate data/eval_data.jsonl from the goal in program.md, using held-out
# questions that do NOT appear in data/training_data.json.
#
# Pure file task: no GPU, no vLLM, no HF_TOKEN needed. Just Claude Code
# authenticated (Bedrock via instance role).
#
# Usage:
#   ./gen_eval.sh [budget_usd]        # default $5 cost cap

set -uo pipefail

BUDGET="${1:-5}"

cd "$(dirname "$0")"

claude -p "Update the evaluation file for this project.
1. Read program.md to get the current TASK (section 1) and the scoring --mode (section 6).
2. Read data/training_data.json and extract EVERY user question in it.
3. Overwrite data/eval_data.jsonl with 10 NEW held-out eval rows that fit the task, using questions that do NOT appear in data/training_data.json and are not trivial rewordings of them (they must be genuinely different, to measure generalization).
4. Use only questions with stable, well-known factual answers (no current time/date/weather).
5. Format each row as one JSON object per line with keys 'prompt' and 'expected', where 'expected' matches the current --mode so a correct response passes. For the joke-then-answer task (--mode contains) use: {\"all\": [\"Actual answer is\", \"<the correct answer>\"], \"any\": [the AI/ML keyword list already used in eval_data.jsonl]}.
6. After writing, print the list of training questions and the list of new eval questions side by side and confirm there is zero overlap." \
  --permission-mode auto --permission-prompts none \
  --allowedTools "Read,Write,Edit" \
  --max-budget-usd "$BUDGET" \
  --output-format json
