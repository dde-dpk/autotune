#!/usr/bin/env bash
# Run the automated autotune loop with NO time limit.
#
# Hands program.md to Claude Code and lets it iterate freely. Bounded only by
# the dev target (match_rate >= 0.95) and a cost cap (safety guardrail, not a
# clock). For a wall-clock-bounded run instead, use ./test_timed.sh.
#
# Usage:
#   ./test_auto.sh [budget_usd]      # default $25 cost cap
#   ./test_auto.sh 50                # looser cost cap
#
# Prereqs: Claude Code authenticated (Bedrock via instance role), HF_TOKEN set,
# and the gemma-lora vLLM container available on the box.

set -uo pipefail

BUDGET="${1:-25}"

cd "$(dirname "$0")"

start=$SECONDS
claude -p "Read program.md and run the customization loop: train on data/training_data.json, serve, score with ./test_manual.sh, tune the train_lora.py flags and training data, keep the best adapter, and iterate. Stop when dev match_rate >= 0.95 or your budget is exhausted. Log each (change, match_rate) and report the best config at the end." \
  --permission-mode auto --permission-prompts none \
  --allowedTools "Bash,Read,Edit" \
  --max-budget-usd "$BUDGET" \
  --output-format json
code=$?
elapsed=$((SECONDS - start))

echo
echo "[test_auto] cost cap=\$$BUDGET  elapsed=${elapsed}s  exit=$code"
exit "$code"
