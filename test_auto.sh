#!/usr/bin/env bash
# Run the automated autotune loop with NO time limit.
#
# Hands program.md to Claude Code and lets it iterate freely. Bounded only by
# the dev target (match_rate >= 0.95) and a cost cap (safety guardrail, not a
# clock). For a wall-clock-bounded run instead, use ./test_timed.sh.
#
# Usage:
#   ./test_auto.sh [adapter_name] [budget_usd]   # defaults: newno, $25
#   ./test_auto.sh jokeans                        # target "jokeans", $25 cap
#   ./test_auto.sh jokeans 50                     # target "jokeans", $50 cap
#
# Prereqs: Claude Code authenticated (Bedrock via instance role), HF_TOKEN set,
# and the gemma-lora vLLM container available on the box.

set -uo pipefail

NAME="${1:-newno}"
BUDGET="${2:-25}"

cd "$(dirname "$0")"

start=$SECONDS
claude -p "Read program.md and run the customization loop for an adapter named $NAME: train on data/training_data.json into ./out/$NAME with --s3-uri s3://loras-root/$NAME, serve, load the $NAME adapter, and score it with ./test_manual.sh $NAME (use the --mode indicated by program.md). Tune the train_lora.py flags and training data, keep the best adapter, and iterate. Stop when match_rate >= 0.95 or your budget is exhausted. Log each (change, match_rate) and report the best config at the end." \
  --permission-mode auto --permission-prompts none \
  --allowedTools "Bash,Read,Edit" \
  --max-budget-usd "$BUDGET" \
  --output-format json
code=$?
elapsed=$((SECONDS - start))

echo
echo "[test_auto] adapter=$NAME  cost cap=\$$BUDGET  elapsed=${elapsed}s  exit=$code"
exit "$code"
