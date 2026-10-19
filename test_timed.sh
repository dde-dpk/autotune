#!/usr/bin/env bash
# Run the automated autotune loop under a wall-clock TIME budget.
#
# Wraps `claude -p` with `timeout -s INT` so the run is cut off at the cap and
# still ends the current turn cleanly (prints a final report). Keeps a cost cap
# as a secondary guardrail.
#
# Usage:
#   ./test_timed.sh [duration] [budget_usd]
#   ./test_timed.sh 30m 25        # defaults: 30-minute cap, $25 cost cap
#   ./test_timed.sh 3m 10         # quick check that the time cap actually fires
#
# Exit code: 0 = loop ended on its own (target met / budget-aware stop);
#            124 = stopped by the time budget. `elapsed` below is the robust signal.

set -uo pipefail

DURATION="${1:-30m}"
BUDGET="${2:-25}"

cd "$(dirname "$0")"

start=$SECONDS
timeout -s INT "$DURATION" claude -p "Read program.md and run the customization loop: train on data/training_data.json, serve, score with ./test_manual.sh, tune the train_lora.py flags and training data, keep the best adapter, and iterate. Stop when dev match_rate >= 0.95 or your budget is exhausted. Log each (change, match_rate) and report the best config at the end." \
  --permission-mode auto --permission-prompts none \
  --allowedTools "Bash,Read,Edit" \
  --max-budget-usd "$BUDGET" \
  --output-format json
code=$?
elapsed=$((SECONDS - start))

echo
echo "[test_timed] time cap=$DURATION  cost cap=\$$BUDGET  elapsed=${elapsed}s  exit=$code"
case "$code" in
  0)   echo "[test_timed] loop ended on its own (target met or stopped within budget)";;
  124) echo "[test_timed] STOPPED by the time budget";;
  *)   echo "[test_timed] exited with code $code (see output above)";;
esac
exit "$code"
