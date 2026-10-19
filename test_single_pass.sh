#!/usr/bin/env bash
# Run ONE automated experiment (no looping) via Claude Code.
#
# Usage:
#   ./test_single_pass.sh [adapter_name]      # default: newno
#   ./test_single_pass.sh jokeans
#
# Prereqs: Claude Code authenticated (Bedrock via instance role), HF_TOKEN set,
# and the gemma-lora vLLM container available on the box.

set -uo pipefail

NAME="${1:-newno}"

cd "$(dirname "$0")"

claude -p "Read program.md. Do ONE experiment only, no looping: docker stop gemma-lora to free the GPU, train the LoRA on data/training_data.json into ./out/$NAME with --s3-uri s3://loras-root/$NAME, docker start gemma-lora and wait for vllm_ready, load the $NAME adapter, then run ./test_manual.sh $NAME (use the --mode indicated by program.md) and report the match_rate." \
  --permission-mode auto --permission-prompts none \
  --allowedTools "Bash,Read,Edit" \
  --max-budget-usd 10 \
  --output-format json
