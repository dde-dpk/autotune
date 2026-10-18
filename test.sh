#!/usr/bin/env bash
# Smoke test for the autotune pipeline.
#
# Verifies the core claim end-to-end: the BASE model scores LOW on the eval set
# while the TRAINED adapter scores HIGH. The gap is the test passing.
#
# Assumes the adapter has already been trained and uploaded to S3, e.g.:
#   python3 train_lora.py --dataset data/training_data.json \
#     --output-dir ./out/newno --epochs 5 --s3-uri s3://loras-root/newno
# and that the vLLM server (gemma-lora container) is running.
#
# Usage:
#   ./test.sh [adapter_name]      # default adapter name: newno
#
# Env overrides: EVAL_FILE, MODE, MGR, PASS_THRESHOLD, BASE_CEILING

set -uo pipefail

MODEL="${1:-newno}"
BASE="google/gemma-3-4b-it"
EVAL_FILE="${EVAL_FILE:-data/eval_data.jsonl}"
MODE="${MODE:-exact}"
MGR="${MGR:-http://localhost:8001}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.90}"   # adapter must score >= this
BASE_CEILING="${BASE_CEILING:-0.10}"       # base must score <= this

say() { printf '\n=== %s ===\n' "$1"; }

# Run the eval harness for a given model id; echo per-prompt output to the
# terminal (stderr) and return just the match_rate float on stdout.
score() {
  python3 eval_adapter.py --model "$1" --eval-file "$EVAL_FILE" --mode "$MODE" \
    | tee /dev/stderr | sed -n 's/^match_rate: \([0-9.]*\).*/\1/p'
}

say "Stage 0: validate data files"
python3 -c "import json;[json.loads(l) for l in open('$EVAL_FILE') if l.strip()]; print('eval ok')" || exit 1
python3 -c "import json; json.load(open('data/training_data.json')); print('train ok')" || exit 1

say "Stage 1: vLLM health"
curl -sf "$MGR/health" >/dev/null || {
  echo "vLLM not reachable at $MGR — start it: docker start gemma-lora (wait for vllm_ready:true)"; exit 1; }
echo "server reachable"

say "Stage 2: baseline — base model should score LOW"
base_score=$(score "$BASE")

say "Stage 3: load adapter '$MODEL'"
curl -s -X POST "$MGR/v1/loras/unload" -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"$MODEL\",\"delete_local\":true}" >/dev/null
curl -s -X POST "$MGR/v1/loras/load" -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"$MODEL\"}"; echo

say "Stage 4: trained adapter should score HIGH"
adapter_score=$(score "$MODEL")

say "RESULT"
echo "base  ($BASE): ${base_score:-?}   (want <= $BASE_CEILING)"
echo "adapter ($MODEL): ${adapter_score:-?}   (want >= $PASS_THRESHOLD)"

awk -v a="${adapter_score:-0}" -v b="${base_score:-1}" \
    -v pt="$PASS_THRESHOLD" -v bc="$BASE_CEILING" 'BEGIN{
  if (a >= pt && b <= bc) { print "\nPASS - pipeline works (base low, adapter high)"; exit 0 }
  print "\nFAIL - see scores above"; exit 1
}'
