cd ~/autotune
claude -p "Read program.md. Do ONE experiment only, no looping: docker stop gemma-lora to free the GPU, train the LoRA on data/training_data.json into ./out/newno, docker start gemma-lora and wait for vllm_ready, load the newno adapter, then run ./test.sh newno and report the match_rate." \
  --permission-mode auto --permission-prompts none \
  --allowedTools "Bash,Read,Edit" \
  --max-budget-usd 10 \
  --output-format json
