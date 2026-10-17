# program.md — autonomous LoRA customization (style / behaviour / format)

An agent (e.g. Claude Code) reads this file, then generates data, tunes the
training flags, evaluates, and iterates — the autoresearch loop, scoped to
*checkable* style/behaviour/format tasks only.

---

## 1. TASK  ← the only part a human edits

> Describe, in plain English, the style / behaviour / format you want the
> fine-tuned model to have. Be specific about what "correct" looks like.

Example:
> The model must answer **every** question with exactly: `No no no... I cant answer!`

Swap in your own task before launching the loop.

---

## 2. OBJECTIVE / METRIC

- Scored by `eval_adapter.py`, which prints `match_rate: <float>` (0.0–1.0, higher is better).
- **Dev target:** `match_rate >= 0.95` on `data/<task>_dev.jsonl`.
- **Final:** once dev target is met, the human scores the frozen `holdout/<task>_eval.jsonl`.
  That holdout number is the real result — the dev score is just for iterating.

---

## 3. WHAT YOU (the agent) MAY CHANGE

- The **training set** at `autotune/data/training_data.json` (chat-format JSON) — you MAY
  edit, add, remove, or rewrite its examples to improve the task behaviour.
- `data/<task>_dev.jsonl` — generate the eval prompts you iterate against
  (DIFFERENT prompts from the training set, to measure generalization — never
  copy training rows into dev).
- `train_lora.py` **flags only**: `--lora-rank` (≤ 32), `--lora-alpha`,
  `--lora-dropout`, `--epochs`, `--lr`, `--max-seq-len`.
- The eval `--mode` and `--target` that best capture the task (see §6).

## 4. WHAT YOU MUST NOT CHANGE

- `app/`, `scripts/`, `Dockerfile`, the serving stack.
- `holdout/*` — frozen, human-owned, final scoring only. Do not read or edit it.
- `eval_adapter.py` — reuse the harness; do not rewrite the scorer.
- `train_lora.py` **code** (edit flags on the command line, not the file).

---

## 5. HOW TO RUN ONE EXPERIMENT

GPU rule: training and the vLLM server can't both own the 24 GB A10G — stop the
server while training, restart it to evaluate.

```bash
# 1. free the GPU
docker stop gemma-lora

# 2. train with the current knobs (training code + data are local to this repo)
python train_lora.py --dataset data/training_data.json --output-dir ./out/<task> --epochs <n>

# 3. serve + load the fresh adapter for eval
docker start gemma-lora                     # wait until :8001/health reports vllm_ready:true
curl -X POST localhost:8001/v1/loras/unload -H 'Content-Type: application/json' \
     -d '{"lora_name":"<task>","delete_local":true}'   # clear any prior copy
# load from the local output dir produced in step 2 (manager resolves S3 by default;
# for a just-trained local adapter, upload with --s3-uri or point the loader at ./out/<task>)

# 4. score it
python eval_adapter.py --model <task> --eval-file data/<task>_dev.jsonl --mode <mode>
```

Read the final `match_rate`. **Keep** the data/flags if it went up; **revert** and
try something else if it didn't.

---

## 6. SCORER GUIDE — pick `--mode` by task type

| Task type | `--mode` | `expected` in the eval JSONL |
|---|---|---|
| Fixed response / exact behaviour | `exact` | the exact target string |
| Persona / style markers | `contains` | a list of required words, e.g. `["arr","matey"]` |
| Format constraint (JSON, pattern, length) | `regex` | a regex, e.g. `^\s*\{.*\}\s*$` |

Keep to these checkable modes. If a task needs subjective judgement
("sound more formal"), either reduce it to markers/format you *can* check, or
stop and ask the human — do not invent an LLM-judge inside the loop.

---

## 7. LOOP PROTOCOL

- Budget: **<N>** experiments or **<M>** minutes — stop when you hit it.
- Keep a running log of `(change made, match_rate)` so decisions are reversible.
- Always keep the best adapter seen so far.
- Stop early when dev `match_rate >= 0.95`; then hand off to the human for the
  holdout score. Report the log and the best config.
