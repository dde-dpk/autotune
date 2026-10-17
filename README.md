# autotune

An autoresearch-style autonomous loop for **LoRA customization**, scoped to
*checkable* style / behaviour / format tasks.

You describe a task in plain English in [`program.md`](program.md); an agent then
generates training data, tunes the fine-tuning hyperparameters, evaluates with a
deterministic scorer, and iterates toward the objective — keeping the best adapter.

## Contents

| File | Purpose |
|---|---|
| [`program.md`](program.md) | The driver. A human fills in the **TASK**; the rest is the fixed contract the agent follows (what it may edit, how to run, the loop protocol). |
| [`train_lora.py`](train_lora.py) | QLoRA fine-tuning script. Trains a LoRA adapter on `data/training_data.json`; the agent tunes its flags. |
| [`requirements-train.txt`](requirements-train.txt) | Training dependencies (torch, transformers, peft, trl, …) for a local venv. |
| [`eval_adapter.py`](eval_adapter.py) | Deterministic eval harness. Queries a running vLLM server and prints a scalar `match_rate`. Modes: `exact`, `contains`, `regex`. |
| [`data/training_data.json`](data/training_data.json) | The training set (chat-format). The agent may edit it to improve the task behaviour. |
| [`data/eval_data.jsonl`](data/eval_data.jsonl) | Example eval set (held-out prompts for the "always refuse" demo task). |

## Scope

Works well for tasks whose success is *checkable without judgement*:

- **Fixed behaviour** (always refuse with a set phrase) → `--mode exact`
- **Persona / style markers** (pirate-speak, formal tone) → `--mode contains`
- **Format constraints** (JSON-only, length, patterns) → `--mode regex`

Capability tasks ("be more accurate") are deliberately out of scope — the eval
there is the hard part and easily gamed.

## Usage

Training is **self-contained** in this repo: [`train_lora.py`](train_lora.py)
(QLoRA fine-tuning) + [`requirements-train.txt`](requirements-train.txt) +
[`data/training_data.json`](data/training_data.json). Set up a venv and install the training deps:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
```

**Serving + scoring** still uses the sibling `gemma-lora-server` project (the vLLM
container on :8000/:8001): train the adapter here, serve it there, then
[`eval_adapter.py`](eval_adapter.py) scores it over HTTP. See
[`program.md`](program.md) for the full loop.
