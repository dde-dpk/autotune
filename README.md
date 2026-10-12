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
| [`eval_adapter.py`](eval_adapter.py) | Deterministic eval harness. Queries a running vLLM server and prints a scalar `match_rate`. Modes: `exact`, `contains`, `regex`. |
| [`data/eval_nono.jsonl`](data/eval_nono.jsonl) | Example eval set (held-out prompts for the "always refuse" demo task). |

## Scope

Works well for tasks whose success is *checkable without judgement*:

- **Fixed behaviour** (always refuse with a set phrase) → `--mode exact`
- **Persona / style markers** (pirate-speak, formal tone) → `--mode contains`
- **Format constraints** (JSON-only, length, patterns) → `--mode regex`

Capability tasks ("be more accurate") are deliberately out of scope — the eval
there is the hard part and easily gamed.

## Usage

The training/serving stack it drives lives in a sibling project
(`gemma-lora-server`): `train_lora.py` produces the adapter, vLLM serves it, and
`eval_adapter.py` scores it. See [`program.md`](program.md) for the full loop.
