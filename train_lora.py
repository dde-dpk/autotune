#!/usr/bin/env python3
"""Fine-tune a LoRA adapter on google/gemma-3-4b-it from a chat-format JSONL file.

Produces adapter_config.json + adapter_model.safetensors ready to serve on the
gemma-lora-server (same base model, rank <= MAX_LORA_RANK). Optionally uploads
the finished adapter to S3.

Dataset format: one JSON object per line, each with a "messages" list of
{"role": "user"|"assistant", "content": "..."} turns. See data/sample.jsonl.

Example:
  python train_lora.py \
    --dataset data/sample.jsonl \
    --output-dir ./out/pirate \
    --epochs 3 \
    --s3-uri s3://loras-root/pirate
"""

import argparse
import subprocess
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# LoRA is applied to the LANGUAGE-model projections only. We deliberately skip
# the vision tower (vLLM ignores vision-tower LoRA for Gemma 3 anyway).
LANG_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="google/gemma-3-4b-it")
    p.add_argument("--dataset", required=True, help="Path to chat-format JSONL (see data/sample_nono.jsonl).")
    p.add_argument("--output-dir", required=True, help="Where to save the adapter.")
    p.add_argument("--hf-token", default=None, help="HF token for the gated Gemma weights (or set HF_TOKEN env).")

    p.add_argument("--lora-rank", type=int, default=32, help="Must be <= the server's MAX_LORA_RANK.")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)

    p.add_argument("--bf16", action="store_true", help="Train in bf16 instead of 4-bit QLoRA (needs more VRAM).")
    p.add_argument("--s3-uri", default=None, help="If set, upload the finished adapter here (s3://...).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.lora_rank > 32:
        print(f"WARNING: --lora-rank {args.lora_rank} exceeds the server default MAX_LORA_RANK=32. "
              f"Raise MAX_LORA_RANK in the server .env or the adapter will fail to load.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)

    # --- dataset: render each example to a single string via the Gemma chat template ---
    ds = load_dataset("json", data_files=args.dataset, split="train")

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"Loaded {len(ds)} training examples.")

    # --- model: 4-bit QLoRA by default (fits 24GB), or full bf16 with --bf16 ---
    quant_config = None
    if not args.bf16:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto",
        token=args.hf_token,
        attn_implementation="eager",  # recommended for Gemma 3 stability
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LANG_TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_len,  # TRL >=0.20 renamed max_seq_length -> max_length
        dataset_text_field="text",
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nAdapter saved to {args.output_dir}")

    if args.s3_uri:
        print(f"Uploading adapter to {args.s3_uri} ...")
        subprocess.run(
            ["aws", "s3", "cp", args.output_dir, args.s3_uri,
             "--recursive", "--exclude", "checkpoint-*/*"],
            check=True,
        )
        name = args.s3_uri.rstrip("/").split("/")[-1]
        print(f"\nDone. Load it on the server with:\n"
              f"  curl -X POST localhost:8001/v1/loras/load "
              f"-H 'Content-Type: application/json' -d '{{\"lora_name\":\"{name}\"}}'")


if __name__ == "__main__":
    main()
