#!/usr/bin/env python3
"""Simple, deterministic evaluation for a served LoRA adapter (or the base model).

Sends each prompt in an eval JSONL to the running vLLM OpenAI server on :8000 and
scores the reply with a checkable rule (no LLM judge). Prints a per-prompt
PASS/FAIL breakdown and a single scalar `match_rate` (0.0-1.0) at the end -- the
comparable number an autoresearch-style loop optimizes.

Scoring modes (--mode), covering style / behaviour / format tasks:
  exact    reply equals `expected` (whitespace/case-insensitive). For fixed-response
           behaviour, e.g. always refuse with a set phrase.
  contains reply contains `expected`. If `expected` is a LIST, ALL items must be
           present. For persona/style marker words, e.g. pirate-speak.
  regex    `expected` is a regex (re.search, IGNORECASE|DOTALL). If a LIST, ALL
           patterns must match. For format constraints, e.g. JSON-only output.

`expected` can also be an OBJECT to combine a required set with an either/or set:
  {"all": [...], "any": [...]}  -> ALL of `all` must match AND at least one of `any`.
Use `any` as a heuristic OR-check, e.g. "does the reply contain an AI joke?" by
listing AI/ML keywords -- any one of them appearing counts as the joke being present.
  {"prompt":"capital of France?",
   "expected":{"all":["Actual answer is","Paris"],
               "any":["inference","gradient","neural","model","token"]}}

Prereqs: vLLM is up and the adapter is loaded, e.g.
  curl -X POST localhost:8001/v1/loras/load -H 'Content-Type: application/json' \
       -d '{"lora_name":"nono"}'

Eval file format: one JSON object per line:
  {"prompt": "some question", "expected": "No no no... I cant answer!"}
  {"prompt": "tell me a story", "expected": ["arr", "matey"]}      # contains mode
  {"prompt": "give me data", "expected": "^\\\\s*\\\\{.*\\\\}\\\\s*$"}       # regex mode
`expected` is optional; if omitted it falls back to --target.

Example:
  python eval_adapter.py --model nono --eval-file data/eval_data.jsonl --mode exact
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request


def normalize(s: str) -> str:
    """Whitespace- and case-insensitive compare so trivial formatting differences
    (trailing newline, double spaces, capitalization) don't count as failures."""
    return " ".join(s.strip().lower().split())


def score_one(out: str, expected, mode: str) -> bool:
    n = normalize(out)

    def present(x):              # substring test (normalized)
        return normalize(x) in n

    def matches(p):              # regex test (raw text)
        return re.search(p, out, re.IGNORECASE | re.DOTALL) is not None

    def check_group(expected, test):
        # expected can be:
        #   str  -> must satisfy `test`
        #   list -> ALL items must satisfy `test`
        #   dict -> ALL of expected["all"] AND at least one of expected["any"]
        #           (use "any" for an OR group, e.g. "is a joke present?" via AI keywords)
        if isinstance(expected, dict):
            all_ok = all(test(x) for x in expected.get("all", []))
            any_list = expected.get("any", [])
            any_ok = (not any_list) or any(test(x) for x in any_list)
            return all_ok and any_ok
        items = expected if isinstance(expected, list) else [expected]
        return all(test(x) for x in items)

    if mode == "exact":
        target = expected[0] if isinstance(expected, list) else expected
        return normalize(out) == normalize(target)
    if mode == "contains":
        return check_group(expected, present)
    if mode == "regex":
        return check_group(expected, matches)
    raise ValueError(f"unknown --mode {mode!r}")


def chat(base_url: str, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="nono", help="Model id to query (loaded lora_name or the base id).")
    p.add_argument("--eval-file", default="data/eval_data.jsonl", help="JSONL of {prompt, expected} rows.")
    p.add_argument("--mode", default="exact", choices=["exact", "contains", "regex"],
                   help="Scoring rule. See module docstring.")
    p.add_argument("--target", default="No no no... I cant answer!",
                   help="Fallback expected value when a row omits `expected`.")
    p.add_argument("--base-url", default="http://localhost:8000", help="vLLM OpenAI server base URL.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = deterministic, so the score is stable.")
    args = p.parse_args()

    examples = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if not examples:
        print(f"No eval examples found in {args.eval_file}", file=sys.stderr)
        sys.exit(1)

    passed = 0
    for ex in examples:
        prompt = ex["prompt"]
        expected = ex.get("expected", args.target)
        try:
            out = chat(args.base_url, args.model, prompt, args.max_tokens, args.temperature)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} querying model {args.model!r}: {e.read().decode()}", file=sys.stderr)
            print("Is vLLM up and the adapter loaded? (POST :8001/v1/loras/load)", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"Could not reach {args.base_url}: {e}", file=sys.stderr)
            sys.exit(1)

        ok = score_one(out, expected, args.mode)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r}\n        -> {out!r}")

    score = passed / len(examples)
    print(f"\nmatch_rate: {score:.3f}  ({passed}/{len(examples)})  mode={args.mode}")


if __name__ == "__main__":
    main()
