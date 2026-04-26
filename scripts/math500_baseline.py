"""
Plain speculative-decode acceptance-length baseline on MATH-500.
No CFG, no tricks — just draft.spec_generate() vs target.

Usage:
  python scripts/math500_baseline.py --n 50
  python scripts/math500_baseline.py --n 500 --block-size 16
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

DRAFT_PATH = (
    "/mnt/data/szf_temp/huggingface/models--z-lab--Qwen3-8B-DFlash-b16"
    "/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4"
)
TARGET_PATH = (
    "/mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-8B"
    "/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of MATH-500 problems to eval")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=16,
                    help="block size for reporting stats (must match model: DFlash-b16 → 16)")
    ap.add_argument("--print-samples", type=int, default=5)
    args = ap.parse_args()

    n_pred = args.block_size - 1

    draft = AutoModel.from_pretrained(
        DRAFT_PATH, trust_remote_code=True, dtype="auto", device_map="cuda:0"
    ).eval()
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, dtype="auto", device_map="cuda:0"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)

    if draft.mask_token_id is None:
        draft.mask_token_id = getattr(tokenizer, "mask_token_id", None) or (
            draft.config.vocab_size - 1
        )

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.n < len(ds):
        ds = ds.select(range(args.n))

    all_acc: list[int] = []
    t_total = time.perf_counter()

    for idx, row in enumerate(tqdm(ds, desc="math500")):
        messages = [{"role": "user", "content": row["problem"]}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
        ).to(draft.device)

        output_ids, acceptance_lengths = draft.spec_generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            target=target,
            stop_token_ids=[tokenizer.eos_token_id],
            return_debug=False,
        )[:2]

        all_acc.extend(acceptance_lengths)
        mean_s = sum(acceptance_lengths) / len(acceptance_lengths) - 1

        if idx < args.print_samples:
            gen = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
            print(f"\n====== sample {idx} ======")
            print(f"Q: {row['problem'][:120]}")
            print(f"A: {gen[:200]}")
            print(f"[sample {idx}] mean_acc={mean_s:.2f}/{n_pred}  ({len(acceptance_lengths)} iters)")

    elapsed = time.perf_counter() - t_total
    actual = [a - 1 for a in all_acc]
    mean_a   = sum(actual) / len(actual)
    top1_pct = sum(1 for a in actual if a >= 1) / len(actual) * 100
    full_pct = sum(1 for a in actual if a == n_pred) / len(actual) * 100

    print("\n================ SUMMARY ================")
    print(f"dataset: MATH-500  samples: {args.n}  block_size: {args.block_size}  "
          f"n_pred: {n_pred}")
    print(f"  mean_acc = {mean_a:.3f}/{n_pred}")
    print(f"  top1%    = {top1_pct:.1f}%  (at least 1 token accepted)")
    print(f"  full%    = {full_pct:.1f}%  (all {n_pred} tokens accepted)")
    print(f"  total_steps = {len(all_acc)}  elapsed = {elapsed:.1f}s")


if __name__ == "__main__":
    main()
