"""
Build a bigram calibration table by generating answers with the target model
(via a running SGLang server) across multiple datasets.

Usage:
  # 1. start SGLang server
  python -m sglang.launch_server \\
      --model /path/to/model --port 30000 --host 0.0.0.0 \\
      --dtype bfloat16 --trust-remote-code

  # 2. build table
  python scripts/build_bigram.py \\
      --tokenizer /path/to/tokenizer \\
      --server-address localhost:30000 \\
      --out bigram_table.pt \\
      [--datasets gsm8k math alpaca metamath] \\
      [--n-per-dataset 500] \\
      [--max-new-tokens 512] \\
      [--concurrency 64] \\
      [--min-count 2] \\
      [--max-per-prev 200]

Multiple --server-address values are supported (requests are round-robin'd).

Built-in dataset presets (--datasets key):
    gsm8k       gsm8k main/train          "question"
    math        lighteval/MATH all/train  "problem"
    alpaca      tatsu-lab/alpaca train     "instruction" [+ "input"]
    ultrachat   HuggingFaceH4/ultrachat_200k train_sft  first user turn
    metamath    meta-math/MetaMathQA train "query"
"""
import argparse
import itertools
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "gsm8k": dict(
        path="gsm8k", name="main", split="train",
        question_fn=lambda row: row["question"],
    ),
    "math": dict(
        path="DigitalLearningGmbH/MATH-lighteval", name="default", split="train",
        question_fn=lambda row: row["problem"],
    ),
    "alpaca": dict(
        path="tatsu-lab/alpaca", name=None, split="train",
        question_fn=lambda row: (
            row["instruction"] + "\n" + row["input"]
            if row.get("input") else row["instruction"]
        ),
    ),
    "ultrachat": dict(
        path="HuggingFaceH4/ultrachat_200k", name=None, split="train_sft",
        question_fn=lambda row: row["messages"][0]["content"],
    ),
    "metamath": dict(
        path="meta-math/MetaMathQA", name=None, split="train",
        question_fn=lambda row: row["query"],
    ),
    "codealpaca": dict(
        path="HuggingFaceH4/CodeAlpaca_20K", name=None, split="train",
        question_fn=lambda row: row["prompt"],
    ),
    "evol_instruct": dict(
        path="WizardLMTeam/WizardLM_evol_instruct_V2_196k", name=None, split="train",
        question_fn=lambda row: row["conversations"][0]["value"],
    ),
}


# ---------------------------------------------------------------------------
# Bigram table
# ---------------------------------------------------------------------------
def build_bigram_table(token_sequences, min_count: int = 2, max_per_prev: int = 200):
    counts: dict = defaultdict(lambda: defaultdict(int))
    for seq in token_sequences:
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1

    table = {}
    for prev, nxt_counts in counts.items():
        filtered = {k: v for k, v in nxt_counts.items() if v >= min_count}
        if not filtered:
            continue
        top = sorted(filtered.items(), key=lambda x: -x[1])[:max_per_prev]
        total = sum(v for _, v in top)
        toks = [k for k, _ in top]
        lps  = [math.log(v / total) for _, v in top]
        table[prev] = (
            torch.tensor(toks, dtype=torch.long),
            torch.tensor(lps,  dtype=torch.float32),
        )
    return table


# ---------------------------------------------------------------------------
# SGLang generation
# ---------------------------------------------------------------------------
def _wait_for_server(addr: str, timeout: int = 300, interval: int = 5) -> None:
    """Poll GET /health until the server responds 200 or timeout expires."""
    import time
    import urllib.request
    import urllib.error
    url = f"http://{addr}/health"
    deadline = time.time() + timeout
    print(f"waiting for server at {addr} …", flush=True)
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            print(f"server {addr} is ready", flush=True)
            return
        except Exception:
            time.sleep(interval)
    raise RuntimeError(f"server {addr} did not become ready within {timeout}s")


def _make_clients(server_addresses: list[str], wait: bool = True) -> list[OpenAI]:
    if wait:
        for addr in server_addresses:
            _wait_for_server(addr)
    return [
        OpenAI(base_url=f"http://{addr}/v1", api_key="EMPTY")
        for addr in server_addresses
    ]


def _call_one(client: OpenAI, model_name: str, question: str, max_new_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_new_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


def generate_sequences(
    server_addresses: list[str],
    model_name: str,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
    concurrency: int,
) -> list[list[int]]:
    """
    Send all questions concurrently to the SGLang server(s), collect
    generated text, tokenise into token-id sequences.
    """
    clients = _make_clients(server_addresses)
    client_cycle = itertools.cycle(clients)

    futures_map = {}
    results = [""] * len(questions)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for idx, q in enumerate(questions):
            client = next(client_cycle)
            fut = pool.submit(_call_one, client, model_name, q, max_new_tokens)
            futures_map[fut] = idx

        for fut in tqdm(as_completed(futures_map), total=len(futures_map),
                        desc="  generating", unit="sample"):
            idx = futures_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = ""
                tqdm.write(f"  warning: sample {idx} failed: {e}")

    # tokenise generated text → token id sequences (prompt excluded)
    seqs = []
    for text in results:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            seqs.append(ids)
    return seqs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True,
                    help="path to tokenizer (same vocab as target model)")
    ap.add_argument("--server-address", nargs="+", required=True,
                    metavar="HOST:PORT",
                    help="SGLang server address(es); requests are round-robin'd")
    ap.add_argument("--model", type=str, default=None,
                    help="model name sent in API requests "
                         "(defaults to tokenizer basename if omitted)")
    ap.add_argument("--out", required=True,
                    help="output path for the .pt bigram table")
    ap.add_argument("--datasets", nargs="+",
                    default=["gsm8k", "math", "alpaca", "metamath"],
                    choices=list(DATASET_REGISTRY),
                    help="datasets to sample from (default: gsm8k math alpaca metamath)")
    ap.add_argument("--n-per-dataset", type=int, default=500,
                    help="questions sampled per dataset (default 500)")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="max tokens generated per question (default 512)")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="concurrent HTTP requests (default 64)")
    ap.add_argument("--min-count", type=int, default=2,
                    help="minimum bigram count to keep (default 2)")
    ap.add_argument("--max-per-prev", type=int, default=200,
                    help="max next-token entries per conditioning token (default 200)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model_name = args.model or args.tokenizer.rstrip("/").split("/")[-1]

    all_seqs = []
    for ds_key in args.datasets:
        cfg = DATASET_REGISTRY[ds_key]
        print(f"\n[{ds_key}] loading {args.n_per_dataset} questions …")
        ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
        ds = ds.shuffle(seed=42).select(range(min(args.n_per_dataset, len(ds))))
        questions = [cfg["question_fn"](row) for row in ds]

        seqs = generate_sequences(
            args.server_address, model_name, tokenizer,
            questions, args.max_new_tokens, args.concurrency,
        )
        tok_count = sum(len(s) for s in seqs)
        print(f"[{ds_key}] {len(seqs)} sequences, {tok_count:,} tokens")
        all_seqs.extend(seqs)

    print(f"\ntotal sequences : {len(all_seqs):,}")
    print(f"total tokens    : {sum(len(s) for s in all_seqs):,}")

    table = build_bigram_table(all_seqs, args.min_count, args.max_per_prev)
    print(f"unique prev tokens  : {len(table):,}")
    print(f"total bigram entries: {sum(len(v[0]) for v in table.values()):,}")

    torch.save(table, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
