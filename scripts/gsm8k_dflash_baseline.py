import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def format_topk_row(tok_ids, logits, tokenizer, highlight_id=None):
    parts = []
    for tid, lg in zip(tok_ids.tolist(), logits.tolist()):
        piece = tokenizer.decode([tid]).replace("\n", "\\n")
        marker = "*" if (highlight_id is not None and tid == highlight_id) else " "
        parts.append(f"{marker}{tid:>6d}:{piece!r:<14s}({lg:+.2f})")
    return " ".join(parts)


def print_comparison(debug_iters, tokenizer, max_iters_to_print=3,
                     max_positions_per_iter=None):
    total_positions = 0
    total_agree = 0
    for it_idx, it in enumerate(debug_iters):
        d_ids = it["draft_top_ids"]
        t_ids = it["target_top_ids"]
        d_lg = it["draft_top_logits"]
        t_lg = it["target_top_logits"]
        accepted = it["acceptance_length"]
        n_pos = d_ids.shape[0]
        for i in range(n_pos):
            total_positions += 1
            if d_ids[i, 0].item() == t_ids[i, 0].item():
                total_agree += 1

        if it_idx >= max_iters_to_print:
            continue
        print(f"\n--- iter {it_idx}  start={it['start']}  "
              f"accepted={accepted}/{it['block_size'] - 1} ---")
        n_print = n_pos if max_positions_per_iter is None else min(n_pos, max_positions_per_iter)
        final_tokens = it.get("final_tokens")
        d_rank = it.get("draft_final_rank")
        t_rank = it.get("target_final_rank")
        for i in range(n_print):
            target_tok = int(it["target_posterior"][i].item())
            draft_tok = int(it["draft_proposed"][i].item())
            final_tok = int(final_tokens[i].item()) if final_tokens is not None else -1
            agree = "OK " if d_ids[i, 0].item() == t_ids[i, 0].item() else "DIFF"
            acc = "acc" if i < accepted else ("ovr" if i == accepted else "rej")
            dr = int(d_rank[i].item()) if d_rank is not None else -1
            tr = int(t_rank[i].item()) if t_rank is not None else -1
            if final_tok < 0:
                final_str = "(beyond end-of-seq)"
            else:
                final_str = f"golden={final_tok} rank[draft]={dr} rank[target]={tr}"
            print(f"  pos {i} [{agree}|{acc}] draft={draft_tok} target={target_tok} {final_str}")
            # marker '*' = token that actually landed in the final output sequence
            hl = None if final_tok < 0 else final_tok
            print(f"    draft : {format_topk_row(d_ids[i], d_lg[i], tokenizer, highlight_id=hl)}")
            print(f"    target: {format_topk_row(t_ids[i], t_lg[i], tokenizer, highlight_id=hl)}")
    return total_positions, total_agree


def print_oracle_comparison(debug_iters, tokenizer, max_iters_to_print=3):
    for it_idx, it in enumerate(debug_iters):
        if it_idx >= max_iters_to_print:
            break
        accepted = it["acceptance_length"]
        n_pos = it["draft_top_ids"].shape[0]
        noise_tokens = it.get("noise_tokens")  # [block_size-1]
        print(f"\n--- oracle iter {it_idx}  start={it['start']}  "
              f"accepted={accepted}/{it['block_size'] - 1} ---")
        ctx = it["context_token_id"]
        print(f"  context(pos0): {ctx} {tokenizer.decode([ctx])!r}")
        for i in range(n_pos):
            noise_tok = int(noise_tokens[i].item()) if noise_tokens is not None else -1
            draft_tok = int(it["draft_proposed"][i].item())
            target_tok = int(it["target_posterior"][i].item())
            acc = "acc" if i < accepted else ("ovr" if i == accepted else "rej")
            noise_str = (f"{noise_tok} {tokenizer.decode([noise_tok]).replace(chr(10), repr(chr(10)))!r}"
                         if noise_tok >= 0 else "mask")
            d_top = it["draft_top_ids"][i]
            d_lg  = it["draft_top_logits"][i]
            t_top = it["target_top_ids"][i]
            t_lg  = it["target_top_logits"][i]
            match = "==" if draft_tok == target_tok else "!="
            print(f"  pos {i} [{acc}]  noise={noise_str}  "
                  f"draft={draft_tok} {match} target={target_tok}")
            print(f"    draft : {format_topk_row(d_top, d_lg, tokenizer, highlight_id=target_tok)}")
            print(f"    target: {format_topk_row(t_top, t_lg, tokenizer, highlight_id=target_tok)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--print-iters", type=int, default=2,
                    help="how many iterations to print detailed top5 for per sample")
    ap.add_argument("--print-samples", type=int, default=5,
                    help="how many gsm8k samples to print detailed top5 for")
    ap.add_argument("--save", type=str, default=None,
                    help="optional path to save debug dump as torch .pt")
    ap.add_argument("--bigram-table", type=str, default=None,
                    help="path to bigram calibration table (.pt) built by build_bigram.py")
    ap.add_argument("--lambda-bigram", type=float, default=1.0,
                    help="weight for bigram log-prob bonus (default 1.0)")
    args = ap.parse_args()

    bigram_table = None
    if args.bigram_table is not None:
        bigram_table = torch.load(args.bigram_table, weights_only=False)
        print(f"loaded bigram table: {len(bigram_table):,} conditioning tokens  "
              f"(lambda={args.lambda_bigram})")

    draft = AutoModel.from_pretrained(
        "/mnt/data/szf_temp/huggingface/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4",
        trust_remote_code=True, dtype="auto", device_map="cuda:0",
    ).eval()
    target = AutoModelForCausalLM.from_pretrained(
        "/mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        dtype="auto", device_map="cuda:0",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        trust_remote_code=True,
    )

    ds = load_dataset("gsm8k", "main", split=f"test[:{args.n}]")

    all_debug = []
    grand_pos = 0
    grand_agree = 0
    all_ref_acc = []
    all_oracle_acc = []
    # per-iter rank vectors (length n_pred each, -1 for positions beyond seq end)
    all_iter_d_ranks = []
    all_iter_t_ranks = []
    for idx, row in enumerate(ds):
        question = row["question"]
        messages = [{"role": "user", "content": question}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False,
        ).to(draft.device)

        print(f"\n============================== sample {idx} ==============================")
        print(f"Q: {question[:200]}")

        output_ids, acceptance_lengths, debug = draft.spec_generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            target=target,
            stop_token_ids=[tokenizer.eos_token_id],
            return_debug=True,
            debug_topk=5,
            bigram_table=bigram_table,
            lambda_bigram=args.lambda_bigram,
        )
        gen = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
        print(f"A: {gen[:300]}")

        pos, agree = print_comparison(
            debug, tokenizer,
            max_iters_to_print=args.print_iters if idx < args.print_samples else 0,
        )
        grand_pos += pos
        grand_agree += agree

        for it in debug:
            d_rank = it.get("draft_final_rank")
            t_rank = it.get("target_final_rank")
            if d_rank is None:
                continue
            all_iter_d_ranks.append(d_rank.tolist())
            all_iter_t_ranks.append(t_rank.tolist())
        print(f"[sample {idx}] top1-agree {agree}/{pos} = "
              f"{(agree / pos * 100 if pos else 0):.1f}%")

        # Oracle-noise run: feed ref output_ids as noise to measure acceptance upper bound.
        _, oracle_acc_lengths, oracle_debug = draft.spec_generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            target=target,
            stop_token_ids=[tokenizer.eos_token_id],
            return_debug=True,
            debug_topk=5,
            oracle_noise_ids=output_ids,
        )
        ref_mean = sum(acceptance_lengths) / len(acceptance_lengths)
        oracle_mean = sum(oracle_acc_lengths) / len(oracle_acc_lengths)
        all_ref_acc.extend(acceptance_lengths)
        all_oracle_acc.extend(oracle_acc_lengths)
        print(f"[sample {idx}] acc  baseline={ref_mean:.2f}  oracle_noise={oracle_mean:.2f}  "
              f"gain={oracle_mean - ref_mean:+.2f}")
        if idx < args.print_samples:
            print_oracle_comparison(oracle_debug, tokenizer,
                                    max_iters_to_print=args.print_iters)

        if args.save is not None:
            all_debug.append({"idx": idx, "question": question, "debug": debug})

    print("\n================ SUMMARY ================")
    print(f"samples: {args.n}")
    print(f"total predicted positions: {grand_pos}")
    print(f"draft-target top1 agreement: {grand_agree}/{grand_pos} = "
          f"{(grand_agree / grand_pos * 100 if grand_pos else 0):.2f}%")
    if all_ref_acc:
        grand_ref = sum(all_ref_acc) / len(all_ref_acc)
        grand_oracle = sum(all_oracle_acc) / len(all_oracle_acc)
        print(f"mean acceptance  baseline={grand_ref:.2f}  oracle_noise={grand_oracle:.2f}  "
              f"gain={grand_oracle - grand_ref:+.2f}")

    print("\n-------- per-block-position rank stats (golden = final sequence) --------")
    print("legend:")
    print("  draft_top1%        = P(draft rank == 1 at this pos)")
    print("  cum_all_top1%      = P(draft rank == 1 at every pos 1..i)  [denom: iters with all those positions valid]")
    print("  prev_one_miss%     = P(exactly one non-top1 among positions 1..i-1 AND current pos i is top1) [denom: iters with positions 1..i valid]")
    n_pred = max((len(r) for r in all_iter_d_ranks), default=0)
    header = (f"{'block_pos':>9s}  {'count':>7s}  "
              f"{'draft_mean':>10s} {'draft_med':>9s} {'draft_p90':>9s} {'draft_top1%':>11s} "
              f"{'cum_all_top1%':>14s} {'prev_one_miss%':>15s}  "
              f"{'tgt_mean':>9s} {'tgt_med':>8s} {'tgt_top1%':>10s}")
    print(header)
    for i in range(n_pred):
        d_at_i = [r[i] for r in all_iter_d_ranks if r[i] > 0]
        t_at_i = [r[i] for r in all_iter_t_ranks if r[i] > 0]
        n = len(d_at_i)
        if n == 0:
            continue
        d_at_i_sorted = sorted(d_at_i)
        t_at_i_sorted = sorted(t_at_i)
        d_mean = sum(d_at_i) / n
        t_mean = sum(t_at_i) / n
        d_med = d_at_i_sorted[n // 2]
        t_med = t_at_i_sorted[n // 2]
        d_p90 = d_at_i_sorted[min(n - 1, int(n * 0.9))]
        d_top1 = sum(1 for v in d_at_i if v == 1) / n * 100
        t_top1 = sum(1 for v in t_at_i if v == 1) / n * 100

        # cumulative all-top1 over draft positions 0..i
        cum_n = cum_all = 0
        for r in all_iter_d_ranks:
            if any(r[j] <= 0 for j in range(i + 1)):
                continue
            cum_n += 1
            if all(r[j] == 1 for j in range(i + 1)):
                cum_all += 1
        cum_pct = (cum_all / cum_n * 100) if cum_n else 0.0

        # exactly one non-top1 among positions 0..i-1 AND current pos i is top1
        if i == 0:
            prev_one_pct = 0.0
        else:
            prev_n = prev_one = 0
            for r in all_iter_d_ranks:
                if any(r[j] <= 0 for j in range(i + 1)):
                    continue
                prev_n += 1
                misses = sum(1 for j in range(i) if r[j] != 1)
                if misses == 1 and r[i] == 1:
                    prev_one += 1
            prev_one_pct = (prev_one / prev_n * 100) if prev_n else 0.0

        print(f"{i + 1:>9d}  {n:>7d}  "
              f"{d_mean:>10.2f} {d_med:>9d} {d_p90:>9d} {d_top1:>10.2f}% "
              f"{cum_pct:>13.2f}% {prev_one_pct:>14.2f}%  "
              f"{t_mean:>9.2f} {t_med:>8d} {t_top1:>9.2f}%")

    if args.save is not None:
        torch.save(all_debug, args.save)
        print(f"saved debug dump -> {args.save}")


if __name__ == "__main__":
    main()
