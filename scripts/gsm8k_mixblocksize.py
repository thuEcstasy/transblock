"""
Evaluate DFlash draft model with N independent bidirectional sub-blocks in one forward.

MODE: original (default, 3 sub-blocks of 16)
  Layout per step:
    sub-block A : [c0, mask×15]  full bidirectional
    sub-block B : [c0, mask×15]  random attention dropout
    sub-block C : [c0, mask×15]  diagonal only (position-only signal)
  Acceptance: standard sequential cumprod over all 3*15=45 draft positions.

MODE: ensemble (1 full bidir + 4 random dropout blocks)
  Layout per step (5 sub-blocks of 16 = 80 positions total):
    block-0    : [c0, mask×15]  full bidirectional
    block-1..4 : [c0, mask×15]  each independent random dropout (fresh every step)
  TWO acceptance strategies are evaluated each step:
    bidir     — use block-0 proposals for target verification (real speculation)
    ensemble  — sum logits of blocks 1-4, sample ensemble proposals, compare vs posterior
                (evaluated analytically against the same target posterior; no extra target pass)
  Reports per-step and aggregate acceptance for both strategies.

Usage:
  python gsm8k_mixblocksize.py --n 50
  python gsm8k_mixblocksize.py --mode ensemble --n 50
  python gsm8k_mixblocksize.py --sub-block-size 16 --n-sub-blocks 3
  python gsm8k_mixblocksize.py --n 5 --print-iters 2 --debug-topk 5
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import time
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

DRAFT_PATH = (
    "/mnt/data/szf_temp/huggingface/models--z-lab--Qwen3-8B-DFlash-b16"
    "/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4"
)
TARGET_PATH = (
    "/mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-8B"
    "/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)

DEFAULT_SUB_BLOCK_SIZE = 16
DEFAULT_N_SUB_BLOCKS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_topk_row(tok_ids, logits, tokenizer, highlight_id=None):
    parts = []
    for tid, lg in zip(tok_ids.tolist(), logits.tolist()):
        piece = tokenizer.decode([tid]).replace("\n", "\\n")
        marker = "*" if (highlight_id is not None and tid == highlight_id) else " "
        parts.append(f"{marker}{tid:>6d}:{piece!r:<14s}({lg:+.2f})")
    return " ".join(parts)


def print_mixblock_debug(
    debug_iters,
    tokenizer,
    n_sub_blocks: int,
    sub_block_size: int,
    max_iters: int = 2,
    max_pos: int = None,
) -> None:
    """Print per-sub-block draft logits vs target logits for each predicted position."""
    n_pred = sub_block_size - 1
    for it_idx, it in enumerate(debug_iters):
        if it_idx >= max_iters:
            break
        start_pos = it["start"]
        acc = it["acceptance_length"]
        print(f"\n--- iter {it_idx}  start={start_pos}  accepted={acc}/{n_pred * n_sub_blocks} ---")
        n_print = n_pred if max_pos is None else min(n_pred, max_pos)
        for pos in range(n_print):
            target_tok = int(it["target_posterior"][pos].item())
            # per-sub-block acceptance indicator (for the sequential check, sub-A only)
            acc_tag = "acc" if pos < acc else ("ovr" if pos == acc else "rej")
            print(f"  pos +{pos + 1} [{acc_tag}]  target={target_tok} "
                  f"{tokenizer.decode([target_tok]).replace(chr(10), repr(chr(10)))!r}")
            for si, sub in enumerate(it["sub_blocks"]):
                d_top_ids = sub["top_ids"][pos]    # [k]
                d_top_lg  = sub["top_logits"][pos]  # [k]
                proposed  = int(sub["proposed"][pos].item())
                prop_mark = "*" if proposed == target_tok else " "
                print(f"    sub-{chr(ord('A') + si)} (proposed={prop_mark}{proposed}): "
                      f"{format_topk_row(d_top_ids, d_top_lg, tokenizer, highlight_id=target_tok)}")
            t_ids = it["target_top_ids"][pos]
            t_lg  = it["target_top_logits"][pos]
            print(f"    target                         : "
                  f"{format_topk_row(t_ids, t_lg, tokenizer, highlight_id=target_tok)}")


def print_ensemble_debug(
    debug_iters,
    tokenizer,
    sub_block_size: int,
    max_iters: int = 2,
    max_pos: int = None,
) -> None:
    """Print per-position logits: bidir / each dropout block / ensemble(sum) / target."""
    n_pred = sub_block_size - 1
    for it_idx, it in enumerate(debug_iters):
        if it_idx >= max_iters:
            break
        start_pos = it["start"]
        b_acc = it["bidir_acc"]
        e_acc = it["ensemble_acc"]
        n_dropout = len(it["dropout_tops"])
        print(
            f"\n--- iter {it_idx}  start={start_pos}  "
            f"bidir_acc={b_acc}/{n_pred}  ensemble_acc={e_acc}/{n_pred} ---"
        )
        n_print = n_pred if max_pos is None else min(n_pred, max_pos)
        for pos in range(n_print):
            target_tok = int(it["target_posterior"][pos].item())
            b_prop = int(it["bidir_proposed"][pos].item())
            e_prop = int(it["ensemble_proposed"][pos].item())
            b_acc_tag  = "acc" if pos < b_acc else ("ovr" if pos == b_acc else "rej")
            e_acc_tag  = "acc" if pos < e_acc else ("ovr" if pos == e_acc else "rej")
            print(
                f"  pos +{pos + 1}  target={target_tok} "
                f"{tokenizer.decode([target_tok]).replace(chr(10), repr(chr(10)))!r}"
            )
            # bidir block
            b_mark = "*" if b_prop == target_tok else " "
            print(
                f"    bidir    [{b_acc_tag}] (proposed={b_mark}{b_prop}): "
                f"{format_topk_row(it['bidir_top_ids'][pos], it['bidir_top_logits'][pos], tokenizer, highlight_id=target_tok)}"
            )
            # individual dropout blocks
            for di, dp in enumerate(it["dropout_tops"]):
                print(
                    f"    dropout{di + 1}          (logits only ): "
                    f"{format_topk_row(dp['top_ids'][pos], dp['top_logits'][pos], tokenizer, highlight_id=target_tok)}"
                )
            # ensemble (sum of dropout logits)
            e_mark = "*" if e_prop == target_tok else " "
            print(
                f"    ensemble [{e_acc_tag}] (proposed={e_mark}{e_prop}): "
                f"{format_topk_row(it['ensemble_top_ids'][pos], it['ensemble_top_logits'][pos], tokenizer, highlight_id=target_tok)}"
            )
            # target ground truth
            print(
                f"    target                              : "
                f"{format_topk_row(it['target_top_ids'][pos], it['target_top_logits'][pos], tokenizer, highlight_id=target_tok)}"
            )


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab = logits.shape
    logits = logits.view(-1, vocab) / temperature
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).view(bsz, seq_len)


def _extract_context(hidden_states, layer_ids: list[int]) -> torch.Tensor:
    return torch.cat([hidden_states[lid + 1] for lid in layer_ids], dim=-1)


# ---------------------------------------------------------------------------
# Per-step attention mask (rebuilt each iteration for stochastic sub-block B)
# ---------------------------------------------------------------------------

def build_noise_mask_for_step(
    n_sub_blocks: int,
    sub_block_size: int,
    dtype: torch.dtype,
    device,
    dropout_prob_b: float = 0.5,
) -> torch.Tensor:
    """
    Additive attention mask [1, 1, total, total] for the noise portion.
    Rebuilt every step so sub-block B gets a fresh random pattern.
    No cross-sub-block attention in any sub-block.

      sub-block A : full bidirectional — all sub_block_size tokens attend to all
      sub-block B : bidirectional base with random attention dropout.
                    Each (i,j) position kept with prob (1 - dropout_prob_b).
                    Diagonal always kept to avoid all-masked rows.
      sub-block C : diagonal only — no inter-position noise attention.
                    Each token sees only itself; all content signal comes from
                    cross-attention to target_hidden + its own positional embedding.
    0.0 = attend, -inf = masked.
    """
    total = n_sub_blocks * sub_block_size
    attend = torch.zeros(total, total, dtype=torch.bool, device=device)

    for si in range(n_sub_blocks):
        s = si * sub_block_size
        if si == 0:
            # Sub-block A: full bidirectional
            attend[s : s + sub_block_size, s : s + sub_block_size] = True
        elif si == 1:
            # Sub-block B: random dropout over every (i,j) pair
            keep = torch.bernoulli(
                torch.full(
                    (sub_block_size, sub_block_size),
                    1.0 - dropout_prob_b,
                    device=device,
                )
            ).bool()
            keep.fill_diagonal_(True)   # never drop self-attention
            attend[s : s + sub_block_size, s : s + sub_block_size] = keep
        else:
            # Sub-block C: diagonal only (position-only signal)
            for i in range(sub_block_size):
                attend[s + i, s + i] = True

    mask = torch.full(
        (1, 1, total, total), torch.finfo(dtype).min, dtype=dtype, device=device
    )
    mask.masked_fill_(attend, 0.0)
    return mask


def make_full_mask(noise_mask: torch.Tensor, ctx_cols: int) -> torch.Tensor:
    """Prepend ctx_cols fully-attended columns to the noise-only mask."""
    bsz, heads, q, kv_noise = noise_mask.shape
    left = torch.zeros(bsz, heads, q, ctx_cols, dtype=noise_mask.dtype, device=noise_mask.device)
    return torch.cat([left, noise_mask], dim=-1)


def build_noise_mask_ensemble(
    n_dropout_blocks: int,
    sub_block_size: int,
    dtype: torch.dtype,
    device,
    dropout_prob: float = 0.5,
) -> torch.Tensor:
    """
    Additive attention mask for the ensemble mode: 1 full bidir + n_dropout_blocks random.

      block-0          : full bidirectional (all sub_block_size positions attend to all)
      block-1..N       : each independently random dropout; diagonal always kept.

    No cross-block attention anywhere.
    0.0 = attend, -inf = masked.
    """
    n_total = 1 + n_dropout_blocks
    total = n_total * sub_block_size
    attend = torch.zeros(total, total, dtype=torch.bool, device=device)

    # block-0: full bidir
    attend[0:sub_block_size, 0:sub_block_size] = True

    # block-1..N: random dropout, fresh each call
    for bi in range(1, n_total):
        s = bi * sub_block_size
        keep = torch.bernoulli(
            torch.full((sub_block_size, sub_block_size), 1.0 - dropout_prob, device=device)
        ).bool()
        keep.fill_diagonal_(True)
        attend[s : s + sub_block_size, s : s + sub_block_size] = keep

    mask = torch.full(
        (1, 1, total, total), torch.finfo(dtype).min, dtype=dtype, device=device
    )
    mask.masked_fill_(attend, 0.0)
    return mask


# ---------------------------------------------------------------------------
# Mixed-blocksize speculative generate
# ---------------------------------------------------------------------------

@torch.inference_mode()
def spec_generate_mixblock(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    n_sub_blocks: int = DEFAULT_N_SUB_BLOCKS,
    dropout_prob_b: float = 0.5,
    return_debug: bool = False,
    debug_topk: int = 5,
) -> tuple[torch.LongTensor, list[int], list[list[int]]]:
    """
    Returns:
      output_ids         : generated token ids
      acceptance_lengths : per-step (acceptance_length + 1) using standard sequential check
      sub_acceptances    : per-step list of per-sub-block acceptance counts
                           (hierarchical check skipping anchor positions, for analysis only)
    dropout_prob_b: probability of dropping each attention connection in sub-block B (default 0.5)
    """
    total_noise = sub_block_size * n_sub_blocks          # 48
    n_pred_per_sub = sub_block_size - 1                  # 15 per sub-block
    # Positions within noise that are ANCHOR tokens (first slot of each sub-block)
    anchor_pos = [i * sub_block_size for i in range(n_sub_blocks)]   # [0, 16, 32]

    device = input_ids.device
    mask_dtype = next(draft.parameters()).dtype
    # noise_mask is rebuilt each step (sub-block B is stochastic)

    orig_impl = draft.config._attn_implementation
    draft.config._attn_implementation = "eager"

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    mask_id = draft.mask_token_id
    if mask_id is None:
        raise ValueError("draft.mask_token_id is None")

    output_ids = torch.full(
        (1, max_length + total_noise), mask_id, dtype=torch.long, device=device
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    past_kv_target = DynamicCache()
    past_kv_draft = DynamicCache()

    # ---- Prefill ----
    out = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_kv_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = _sample(out.logits, temperature)
    target_hidden = _extract_context(out.hidden_states, draft.target_layer_ids)

    acceptance_lengths: list[int] = []
    all_sub_acceptances: list[list[int]] = []
    debug_iters: list[dict] = [] if return_debug else None
    start = num_input_tokens
    t0 = time.perf_counter()
    pbar = tqdm(total=max_new_tokens, desc="mixblock", unit="tok")

    while start < max_length:
        # ---- Build block_ids ----
        # All anchor positions get the context token (c0 = last accepted token at `start`)
        block_ids = torch.full(
            (1, total_noise), mask_id, dtype=torch.long, device=device
        )
        c0 = output_ids[0, start].item()
        for ap in anchor_pos:
            block_ids[0, ap] = c0

        noise_emb = target.model.embed_tokens(block_ids)  # [1, 48, hidden]

        # ---- Build shared position IDs: all sub-blocks use start..start+sub_block_size-1 ----
        # position_ids layout: [cached_len..start-1 | start..start+15 | start..start+15 | ...]
        # apply_rotary_pos_emb uses cos[..., -q_len:, :] for q, so q (total_noise tokens)
        # naturally gets the repeated sub-block positions regardless of prefix length.
        cached_len = past_kv_draft.get_seq_length()
        prefix_pos = torch.arange(cached_len, start, device=device)
        block_sub_pos = torch.arange(start, start + sub_block_size, device=device)
        custom_pos_ids = torch.cat(
            [prefix_pos, block_sub_pos.repeat(n_sub_blocks)]
        ).unsqueeze(0)  # [1, (start-cached_len) + total_noise]

        # ---- Rebuild mask each step: sub-block B gets a fresh random dropout pattern ----
        noise_mask = build_noise_mask_for_step(
            n_sub_blocks, sub_block_size, mask_dtype, device,
            dropout_prob_b=dropout_prob_b,
        )
        full_mask = make_full_mask(noise_mask, ctx_cols=start)

        draft_out = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=custom_pos_ids,
            attention_mask=full_mask,
            past_key_values=past_kv_draft,
            use_cache=True,
        )
        past_kv_draft.crop(start)

        # ---- Per-sub-block logits and sampling ----
        n_pred = sub_block_size - 1
        sub_draft_logits = []
        for si in range(n_sub_blocks):
            pred_s = si * sub_block_size + 1   # skip anchor at si * sub_block_size
            pred_e = (si + 1) * sub_block_size
            sub_logits = target.lm_head(draft_out[:, pred_s:pred_e, :])  # [1, n_pred, V]
            sub_draft_logits.append(sub_logits)
            block_ids[:, pred_s:pred_e] = _sample(sub_logits)

        # ---- Target verification ----
        block_pos = position_ids[:, start : start + total_noise]
        out = target(
            block_ids,
            position_ids=block_pos,
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = _sample(out.logits, temperature)  # [1, 48]

        # ---- Standard sequential acceptance (positions 1..47 vs posterior 0..46) ----
        acceptance_length = int(
            (block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )

        # ---- Hierarchical sub-block acceptance (for analysis, skip anchor positions) ----
        sub_accs: list[int] = []
        for si in range(n_sub_blocks):
            p_start = si * sub_block_size + 1          # first prediction in this sub-block
            p_end = (si + 1) * sub_block_size          # exclusive
            # posterior[p_start-1 .. p_end-2] predicts positions p_start..p_end-1
            draft_preds = block_ids[:, p_start:p_end]
            target_preds = posterior[:, p_start - 1 : p_end - 1]
            acc = int((draft_preds == target_preds).cumprod(dim=1).sum(dim=1)[0].item())
            sub_accs.append(acc)
            if acc < n_pred_per_sub:
                break  # stop as soon as a sub-block is not fully accepted
        # Pad to n_sub_blocks for uniform shape
        while len(sub_accs) < n_sub_blocks:
            sub_accs.append(-1)  # -1 = not reached
        all_sub_acceptances.append(sub_accs)

        # ---- Debug collection ----
        if return_debug:
            # target logits for first n_pred positions (sub-block A range in target)
            t_logits = out.logits[0, :n_pred]             # [n_pred, V]
            t_top = torch.topk(t_logits, k=debug_topk, dim=-1)
            sub_debug = []
            for si, sl in enumerate(sub_draft_logits):
                d_top = torch.topk(sl[0], k=debug_topk, dim=-1)
                sub_debug.append({
                    "top_ids":    d_top.indices.cpu(),          # [n_pred, k]
                    "top_logits": d_top.values.float().cpu(),   # [n_pred, k]
                    "proposed":   block_ids[0, si * sub_block_size + 1 :
                                               (si + 1) * sub_block_size].cpu(),  # [n_pred]
                })
            debug_iters.append({
                "start":            start,
                "acceptance_length": acceptance_length,
                "sub_blocks":       sub_debug,
                "target_top_ids":   t_top.indices.cpu(),      # [n_pred, k]
                "target_top_logits": t_top.values.float().cpu(),
                "target_posterior": posterior[0, :n_pred].cpu(),
            })

        # ---- Commit ----
        output_ids[:, start : start + acceptance_length + 1] = block_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        step_adv = acceptance_length + 1
        start += step_adv
        acceptance_lengths.append(acceptance_length + 1)
        pbar.update(step_adv)

        elapsed = time.perf_counter() - t0
        pbar.set_postfix({
            "sub_acc": "/".join(str(a) for a in sub_accs if a >= 0),
            "acc": f"{acceptance_length}/{total_noise - 1}",
            "tok/s": f"{(start - num_input_tokens) / elapsed:.1f}",
        })

        past_kv_target.crop(start)
        target_hidden = _extract_context(out.hidden_states, draft.target_layer_ids)[
            :, :step_adv, :
        ]

        if stop_token_ids and any(
            sid in output_ids[:, num_input_tokens:] for sid in stop_token_ids
        ):
            break

    pbar.close()
    draft.config._attn_implementation = orig_impl

    elapsed = time.perf_counter() - t0
    total_new = start - num_input_tokens
    n_pred_total = n_pred_per_sub * n_sub_blocks
    print(
        f"[mixblock] {total_new} tok in {elapsed:.2f}s | "
        f"{total_new / elapsed:.1f} tok/s | "
        f"mean acc {sum(acceptance_lengths) / len(acceptance_lengths):.2f}/{n_pred_total} | "
        f"{len(acceptance_lengths)} iters"
    )

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_id]
    if stop_token_ids:
        stop_t = torch.tensor(stop_token_ids, device=device)
        stops = torch.isin(output_ids[0][num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if stops.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stops[0] + 1]

    if return_debug:
        return output_ids, acceptance_lengths, all_sub_acceptances, debug_iters
    return output_ids, acceptance_lengths, all_sub_acceptances


# ---------------------------------------------------------------------------
# Ensemble speculative generate: 1 bidir block + 4 random-dropout blocks
# ---------------------------------------------------------------------------

@torch.inference_mode()
def spec_generate_ensemble(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    n_dropout_blocks: int = 4,
    dropout_prob: float = 0.5,
    return_debug: bool = False,
    debug_topk: int = 5,
) -> tuple:
    """
    5 sub-blocks per step: 1 full bidir (block-0) + 4 random dropout (blocks 1-4).

    Target verification uses block-0 proposals (the real speculation chain).
    Ensemble logits = sum of logits from blocks 1-4; ensemble proposals are sampled
    from this and compared against the same target posterior (no extra target pass).

    Returns:
      output_ids        : generated token ids (from bidir chain)
      bidir_acc_lens    : per-step acceptance lengths from block-0 (standard seq check)
      ensemble_acc_lens : per-step acceptance lengths from ensemble proposals vs posterior
    """
    n_total_blocks = 1 + n_dropout_blocks
    total_noise = sub_block_size * n_total_blocks
    n_pred = sub_block_size - 1

    device = input_ids.device
    mask_dtype = next(draft.parameters()).dtype

    orig_impl = draft.config._attn_implementation
    draft.config._attn_implementation = "eager"

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    mask_id = draft.mask_token_id
    if mask_id is None:
        raise ValueError("draft.mask_token_id is None")

    output_ids = torch.full(
        (1, max_length + total_noise), mask_id, dtype=torch.long, device=device
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    past_kv_target = DynamicCache()
    past_kv_draft = DynamicCache()

    # ---- Prefill ----
    out = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_kv_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = _sample(out.logits, temperature)
    target_hidden = _extract_context(out.hidden_states, draft.target_layer_ids)

    bidir_acc_lens: list[int] = []
    ensemble_acc_lens: list[int] = []
    debug_iters: list[dict] = [] if return_debug else None

    start = num_input_tokens
    t0 = time.perf_counter()
    pbar = tqdm(total=max_new_tokens, desc="ensemble", unit="tok")

    while start < max_length:
        # ---- Build block_ids: anchor at each sub-block start, rest mask ----
        block_ids = torch.full(
            (1, total_noise), mask_id, dtype=torch.long, device=device
        )
        c0 = output_ids[0, start].item()
        for bi in range(n_total_blocks):
            block_ids[0, bi * sub_block_size] = c0

        noise_emb = target.model.embed_tokens(block_ids)

        # ---- Position IDs: all sub-blocks share start..start+sub_block_size-1 ----
        cached_len = past_kv_draft.get_seq_length()
        prefix_pos = torch.arange(cached_len, start, device=device)
        block_sub_pos = torch.arange(start, start + sub_block_size, device=device)
        custom_pos_ids = torch.cat(
            [prefix_pos, block_sub_pos.repeat(n_total_blocks)]
        ).unsqueeze(0)

        # ---- Mask: fresh random dropout for blocks 1-4 each step ----
        noise_mask = build_noise_mask_ensemble(
            n_dropout_blocks, sub_block_size, mask_dtype, device, dropout_prob=dropout_prob
        )
        full_mask = make_full_mask(noise_mask, ctx_cols=start)

        draft_out = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=custom_pos_ids,
            attention_mask=full_mask,
            past_key_values=past_kv_draft,
            use_cache=True,
        )
        past_kv_draft.crop(start)

        # ---- Per-block logits ----
        # block-0: bidir proposals (used for target verification)
        bidir_logits = target.lm_head(draft_out[:, 1:sub_block_size, :])      # [1, n_pred, V]
        bidir_proposed = _sample(bidir_logits)                                  # [1, n_pred]
        block_ids[:, 1:sub_block_size] = bidir_proposed

        # blocks 1-4: sum logits → ensemble proposals
        ensemble_logits = torch.zeros_like(bidir_logits)
        dropout_logits_list = []
        for bi in range(1, n_total_blocks):
            s = bi * sub_block_size
            bl = target.lm_head(draft_out[:, s + 1 : s + sub_block_size, :])  # [1, n_pred, V]
            dropout_logits_list.append(bl)
            ensemble_logits = ensemble_logits + bl
        ensemble_proposed = _sample(ensemble_logits)                            # [1, n_pred]

        # ---- Target verification (on bidir proposals) ----
        block_pos = position_ids[:, start : start + total_noise]
        out = target(
            block_ids,
            position_ids=block_pos,
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = _sample(out.logits, temperature)   # [1, total_noise]

        # ---- Bidir acceptance: sequential check on positions 1..sub_block_size-1 ----
        bidir_acc = int(
            (bidir_proposed == posterior[:, :n_pred]).cumprod(dim=1).sum(dim=1)[0].item()
        )

        # ---- Ensemble acceptance: compare ensemble proposals vs same posterior ----
        # posterior[:, :n_pred] = target's response to bidir_proposed[0..n_pred-1]
        # For greedy (temp=0), posterior is deterministic given context → valid comparison.
        ensemble_acc = int(
            (ensemble_proposed == posterior[:, :n_pred]).cumprod(dim=1).sum(dim=1)[0].item()
        )

        # ---- Debug collection ----
        if return_debug:
            t_logits = out.logits[0, :n_pred]
            t_top = torch.topk(t_logits, k=debug_topk, dim=-1)
            bd_top = torch.topk(bidir_logits[0], k=debug_topk, dim=-1)
            ens_top = torch.topk(ensemble_logits[0], k=debug_topk, dim=-1)
            dp_tops = [torch.topk(bl[0], k=debug_topk, dim=-1) for bl in dropout_logits_list]
            debug_iters.append({
                "start": start,
                "bidir_acc": bidir_acc,
                "ensemble_acc": ensemble_acc,
                "bidir_top_ids":    bd_top.indices.cpu(),
                "bidir_top_logits": bd_top.values.float().cpu(),
                "bidir_proposed":   bidir_proposed[0].cpu(),
                "ensemble_top_ids":    ens_top.indices.cpu(),
                "ensemble_top_logits": ens_top.values.float().cpu(),
                "ensemble_proposed":   ensemble_proposed[0].cpu(),
                "dropout_tops": [
                    {"top_ids": dp.indices.cpu(), "top_logits": dp.values.float().cpu()}
                    for dp in dp_tops
                ],
                "target_top_ids":    t_top.indices.cpu(),
                "target_top_logits": t_top.values.float().cpu(),
                "target_posterior":  posterior[0, :n_pred].cpu(),
            })

        # ---- Commit bidir chain ----
        output_ids[:, start : start + bidir_acc + 1] = block_ids[:, : bidir_acc + 1]
        output_ids[:, start + bidir_acc + 1] = posterior[:, bidir_acc]

        step_adv = bidir_acc + 1
        start += step_adv
        bidir_acc_lens.append(bidir_acc + 1)
        ensemble_acc_lens.append(ensemble_acc + 1)
        pbar.update(step_adv)

        elapsed = time.perf_counter() - t0
        pbar.set_postfix({
            "bidir": f"{bidir_acc}/{n_pred}",
            "ens":   f"{ensemble_acc}/{n_pred}",
            "tok/s": f"{(start - num_input_tokens) / elapsed:.1f}",
        })

        past_kv_target.crop(start)
        target_hidden = _extract_context(out.hidden_states, draft.target_layer_ids)[
            :, :step_adv, :
        ]

        if stop_token_ids and any(
            sid in output_ids[:, num_input_tokens:] for sid in stop_token_ids
        ):
            break

    pbar.close()
    draft.config._attn_implementation = orig_impl

    elapsed = time.perf_counter() - t0
    total_new = start - num_input_tokens
    print(
        f"[ensemble] {total_new} tok in {elapsed:.2f}s | "
        f"{total_new / elapsed:.1f} tok/s | "
        f"bidir mean acc {sum(bidir_acc_lens) / len(bidir_acc_lens) - 1:.2f}/{n_pred} | "
        f"ensemble mean acc {sum(ensemble_acc_lens) / len(ensemble_acc_lens) - 1:.2f}/{n_pred} | "
        f"{len(bidir_acc_lens)} iters"
    )

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_id]
    if stop_token_ids:
        stop_t = torch.tensor(stop_token_ids, device=device)
        stops = torch.isin(output_ids[0][num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if stops.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stops[0] + 1]

    if return_debug:
        return output_ids, bidir_acc_lens, ensemble_acc_lens, debug_iters
    return output_ids, bidir_acc_lens, ensemble_acc_lens


def print_ensemble_stats(
    all_bidir: list[int],
    all_ensemble: list[int],
    n_pred: int,
) -> None:
    actual_bidir = [a - 1 for a in all_bidir]
    actual_ens   = [a - 1 for a in all_ensemble]
    n = len(actual_bidir)
    mean_b = sum(actual_bidir) / n
    mean_e = sum(actual_ens) / n
    print("\n-------- ensemble mode acceptance --------")
    print(f"  {'strategy':<12s}  {'mean acc':>9s}  {'top1%':>7s}  {'full%':>7s}")
    for label, accs in [("bidir", actual_bidir), ("ensemble", actual_ens)]:
        top1_pct  = sum(1 for a in accs if a >= 1) / n * 100
        full_pct  = sum(1 for a in accs if a == n_pred) / n * 100
        mean_a    = sum(accs) / n
        print(f"  {label:<12s}  {mean_a:>8.2f}/{n_pred}  {top1_pct:>6.1f}%  {full_pct:>6.1f}%")
    print(f"\n  ensemble gain vs bidir: {mean_e - mean_b:+.3f} tokens/step  ({n} iters)")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(
    all_acc: list[int],
    all_sub_acc: list[list[int]],
    n_sub_blocks: int,
    sub_block_size: int,
) -> None:
    n_pred_per_sub = sub_block_size - 1
    n_pred_total = n_pred_per_sub * n_sub_blocks

    print("\n-------- overall acceptance --------")
    # acceptance_lengths stores acc+1; acc = a-1 in [0, total_noise-1]
    actual_accs = [a - 1 for a in all_acc]
    mean_acc = sum(actual_accs) / len(actual_accs)
    print(f"  mean acceptance: {mean_acc:.2f} / {n_pred_total}  ({len(all_acc)} iters)")

    # Distribution at sub-block boundaries
    print("\n  sequential acceptance distribution:")
    cap = n_sub_blocks * sub_block_size - 1  # = 47
    buckets = {}
    for a in actual_accs:
        buckets[a] = buckets.get(a, 0) + 1
    # Show cumulative P(acc >= k) at key thresholds
    thresholds = [(si * sub_block_size - 1, f"≥A{si+1}_full") for si in range(1, n_sub_blocks)]
    for thresh, label in thresholds:
        count = sum(v for k, v in buckets.items() if k >= thresh)
        pct = count / len(all_acc) * 100
        print(f"    P(acc ≥ {thresh:2d}) [{label}]: {count}/{len(all_acc)} = {pct:.1f}%")

    print("\n-------- per-sub-block hierarchical acceptance (ignoring anchor slots) --------")
    print(
        f"{'sub':>4}  {'eligible':>8}  {'all_acc%':>9}  {'mean_acc':>9}"
    )
    for si in range(n_sub_blocks):
        # eligible = iters where all previous sub-blocks were fully accepted
        # sub_acc[si] = -1 means not reached (not eligible)
        eligible = [row[si] for row in all_sub_acc if row[si] >= 0]
        n_elig = len(eligible)
        if n_elig == 0:
            print(f"   {chr(ord('A') + si)}   {'---':>8}  {'---':>9}  {'---':>9}")
            continue
        n_full = sum(1 for a in eligible if a == n_pred_per_sub)
        mean_in = sum(eligible) / n_elig
        full_pct = n_full / n_elig * 100
        print(
            f"   {chr(ord('A') + si)}   {n_elig:>8d}  {full_pct:>8.1f}%  {mean_in:>8.2f}/{n_pred_per_sub}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--mode", choices=["original", "ensemble"], default="original",
                    help="original: 3-block A/B/C; ensemble: 1 bidir + 4 random-dropout")
    ap.add_argument("--sub-block-size", type=int, default=DEFAULT_SUB_BLOCK_SIZE,
                    help="slots per sub-block including anchor (default 16 → 15 predictions)")
    ap.add_argument("--n-sub-blocks", type=int, default=DEFAULT_N_SUB_BLOCKS,
                    help="[original mode] number of independent sub-blocks (default 3)")
    ap.add_argument("--n-dropout-blocks", type=int, default=4,
                    help="[ensemble mode] number of random-dropout blocks (default 4)")
    ap.add_argument("--print-samples", type=int, default=3)
    ap.add_argument("--print-iters", type=int, default=2,
                    help="debug iters to print per sample (0 to disable)")
    ap.add_argument("--debug-topk", type=int, default=5,
                    help="top-k tokens to show in debug output")
    ap.add_argument("--dropout-b", type=float, default=0.5,
                    help="attention dropout probability (used in both modes, default 0.5)")
    args = ap.parse_args()

    sub_block_size = args.sub_block_size

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
        print(f"resolved mask_token_id = {draft.mask_token_id}")

    ds = load_dataset("gsm8k", "main", split=f"test[:{args.n}]")

    if args.mode == "ensemble":
        n_dropout_blocks = args.n_dropout_blocks
        n_total = 1 + n_dropout_blocks
        n_pred = sub_block_size - 1
        print(
            f"[ensemble mode] 1 bidir + {n_dropout_blocks} random-dropout blocks × "
            f"{sub_block_size} slots  →  total_noise={n_total * sub_block_size}  "
            f"n_pred_per_block={n_pred}"
        )
        print(f"  dropout_prob={args.dropout_b}  comparing bidir vs ensemble(sum of {n_dropout_blocks} logits)")

        all_bidir_acc: list[int] = []
        all_ensemble_acc: list[int] = []

        for idx, row in enumerate(ds):
            messages = [{"role": "user", "content": row["question"]}]
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
            ).to(draft.device)

            print(f"\n====== sample {idx} ======")
            if idx < args.print_samples:
                print(f"Q: {row['question'][:120]}")

            do_debug = args.print_iters > 0 and idx < args.print_samples
            ret = spec_generate_ensemble(
                draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                sub_block_size=sub_block_size,
                n_dropout_blocks=n_dropout_blocks,
                dropout_prob=args.dropout_b,
                return_debug=do_debug,
                debug_topk=args.debug_topk,
            )
            if do_debug:
                output_ids, b_lens, e_lens, dbg = ret
            else:
                output_ids, b_lens, e_lens = ret
                dbg = []

            if idx < args.print_samples:
                gen = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
                print(f"A: {gen[:300]}")
            bm = sum(b_lens) / len(b_lens) - 1
            em = sum(e_lens) / len(e_lens) - 1
            print(
                f"[sample {idx}] bidir={bm:.2f}/{n_pred}  ensemble={em:.2f}/{n_pred}  "
                f"gain={em - bm:+.3f}  ({len(b_lens)} iters)"
            )
            if dbg:
                print_ensemble_debug(
                    dbg, tokenizer,
                    sub_block_size=sub_block_size,
                    max_iters=args.print_iters,
                )
            all_bidir_acc.extend(b_lens)
            all_ensemble_acc.extend(e_lens)

        print("\n================ SUMMARY ================")
        print(f"samples: {args.n}  sub_block_size: {sub_block_size}  n_dropout_blocks: {n_dropout_blocks}")
        print_ensemble_stats(all_bidir_acc, all_ensemble_acc, n_pred)

    else:
        # original mode
        n_sub_blocks = args.n_sub_blocks
        n_pred_per_sub = sub_block_size - 1
        total_noise = sub_block_size * n_sub_blocks
        print(
            f"layout: {n_sub_blocks} × [{sub_block_size} slots = anchor + {n_pred_per_sub} preds]  "
            f"→ total_noise={total_noise}, total_preds={n_pred_per_sub * n_sub_blocks}"
        )
        print(
            f"mask patterns: A=bidir  B=random_dropout(p={args.dropout_b})  C=diagonal_only"
        )

        all_acc: list[int] = []
        all_sub_acc: list[list[int]] = []

        for idx, row in enumerate(ds):
            messages = [{"role": "user", "content": row["question"]}]
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
            ).to(draft.device)

            print(f"\n====== sample {idx} ======")
            if idx < args.print_samples:
                print(f"Q: {row['question'][:120]}")

            do_debug = args.print_iters > 0 and idx < args.print_samples
            ret = spec_generate_mixblock(
                draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                sub_block_size=sub_block_size,
                n_sub_blocks=n_sub_blocks,
                dropout_prob_b=args.dropout_b,
                return_debug=do_debug,
                debug_topk=args.debug_topk,
            )
            if do_debug:
                output_ids, acc_lens, sub_accs, debug_iters = ret
            else:
                output_ids, acc_lens, sub_accs = ret
                debug_iters = []

            if idx < args.print_samples:
                gen = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
                print(f"A: {gen[:300]}")
            n_pred_total = n_pred_per_sub * n_sub_blocks
            print(
                f"[sample {idx}] mean acc = {sum(acc_lens) / len(acc_lens) - 1:.2f}/{n_pred_total}  "
                f"({len(acc_lens)} iters)"
            )
            if debug_iters:
                print_mixblock_debug(
                    debug_iters, tokenizer,
                    n_sub_blocks=n_sub_blocks,
                    sub_block_size=sub_block_size,
                    max_iters=args.print_iters,
                )
            all_acc.extend(acc_lens)
            all_sub_acc.extend(sub_accs)

        print("\n================ SUMMARY ================")
        print(f"samples: {args.n}  sub_block_size: {sub_block_size}  n_sub_blocks: {n_sub_blocks}")
        print_stats(all_acc, all_sub_acc, n_sub_blocks, sub_block_size)


if __name__ == "__main__":
    main()
