"""
Two-block Classifier Free Guidance (CFG) speculative decoding.

Layout per step (2 sub-blocks of `sub_block_size`):
  block-0 (unconditional): [c0, mask×(B-1)]
    - noise tokens attend to each other: full bidir within block-0
    - NOT visible to KV cache (context tokens) — this is the CFG "uncond" pass
  block-1 (conditional):   [c0, mask×(B-1)]
    - noise tokens attend to each other: full bidir within block-1
    - CAN attend to KV cache (context tokens) — this is the CFG "cond" pass

CFG logits (gamma=2):
  logits_cfg = logits_uncond + gamma * (logits_cond - logits_uncond)
             = 2 * logits_cond - logits_uncond

Draft proposals are sampled from logits_cfg and verified against the target model.

Usage:
  python gsm8k_cfg.py --n 50
  python gsm8k_cfg.py --n 50 --gamma 2.0 --sub-block-size 16
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import time

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
DEFAULT_GAMMA = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab = logits.shape
    logits = logits.view(-1, vocab) / temperature
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).view(bsz, seq_len)


def _extract_context(hidden_states, layer_ids: list[int]) -> torch.Tensor:
    return torch.cat([hidden_states[lid + 1] for lid in layer_ids], dim=-1)


def build_cfg_full_mask(
    sub_block_size: int,
    ctx_len: int,
    dtype: torch.dtype,
    device,
    uncond_ctx_tokens: int = 0,
    causal_cond: bool = False,
) -> torch.Tensor:
    """
    Full additive attention mask [1, 1, 2*B, ctx_len + 2*B].

    Columns [0 .. ctx_len-1]            = context (KV cache) positions
    Columns [ctx_len .. ctx_len+B-1]    = block-0 noise positions
    Columns [ctx_len+B .. ctx_len+2B-1] = block-1 noise positions

    Row groups (queries):
      block-0 (rows 0..B-1)  — unconditional:
        ctx cols [0 .. ctx_len-uncond_ctx_tokens-1] : attend (0.0)  all except local tail
        ctx cols [ctx_len-uncond_ctx_tokens .. ctx_len-1] : MASKED  (-inf)  local tail excluded
        block-0 cols : attend  (0.0)   full bidir within block
        block-1 cols : MASKED  (-inf)  no cross-block

      block-1 (rows B..2B-1) — conditional:
        ctx columns  : attend  (0.0)   full context
        block-0 cols : MASKED  (-inf)  no cross-block
        block-1 cols : attend  (0.0)   full bidir  [causal_cond=False]
                     : causal  (tril)  [causal_cond=True]  position i sees 0..i

    0.0 = attend, -inf = masked.
    """
    B = sub_block_size
    total_q = 2 * B
    total_kv = ctx_len + 2 * B
    neg_inf = torch.finfo(dtype).min

    mask = torch.full((1, 1, total_q, total_kv), neg_inf, dtype=dtype, device=device)

    # block-0 (unconditional): attend to all ctx EXCEPT last uncond_ctx_tokens (local tail)
    uncond_ctx_end = max(0, ctx_len - uncond_ctx_tokens) if uncond_ctx_tokens > 0 else ctx_len
    if uncond_ctx_end > 0:
        mask[0, 0, 0:B, 0:uncond_ctx_end] = 0.0
    mask[0, 0, 0:B, ctx_len : ctx_len + B] = 0.0

    # block-1 (conditional): full ctx always visible
    mask[0, 0, B : 2 * B, 0:ctx_len] = 0.0
    if causal_cond:
        # causal within block-1: position B+i attends to block-1[0..i]
        causal_block = torch.where(
            torch.tril(torch.ones(B, B, dtype=torch.bool, device=device)),
            torch.zeros(B, B, dtype=dtype, device=device),
            torch.full((B, B), neg_inf, dtype=dtype, device=device),
        )
        mask[0, 0, B : 2 * B, ctx_len + B : ctx_len + 2 * B] = causal_block
    else:
        mask[0, 0, B : 2 * B, ctx_len + B : ctx_len + 2 * B] = 0.0

    return mask


# ---------------------------------------------------------------------------
# Layer-level CFG helpers
# ---------------------------------------------------------------------------

from contextlib import contextmanager

@contextmanager
def _null_ctx_at_layer(draft: torch.nn.Module, layer_idx: int):
    """Zero out target_hidden for draft.layers[layer_idx].self_attn during forward."""
    attn = draft.layers[layer_idx].self_attn
    _zeros: list[torch.Tensor] = []

    def pre_hook(module, args, kwargs):
        th = kwargs.get("target_hidden")
        if th is not None:
            if not _zeros or _zeros[0].shape != th.shape:
                _zeros.clear()
                _zeros.append(torch.zeros_like(th))
            kwargs["target_hidden"] = _zeros[0]
        return args, kwargs

    handle = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def _single_block_logits(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    c0: int,
    start: int,
    sub_block_size: int,
    target_hidden: torch.Tensor,
    mask_id: int,
    mask_dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """
    Run ONE noise block (B tokens) through draft without KV cache.
    Returns lm_head logits for positions 1..B-1 (the n_pred predictions).

    pos_ids must cover BOTH ctx positions (for k_ctx from target_hidden) and
    noise positions, because apply_rotary_pos_emb is applied to the full
    concatenated k = cat([k_ctx, k_noise]).

    target_hidden represents the accepted tokens at positions [start-ctx_len..start-1].
    So the ctx positions are torch.arange(start - ctx_len, start).
    """
    B = sub_block_size
    ctx_len = target_hidden.shape[1]  # typically step_adv from previous step

    block_ids = torch.full((1, B), mask_id, dtype=torch.long, device=device)
    block_ids[0, 0] = c0
    noise_emb = target.model.embed_tokens(block_ids)  # [1, B, hidden]

    # pos_ids: ctx positions [start-ctx_len..start-1] then noise [start..start+B-1]
    ctx_start = start - ctx_len
    pos_ids = torch.cat([
        torch.arange(ctx_start, start, device=device),
        torch.arange(start, start + B, device=device),
    ]).unsqueeze(0)  # [1, ctx_len + B]

    # attn_mask: B queries attend to [ctx_len + B] keys
    # Allow full ctx (hook zeros target_hidden at the specific layer)
    # Full bidir within the noise block
    attn_mask = torch.zeros(1, 1, B, ctx_len + B, dtype=mask_dtype, device=device)

    draft_out = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_emb,
        position_ids=pos_ids,
        attention_mask=attn_mask,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )  # [1, B, hidden]
    return target.lm_head(draft_out[:, 1:, :])  # [1, n_pred, V]


# ---------------------------------------------------------------------------
# CFG speculative generate
# ---------------------------------------------------------------------------

@torch.inference_mode()
def spec_generate_cfg(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    gamma: float = DEFAULT_GAMMA,
    uncond_ctx_tokens: int = 32,
    use_cfg: bool = True,
    causal_cond: bool = False,
    return_debug: bool = False,
    debug_topk: int = 5,
) -> tuple:
    """
    Two-block (attention-mask) speculative decode.

    use_cfg=True  (default): 2-block joint forward; proposals from CFG logits
                             = uncond + gamma*(cond-uncond).
    use_cfg=False: single block-1 forward (full context); proposals from
                   cond logits only — no guidance.
    causal_cond=True: block-1 uses causal attention within itself instead of bidir.
                      position i can attend to block-1[0..i], enabling left-to-right
                      information flow within the draft block.

    For a fair cond vs cfg comparison, call this function TWICE on the same
    input_ids — once with use_cfg=False and once with use_cfg=True — so each
    run has its own independent KV caches and target verification.

    Returns:
      output_ids, acceptance_lengths[, debug_iters]
    """
    B = sub_block_size
    n_pred = B - 1
    label = "block-cfg" if use_cfg else "block-cond"

    device = input_ids.device
    mask_dtype = next(draft.parameters()).dtype

    orig_impl = draft.config._attn_implementation
    draft.config._attn_implementation = "eager"

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    mask_id = draft.mask_token_id
    if mask_id is None:
        raise ValueError("draft.mask_token_id is None")

    buf_extra = 2 * B if use_cfg else B
    output_ids = torch.full(
        (1, max_length + buf_extra), mask_id, dtype=torch.long, device=device
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
    debug_iters: list[dict] = [] if return_debug else None

    start = num_input_tokens
    t0 = time.perf_counter()
    pbar = tqdm(total=max_new_tokens, desc=label, unit="tok")

    while start < max_length:
        c0 = output_ids[0, start].item()

        cached_len = past_kv_draft.get_seq_length()
        prefix_pos = torch.arange(cached_len, start, device=device)
        block_sub_pos = torch.arange(start, start + B, device=device)

        if use_cfg:
            # ---- 2-block joint forward ----
            block_ids = torch.full((1, 2 * B), mask_id, dtype=torch.long, device=device)
            block_ids[0, 0] = c0   # block-0 anchor
            block_ids[0, B] = c0   # block-1 anchor
            noise_emb = target.model.embed_tokens(block_ids)
            custom_pos_ids = torch.cat(
                [prefix_pos, block_sub_pos.repeat(2)]
            ).unsqueeze(0)
            full_mask = build_cfg_full_mask(
                B, ctx_len=start, dtype=mask_dtype, device=device,
                uncond_ctx_tokens=uncond_ctx_tokens,
                causal_cond=causal_cond,
            )
            draft_out = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_emb,
                position_ids=custom_pos_ids,
                attention_mask=full_mask,
                past_key_values=past_kv_draft,
                use_cache=True,
                is_causal=False,
            )
            past_kv_draft.crop(start)
            uncond_logits = target.lm_head(draft_out[:, 1:B, :])
            cond_logits   = target.lm_head(draft_out[:, B + 1 : 2 * B, :])
            proposal_logits = uncond_logits + gamma * (cond_logits - uncond_logits)
        else:
            # ---- Single block-1 forward (cond only, full context) ----
            block_ids = torch.full((1, B), mask_id, dtype=torch.long, device=device)
            block_ids[0, 0] = c0
            noise_emb = target.model.embed_tokens(block_ids)
            cond_pos_ids = torch.cat([prefix_pos, block_sub_pos]).unsqueeze(0)
            if causal_cond:
                neg_inf = torch.finfo(mask_dtype).min
                cond_mask = torch.full((1, 1, B, start + B), neg_inf, dtype=mask_dtype, device=device)
                cond_mask[0, 0, :, :start] = 0.0  # full ctx visible
                causal_self = torch.where(
                    torch.tril(torch.ones(B, B, dtype=torch.bool, device=device)),
                    torch.zeros(B, B, dtype=mask_dtype, device=device),
                    torch.full((B, B), neg_inf, dtype=mask_dtype, device=device),
                )
                cond_mask[0, 0, :, start:] = causal_self
            else:
                cond_mask = torch.zeros(1, 1, B, start + B, dtype=mask_dtype, device=device)
            cond_out = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_emb,
                position_ids=cond_pos_ids,
                attention_mask=cond_mask,
                past_key_values=past_kv_draft,
                use_cache=True,
                is_causal=False,
            )
            past_kv_draft.crop(start)
            cond_logits     = target.lm_head(cond_out[:, 1:, :])
            proposal_logits = cond_logits
            uncond_logits   = None

        # ---- Sample, verify, accept ----
        draft_proposed = _sample(proposal_logits)  # [1, n_pred]
        verify_ids = torch.full((1, B), mask_id, dtype=torch.long, device=device)
        verify_ids[0, 0] = c0
        verify_ids[:, 1:] = draft_proposed

        out = target(
            verify_ids,
            position_ids=position_ids[:, start : start + B],
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = _sample(out.logits, temperature)  # [1, B]

        acceptance_length = int(
            (draft_proposed == posterior[:, :n_pred]).cumprod(dim=1).sum(dim=1)[0].item()
        )

        # ---- Debug collection ----
        if return_debug:
            # Oracle: true target tokens via autoregressive greedy decoding on correct prefix.
            # past_kv_target now holds positions 0..start+B-1, but positions start..start+B-1
            # may be conditioned on wrong draft tokens. Snapshot correct prefix 0..start-1.
            import copy
            oracle_cache = copy.deepcopy(past_kv_target)
            oracle_cache.crop(start)

            oracle_true_tokens: list[int] = []
            oracle_logits_list: list[torch.Tensor] = []
            _oracle_next = int(c0)
            for _oi in range(n_pred):
                _oid = torch.tensor([[_oracle_next]], dtype=torch.long, device=device)
                _opos = torch.tensor([[start + _oi]], device=device)
                _o = target(
                    _oid,
                    position_ids=_opos,
                    past_key_values=oracle_cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
                _olg = _o.logits[0, 0, :].detach().float().cpu()
                oracle_logits_list.append(_olg)
                _oracle_next = int(_olg.argmax().item())
                oracle_true_tokens.append(_oracle_next)
            del oracle_cache

            oracle_true = torch.tensor(oracle_true_tokens, dtype=torch.long)  # [n_pred]

            # ---- Alignment check --------------------------------------------------------
            # Invariant (greedy only): for positions 0..acceptance_length, the AR verify
            # pass and the oracle use the SAME correct prefix, so posterior must equal
            # oracle_true. Any mismatch indicates a one-token index offset somewhere.
            if temperature < 1e-5:
                for _ci in range(min(acceptance_length + 1, n_pred)):
                    _p = int(posterior[0, _ci].item())
                    _o = int(oracle_true_tokens[_ci])
                    if _p != _o:
                        print(
                            f"  [ALIGN BUG] iter start={start} pos+{_ci+1}: "
                            f"posterior={_p} != oracle={_o}  acc={acceptance_length}"
                        )
            # ---- End alignment check -----------------------------------------------------

            # Rank of oracle true token in cond / proposal (1 = top-1)
            _cond_l = cond_logits[0].float().cpu()      # [n_pred, V]
            _prop_l = proposal_logits[0].float().cpu()  # [n_pred, V]
            _idx    = torch.arange(n_pred)
            _cond_at_true = _cond_l[_idx, oracle_true]  # [n_pred]
            _prop_at_true = _prop_l[_idx, oracle_true]  # [n_pred]
            cond_rank = (_cond_l > _cond_at_true.unsqueeze(-1)).sum(-1) + 1  # [n_pred]
            prop_rank = (_prop_l > _prop_at_true.unsqueeze(-1)).sum(-1) + 1  # [n_pred]

            # target* uses oracle logits (correct AR prefix), not the verify pass
            oracle_logits_tensor = torch.stack(oracle_logits_list, dim=0)  # [n_pred, V] on CPU
            t_top        = torch.topk(oracle_logits_tensor, k=debug_topk, dim=-1)
            proposal_top = torch.topk(proposal_logits[0], k=debug_topk, dim=-1)
            cond_top     = torch.topk(cond_logits[0], k=debug_topk, dim=-1)

            # Rank of AR verify result in oracle logits
            ar_verify_toks = posterior[0, :n_pred].cpu()  # [n_pred]
            _av_at_oracle  = oracle_logits_tensor[torch.arange(n_pred), ar_verify_toks]
            ar_verify_rank = (oracle_logits_tensor > _av_at_oracle.unsqueeze(-1)).sum(-1) + 1

            entry = {
                "start": start,
                "acceptance_length": acceptance_length,
                "draft_proposed": draft_proposed[0].cpu(),
                "oracle_true_tokens": oracle_true,          # [n_pred] TRUE target tokens
                "cond_rank": cond_rank,                     # [n_pred] rank of true tok in cond
                "prop_rank": prop_rank,                     # [n_pred] rank of true tok in proposal
                "ar_verify_tokens": ar_verify_toks,         # [n_pred] AR verify proposed tokens
                "ar_verify_rank":   ar_verify_rank,         # [n_pred] rank of AR verify tok in oracle
                "proposal_top_ids":    proposal_top.indices.cpu(),
                "proposal_top_logits": proposal_top.values.float().cpu(),
                "cond_top_ids":    cond_top.indices.cpu(),
                "cond_top_logits": cond_top.values.float().cpu(),
                "target_top_ids":    t_top.indices.cpu(),
                "target_top_logits": t_top.values.float().cpu(),
            }
            if use_cfg and uncond_logits is not None:
                uncond_top = torch.topk(uncond_logits[0], k=debug_topk, dim=-1)
                entry["uncond_top_ids"]    = uncond_top.indices.cpu()
                entry["uncond_top_logits"] = uncond_top.values.float().cpu()
            debug_iters.append(entry)

        # ---- Commit accepted tokens ----
        output_ids[:, start : start + acceptance_length + 1] = verify_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        step_adv = acceptance_length + 1
        start += step_adv
        acceptance_lengths.append(acceptance_length + 1)
        pbar.update(step_adv)

        elapsed = time.perf_counter() - t0
        pbar.set_postfix({
            "acc": f"{acceptance_length}/{n_pred}",
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
    mean_acc = sum(acceptance_lengths) / len(acceptance_lengths) - 1
    print(
        f"[{label}] {total_new} tok in {elapsed:.2f}s | "
        f"{total_new / elapsed:.1f} tok/s | "
        f"acc {mean_acc:.2f}/{n_pred} | "
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
        return output_ids, acceptance_lengths, debug_iters
    return output_ids, acceptance_lengths


# ---------------------------------------------------------------------------
# Layer-level CFG speculative generate
# ---------------------------------------------------------------------------

@torch.inference_mode()
def spec_generate_layer_cfg(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    gamma: float = DEFAULT_GAMMA,
    use_cfg: bool = False,
    return_debug: bool = False,
    debug_topk: int = 5,
) -> tuple:
    """
    Layer-level speculative decode.

    use_cfg=False: proposals from cond logits only (full context, no guidance).
    use_cfg=True:  proposals from CFG = uncond + gamma*(cond-uncond),
                   where uncond_i = cond with target_hidden zeroed at layer i,
                   uncond = mean over all draft layers.

    For a fair cond vs cfg comparison, call this function TWICE on the same
    input_ids — once with use_cfg=False and once with use_cfg=True — so each
    run has its own independent KV caches and target verification.

    Returns:
      output_ids, acceptance_lengths[, debug_iters]
    """
    B = sub_block_size
    n_pred = B - 1
    n_draft_layers = len(draft.layers)
    label = "layer-cfg" if use_cfg else "layer-cond"

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
        (1, max_length + B), mask_id, dtype=torch.long, device=device
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
    debug_iters: list[dict] = [] if return_debug else None

    start = num_input_tokens
    t0 = time.perf_counter()
    pbar = tqdm(total=max_new_tokens, desc=label, unit="tok")

    while start < max_length:
        c0 = output_ids[0, start].item()

        # ---- Cond pass: 1 block, WITH draft KV cache ----
        block_ids = torch.full((1, B), mask_id, dtype=torch.long, device=device)
        block_ids[0, 0] = c0
        noise_emb = target.model.embed_tokens(block_ids)

        cached_len = past_kv_draft.get_seq_length()
        prefix_pos = torch.arange(cached_len, start, device=device)
        block_pos = torch.arange(start, start + B, device=device)
        cond_pos_ids = torch.cat([prefix_pos, block_pos]).unsqueeze(0)

        cond_full_mask = torch.zeros(1, 1, B, start + B, dtype=mask_dtype, device=device)

        cond_out = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=cond_pos_ids,
            attention_mask=cond_full_mask,
            past_key_values=past_kv_draft,
            use_cache=True,
            is_causal=False,
        )
        past_kv_draft.crop(start)

        cond_logits = target.lm_head(cond_out[:, 1:, :])  # [1, n_pred, V]

        # ---- CFG: N uncond passes (one per draft layer), no KV cache ----
        if use_cfg:
            uncond_logits_list = []
            for li in range(n_draft_layers):
                with _null_ctx_at_layer(draft, li):
                    layer_logits = _single_block_logits(
                        draft, target, c0, start, B, target_hidden,
                        mask_id, mask_dtype, device,
                    )
                uncond_logits_list.append(layer_logits)
            uncond_logits = torch.stack(uncond_logits_list, dim=0).mean(dim=0)
            proposal_logits = uncond_logits + gamma * (cond_logits - uncond_logits)
        else:
            uncond_logits_list = []
            uncond_logits = None
            proposal_logits = cond_logits

        # ---- Sample, verify, accept ----
        draft_proposed = _sample(proposal_logits)  # [1, n_pred]
        block_ids[:, 1:] = draft_proposed

        out = target(
            block_ids,
            position_ids=position_ids[:, start : start + B],
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = _sample(out.logits, temperature)  # [1, B]

        acceptance_length = int(
            (draft_proposed == posterior[:, :n_pred]).cumprod(dim=1).sum(dim=1)[0].item()
        )

        # ---- Debug collection ----
        if return_debug:
            t_logits = out.logits[0, :n_pred]
            t_top = torch.topk(t_logits, k=debug_topk, dim=-1)
            proposal_top = torch.topk(proposal_logits[0], k=debug_topk, dim=-1)
            cond_top = torch.topk(cond_logits[0], k=debug_topk, dim=-1)
            entry = {
                "start": start,
                "acceptance_length": acceptance_length,
                "draft_proposed": draft_proposed[0].cpu(),
                "target_posterior": posterior[0, :n_pred].cpu(),
                "proposal_top_ids": proposal_top.indices.cpu(),
                "proposal_top_logits": proposal_top.values.float().cpu(),
                "cond_top_ids": cond_top.indices.cpu(),
                "cond_top_logits": cond_top.values.float().cpu(),
                "target_top_ids": t_top.indices.cpu(),
                "target_top_logits": t_top.values.float().cpu(),
            }
            if use_cfg and uncond_logits is not None:
                uncond_top = torch.topk(uncond_logits[0], k=debug_topk, dim=-1)
                per_layer_tops = [
                    torch.topk(ll[0], k=debug_topk, dim=-1) for ll in uncond_logits_list
                ]
                entry["uncond_top_ids"] = uncond_top.indices.cpu()
                entry["uncond_top_logits"] = uncond_top.values.float().cpu()
                entry["per_layer_top_ids"] = [t.indices.cpu() for t in per_layer_tops]
                entry["per_layer_top_logits"] = [t.values.float().cpu() for t in per_layer_tops]
            debug_iters.append(entry)

        # ---- Commit ----
        output_ids[:, start : start + acceptance_length + 1] = block_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        step_adv = acceptance_length + 1
        start += step_adv
        acceptance_lengths.append(acceptance_length + 1)
        pbar.update(step_adv)

        elapsed = time.perf_counter() - t0
        pbar.set_postfix({
            "acc": f"{acceptance_length}/{n_pred}",
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
    mean_acc = sum(acceptance_lengths) / len(acceptance_lengths) - 1
    print(
        f"[{label}] {total_new} tok in {elapsed:.2f}s | "
        f"{total_new / elapsed:.1f} tok/s | "
        f"acc {mean_acc:.2f}/{n_pred} | "
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
        return output_ids, acceptance_lengths, debug_iters
    return output_ids, acceptance_lengths


# ---------------------------------------------------------------------------
# Debug printing
# ---------------------------------------------------------------------------

def format_topk_row(tok_ids, logits, tokenizer, highlight_id=None):
    parts = []
    for tid, lg in zip(tok_ids.tolist(), logits.tolist()):
        piece = tokenizer.decode([tid]).replace("\n", "\\n")
        marker = "*" if (highlight_id is not None and tid == highlight_id) else " "
        parts.append(f"{marker}{tid:>6d}:{piece!r:<14s}({lg:+.2f})")
    return " ".join(parts)


def print_cfg_debug(debug_iters, tokenizer, max_iters: int = 2, max_pos: int = None) -> None:
    for it_idx, it in enumerate(debug_iters):
        if it_idx >= max_iters:
            break
        acc = it["acceptance_length"]
        n_pos = it["draft_proposed"].shape[0]
        n_print = n_pos if max_pos is None else min(n_pos, max_pos)
        print(
            f"\n--- iter {it_idx}  start={it['start']}  "
            f"acc={acc}/{n_pos} ---"
        )
        has_oracle = "oracle_true_tokens" in it
        oracle_toks = it["oracle_true_tokens"] if has_oracle else it.get("target_posterior")
        n_oracle = len(oracle_toks) if oracle_toks is not None else 0
        def _tok_str(tid: int) -> str:
            return tokenizer.decode([tid]).replace("\n", "\\n").replace("\r", "\\r")

        for i in range(n_print):
            true_tok  = int(oracle_toks[i].item()) if has_oracle else int(it["target_posterior"][i].item())
            draft_tok = int(it["draft_proposed"][i].item())
            tag   = "acc" if i < acc else ("ovr" if i == acc else "rej")
            match = "==" if draft_tok == true_tok else "!="
            c_rank   = int(it["cond_rank"][i].item())      if "cond_rank"      in it else "?"
            p_rank   = int(it["prop_rank"][i].item())      if "prop_rank"      in it else "?"
            av_rank  = int(it["ar_verify_rank"][i].item()) if "ar_verify_rank" in it else "?"
            av_tok   = int(it["ar_verify_tokens"][i].item()) if "ar_verify_tokens" in it else "?"
            true_str  = _tok_str(true_tok)
            draft_str = _tok_str(draft_tok)
            # Shift markers: did draft predict the oracle token for an adjacent position?
            shift_tag = ""
            if has_oracle and draft_tok != true_tok:
                prev_true = int(oracle_toks[i - 1].item()) if i > 0 else None
                next_true = int(oracle_toks[i + 1].item()) if i < n_oracle - 1 else None
                if draft_tok == next_true:
                    shift_tag = "  [>>fwd]"
                elif draft_tok == prev_true:
                    shift_tag = "  [<<bck]"
            shift_warn = ""
            if tag == "acc" and c_rank != 1:
                shift_warn = "  [!cond top-1 mismatch on accepted pos]"
            av_str = _tok_str(av_tok) if isinstance(av_tok, int) else "?"
            print(f"  pos +{i + 1}  [{tag}]  "
                  f"draft={draft_tok}({draft_str!r}){match}true={true_tok}({true_str!r})  "
                  f"cond_rank={c_rank}  prop_rank={p_rank}  "
                  f"ar_verify={av_tok}({av_str!r})(oracle_rank={av_rank})"
                  f"{shift_tag}{shift_warn}")
            # * is placed on true_tok (= oracle top-1) in all rows
            if "uncond_top_ids" in it:
                print(f"    uncond  : {format_topk_row(it['uncond_top_ids'][i], it['uncond_top_logits'][i], tokenizer, highlight_id=true_tok)}")
            print(f"    cond    : {format_topk_row(it['cond_top_ids'][i], it['cond_top_logits'][i], tokenizer, highlight_id=true_tok)}")
            print(f"    proposal: {format_topk_row(it['proposal_top_ids'][i], it['proposal_top_logits'][i], tokenizer, highlight_id=true_tok)}")
            print(f"    target* : {format_topk_row(it['target_top_ids'][i], it['target_top_logits'][i], tokenizer, highlight_id=true_tok)}")


# ---------------------------------------------------------------------------
# Shift-frequency analysis
# ---------------------------------------------------------------------------

def print_shift_stats(debug_iters_list: list[list[dict]], labels: list[str]) -> None:
    """
    For each set of debug iterations, compute how often the draft predicts the
    oracle token belonging to an adjacent position (±1).

    Counts are broken down by:
      - overall
      - by position within block (pos+1 .. pos+n_pred)
      - by tag (acc / ovr+rej combined)
    """
    for debug_iters, label in zip(debug_iters_list, labels):
        if not debug_iters or "oracle_true_tokens" not in debug_iters[0]:
            continue
        n_pred = len(debug_iters[0]["draft_proposed"])
        # per-position counters [n_pred]: total, fwd, bck
        pos_total = [0] * n_pred
        pos_fwd   = [0] * n_pred
        pos_bck   = [0] * n_pred
        # by tag (rej/ovr = wrong, acc = correct by definition)
        wrong_total = wrong_fwd = wrong_bck = 0

        for it in debug_iters:
            acc       = it["acceptance_length"]
            oracle_t  = it["oracle_true_tokens"]   # [n_pred]
            draft_t   = it["draft_proposed"]        # [n_pred]
            n_o = len(oracle_t)
            for i in range(n_pred):
                d = int(draft_t[i].item())
                true_i = int(oracle_t[i].item())
                tag = "acc" if i < acc else "wrong"
                pos_total[i] += 1
                prev_true = int(oracle_t[i - 1].item()) if i > 0         else None
                next_true = int(oracle_t[i + 1].item()) if i < n_o - 1   else None
                is_fwd = (d != true_i) and (d == next_true)
                is_bck = (d != true_i) and (d == prev_true)
                if is_fwd:
                    pos_fwd[i] += 1
                if is_bck:
                    pos_bck[i] += 1
                if tag == "wrong":
                    wrong_total += 1
                    if is_fwd: wrong_fwd += 1
                    if is_bck: wrong_bck += 1

        total_pos = sum(pos_total)
        total_fwd = sum(pos_fwd)
        total_bck = sum(pos_bck)
        print(f"\n{'='*60}")
        print(f"[shift stats: {label}]  {len(debug_iters)} iters  {total_pos} positions")
        if total_pos == 0:
            continue
        print(f"  overall   fwd(+1)={total_fwd}/{total_pos} ({100*total_fwd/total_pos:.1f}%)  "
              f"bck(-1)={total_bck}/{total_pos} ({100*total_bck/total_pos:.1f}%)")
        if wrong_total > 0:
            print(f"  wrong-only fwd={wrong_fwd}/{wrong_total} ({100*wrong_fwd/wrong_total:.1f}%)  "
                  f"bck={wrong_bck}/{wrong_total} ({100*wrong_bck/wrong_total:.1f}%)")
        print("  by block position:")
        for i in range(n_pred):
            n = pos_total[i]
            if n == 0:
                continue
            fwd_r = 100 * pos_fwd[i] / n
            bck_r = 100 * pos_bck[i] / n
            bar_f = ">" * pos_fwd[i]
            bar_b = "<" * pos_bck[i]
            print(f"    pos+{i+1:2d}  fwd={pos_fwd[i]:3d}/{n} ({fwd_r:4.1f}%)  "
                  f"bck={pos_bck[i]:3d}/{n} ({bck_r:4.1f}%)  {bar_f}{bar_b}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--mode", choices=["block-cfg", "layer-cfg"], default="block-cfg",
                    help="block-cfg: 2-block attention-mask CFG; layer-cfg: per-draft-layer forward-hook CFG")
    ap.add_argument("--sub-block-size", type=int, default=DEFAULT_SUB_BLOCK_SIZE,
                    help="slots per sub-block including anchor (default 16 → 15 predictions)")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                    help="CFG guidance scale (default 2.0; 1.0 = cond only, no guidance)")
    ap.add_argument("--uncond-ctx-tokens", type=int, default=2,
                    help="[block-cfg] number of trailing local ctx tokens EXCLUDED from uncond block (default 2, 0=uncond sees all ctx)")
    ap.add_argument("--causal-cond", action="store_true",
                    help="[block-cfg] use causal (left-to-right) attention within cond block instead of full bidir")
    ap.add_argument("--print-samples", type=int, default=10)
    ap.add_argument("--print-iters", type=int, default=10,
                    help="debug iters to print per sample (0 to disable)")
    ap.add_argument("--debug-topk", type=int, default=5)
    args = ap.parse_args()

    B = args.sub_block_size
    n_pred = B - 1

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

    if args.mode == "layer-cfg":
        n_layers = len(draft.layers)
        print(
            f"[layer-cfg] sub_block_size={B}  n_pred={n_pred}  gamma={args.gamma}  "
            f"n_draft_layers={n_layers}\n"
            f"  cond:   1 pass, full ctx, WITH draft KV cache\n"
            f"  uncond: {n_layers} passes (one per layer, ctx zeroed at that layer), NO draft KV cache\n"
            f"  uncond = mean of {n_layers} layer-ablated logits\n"
            f"  cfg = uncond + {args.gamma} * (cond - uncond)"
        )
    else:
        print(
            f"[block-cfg] sub_block_size={B}  n_pred={n_pred}  gamma={args.gamma}  "
            f"excl_local_ctx={args.uncond_ctx_tokens}\n"
            f"  block-0 (uncond): full bidir, all ctx EXCEPT last {args.uncond_ctx_tokens} local tokens VISIBLE\n"
            f"  block-1 (cond)  : {'causal' if args.causal_cond else 'full bidir'}, full ctx VISIBLE\n"
            f"  cfg = uncond + {args.gamma} * (cond - uncond)"
        )

    ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{args.n}]")

    all_acc: list[int] = []
    all_cond_acc: list[int] = []
    all_dbg_cond: list[dict] = []
    all_dbg_cfg:  list[dict] = []

    for idx, row in enumerate(ds):
        messages = [{"role": "user", "content": row["problem"]}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
        ).to(draft.device)

        print(f"\n====== sample {idx} ======")
        if idx < args.print_samples:
            print(f"Q: {row['problem'][:120]}")

        do_debug = args.print_iters > 0 and idx < args.print_samples
        if args.mode == "layer-cfg":
            # cond run: independent decode with cond proposals only
            ret_cond = spec_generate_layer_cfg(
                draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                sub_block_size=B,
                gamma=args.gamma,
                use_cfg=False,
                return_debug=do_debug,
                debug_topk=args.debug_topk,
            )
            # cfg run: independent decode with CFG proposals
            ret_cfg = spec_generate_layer_cfg(
                draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                sub_block_size=B,
                gamma=args.gamma,
                use_cfg=True,
                return_debug=do_debug,
                debug_topk=args.debug_topk,
            )
            if do_debug:
                output_ids, cond_lens, dbg = ret_cond
                _, acc_lens, dbg_cfg = ret_cfg
            else:
                output_ids, cond_lens = ret_cond
                _, acc_lens = ret_cfg
                dbg = []
                dbg_cfg = []
        else:
            # cond run: single block-1, full context, independent decode
            ret_cond = spec_generate_cfg(
                draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                sub_block_size=B,
                gamma=args.gamma,
                uncond_ctx_tokens=args.uncond_ctx_tokens,
                use_cfg=False,
                causal_cond=args.causal_cond,
                return_debug=do_debug,
                debug_topk=args.debug_topk,
            )
            # cfg run: 2-block joint forward, independent decode
            ret_cfg = spec_generate_cfg(
                draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                sub_block_size=B,
                gamma=args.gamma,
                uncond_ctx_tokens=args.uncond_ctx_tokens,
                use_cfg=True,
                causal_cond=args.causal_cond,
                return_debug=do_debug,
                debug_topk=args.debug_topk,
            )
            if do_debug:
                output_ids, cond_lens, dbg = ret_cond
                _, acc_lens, dbg_cfg = ret_cfg
            else:
                output_ids, cond_lens = ret_cond
                _, acc_lens = ret_cfg
                dbg = []
                dbg_cfg = []

        if idx < args.print_samples:
            gen = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
            print(f"A: {gen[:300]}")

        mean_cfg_s = sum(acc_lens) / len(acc_lens) - 1
        mean_cond_s = sum(cond_lens) / len(cond_lens) - 1
        print(
            f"[sample {idx}] cfg={mean_cfg_s:.2f}/{n_pred}  "
            f"cond={mean_cond_s:.2f}/{n_pred}  "
            f"delta={mean_cfg_s - mean_cond_s:+.2f}  ({len(acc_lens)} iters)"
        )

        if dbg:
            print("\n[cond debug]")
            print_cfg_debug(dbg, tokenizer, max_iters=args.print_iters)
        if dbg_cfg:
            print("\n[cfg debug]")
            print_cfg_debug(dbg_cfg, tokenizer, max_iters=args.print_iters)

        all_acc.extend(acc_lens)
        all_cond_acc.extend(cond_lens)
        all_dbg_cond.extend(dbg)
        all_dbg_cfg.extend(dbg_cfg)

    if all_dbg_cond or all_dbg_cfg:
        print_shift_stats(
            [all_dbg_cond, all_dbg_cfg],
            ["cond", "cfg"],
        )

    print("\n================ SUMMARY ================")
    print(f"samples: {args.n}  mode: {args.mode}  sub_block_size: {B}  gamma: {args.gamma}")

    def _stats(lens, label, n_pred):
        actual = [a - 1 for a in lens]
        mean_a = sum(actual) / len(actual)
        top1_pct = sum(1 for a in actual if a >= 1) / len(actual) * 100
        full_pct = sum(1 for a in actual if a == n_pred) / len(actual) * 100
        print(f"  {label:<6s}  mean={mean_a:.2f}/{n_pred}  top1%={top1_pct:.1f}%  full%={full_pct:.1f}%")
        return mean_a

    mean_cfg_total  = _stats(all_acc,      "cfg",  n_pred)
    mean_cond_total = _stats(all_cond_acc, "cond", n_pred)
    print(f"  delta (cfg - cond) = {mean_cfg_total - mean_cond_total:+.3f} tokens/step")


if __name__ == "__main__":
    main()
