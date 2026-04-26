# coding=utf-8
"""BlockFlash Stage-1: train a draft model to predict sub-block anchor positions only.

Layout (block_size=16, sub_block_size=4, n_sub=4):

  Main block starting at anchor position t:
    seq:   [t  | t+1 t+2 t+3 | t+4 | t+5 t+6 t+7 | t+8 | t+9 t+10 t+11 | t+12 | ...]
    role:  [c0 |  (internal) | a1  |   (internal) | a2  |    (internal)  | a3   | ...]
    noise: [c0 |     ------  | MASK|     ------   | MASK|     ------      | MASK |    ]

  This Stage-1 model predicts ONLY the anchor tokens {a1, a2, a3} using causal
  attention among the n_sub anchors within each main block:
    a1 attends to: context[0..t+3] (strictly before a1) + c0
    a2 attends to: context[0..t+7] + c0 + a1_true (teacher-forced)
    a3 attends to: context[0..t+11] + c0 + a1_true + a2_true

  At inference this becomes causal speculation:
    a1_spec = argmax logits given c0 + prefix
    a2_spec = argmax logits given c0 + a1_spec + prefix
    a3_spec = argmax logits given c0 + a1_spec + a2_spec + prefix

Causality guarantee: the target model verification is always causal (AR), so the
accepted output is identical to standard autoregressive decoding regardless of
whether the speculated anchors are correct.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------

def create_anchor_causal_sdpa_mask(
    anchor_positions: torch.Tensor,   # [B, N_blocks]
    block_keep_mask: torch.Tensor,    # [B, N_blocks]
    S: int,                           # context (sequence) length
    sub_block_size: int,
    n_sub: int,                       # block_size // sub_block_size
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """
    Additive SDPA attention mask  [B, 1, Q_LEN, KV_LEN]  (0=attend, -inf=mask).

    Query layout  (Q_LEN = N_blocks * n_sub):
      block_0: [a0_0, a0_1, ..., a0_{n_sub-1}],  block_1: [...], ...

    KV layout  (KV_LEN = S + Q_LEN):
      [context_0 ... context_{S-1} | same query layout as above]

    Rules for query q belonging to (block n, sub-anchor i):
      anchor_seq_pos = anchor_positions[b, n] + i * sub_block_size

      Context  (kv < S):  visible iff kv < anchor_seq_pos   (strictly causal)
      Anchors  (kv >= S):  visible iff same block n  AND  kv_sub <= i  (causal among anchors)
      Invalid blocks (block_keep_mask=False): all masked.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * n_sub
    KV_LEN = S + Q_LEN
    neg_inf = torch.finfo(dtype).min

    # Per-query block and sub-anchor indices
    q_idx   = torch.arange(Q_LEN, device=device)
    q_block = q_idx // n_sub   # [Q_LEN]
    q_sub   = q_idx % n_sub    # [Q_LEN]

    # Sequence position of each query anchor  [B, Q_LEN]
    anchor_seq_pos = (
        anchor_positions[:, q_block]            # [B, Q_LEN]
        + q_sub.unsqueeze(0) * sub_block_size   # [1, Q_LEN]
    )

    # --- Context part ---
    # ctx_kv [1, 1, S]; asp [B, Q_LEN, 1]
    ctx_kv = torch.arange(S, device=device).view(1, 1, S)
    ctx_visible = ctx_kv < anchor_seq_pos.unsqueeze(2)  # [B, Q_LEN, S]

    # --- Anchor part ---
    kv_block = q_block  # [Q_LEN]  — same layout for KV anchors
    kv_sub   = q_sub    # [Q_LEN]
    same_block   = q_block.unsqueeze(1) == kv_block.unsqueeze(0)  # [Q_LEN, Q_LEN]
    causal_sub   = kv_sub.unsqueeze(0)  <= q_sub.unsqueeze(1)     # [Q_LEN, Q_LEN]
    anch_visible = (same_block & causal_sub).unsqueeze(0)          # [1, Q_LEN, Q_LEN]

    # Validity: invalid blocks see nothing
    q_valid = block_keep_mask[:, q_block]  # [B, Q_LEN]

    # Build mask  [B, 1, Q_LEN, KV_LEN]
    mask = torch.full((B, 1, Q_LEN, KV_LEN), neg_inf, dtype=dtype, device=device)
    # context columns
    mask[:, 0, :, :S] = torch.where(
        ctx_visible & q_valid.unsqueeze(2),
        torch.zeros(1, device=device, dtype=dtype),
        torch.tensor(neg_inf, device=device, dtype=dtype),
    )
    # anchor columns
    mask[:, 0, :, S:] = torch.where(
        anch_visible.expand(B, -1, -1) & q_valid.unsqueeze(2),
        torch.zeros(1, device=device, dtype=dtype),
        torch.tensor(neg_inf, device=device, dtype=dtype),
    )
    return mask


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

class OnlineBlockFlashAnchorModel(nn.Module):
    """BlockFlash Stage-1 training wrapper.

    Trains the draft model to predict ONLY the n_sub-1 anchor tokens per main
    block (i.e., positions sub_block_size, 2*sub_block_size, ...,
    (n_sub-1)*sub_block_size past the block start c0).

    Uses causal attention among the n_sub anchors within each block, matching
    the inference protocol exactly.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        sub_block_size: int = 4,
        num_anchors: int = 512,
    ):
        super().__init__()
        assert block_size % sub_block_size == 0, \
            f"block_size ({block_size}) must be divisible by sub_block_size ({sub_block_size})"
        self.draft_model      = draft_model
        self.lm_head          = target_lm_head
        self.embed_tokens     = target_embed_tokens
        self.mask_token_id    = mask_token_id
        self.block_size       = block_size
        self.sub_block_size   = sub_block_size
        self.n_sub            = block_size // sub_block_size
        self.num_anchors      = num_anchors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample main-block c0 positions; returns (anchors [B,N], keep [B,N])."""
        bsz = loss_mask.shape[0]
        # Need block_size tokens after anchor to label all n_sub anchors
        max_anchor = max(seq_len - self.block_size, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError("Not enough valid positions. Check data preprocessing.")

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))
        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = (
            torch.arange(max_n, device=device).unsqueeze(0)
            < valid_counts.unsqueeze(1).clamp(max=max_n)
        )
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
        return anchors, keep_mask

    def _anchor_seq_positions(
        self, anchor_positions: torch.Tensor
    ) -> torch.Tensor:
        """Sequence positions of all n_sub anchors per block.  [B, N*n_sub]"""
        B, N = anchor_positions.shape
        device = anchor_positions.device
        n_sub = self.n_sub
        q_block = torch.arange(N * n_sub, device=device) // n_sub   # [Q_LEN]
        q_sub   = torch.arange(N * n_sub, device=device) % n_sub    # [Q_LEN]
        return (
            anchor_positions[:, q_block]              # [B, Q_LEN]
            + q_sub.unsqueeze(0) * self.sub_block_size
        )

    def _make_noise_embedding(
        self, input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Noise token IDs  [B, Q_LEN]:
          sub-anchor 0  (c0) : real token from input_ids
          sub-anchors 1..n_sub-1 : MASK  (to be predicted)
        """
        B, N = anchor_positions.shape
        n_sub, S = self.n_sub, self.sub_block_size
        Q_LEN  = N * n_sub
        device = input_ids.device
        seq_len = input_ids.size(1)

        q_sub = torch.arange(Q_LEN, device=device) % n_sub  # [Q_LEN]
        is_c0 = q_sub == 0  # [Q_LEN]  — which queries are c0 positions

        # All start as MASK
        noise_ids = torch.full((B, Q_LEN), self.mask_token_id, dtype=torch.long, device=device)

        # Fill c0 positions with real tokens
        if is_c0.any():
            c0_q_indices = is_c0.nonzero(as_tuple=True)[0]   # [N]  indices in Q_LEN
            # anchor block ids for these c0 queries
            c0_block_ids = c0_q_indices // n_sub              # [N]
            c0_seq_pos   = anchor_positions[:, c0_block_ids]  # [B, N]
            c0_seq_pos   = c0_seq_pos.clamp(0, seq_len - 1)
            c0_tokens    = torch.gather(input_ids, 1, c0_seq_pos)          # [B, N]
            valid = block_keep_mask[:, c0_block_ids]                        # [B, N]
            c0_tokens = torch.where(valid, c0_tokens,
                                    torch.tensor(self.mask_token_id, device=device))
            noise_ids[:, c0_q_indices] = c0_tokens

        return self.embed_tokens(noise_ids)  # [B, Q_LEN, hidden]

    def _make_position_ids(
        self, anchor_positions: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Full position IDs  [B, seq_len + Q_LEN]  (context || anchors)."""
        device = anchor_positions.device
        B = anchor_positions.size(0)
        ctx_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        anch_pos = self._anchor_seq_positions(anchor_positions)  # [B, Q_LEN]
        return torch.cat([ctx_pos, anch_pos], dim=1)

    def _labels_and_weights(
        self, input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          target_ids   [B, Q_LEN] — token to predict at each query position
          weight_mask  [B, Q_LEN] — 1.0 for positions we train on, 0 otherwise

        We train on sub-anchors 1..n_sub-1 (skip c0 at index 0).
        Also apply original loss_mask at the target token's sequence position.
        """
        B, N = anchor_positions.shape
        n_sub = self.n_sub
        Q_LEN = N * n_sub
        seq_len = input_ids.size(1)
        device  = input_ids.device

        q_block = torch.arange(Q_LEN, device=device) // n_sub  # [Q_LEN]
        q_sub   = torch.arange(Q_LEN, device=device) % n_sub   # [Q_LEN]

        # Sequence positions of targets
        target_seq_pos = (
            anchor_positions[:, q_block]               # [B, Q_LEN]
            + q_sub.unsqueeze(0) * self.sub_block_size
        ).clamp(0, seq_len - 1)

        target_ids = torch.gather(input_ids, 1, target_seq_pos)  # [B, Q_LEN]

        # Weight: predict only sub-anchors 1..n_sub-1 that are in-bounds and valid
        is_prediction  = (q_sub > 0).unsqueeze(0)                   # [1, Q_LEN]
        in_bounds      = (
            anchor_positions[:, q_block]
            + q_sub.unsqueeze(0) * self.sub_block_size
        ) < seq_len                                                  # [B, Q_LEN]
        block_valid    = block_keep_mask[:, q_block]                # [B, Q_LEN]
        seq_loss_valid = torch.gather(loss_mask.float(), 1, target_seq_pos) > 0.5

        weight_mask = (
            is_prediction & in_bounds & block_valid & seq_loss_valid
        ).float()

        return target_ids, weight_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, seq_len]
        hidden_states: torch.Tensor,  # [B, seq_len, n_layers*hidden]
        loss_mask: torch.Tensor,      # [B, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bsz, seq_len = input_ids.shape
        device = input_ids.device
        dtype  = next(self.draft_model.parameters()).dtype

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        N     = anchor_positions.size(1)
        n_sub = self.n_sub
        Q_LEN = N * n_sub

        noise_emb   = self._make_noise_embedding(input_ids, anchor_positions, block_keep_mask)
        full_pos_ids = self._make_position_ids(anchor_positions, seq_len)
        attn_mask    = create_anchor_causal_sdpa_mask(
            anchor_positions, block_keep_mask,
            seq_len, self.sub_block_size, n_sub,
            dtype, device,
        )

        output_hidden = self.draft_model(
            position_ids=full_pos_ids,
            noise_embedding=noise_emb,
            target_hidden=hidden_states,
            attention_mask=attn_mask,
        )  # [B, Q_LEN, hidden]

        logits = self.lm_head(output_hidden)  # [B, Q_LEN, vocab]

        target_ids, weight_mask = self._labels_and_weights(
            input_ids, anchor_positions, block_keep_mask, loss_mask
        )

        flat_logits  = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_count    = flat_weights.sum() + 1e-6
        loss           = (loss_per_token * flat_weights).sum() / valid_count

        with torch.no_grad():
            pred_ids = flat_logits.argmax(-1)
            correct  = (pred_ids == flat_targets) & (flat_weights > 0.5)
            accuracy = correct.sum().float() / valid_count

        return loss, accuracy
