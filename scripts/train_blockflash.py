#!/usr/bin/env python3
# coding=utf-8
"""BlockFlash Stage-1 Training: predict only sub-block anchor positions.

Architecture:
  - Target model : large AR model (e.g. Qwen3-8B) — frozen, provides hidden states
  - Draft model  : based on Qwen3-0.6B architecture with DFlash attention
  - Predicts     : tokens at positions sub_block_size, 2*sub_block_size, ...,
                   (n_sub-1)*sub_block_size within each main block
  - Attention    : CAUSAL among the n_sub anchors — matches inference exactly

Usage:
  torchrun --standalone --nproc_per_node 1 scripts/train_blockflash.py \\
    --target-model-path /path/to/Qwen3-8B \\
    --draft-base-model-path /path/to/Qwen3-0.6B \\
    --train-data-path data/train.json \\
    --output-dir outputs/blockflash-anchor \\
    --block-size 16 --sub-block-size 4 \\
    --num-draft-layers 1 --batch-size 2
"""

import argparse
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.blockflash import OnlineBlockFlashAnchorModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train BlockFlash Stage-1 (anchor-only)")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True,
                             help="Path to frozen target AR model (e.g. Qwen3-8B)")
    model_group.add_argument("--draft-base-model-path", type=str, required=True,
                             help="Path to base model for draft architecture (e.g. Qwen3-0.6B)")
    model_group.add_argument(
        "--target-model-backend", type=str, default="hf", choices=["sglang", "hf"],
    )
    model_group.add_argument("--draft-config-path", type=str, default=None,
                             help="Resume: path to existing draft config.json")
    model_group.add_argument("--block-size", type=int, default=16,
                             help="Full speculation block size (default 16)")
    model_group.add_argument("--sub-block-size", type=int, default=4,
                             help="Sub-block size; must divide block_size evenly (default 4)")
    model_group.add_argument("--num-draft-layers", type=int, default=1,
                             help="Number of transformer layers in draft model")
    model_group.add_argument("--mask-token-id", type=int, default=None)
    model_group.add_argument(
        "--attention-backend", type=str, default="sdpa",
        choices=["eager", "sdpa"],
        help="Attention backend for draft model. sdpa recommended for anchor-only mask.",
    )
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--num-anchors", type=int, default=512,
                             help="Max main-block anchor positions sampled per sequence")
    model_group.add_argument("--embedding-key", type=str, default=None)
    model_group.add_argument("--lm-head-key", type=str, default=None)

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc", type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_models(args) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
    """Build frozen target model and trainable BlockFlash draft model."""
    assert args.block_size % args.sub_block_size == 0, (
        f"block_size ({args.block_size}) must be divisible by "
        f"sub_block_size ({args.sub_block_size})"
    )
    print_on_rank0(
        f"Loading target model from {args.target_model_path} "
        f"using {args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    # Draft config: take architecture from Qwen3-0.6B, override key fields
    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from checkpoint: {args.draft_config_path}")
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config  = AutoConfig.from_pretrained(args.draft_base_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size        = args.block_size
        # num_target_layers drives the fc projection dim
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0(
            f"Draft architecture from {args.draft_base_model_path}: "
            f"hidden={draft_config.hidden_size}, layers={draft_config.num_hidden_layers}"
        )

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    # Force sdpa — the anchor-only causal mask is not flex_attention native
    draft_config._attn_implementation = args.attention_backend

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    n_sub = args.block_size // args.sub_block_size
    print_on_rank0(
        f"BlockFlash config: block_size={args.block_size}, sub_block_size={args.sub_block_size}, "
        f"n_sub={n_sub} (predicts {n_sub-1} anchors per block)\n"
        f"Draft params: {sum(p.numel() for p in draft_model.parameters()):,}\n"
        f"target_layer_ids: {draft_model.target_layer_ids}"
    )

    return target_model, draft_model


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    import hashlib
    cache_key = hashlib.md5(
        f"{args.train_data_path}-{args.max_length}-"
        f"{args.chat_template}-{args.target_model_path}".encode()
    ).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    # Need at least block_size tokens with loss signal
    min_loss_tokens = args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(args, epoch, step, bf_model, draft_model, optimizer):
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(bf_model, StateDictType.FULL_STATE_DICT):
        state_dict = bf_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            # Copy the dflash.py modeling file so checkpoint is self-contained
            modeling_src = os.path.join(
                os.path.dirname(__file__), "..", "specforge", "modeling", "draft", "dflash.py"
            )
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, os.path.join(save_dir, "dflash.py"))

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def record_metrics(args, loss, accuracy, global_step, tracker, optimizer,
                   train_dataloader=None, mode="train"):
    logdict = {}
    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
    logdict[f"{mode}/loss"]     = loss
    logdict[f"{mode}/accuracy"] = accuracy

    total = args.num_epochs * len(train_dataloader) // args.accumulation_steps
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}/{total}  "
        f"Loss: {loss:.4f}  Acc: {accuracy:.4f}"
    )
    tracker.log(logdict, step=global_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    # --- Resume ---
    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Resuming from checkpoint: {draft_model_last_checkpoint}")
        if draft_model_last_checkpoint:
            cfg_path = os.path.join(draft_model_last_checkpoint, "config.json")
            if os.path.exists(cfg_path):
                args.draft_config_path = cfg_path

    target_model, draft_model = build_models(args)

    resume_state = None
    if draft_model_last_checkpoint:
        loaded = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded.state_dict())
        del loaded
        training_state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            resume_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
            print_on_rank0(
                f"Resuming from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )

    # --- Tokenizer + mask token ---
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"]    = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    draft_model.config.dflash_config["sub_block_size"]   = args.sub_block_size

    # --- Data ---
    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps     = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    # --- Target embeddings + head (frozen) ---
    print_on_rank0("Loading target embeddings and lm_head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    # --- BlockFlash training wrapper ---
    bf_model = OnlineBlockFlashAnchorModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=args.block_size,
        sub_block_size=args.sub_block_size,
        num_anchors=args.num_anchors,
    )

    bf_model = FSDP(
        bf_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    if resume_state is not None:
        optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer state: epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)

    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    # --- Training loop ---
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        progress_bar = (
            tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=True)
            if dist.get_rank() == 0 else train_dataloader
        )

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids      = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask      = data["loss_mask"].cuda()

            target_output  = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states  = target_output.hidden_states.cuda()

            loss, accuracy = bf_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log  = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log  = acc_log  / dist.get_world_size()
                record_metrics(
                    args, loss_log.item(), acc_log.item(),
                    global_step, tracker, optimizer, train_dataloader, mode="train",
                )

            if dist.get_rank() == 0:
                elapsed   = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc":  f"{accuracy.item():.4f}",
                    "t":    f"{elapsed:.2f}s",
                })

            if eval_dataloader and global_step % args.eval_interval == 0:
                draft_model.eval()
                eval_losses, eval_accs = [], []
                with torch.no_grad():
                    for eval_data in eval_dataloader:
                        e_input_ids = eval_data["input_ids"].cuda()
                        e_attn_mask = eval_data["attention_mask"].cuda()
                        e_loss_mask = eval_data["loss_mask"].cuda()
                        e_target    = target_model.generate_dflash_data(
                            e_input_ids, e_attn_mask, e_loss_mask
                        )
                        e_hidden = e_target.hidden_states.cuda()
                        e_loss, e_acc = bf_model(
                            input_ids=e_input_ids,
                            hidden_states=e_hidden,
                            loss_mask=e_loss_mask,
                        )
                        eval_losses.append(e_loss.item())
                        eval_accs.append(e_acc.item())
                record_metrics(
                    args,
                    sum(eval_losses) / len(eval_losses),
                    sum(eval_accs)   / len(eval_accs),
                    global_step, tracker, None, train_dataloader, mode="eval",
                )
                draft_model.train()

            if global_step % args.save_interval == 0:
                save_checkpoint(args, epoch, global_step, bf_model, draft_model, optimizer)

    save_checkpoint(args, args.num_epochs, global_step, bf_model, draft_model, optimizer)
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
