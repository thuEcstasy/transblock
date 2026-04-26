#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_PROJECT=dflash-reproduce
export WANDB_MODE=offline
export WANDB_API_KEY=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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
NUM_GPUS=4

ATTENTION_BACKEND=${2:-flex_attention}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-draft-config.json \
    --train-data-path $ROOT_DIR/cache/dataset/nemotron-v2_qwen3-4b_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-nemotron \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-4b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --sub-block-size 4 \
    --wandb-name qwen3-4b-dflash-nemotron \
    --embedding-key model.embed_tokens.weight \
    --trust-remote-code \
    --sglang-mem-fraction-static 0.3 \
    --cache-dir /mnt/data/szf_temp/cache \
