# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

SpecForge is a training framework for speculative decoding draft models (EAGLE3, DFlash) designed to integrate with the SGLang serving framework. It trains lightweight draft models that predict multiple tokens ahead to accelerate LLM inference.

## Commands

### Install
```bash
uv venv -p 3.11 && source .venv/bin/activate
MAX_JOBS=8 uv pip install -v ".[fa]" --prerelease=allow --no-build-isolation
```

### Test
```bash
python -m unittest discover -s ./tests -p "test_*.py" -v
# single test file:
python -m unittest tests/test_data/test_dataset.py -v
```

### Lint
```bash
pre-commit run --all-files --show-diff-on-failure
```
Hooks: autoflake, isort, black, ruff (F401), clang-format, nbstripout.

### Training
```bash
torchrun --standalone --nproc_per_node NUM_GPUS scripts/train_eagle3.py [args]
torchrun --standalone --nproc_per_node NUM_GPUS scripts/train_dflash.py [args]
# or use shell examples:
bash examples/run_qwen3_4b_dflash_online.sh [NUM_GPUS] [TP_SIZE]
```

### Evaluation / Debug
```bash
python scripts/gsm8k_dflash_baseline.py --n 50 [--bigram-table path.pt] [--lambda-bigram 1.0]
python scripts/build_bigram.py --tokenizer /path/to/tokenizer --out bigram.pt
```

## Architecture

### Two Training Paradigms

**EAGLE3** (`specforge/core/eagle3.py`, trained by `scripts/train_eagle3.py`):
- Autoregressive draft model; takes concatenated hidden states from 3 target layers (layer 1, layer N//2, layer N-4) as additional input, projected to hidden size.
- Supports online (live target inference) and offline (pre-extracted hidden states) modes.
- Uses `LogSoftmaxLoss`.

**DFlash** (`specforge/core/dflash.py`, trained by `scripts/train_dflash.py`):
- Non-autoregressive block diffusion model; predicts `block_size` tokens simultaneously from masked (noise) inputs.
- Drafter uses bidirectional attention (`is_causal=False`) over a block where all non-context positions are `mask_token_id` embeddings — this means positions are predicted independently (no inter-position signal).
- Target hidden states are extracted from selected layers, concatenated, and projected via `fc` + `hidden_norm` before being used as context in each drafter layer.
- The deployed drafter lives in the HuggingFace model snapshot alongside `dflash.py` (custom `trust_remote_code` model). The copy at `specforge/modeling/draft/dflash.py` is the training version.

### Model Loading Flow

`AutoModel.from_pretrained(..., trust_remote_code=True)` loads the snapshot-local `dflash.py` at inference time. Changes to inference behavior must be made in the snapshot copy:
```
/mnt/data/szf_temp/huggingface/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/.../dflash.py
```
After editing, Python's module cache must be cleared in the same process (`del sys.modules[...]`) or a fresh process started.

### Target Layer Extraction

Both paradigms use `extract_context_feature(hidden_states, target_layer_ids)` from `specforge/modeling/utils.py` to pull hidden states from specific target layers. Layer indices are specified in the draft model config (`dflash_config.target_layer_ids` / `eagle3_config`).

### Distributed Strategy

`specforge/distributed.py` manages orthogonal process groups: Tensor Parallelism (TP), Data Parallelism (DP), Sequence Parallelism (SP — Ulysses and Ring modes). Initialized via `init_distributed(tp_size, sp_size, ...)`.

### Key DFlash Inference Arguments (`spec_generate`)

| Arg | Purpose |
|-----|---------|
| `oracle_noise_ids` | Feed verified tokens as noise instead of masks (upper-bound diagnostic) |
| `bigram_table` | Dict `{prev_tok → (LongTensor indices, FloatTensor log_probs)}` for left-to-right bigram calibration of draft logits |
| `lambda_bigram` | Weight for bigram log-prob bonus |
| `return_debug` | Returns per-iteration top-k logits, acceptance info, noise tokens |

### Model Configs

JSON configs in `configs/` define draft architecture. Key DFlash fields:
```json
{
  "block_size": 16,
  "dflash_config": {
    "mask_token_id": null,
    "target_layer_ids": [1, 9, 17, 25, 33]
  }
}
```
`mask_token_id` is resolved at runtime if null.

### Data Pipeline

`specforge/data/` handles conversation parsing (ShareGPT, HuggingFace Harmony, thinking-model formats) via a `ChatTemplate` registry. Loss masking is applied to non-assistant turns. Training data is prepared via `scripts/prepare_data.py` or `scripts/prepare_hidden_states.py` (offline mode only).
