# python scripts/regenerate_train_data.py \
#     --model /mnt/data/szf_temp/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
#     --is-reasoning-model \
#     --concurrency 128 \
#     --max-tokens 2048 \
#     --temperature 0.8 \
#     --server-address localhost:30002 localhost:30003 \
#     --input-file-path ./cache/dataset/nemotron-v2_train.jsonl \
#     --output-file-path ./cache/dataset/nemotron-v2_qwen3p5-4b_regen.jsonl \
#     --resume

python scripts/regenerate_train_data.py \
    --model /mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    --is-reasoning-model \
    --concurrency 256 \
    --max-tokens 2048 \
    --temperature 0 \
    --server-address localhost:30002 localhost:30003 \
    --input-file-path ./cache/dataset/nemotron-v2_train.jsonl \
    --output-file-path ./cache/dataset/nemotron-v2_qwen3-4b_regen.jsonl \
    --resume
