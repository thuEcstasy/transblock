CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
    --model /mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --port 6666 --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
    2>&1 | tee /mnt/data/szf_temp/SpecForge/build_bigram_0.log &

CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model /mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --port 6667 --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
    2>&1 | tee /mnt/data/szf_temp/SpecForge/build_bigram_1.log &

python scripts/build_bigram.py \
    --tokenizer /mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --server-address localhost:6666 localhost:6667 \
    --out /mnt/data/szf_temp/SpecForge/gsm8k_bigram.pt \
    --datasets gsm8k math alpaca metamath codealpaca evol_instruct \
    --n-per-dataset 500000 \
    --min-count 1 \
    --concurrency 128