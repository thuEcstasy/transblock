MODEL=/mnt/data/szf_temp/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \


# CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
#     --model $MODEL --dtype bfloat16 --mem-frac=0.8 --port 30000 &

# CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
#     --model $MODEL --dtype bfloat16 --mem-frac=0.8 --port 30001 &

CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server \
    --model $MODEL --dtype bfloat16 --mem-frac=0.8 --port 30002 &

CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server \
    --model $MODEL --dtype bfloat16 --mem-frac=0.8 --port 30003 &

wait