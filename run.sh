prompt="A beautiful view over the park, realistic, high quality, 8k, detailed, cinematic lighting, hyper realistic, photo realistic, award winning photography"
negative_prompt="vegetation, trees, plants, flowers, grass"
seed=$(shuf -i 1-1000000 -n 1)

CUDA_VISIBLE_DEVICES=0 python3 ours.py \
    --prompt "$prompt" \
    --negative_prompt "$negative_prompt" \
    --seed $seed &

CUDA_VISIBLE_DEVICES=2 python3 vanilla_sd.py \
    --prompt "$prompt" \
    --negative_prompt "$negative_prompt" \
    --seed $seed &

wait