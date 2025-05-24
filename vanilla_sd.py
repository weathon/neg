# %%
import torch
from diffusers import StableDiffusion3Pipeline
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="winter landscape in BC, Canada")
parser.add_argument("--negative_prompt", type=str, default="water body is visible in the image")
parser.add_argument("--seed", type=int, default=1747891046)
args = parser.parse_args()
prompt = args.prompt
negative_prompt = args.negative_prompt
seed = args.seed

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# %%
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5,
    generator=torch.manual_seed(seed),
).images[0]
image.save("vanilla.png")

