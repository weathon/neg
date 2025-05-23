# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

import torch
from diffusers import StableDiffusion3Pipeline as StableDiffusion3PipelineOriginal

pipe_original = StableDiffusion3PipelineOriginal.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe_original = pipe_original.to("cuda")

# %%
negative_prompt = "there are trees in the image"
pipe.tokenizer.tokenize(negative_prompt)
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()
    block.attn.processor.neg_prompt_len=len(pipe.tokenizer.tokenize(negative_prompt)) + 1

# %%
import os
import random
import shutil

prompts = [
    "A typical suburban neighborhood in the USA, detached single-family homes, neatly trimmed lawns, two-car garages, parked cars, sidewalks, mailboxes at the edge of driveways, clear blue sky, daytime, summer season",
]

hp = []
seed = int(random.random() * 1000000)
af=5.625615976406609; ws=0.5673401323526025; no=-7.67718443431245; gc=7.641076639352483

image = pipe(
    prompt=prompts[0],
    negative_prompt=negative_prompt, 
    num_inference_steps=32,
    avoidance_factor=af,
    weight_scale=ws,
    negative_offset=no,
    guidance_scale=gc,
    generator=torch.manual_seed(seed),
).images[0] 
image.save(f"ours.png")

image = pipe_original(
    prompt=prompts[0],
    negative_prompt=negative_prompt,
    num_inference_steps=32,
    guidance_scale=gc,
    generator=torch.manual_seed(seed),
).images[0]
image.save(f"original.png")