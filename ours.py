# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# %%
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="winter landscape in BC, Canada")
parser.add_argument("--negative_prompt", type=str, default="water body is visible in the image")
parser.add_argument("--seed", type=int, default=1747891046)
args = parser.parse_args()
prompt = args.prompt
negative_prompt = args.negative_prompt
seed = args.seed

len(pipe.tokenizer_3.tokenize(negative_prompt)), len(pipe.tokenizer_3.tokenize(negative_prompt)), len(pipe.tokenizer_3.tokenize(negative_prompt))

# %%
for block in pipe.transformer.transformer_blocks:
    block.attn.processor.neg_prompt_len=max([
        len(pipe.tokenizer.tokenize(negative_prompt)), 
        len(pipe.tokenizer_2.tokenize(negative_prompt)),
        len(pipe.tokenizer_3.tokenize(negative_prompt))
    ]) + 1 

# %%
import time
import numpy as np
from PIL import Image 
# seed = 1747891046#int(time.time())

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=32,
    avoidance_factor=2,
    guidance_scale=8,
    # generator=torch.manual_seed(1747891046),  
).images[0] 


image.save("ours.png") 