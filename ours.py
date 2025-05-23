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

for block in pipe.transformer.transformer_blocks:
    block.attn.processor.neg_prompt_len=max([
        len(pipe.tokenizer.tokenize(negative_prompt)), 
        len(pipe.tokenizer_2.tokenize(negative_prompt)),
        len(pipe.tokenizer_3.tokenize(negative_prompt))
    ]) + 1 


image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=32,
    avoidance_factor=0.7,
    guidance_scale=6,
    negative_offset=-1,
    generator=torch.manual_seed(seed),  
).images[0] 


image.save("ours.png") 