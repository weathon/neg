# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# %%
negative_prompt = "there are trees in the image"
pipe.tokenizer.tokenize(negative_prompt)

# %%
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()
    block.attn.processor.neg_prompt_len=len(pipe.tokenizer.tokenize(negative_prompt)) + 1

# %%
import os
os.makedirs("sweeps", exist_ok=True)
for guidance_scale in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for avoidance_factor in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        image = pipe(
            "a beautiful landscape in Canada, in the style of Bob Ross",
            negative_prompt=negative_prompt,
            num_inference_steps=16,
            avoidance_factor=avoidance_factor,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(46), 
        ).images[0] 
        image.save(f"sweeps/avoidance_factor_{avoidance_factor}_guidance_scale_{guidance_scale}.png")

