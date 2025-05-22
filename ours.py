# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# %%
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()


prompt = "winter landscape in BC, Canada"
negative_prompt = "water body is visible in the image"
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
seed = 1747891046#int(time.time())

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=32,
    avoidance_factor=2,
    guidance_scale=8,
    generator=torch.manual_seed(seed),  
).images[0] 


# %%


# %%
seed

# %%
Image.fromarray(np.concatenate([np.array(image1), np.array(image2)], axis=1))

# %%
# import cv2, pylab
# img1 = np.array(image1)
# img2 = np.array(image2)
# pylab.subplot(1, 2, 1)
# pylab.imshow(img1)
# pylab.title("A")
# pylab.axis('off')
# pylab.subplot(1, 2, 2) 
# pylab.imshow(img2) 
# pylab.title("B")
# pylab.axis('off') 
# pylab.suptitle(f"Prompt: {prompt}\nNegative Prompt: {negative_prompt}\nSeed: {seed}")
# pylab.tight_layout() 

# %%
import pylab 
map = torch.stack(pipe.neg_maps)[-3].mean((0,1,2,3)).reshape(64, 64).cpu().float().numpy()
pylab.imshow(map)
pylab.colorbar()

# %%
pylab.plot(torch.stack(pipe.neg_maps).mean((1,2,3,4)).abs().mean(-1).cpu().float().numpy())


