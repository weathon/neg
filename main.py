# %%
import torch
from pipeline_flux import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# %%
from flux_processor import FluxAttnProcessor2_0
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = FluxAttnProcessor2_0()

# %%
import time
image = pipe(
    prompt="a beautiful landscape with mountains, in the style of Bob Ross",
    width=512,
    height=512,
    negative_prompt="tree",
    num_inference_steps=50,
    true_cfg_scale=4,
    generator=torch.manual_seed(1989),
).images[0]
image 

# %%
import numpy as np
import pylab
nmap = np.array([i.cpu().float() for i in pipe.negative_attn_maps]).mean(0).reshape(64, 64)
pylab.imshow(nmap)


