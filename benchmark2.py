# %%
import os
import json
import wandb
import subprocess
import random
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
import io
from openai import OpenAI
import base64
from pydantic import BaseModel
import wandb 
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to("cuda")

# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from diffusers import StableDiffusion3Pipeline as StableDiffusion3PipelineVanilla

from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()

wandb.init(project="bench")
with open("prompts.json", "r") as f:
    prompts = json.load(f)

ours_pos_scores = []
vanilla_pos_scores = []
ours_neg_scores = []
vanilla_neg_scores = []

for i in range(5):
    for prompt in prompts:
        import time
        seed = int(time.time())
        positive_prompt = prompt["pos"]
        negative_prompt = prompt["neg"]
        
        print(f"Using seed: {seed}")    

        for block in pipe.transformer.transformer_blocks:
            block.attn.processor.neg_prompt_len=max([
                len(pipe.tokenizer.tokenize(negative_prompt)), 
                len(pipe.tokenizer_2.tokenize(negative_prompt)),
                len(pipe.tokenizer_3.tokenize(negative_prompt))
            ]) + 1 


        image_ours = pipe(
            positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=16,
            avoidance_factor=10, 
            guidance_scale=7, 
            negative_offset=-8.2, #-8
            clamp_value=20, 
            generator=torch.manual_seed(seed),  
        ).images

        negative_guidance_scales = pipe.negative_guidance_scales
        weight_maps = pipe.weight_maps
        image_vanilla = pipe(
            positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=16,
            guidance_scale=7, 
            vanilla=True,
            generator=torch.manual_seed(seed),
        ).images

        img = Image.fromarray(
                    np.concatenate(
                        [np.array(image_ours[-1]), np.array(image_vanilla[-1])], axis=1
                    ) 
        ) 

        # %%
        import pylab
        pylab.imshow(weight_maps[5].mean(0).mean(0).cpu().float().numpy())
        pylab.colorbar()

        # %%
        
        questions = ["there are " + negative_prompt + " in the image", positive_prompt]
        scores = []
        for image in [image_ours[-1], image_vanilla[-1]]:
            for question in questions:
                inputs = processor(image, question, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    itm_scores = torch.nn.functional.softmax(model(**inputs)[0], dim=-1)[0]
                scores.append(itm_scores[1].item())
        ours_pos_scores.append(scores[1])
        vanilla_pos_scores.append(scores[3])
        ours_neg_scores.append(scores[0])
        vanilla_neg_scores.append(scores[2])
        
        wandb.log({"image": wandb.Image(img, caption=f"{negative_prompt}"),
                   "ours_pos_score": np.mean(ours_pos_scores),
                    "vanilla_pos_score": np.mean(vanilla_pos_scores),
                    "ours_neg_score": np.mean(ours_neg_scores),
                    "vanilla_neg_score": np.mean(vanilla_neg_scores)
                })
                        
        