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
import sys

sys.path.append("..")
# %%
import torch
from sd_pipeline_train import StableDiffusion3Pipeline
from diffusers import StableDiffusion3Pipeline as StableDiffusion3PipelineVanilla

from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()
    
import wandb
import torch
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to("cuda")



positive_prompt = ["A bright, spacious kitchen with an island, stainless steel appliances, a large window, and a decorative fruit bowl on the counter.",
                    "a wide sandy beach under a sunny sky",
                    "a beautiful Canada landscape, in bob ross style, with a river, mountains",
                    "A quiet reading room with rows of wooden shelves, cozy armchairs and large windows letting in soft daylight, evoking a tranquil and studious atmosphere for visitors.",
                    "A modern kitchen, sleek countertops, spacious island, large windows, bright and airy, minimalist design, ready for culinary activities."]
negative_prompt = ["food",
                    "ocean",
                    "trees",
                    "book",
                    "faucets"] 

# %%
wandb.init(project="train")
from adapter import Adapter
adapter = Adapter()
adapter.train_init()

# %%
for param in pipe.transformer.parameters():
    param.requires_grad = False
    
for param in adapter.parameters():
    param.requires_grad = True

def process_image_with_grad(image, image_mean, image_std):
    # resize
    image = torch.nn.functional.interpolate(
        image, size=(224, 224), mode='bilinear', align_corners=False
    )
    # normalize
    image = (image - torch.tensor(image_mean).view(-1, 1, 1)) / torch.tensor(image_std).view(-1, 1, 1)
    return image
    
    
optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-5)
pipe.transformer.enable_gradient_checkpointing()


loss_fn = torch.nn.CrossEntropyLoss()

seed = 42
for epoch in range(200):
    for idx in range(len(positive_prompt)):
        for block in pipe.transformer.transformer_blocks:
            block.attn.processor.neg_prompt_len=max([
                len(pipe.tokenizer.tokenize(negative_prompt[idx])), 
                len(pipe.tokenizer_2.tokenize(negative_prompt[idx])),
                len(pipe.tokenizer_3.tokenize(negative_prompt[idx]))
            ]) + 1 
        
        optimizer.zero_grad(set_to_none=True)
        pipe.transformer.zero_grad(set_to_none=True)
        pipe.vae.zero_grad(set_to_none=True) # forgot this
        pipe.text_encoder.zero_grad(set_to_none=True)
        pipe.text_encoder_2.zero_grad(set_to_none=True)
        pipe.text_encoder_3.zero_grad(set_to_none=True)
        
        model.zero_grad(set_to_none=True)
        
        
        image_ours = pipe(
            positive_prompt[idx],
            negative_prompt=negative_prompt[idx],
            num_inference_steps=13,
            adapter=adapter.cuda(),
            # generator=torch.manual_seed(seed),
        ).images



        pil = Image.fromarray((image_ours.detach().permute(0, 2, 3, 1)[0].cpu().float().numpy() * 255).astype(np.uint8)) 


        img = image_ours[0].unsqueeze(0).cpu()


        text = ["an image of " + positive_prompt[idx] + ", photorealistic, high quality, normal lighting",
                "bad quality, low resolution, noisy, artifacts, bad lighting, unnatural colors"
                , "an image with " + negative_prompt[idx] + " in it"]
        inputs = processor(text=text, return_tensors="pt", padding="longest", do_rescale=False).to("cuda")
        pixel_values = process_image_with_grad(img, processor.image_processor.image_mean, processor.image_processor.image_std).cuda()
        scores = model(pixel_values=pixel_values, **inputs)
        loss = loss_fn(scores.logits_per_image[0]/10, torch.tensor([1., 0., 0.]).to("cuda"))
        print(torch.nn.functional.softmax(scores.logits_per_image[0]/10, dim=-1))
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        
        wandb.log({"loss": loss.item(), "image": wandb.Image(pil, caption=negative_prompt[idx])})


