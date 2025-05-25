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

client = OpenAI()

class Score(BaseModel):
    id_better_positive: int
    id_better_negative: int
    id_better_quality: int
    
    
def compare(image1, image2, pos_prompt, neg_prompt):
    buffer = io.BytesIO()
    image1.save(buffer, format="PNG")
    buffer.seek(0)
    base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
    base64_1 = f"data:image/png;base64,{base64_1}"
    
    buffer = io.BytesIO()
    image2.save(buffer, format="PNG")
    buffer.seek(0)
    base64_2 = base64.b64encode(buffer.read()).decode('utf-8')
    base64_2 = f"data:image/png;base64,{base64_2}"
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Which image has better following for positive prompt, negative prompt (i.e. avoiding the negative prompt), and better qualit. Positive prompt: " + pos_prompt + ", negative prompt: " + neg_prompt + ". Answer which image is better using their id (0 or 1). For tie, answer -1."},
                {
                "type": "image_url",
                "image_url": {
                    "url": base64_1,
                },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": base64_2,
                },
                },
            ],
        }],
        response_format=Score
    )
    data = completion.choices[0].message.parsed
    return data


# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from diffusers import StableDiffusion3Pipeline as StableDiffusion3PipelineVanilla

from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()
    
pipe_vanilla = StableDiffusion3PipelineVanilla.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe_vanilla = pipe_vanilla.to("cuda")

with open("prompts.json", 'r') as f:
    prompts = json.load(f)

wandb.init(project="sd3-benchmark")


win_positive = 0
win_negative = 0
win_quality = 0
total_positive = 0
total_negative = 0
total_quality = 0

for _ in range(10):
    for prompt in prompts:
        positive_prompt = prompt["positive_prompt"] + " 4k, high quality, masterpiece, best quality, 8k, realistic, beautiful"
        negative_prompt = "there are " + prompt["negative_prompt"]
        
        print(f"Positive Prompt: {positive_prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        
        seed = random.randint(0, 2**32 - 1)

        for block in pipe.transformer.transformer_blocks:
            block.attn.processor.neg_prompt_len=max([
                len(pipe.tokenizer.tokenize(negative_prompt)), 
                len(pipe.tokenizer_2.tokenize(negative_prompt)),
                len(pipe.tokenizer_3.tokenize(negative_prompt))
            ]) + 1 


        image = pipe(
            positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=16,
            avoidance_factor=6.4,
            guidance_scale=7, 
            negative_offset=-4,
            clamp_value=13,
            generator=torch.manual_seed(seed),  
        ).images[0] 
        image.save("ours.png") 
        
        image = pipe_vanilla(
            positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=16,
            guidance_scale=7, 
            generator=torch.manual_seed(seed),
        ).images[0]
        image.save("vanilla.png")
        
        img = Image.fromarray(
            np.concatenate(
                [Image.open("ours.png"), Image.open("vanilla.png")], axis=1
            )
        )
        wandb.log({
            "image": wandb.Image(img, caption=f"Negative Prompt: {negative_prompt}")
        })
        