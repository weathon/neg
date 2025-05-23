# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import numpy as np

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
import wandb

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
try:
    shutil.rmtree("random_sweeps")
    print("Deleted previous random_sweeps directory")
except: 
    pass
os.makedirs("random_sweeps", exist_ok=True)

# "a beautiful landscape in Canada",
# "a serene at sunset on land",
# "a mountain range",

prompts = [
    "Suburban neighborhood in the USA",
    "A bustling city park in Tokyo at night",
    "A tranquil mountain scene in the Maldives",
]


from openai import OpenAI
import base64
import io
from pydantic import BaseModel
import json

client = OpenAI()

class Score(BaseModel):
    reasoning: str
    score: float

def gpt_rating(image, prompt):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode('utf-8')
    base64_image = f"data:image/png;base64,{base64_image}"
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "How well does the image match the prompt? Rate from 0 to 10, with 0 being not at all and 10 being perfect match. The prompt is: " + prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": base64_image,
                    "detail": "low"
                },
            },
            ],
        }],
        response_format=Score
    )

    data = completion.choices[0].message.parsed.score
    return data


hp = []
def run():
    wandb.init()
    af = wandb.config["avoidance_factor"]
    ws = wandb.config["weight_scale"]
    no = wandb.config["negative_offset"]
    gc = wandb.config["guidance_scale"]
    print(f"Running with af: {af}, ws: {ws}, no: {no}, gc: {gc}")
    positive_scores = []
    negative_scores = []
    
    for i in range(2):
        for prompt in prompts:
            seed = int(random.random() * 1000000)
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=32,
                avoidance_factor=af,
                weight_scale=ws,
                negative_offset=no,
                guidance_scale=gc,
                generator=torch.manual_seed(seed),
            ).images[0] 
            ps = gpt_rating(image, prompt)
            ns = gpt_rating(image, negative_prompt)
            positive_scores.append(ps)
            negative_scores.append(ns)
            wandb.log({"positive_score": np.mean(positive_scores), "negative_score": np.mean(negative_scores), "score": np.mean(positive_scores) - np.mean(negative_scores), "image": wandb.Image(image)})


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "avoidance_factor": {
            "max": 4,
            "min": 0.1,
            "distribution": "uniform",
        },
        "weight_scale": {
            "max": 2,
            "min": 0.1,
            "distribution": "uniform",
        },
        "negative_offset": {
            "max": 4,
            "min": -4,
            "distribution": "uniform",
        },
        "guidance_scale": {
            "max": 10,
            "min": 3,
            "distribution": "uniform",
        },
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep")

wandb.agent(sweep_id, function=run)