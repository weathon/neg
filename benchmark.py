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
    print(completion.choices[0].message.parsed.reasoning)
    return data


with open("prompts.json", 'r') as f:
    prompts = json.load(f)

wandb.init(project="sd3-benchmark")

for _ in range(10):
    for prompt in prompts:
        positive_prompt = prompt["positive_prompt"] + " 4k, high quality, masterpiece, best quality, 8k, realistic, detailed, intricate, beautiful, cinematic lighting"
        negative_prompt = prompt["negative_prompt"]
        
        print(f"Positive Prompt: {positive_prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        
        seed = random.randint(0, 2**32 - 1)
        cmd = [
            "python", "ours.py",
            "--prompt", positive_prompt,
            "--negative_prompt", negative_prompt,
            "--seed", str(seed)
        ]
        subprocess.run(cmd)
        ours_pos_score = gpt_rating(Image.open("ours.png"), positive_prompt)
        ours_neg_score = gpt_rating(Image.open("ours.png"), negative_prompt)
        ours_score = ours_pos_score - ours_neg_score
        
        cmd = [
            "python", "vanilla_sd.py",
            "--prompt", positive_prompt,
            "--negative_prompt", negative_prompt, 
            "--seed", str(seed)
        ]
        subprocess.run(cmd)
        vanilla_pos_score = gpt_rating(Image.open("vanilla.png"), positive_prompt)
        vanilla_neg_score = gpt_rating(Image.open("vanilla.png"), negative_prompt)
        vanilla_score = vanilla_pos_score - vanilla_neg_score
        
        img = Image.fromarray(
            np.concatenate(
                [Image.open("ours.png"), Image.open("vanilla.png")], axis=1
            )
        )
        wandb.log({
            "image": wandb.Image(img, caption=f"Negative Prompt: {negative_prompt}"),
            "ours_score": ours_score,
            "vanilla_score": vanilla_score,
            "ours_pos_score": ours_pos_score,
            "ours_neg_score": ours_neg_score,
            "vanilla_pos_score": vanilla_pos_score,
            "vanilla_neg_score": vanilla_neg_score
        })
        