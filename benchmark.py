import json
import wandb
import eval
import subprocess
import random
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np


with open("prompts.json", 'r') as f:
    prompts = json.load(f)

wandb.init(project="sd3-benchmark")

for _ in range(10):
    for prompt in prompts:
        positive_prompt = prompt["positive_prompt"] + " 4k, high quality, masterpiece, best quality, 8k, realistic, detailed, intricate, beautiful, cinematic lighting"
        negative_prompt = "there are " + prompt["negative_prompt"] + " in the image" 
        
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
        ours_pos_score = torch.sigmoid(eval.get_score(Image.open("ours.png"), positive_prompt)[0][1]).item()
        ours_neg_score = torch.sigmoid(eval.get_score(Image.open("ours.png"), negative_prompt)[0][1]).item()
        ours_score = ours_pos_score - ours_neg_score
        
        cmd = [
            "python", "vanilla_sd.py",
            "--prompt", positive_prompt,
            "--negative_prompt", negative_prompt, 
            "--seed", str(seed)
        ]
        subprocess.run(cmd)
        vanilla_pos_score = torch.sigmoid(eval.get_score(Image.open("vanilla.png"), positive_prompt)[0][1]).item()
        vanilla_neg_score = torch.sigmoid(eval.get_score(Image.open("vanilla.png"), negative_prompt)[0][1]).item()
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