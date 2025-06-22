import io
import json
import base64
from typing import Dict, List

import torch
from PIL import Image
import wandb
import os
from reinforcement import judge, prompts
import dotenv
from openai import OpenAI
from pydantic import BaseModel

from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0

# Global seed used for all generations
META_SEED = 1989

# Load environment variables and initialize Gemini client
dotenv.load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

PROMPTS = prompts.detailed_prompts


class Score(BaseModel):
    positive: float
    negative: float
    quality: float


def load_pipe() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    ).to("cuda")
    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = JointAttnProcessor2_0()
    return pipe


pipe = load_pipe()
import random

def run() -> None:
    wandb.init(project="sd3-sweep")
    cfg = wandb.config
    scores = []
    neg_scores = []
    reward_scores = []
    random.seed(META_SEED)
    for pair in PROMPTS:
        pos = pair["positive"]
        neg = pair["negative"]
        for block in pipe.transformer.transformer_blocks:
            block.attn.processor.neg_prompt_len = len(pipe.tokenizer.tokenize(neg)) + 1
        seed = random.randint(0, 2**32 - 1)
        image = pipe(
            pos,
            negative_prompt=neg,
            num_inference_steps=16,
            width=512,
            height=512,
            guidance_scale=6,
            generator=torch.manual_seed(seed),
            avoidance_factor=cfg.avoidance_factor,
            negative_offset=cfg.negative_offset,
            clamp_value=cfg.clamp_value,
            start_step=cfg.start_step,
            end_step=cfg.end_step,
        ).images[0]

        neg_score, reward_score = judge.eval(image, pos, neg, return_ind=True)
        total = neg_score + reward_score
        scores.append(total)
        neg_scores.append(neg_score)
        reward_scores.append(reward_score)
        wandb.log({
            "image": wandb.Image(image, caption=f"neg: {neg}"),
            "total_score": total,
            "negative_score": neg_score,
            "reward_score": reward_score,
        })

    wandb.log({"mean_score": sum(scores) / len(scores), "negative_mean": sum(neg_scores) / len(neg_scores), "reward_mean": sum(reward_scores) / len(reward_scores)})


sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "mean_score"},
    "parameters": {
        "avoidance_factor": {"min": 1500.0, "max": 5000.0, "distribution": "uniform"},
        "negative_offset": {"min": -0.2, "max": -0.0, "distribution": "uniform"},
        "clamp_value": {"min": 10.0, "max": 20.0, "distribution": "uniform"},
        "start_step": {"min": 1, "max": 7, "distribution": "int_uniform"},
        "end_step": {"min": -7, "max": -1, "distribution": "int_uniform"},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project="sd3-sweep")
    wandb.agent(sweep_id, function=run)
