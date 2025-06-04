import io
import json
import base64
from typing import Dict, List

import torch
from PIL import Image
import wandb
from openai import OpenAI
from pydantic import BaseModel

from sd_pipeline import StableDiffusion3Pipeline
from sd_processor import JointAttnProcessor2_0

# Global seed used for all generations
SEED = 1989

# Initialize OpenAI client
client = OpenAI()

# Example prompt pairs used for evaluation
PROMPTS: List[Dict[str, str]] = [
    {
        "positive": "A modern living room designed for relaxation and entertainment, featuring comfortable seating, stylish decor, and ample natural light.",
        "negative": "television",
    },
    {
        "positive": "A large outdoor music concert at night, with a brightly lit stage and an energetic crowd, a vibrant light show illuminating the performers.",
        "negative": "speakers",
    },
    {"positive": "A bustling train station interior during peak travel time.", "negative": "trains"},
    {
        "positive": "A bustling city square on a sunny afternoon, with many people milling about, waiting for friends or simply observing the vibrant urban life. There are areas specifically provided for people to sit and wait, providing comfort and a vantage point for people-watching.",
        "negative": "benches",
    },
    {
        "positive": "A tranquil tropical beach scene, with a gently swaying hammock tied between unseen objects, soft sand, and a clear turquoise ocean under a bright sky.",
        "negative": "palm trees",
    },
    {
        "positive": "A wide urban avenue at night, with multiple lanes stretching into the distance, overhead streetlights illuminating the asphalt, and tall buildings lining both sides.",
        "negative": "cars",
    },
    {
        "positive": "A vibrant city street at rush hour, with glowing signs, blurred streaks of light, and the warm glow of storefronts reflecting on the street.",
        "negative": "cars",
    },
    {
        "positive": "A lively sports arena filled with cheering fans, on game day, with vibrant team colors and a large scoreboard.",
        "negative": "athletes",
    },
    {
        "positive": "An iconic, bustling amusement park at twilight, with many thrilling rides illuminated against the sky, laughter and music filling the air, colorful stalls and happy visitors.",
        "negative": "Ferris wheel",
    },
    {"positive": "A serene and minimalist bedroom, designed for quiet rest.", "negative": "nightstand"},
    {"positive": "A quiet forest path along a river, ground-level view.", "negative": "rocks"},
    {"positive": "A majestic mountain vista beneath clear skies.", "negative": "trees"},
    {"positive": "A historic downtown street lined with old brick shops at sunset.", "negative": "cars"},
    {"positive": "A minimalist workspace with a spacious desk and a large window.", "negative": "books"},
    {"positive": "A wide sandy beach at sunset with gentle waves.", "negative": "people"},
    {"positive": "A peaceful meadow of wildflowers under a clear sky.", "negative": "animals"},
    {"positive": "An old stone bridge crossing a calm river.", "negative": "boats"},
    {"positive": "A lively open-air market with colorful stalls at midday.", "negative": "vehicles"},
    {"positive": "A sleek modern kitchen with stainless steel appliances.", "negative": "faucets"},
    {"positive": "A narrow cobblestone alleyway with tall brick walls, evening light.", "negative": "graffiti"},
]

class Score(BaseModel):
    positive: float
    negative: float
    quality: float


def ask_gpt(image: Image.Image, pos: str, neg: str) -> Score:
    """Use GPT-4o to score image adherence."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Rate this image from 0 to 10 for how well it matches the positive prompt '"
                            + pos
                            + "', how well it avoids the negative prompt '"
                            + neg
                            + "' (10 means completely absent), and its overall quality."
                            "Use the provided function to record your scores."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "store_scores",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "positive": {"type": "number"},
                            "negative": {"type": "number"},
                            "quality": {"type": "number"},
                        },
                        "required": ["positive", "negative", "quality"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "store_scores"}},
    )
    args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    data = {"positive": args["positive"], "negative": args["negative"], "quality": args["quality"]}
    return Score(**data)


def load_pipe() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    ).to("cuda")
    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = JointAttnProcessor2_0()
    return pipe


pipe = load_pipe()


def run() -> None:
    wandb.init(project="sd3-sweep")
    cfg = wandb.config
    scores = []

    for pair in PROMPTS:
        pos = pair["positive"]
        neg = pair["negative"]
        for block in pipe.transformer.transformer_blocks:
            block.attn.processor.neg_prompt_len = len(pipe.tokenizer.tokenize(neg)) + 1
        image = pipe(
            pos,
            negative_prompt=neg,
            num_inference_steps=16,
            guidance_scale=6,
            generator=torch.manual_seed(SEED),
            avoidance_factor=cfg.avoidance_factor,
            negative_offset=cfg.negative_offset,
            clamp_value=cfg.clamp_value,
            start_step=cfg.start_step,
            end_step=cfg.end_step,
        ).images[0]

        score = ask_gpt(image, pos, neg)
        total = score.positive + score.negative + score.quality
        scores.append(total)
        wandb.log({
            "image": wandb.Image(image, caption=f"neg: {neg}"),
            "positive_score": score.positive,
            "negative_score": score.negative,
            "quality_score": score.quality,
            "total_score": total,
        })

    wandb.log({"mean_score": sum(scores) / len(scores)})


sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "mean_score"},
    "parameters": {
        "avoidance_factor": {"min": 1900.0, "max": 5700.0, "distribution": "uniform"},
        "negative_offset": {"min": -0.1125, "max": -0.0375, "distribution": "uniform"},
        "clamp_value": {"min": 15.0, "max": 45.0, "distribution": "uniform"},
        "start_step": {"min": 1, "max": 5, "distribution": "int_uniform"},
        "end_step": {"min": -3, "max": -1, "distribution": "int_uniform"},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project="sd3-sweep")
    wandb.agent(sweep_id, function=run)
