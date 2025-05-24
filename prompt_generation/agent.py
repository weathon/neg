from agents import Agent, ModelSettings, function_tool, Runner
from typing import List
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline as StableDiffusion3PipelineVanilla
import random
import os
import asyncio
import dotenv   
from pydantic import BaseModel
import wandb

wandb.init(project="prompt-generation")
dotenv.load_dotenv()


from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

pipe = StableDiffusion3PipelineVanilla.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")


class Output(BaseModel):
    reason: str
    positive_prompt: str
    negative_prompt: str
    num_of_images: int
    exit: bool

def get_image_base64(image: Image.Image) -> str:
    import io
    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def generate_images(positive_prompt: str, negative_prompt: str, num_of_images: int) -> List[Image.Image]:
    images = pipe(
        positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=16,
        guidance_scale=7, 
        num_images_per_prompt=num_of_images,
    ).images
    for i, image in enumerate(images):
        image.save(f"images/{negative_prompt.replace(' ', '_')}_{i}.jpg")
    return images

with open("system_prompt.txt", "r") as f:
    system_prompt = f.read()
    
    
generated = """
Positive Prompt: A grand medieval feast set in a great hall, filled with long tables covered in bountiful platters of food, goblets, and flickering light, with knights and nobles enjoying the lavish spread.  
Negative Prompt: Chalices, candles

Positive Prompt: A classic British breakfast, featuring a diverse selection of cooked delights, perfect for a filling and flavorful start to the day.  
Negative Prompt: Egg, sausage

Positive Prompt: A beautifully set dinner table, with elegant arrangements, a variety of enticing dishes, and delicate decor that speaks to the sophistication of the meal.  
Negative Prompt: Flowers, wine glasses

Positive Prompt: An inspiring art workshop filled with creativity, featuring a wide range of tools and materials used by participants working on their individual projects.  
Negative Prompt: Paintbrush, canvases

Positive Prompt: A charming antique store with shelves and tables filled with unique and historical treasures, offering a glimpse into the past.  
Negative Prompt: Lamps, old books"""

for i in range(100):    
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Example prompts (do not copy these) " + generated},
        ]

    while True:
        completion = client.beta.chat.completions.parse(
            model="gemini-2.5-pro-preview-05-20",
            messages=messages,
            response_format=Output, 
        )
        print(completion.choices[0].message)
        if completion.choices[0].message.parsed.exit:
            break
        positive_prompt = completion.choices[0].message.parsed.positive_prompt
        negative_prompt = completion.choices[0].message.parsed.negative_prompt
        num_of_images = completion.choices[0].message.parsed.num_of_images
        print(f"Positive Prompt: {positive_prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        print(f"Number of Images: {num_of_images}")
        images = generate_images(positive_prompt, negative_prompt, num_of_images)
        messages.append({"role": "assistant", "content": completion.choices[0].message.content})

        messages.append({
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{get_image_base64(image)}",
            }
            }
        for image in images],
        })
        
    with open("prompts.txt", "a") as f:
        f.write(f"Positive Prompt: {positive_prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write("\n")
    wandb.log({"image": wandb.Image(images[0], caption=f"Negative Prompt: {negative_prompt}")})
    generated += f"\nPositive Prompt: {positive_prompt}\nNegative Prompt: {negative_prompt}\n"