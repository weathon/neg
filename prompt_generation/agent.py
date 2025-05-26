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
# client = OpenAI()

pipe = StableDiffusion3PipelineVanilla.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")


class Output(BaseModel):
    reason: str
    positive_prompt: str
    negative_prompt: str
    num_of_images: int
    exit: bool = False
    record: bool = False

import io
import base64
def get_image_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    return base64.b64encode(buffered.read()).decode("utf-8")
    


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
    

for i in range(20):    
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
        ]

    while True:
        
        completion = client.beta.chat.completions.parse(
            model="gemini-2.5-flash-preview-05-20",
            messages=messages,
            response_format=Output, 
            temperature=0.2,
        )
        print(completion.choices[0].message)
        if completion.choices[0].message.parsed.exit:
            record = completion.choices[0].message.parsed.record
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
        
        
    if record:
        with open("prompts.txt", "a") as f:
            f.write(f"Positive Prompt: {positive_prompt}\n")
            f.write(f"Negative Prompt: {negative_prompt}\n")
            f.write("\n")
            negative_prompt = ""
            positive_prompt = ""
            
        wandb.log({"image": wandb.Image(images[0], caption=f"Negative Prompt: {negative_prompt}")})
        generated += f"\nPositive Prompt: {positive_prompt}\nNegative Prompt: {negative_prompt}\n"