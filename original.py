import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A women lying on bed during daytime, realistic style, high detail, full-body view from a side angle, full body visible, no cloth on, nude",
    negative_prompt="the person is wearing cloth",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
image.save("capybara.png")
