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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
import torch
from sd_pipeline import StableDiffusion3Pipeline
from diffusers import StableDiffusion3Pipeline as StableDiffusion3PipelineVanilla
from diffusers import FlowMatchEulerDiscreteScheduler
from sd_transformer import SD3Transformer2DModel
from sd_processor import JointAttnProcessor2_0
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
transformer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16, subfolder="transformer")

pipe = StableDiffusion3Pipeline(transformer=transformer,
                                scheduler=pipe.scheduler,
                                vae=pipe.vae,
                                text_encoder=pipe.text_encoder,
                                text_encoder_2=pipe.text_encoder_2,
                                text_encoder_3=pipe.text_encoder_3,
                                tokenizer=pipe.tokenizer,
                                tokenizer_2=pipe.tokenizer_2,
                                tokenizer_3=pipe.tokenizer_3)
                                
pipe = pipe.to("cuda")
for block in pipe.transformer.transformer_blocks:
    block.attn.processor = JointAttnProcessor2_0()

# %%
# use different model for positive and negative detection? CLIP for positive and detection for negative 

# %%
# remove in image2image

# %%
# positive_prompt = ["a wide sandy beach under a sunny sky",
#                    "a beautiful Canada landscape, in bob ross style, with a river, mountains",
#                    "A quiet reading room with rows of wooden shelves, cozy armchairs and large windows letting in soft daylight, evoking a tranquil and studious atmosphere for visitors.",
#                    "A bustling cityscape at dusk with neon lights illuminating the streets, view from a pedestrian bridge",
#                    "A modern kitchen, sleek countertops, spacious island, large windows, bright and airy, minimalist design, ready for culinary activities."]
# negative_prompt = ["ocean"
#                   "trees",
#                   "book",
#                   "cars",
#                   "faucets"]

# %%


# %%
# images = []
# seed = 64 #8964
# os.makedirs("samples", exist_ok=True)
# for sample in range(20):
#     avoidance_factor = random.uniform(10, 15)
#     negative_offset = random.uniform(-8, -4)
#     clamp_value = random.uniform(8, 13)
#     print(f"Sample {sample}, avoidance_factor: {avoidance_factor}, negative_offset: {negative_offset}, clamp_value: {clamp_value}")
#     row = []
#     for i in range(len(positive_prompt)):
#         for block in pipe.transformer.transformer_blocks:
#             block.attn.processor.neg_prompt_len=max([
#                 len(pipe.tokenizer.tokenize(negative_prompt[i])), 
#                 len(pipe.tokenizer_2.tokenize(negative_prompt[i])),
#                 len(pipe.tokenizer_3.tokenize(negative_prompt[i]))
#             ]) + 1 

#         image_ours = pipe(
#             positive_prompt[i],
#             negative_prompt=negative_prompt[i],
#             num_inference_steps=16,
#             avoidance_factor=avoidance_factor,
#             guidance_scale=7,
#             negative_offset=negative_offset,
#             start_step=0,
#             clamp_value=clamp_value, 
#             generator=torch.manual_seed(seed), 
#         ).images
#         image_ours = image_ours[0]
#         image_ours.save("samples/ours_{}_{}.png".format(sample, i))
#         row.append(image_ours)
#     images.append(row)

# %%
avoidance_factor = 11.4587087402798; negative_offset = -6.514771096534329; clamp_value = 10.41918600862381

# %%
# Image.fromarray(np.hstack([np.vstack([np.array(img) for img in row]) for row in images])).save("avoidance_example.png")

# %%
# with open("noun.txt", "r") as f:
#     lines = f.read().split("\n")
#     positive_prompt = [i.split(".")[0] + ". photorealistic, high quality, 8k, detailed, realistic" for i in lines if i.strip() != ""]
#     negative_prompt = [i.split(".")[1][1:] for i in lines if i.strip() != ""]
# # positive_prompt = ["In the corridor, a student used a clamp to strangle another student's neck, trying to show his strength. The mess beside the shelves and the screams of the students made the scene full of panic. Blood is everywhere"]
# # negative_prompt = ["the cloth is clean, non-violent, bloodless, ungraphic"]

# %%
positive_prompt = ["a Canadian National train at mid night stoped at a winter mountain station, photorealistic, high quality, 8k, detailed, realistic"]
negative_prompt = ["tree"]
negative_prompt_effective = ["tree"]

# %%
# from mod import check_moderation 

# %%
import time
import copy
idx = 0
seed = int(time.time())
print(f"Using seed: {seed}") 
print("Positive prompts:", positive_prompt[idx])
print("Negative prompts:", negative_prompt[idx])
print("Effective negative prompts:", negative_prompt_effective[idx])
    
for block in pipe.transformer.transformer_blocks:
    block.attn.processor.neg_prompt_len=max([
        len(pipe.tokenizer.tokenize(negative_prompt_effective[idx])), 
    ])

for block in pipe.transformer.transformer_blocks:
    block.attn.processor.neg_prompt_len_3=max([
        len(pipe.tokenizer_3.tokenize(negative_prompt_effective[idx])), 
    ])
    
image_vanilla = pipe(
    positive_prompt[idx],
    negative_prompt=negative_prompt[idx],
    num_inference_steps=16,
    guidance_scale=6, 
    vanilla=True,
    generator=torch.manual_seed(seed),  
).images

image_ours = pipe(
    positive_prompt[idx],
    negative_prompt=negative_prompt[idx],
    num_inference_steps = 16,
    avoidance_factor = 3800, 
    negative_offset = -75/1000,
    clamp_value = 30,
    start_step=3,
    end_step=-2,
    guidance_scale = 6,
    generator=torch.manual_seed(seed),
).images


weights = pipe.transformer.transformer_blocks[0].attn.processor.attn_weight
negative_guidance_scales = pipe.negative_guidance_scales
weight_maps = copy.deepcopy(pipe.weight_maps)


# image_ban = pipe(
#     positive_prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=30,
#     guidance_scale=5, 
#     start_step=5,
#     end_step=15,
#     vanilla=True,
#     generator=torch.manual_seed(seed),  
# ).images

    
Image.fromarray(
            np.concatenate( 
                [np.array(image_ours[-1]), np.array(image_vanilla[-1])], axis=1
                # [np.array(image_ours[-1]), np.array(ima`ge_vanilla[-1]), np.array(image_ban[-1])], axis=1
            ) 
 )

# %%


# %%
# check_moderation(image_ours[0]), check_moderation(image_vanilla[0])

# %%
image_ours[0].save("ours.png") 

# %%
weight_maps[-7].shape 

# %%
import pylab 
pylab.figure(figsize=(40, 10))
for i in range(12):
    pylab.subplot(2, 6, i + 1)
    pylab.imshow(weight_maps[i].mean(0).mean(0).cpu().float().numpy(), vmin=0, vmax=30)
    pylab.axis("off")
pylab.colorbar()

# %%
# import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForImageTextRetrieval

# processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
# model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to("cuda")

# raw_image = image_ours[-1]

# question = "there are " + negative_prompt + " visible in the image"
# # question = positive_prompt
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda")


# with torch.no_grad():
#     itm_scores = torch.nn.functional.softmax(model(**inputs)[0], dim=-1)
#     cosine_score = model(**inputs, use_itm_head=False)[0]
# itm_scores, cosine_score


# %%
# import requests
# from PIL import Image
# import torch
# from PIL import ImageDraw

# from transformers import Owlv2Processor, Owlv2ForObjectDetection

# processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
# model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# %%
# text_labels = [[negative_prompt]]
# raw_image = image_ours[-1].copy()
# inputs = processor(text=text_labels, images=raw_image, return_tensors="pt")
# outputs = model(**inputs)

# # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
# target_sizes = torch.tensor([(raw_image.height, raw_image.width)])
# # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
# results = processor.post_process_grounded_object_detection(
#     outputs=outputs, target_sizes=target_sizes, threshold=0.13, text_labels=text_labels
# )

# # Retrieve predictions for the first image for the corresponding text queries
# result = results[0]
# boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
# for box, score, text_label in zip(boxes, scores, text_labels):
#     box = [round(i, 2) for i in box.tolist()]
#     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")

# # draw boxes 
# draw = ImageDraw.Draw(raw_image)
# for box, score, text_label in zip(boxes, scores, text_labels):
#     box = [round(i, 2) for i in box.tolist()]
#     draw.rectangle(box, outline="red", width=3)
#     draw.text((box[0], box[1]), f"{text_label} {round(score.item(), 3)}", fill="red")
# print(scores.max())
# raw_image


