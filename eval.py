import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco", torch_dtype=torch.float16).to("cuda")

def get_score(img, text):
    inputs = processor(img, text, return_tensors="pt").to("cuda", torch.float16)
    itm_scores = model(**inputs)[0]
    return itm_scores