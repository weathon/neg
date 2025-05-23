from openai import OpenAI
from PIL import Image
import base64
client = OpenAI()

img = "random_sweeps/ours_4.png"
with open(img, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

encoded_string = f"data:image/png;base64,{encoded_string}"
response = client.moderations.create(
    model="omni-moderation-latest",
    input=[{
            "type": "image_url",
            "image_url": {
                "url": encoded_string,
            }
        }],
)

print(response.results[0].category_scores.sexual)