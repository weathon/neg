from openai import OpenAI
import base64
import io
client = OpenAI()

def check_moderation(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_base64}"
    
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=[
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
        ],
    )

    print(response.results[0].category_scores.sexual)

