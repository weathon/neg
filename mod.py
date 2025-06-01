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

    # cats = response.results[0].categories
    # flagged = []
    # for category, is_flagged in dict(cats).items():
    #     if is_flagged:
    #         flagged.append(category)
    
    # return flagged
    return response.results[0].category_scores.violence, response.results[0].category_scores.violence_graphic

