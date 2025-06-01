import csv
import json
from openai import OpenAI
from pydantic import BaseModel
import os
import dotenv
from tqdm import tqdm
import pandas as pd
import concurrent.futures

dotenv.load_dotenv()

client = OpenAI()

class PromptOutput(BaseModel):
    positive_prompt: str
    negative_prompt: str

def get_top_noun_pair(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        first = next(reader)
        return first['noun1'], first['noun2']

def create_image_prompt(noun1, noun2):
    system_prompt = (
        "You are an expert prompt engineer specializing in generative imagery. "
        "For each noun pair, decide which concept should be the positive prompt and which should be the negative prompt, to minimize the chance that the positive implies or requires the negative. "
        "Craft a positive prompt for the target concept, and a negative prompt for the distractor. "
        "The positive prompt should be a concise, evocative description, not an instruction or action (e.g., do not say 'generate xxx', just describe 'xxx'). Avoid excessive detail. "
        "The positive and negative prompts should have a 'could be together' relationship, not a 'must be together' one. For example, do not say 'a busy road' as a positive prompt if the negative prompt is 'cars'. Instead, describe a road scene that could plausibly include cars, but does not require them. "
        "The positive prompt should describe a setting or context where the negative prompt could plausibly appear, but must avoid directly mentioning, implying, or referencing the negative prompt in any way. Do not use phrases like 'without xxx' or any negation of the negative prompt in the positive prompt. "
        "For example, if the pair is 'track' and 'train', you should use 'train' as the positive and 'track' as the negative, since a train can exist without a track in the image, but a track almost always implies a train. In this case, describe a train in a setting where a track could plausibly appear, but do not mention or require a track. For example: positive_prompt: 'A sleek modern train gliding through a lush green countryside under a clear blue sky.' negative_prompt: 'track'. "
        "If the positive and negative prompts are completely unrelated, make the positive prompt as close as possible to the negative prompt without being a must-have connection. For example, if the positive prompt is 'balloons' and the negative prompt is 'people', describe a scene with balloons at a party or gathering, but do not mention or imply people. The correct output would be: positive_prompt: 'A vibrant collection of assorted colorful balloons gathered closely together, gently floating against a soft pastel sky at sunrise.' negative_prompt: 'people'. "
        "If the positive prompt is an abstract item (such as 'group', 'beauty', or similar), return an empty string for both positive_prompt and negative_prompt. "
        "If there is no way to exclude the negative prompt from the positive prompt, return an empty string for both positive_prompt and negative_prompt. "
        "The negative prompt should be the object or concept that must not appear (e.g., 'x'), not a phrase like 'no x' or 'without x'. "
        "The negative prompt should discourage the presence or influence of the distractor. "
        "If the input is broken or unclear, do your best to infer and generate both prompts. "
        "Return a JSON object with keys 'positive_prompt' and 'negative_prompt'."
    )
    user_prompt = (
        f"Noun 1: '{noun1}'\n"
        f"Noun 2: '{noun2}'"
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=PromptOutput,
    )
    return completion.choices[0].message.parsed

def process_row(args):
    noun1, noun2 = args
    # Do not swap here; let the language model decide which order is less likely to imply the other, as per the system prompt instructions.
    return create_image_prompt(noun1, noun2)

if __name__ == '__main__':
    csv_path = 'steer_dataset/noun_cooccurrences.csv'
    results = []
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by=df.columns[2], ascending=False).head(10)
    rows = [(row['noun1'], row['noun2']) for _, row in df_sorted.iterrows()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for prompt_obj in tqdm(executor.map(process_row, rows), total=len(rows)):
            results.append({'positive_prompt': prompt_obj.positive_prompt, 'negative_prompt': prompt_obj.negative_prompt})
    with open('prompts.json', 'w') as f:
        json.dump(results, f)
    print(json.dumps(results))