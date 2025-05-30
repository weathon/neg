import re
import json

input_file = "prompts.txt"
output_file = "prompts_converted.jsonl"

pattern = re.compile(r'--prompt\s+"([^"]+)".*?--negative_prompt\s+([\w\s,:"]+?)(?:\s|--|$)')

results = []
with open(input_file, "r") as f:
    for line in f:
        if "--negative_prompt" not in line:
            continue
        match = pattern.search(line)
        if match:
            pos_prompt = match.group(1).strip()
            neg_prompt = match.group(2).strip().strip('"')
            results.append({"pos_prompt": pos_prompt, "neg_prompt": neg_prompt})

with open(output_file, "w") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")