import re
import json
import argparse
import shlex

input_file = "prompts.txt"
output_file = "prompts_converted.jsonl"

results = []

class NoExitArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)

with open(input_file, "r") as f:
    for line in f:
        # Remove leading/trailing whitespace and leading '#'
        line = line.strip()
        if not line:
            continue  # skip empty lines
        if line.startswith('#'):
            line = line[1:].strip()
        if line.startswith('python '):
            line = line[len('python '):].strip()
        # Remove filename (e.g., scripts/daamwandb.py) if present
        if line.startswith('scripts/'):
            parts = line.split(' ', 1)
            if len(parts) > 1:
                line = parts[1].strip()
            else:
                continue
        if not line or "--negative_prompt" not in line:
            continue
        # Remove inline comments after the command (if any)
        if ' --' in line:
            # Only keep up to the last --wandb or end of command
            line = line.split('--wandb')[0].strip()
        parser = NoExitArgumentParser(add_help=False)
        parser.add_argument('--prompt', type=str, required=True)
        parser.add_argument('--negative_prompt', type=str, required=True)
        parser.add_argument('extra', nargs=argparse.REMAINDER)  # ignore additional args
        print(f"Parsing line: {line}")
        args, _ = parser.parse_known_args(shlex.split(line))
        pos_prompt = args.prompt.strip()
        neg_prompt = args.negative_prompt.strip()
        results.append({"pos_prompt": pos_prompt, "neg_prompt": neg_prompt})

with open(output_file, "w") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")