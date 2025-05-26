with open("prompts.txt", "r") as f:
    prompts = f.read().split("\n\n")
    
data = []
for prompt in prompts:
    if not prompt.strip():
        continue
    pos, neg = prompt.split("\n")
    pos = pos.strip().replace("Positive Prompt: ", "")
    neg = neg.strip().replace("Negative Prompt: ", "")
    
    print(f"p: {pos}")
    print(f"n: {neg}")
    
    data.append({
        "pos": pos,
        "neg": neg
    })

with open("prompts.json", "w") as f:
    import json
    json.dump(data, f, indent=4)