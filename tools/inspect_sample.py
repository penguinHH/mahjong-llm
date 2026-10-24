import json
import re

p = "/home/user/mahjong-llm/work/smoke40/sft_riichi.jsonl"
with open(p, encoding="utf-8") as f:
    first = json.loads(f.readline())

ins = first["instruction"]
print("=== instruction length:", len(ins))
print("=== instruction tail (2200 chars) ===")
print(ins[-2200:])
print("=== output:", first["output"])

nums = re.findall(r"^(\d+)\.", ins, re.M)
print("=== numbered field lines:", nums)

# sample 2000th line for variety (riichi mid-game etc.)
with open(p, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i == 2000:
            it = json.loads(line)
            tail = it["instruction"][-1800:]
            print("=== sample #2000 tail ===")
            print(tail)
            print("=== sample #2000 output:", it["output"])
            break

n = 0
with open(p, encoding="utf-8") as f:
    for _ in f:
        n += 1
print("=== total sft lines:", n)
