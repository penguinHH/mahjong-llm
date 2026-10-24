#!/usr/bin/env python3
"""渲染几张测试集样图，存到 vlm/samples/（本地查看用）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/vlm/render_samples.py [行号 ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_board import fields, render  # noqa: E402

SRC = os.path.expanduser("~/mahjong-llm/work/sft/riichi_test5k.jsonl")
OUT = "/mnt/d/mahjong-llm/vlm/samples"
rows = [int(x) for x in sys.argv[1:]] or [3, 0, 17]
os.makedirs(OUT, exist_ok=True)
lines = open(SRC, encoding="utf-8").readlines()
for r in rows:
    d = json.loads(lines[r])
    f = fields(d["instruction"])
    im = render(f)
    p = f"{OUT}/test{r}.png"
    im.save(p)
    print(p, im.size, "| 手牌", " ".join(f["my hands"]), "| 合法", f["Legal actions"][:6], "| 牌谱", d["output"])
