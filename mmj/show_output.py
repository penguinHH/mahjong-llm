#!/usr/bin/env python3
"""看一个局面上编码器 / 策略头的实际输出（CPU 上跑，不占显卡）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/show_output.py [ckpt 名] [测试集行号]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actions import ACTIONS, legal_mask, norm  # noqa: E402
from encoder import EncoderPolicy  # noqa: E402
from state_features import KINDS, N_PLANES, N_SCALARS, P  # noqa: E402

D = os.path.expanduser("~/mahjong-llm/work/mmj")
SFT = os.path.expanduser("~/mahjong-llm/work/sft")
name = sys.argv[1] if len(sys.argv) > 1 else "encoder_100k_c192b10.pt"
row = int(sys.argv[2]) if len(sys.argv) > 2 else 3

ck = torch.load(f"{D}/{name}", map_location="cpu", weights_only=False)
cfg = ck["enc_cfg"]
model = EncoderPolicy(N_PLANES, N_SCALARS, cfg["ch"], cfg["blocks"])
model.load_state_dict(ck["state_dict"])
model.eval()

tp, ts = np.load(f"{D}/test_planes.npy"), np.load(f"{D}/test_scalars.npy")
text = [json.loads(l) for l in open(f"{D}/test_text.jsonl", encoding="utf-8")][row]
meta = {m["md5"]: m for m in map(json.loads, open(f"{SFT}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}[text["md5"]]
pl, sc = torch.from_numpy(tp[row:row + 1]), torch.from_numpy(ts[row:row + 1])
mask = torch.from_numpy(legal_mask(tp[row:row + 1], ts[row:row + 1]))

with torch.no_grad():
    tiles, glob = model.enc(pl, sc)
    logits = model(pl, sc, mask)[0]
prob = torch.softmax(logits, -1)

hand = [KINDS[i] for i in range(34) for _ in range(sum(int(tp[row, P[f"hand>={c}"], i]) for c in (1, 2, 3, 4)))]
print(f"测试集第 {row} 条  手牌: {' '.join(hand)}")
print(f"\n① 编码器输出（以后经投影层变成 LLM 的 35 个局面 token）")
print(f"   tiles: {tuple(tiles.shape)}  ← 34 种牌各一个 {tiles.shape[-1]} 维向量")
print(f"   glob : {tuple(glob.shape)}  ← 整个局面一个 {glob.shape[-1]} 维向量")
print(f"   例：'{KINDS[4]}' 那一格向量的前 6 维 {[round(x, 2) for x in tiles[0, 4, :6].tolist()]} …")
print(f"\n② 策略头输出：46 个动作的分数 → 只在合法动作里做 softmax 得到概率")
order = prob.argsort(descending=True)
for k in order[:min(6, int(mask.sum()))].tolist():
    a = ACTIONS[k]
    tags = []
    if a == meta["mortal"]:
        tags.append("Mortal 首选")
    if a == norm(json.loads(text["output"])["action"]):
        tags.append("牌谱玩家")
    q = meta["q"].get(a)
    print(f"   {a:<12} 概率 {prob[k].item():6.1%}   Mortal Q {q:+.2f}   {' / '.join(tags)}")
print(f"\n   → 模型选 {ACTIONS[order[0].item()]}（概率最高的合法动作）")
