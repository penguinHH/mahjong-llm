#!/usr/bin/env python3
"""第 1 步（续）：把训练集 / 测试集整批转成局面张量，存成 npy，并做全量自检。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/build_state_dataset.py

输入（行对齐）：
  train: sft/riichi_100k_mortal_reason.jsonl（答案 = 理由 + Mortal 动作）
  test : sft/riichi_test5k.jsonl（答案 = 牌谱动作；与其它模型同一批 5000 条）
输出 work/mmj/：
  {split}_planes.npy  uint8   [N, N_PLANES, 34]
  {split}_scalars.npy float32 [N, N_SCALARS]
  {split}_text.jsonl  {"md5", "instruction"(短指令), "output"}，与 npy 行对齐
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_features import N_PLANES, N_SCALARS, PLANES, SCALARS, check, encode, fields, short_instruction  # noqa: E402

SFT = os.path.expanduser("~/mahjong-llm/work/sft")
OUT = os.path.expanduser("~/mahjong-llm/work/mmj")
SPLITS = {"train": "riichi_100k_mortal_reason.jsonl", "test": "riichi_test5k.jsonl"}


def work(line):
    r = json.loads(line)
    f = fields(r["instruction"])
    pl, sc = encode(f)
    bad = check(f, pl, sc)
    text = {"md5": hashlib.md5(r["instruction"].encode()).hexdigest(),
            "instruction": short_instruction(f), "output": r["output"]}
    return pl, sc, text, bad


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"平面 {N_PLANES} 个，标量 {N_SCALARS} 个")
    for split, fn in SPLITS.items():
        lines = open(f"{SFT}/{fn}", encoding="utf-8").readlines()
        with mp.Pool(8) as pool:
            res = pool.map(work, lines, chunksize=500)
        planes = np.stack([r[0] for r in res])
        scalars = np.stack([r[1] for r in res])
        np.save(f"{OUT}/{split}_planes.npy", planes)
        np.save(f"{OUT}/{split}_scalars.npy", scalars)
        with open(f"{OUT}/{split}_text.jsonl", "w", encoding="utf-8") as fo:
            for r in res:
                fo.write(json.dumps(r[2], ensure_ascii=False) + "\n")
        bad = [b for r in res for b in r[3]]
        print(f"\n[{split}] {len(res)} 条  planes {planes.shape} {planes.nbytes / 1e6:.0f}MB  scalars {scalars.shape}")
        print(f"  自检不一致：{len(bad)} 项 {sorted(set(bad))[:10]}")
        print(f"  平面平均非零格数：{(planes > 0).sum(axis=(1, 2)).mean():.1f} / {N_PLANES * 34}；"
              f"取值最大 {planes.max()}")
        print(f"  标量范围：min {scalars.min():.2f}  max {scalars.max():.2f}；"
              f"从不为 0 的标量：{[SCALARS[i] for i in np.flatnonzero((scalars != 0).all(0))][:6]}；"
              f"恒为 0：{[SCALARS[i] for i in np.flatnonzero((scalars == 0).all(0))]}")
        never = [PLANES[i] for i in np.flatnonzero((planes == 0).all(axis=(0, 2)))]
        print(f"  从未出现的平面：{never}")
        lens = sorted(len(r[2]["instruction"]) for r in res)
        print(f"  短指令长度中位数 {lens[len(lens) // 2]} 字符（原 prompt 约 3440）")
        print("  例：", json.dumps(res[0][2], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
