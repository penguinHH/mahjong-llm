#!/usr/bin/env python3
"""单张牌桌截图看不到的信息，对编码器决策有多大影响（CPU 上跑，几分钟）。

把测试集里"截图里看不到"的特征清零，再让训好的编码器出动作，和不清零时比一致率：
  tedashi   —— 各家最后一张手切牌（手切 / 摸切在静态画面上分不出）
  furiten   —— 振听标志（同巡振听、立直后见逃要靠历史，画面上推不全；清零是最坏情况）

    PY=~/mahjong-llm/venv-train/bin/python
    $PY -I /mnt/d/mahjong-llm/mmj/ablate_unseen.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actions import ACTIONS, legal_mask, norm  # noqa: E402
from encoder import EncoderPolicy  # noqa: E402
from state_features import N_PLANES, N_SCALARS, P, S  # noqa: E402
from train_encoder import test_report  # noqa: E402

D = os.path.expanduser("~/mahjong-llm/work/mmj")
SFT = os.path.expanduser("~/mahjong-llm/work/sft")
TEDASHI = [P[f"tedashi{p}"] for p in range(4)]


def predict(model, planes, scalars, bs=1000):
    mask = torch.from_numpy(legal_mask(planes, scalars))
    out = []
    with torch.no_grad():
        for i in range(0, len(planes), bs):
            out.append(model(torch.from_numpy(planes[i:i + bs]), torch.from_numpy(scalars[i:i + bs]),
                             mask[i:i + bs]).argmax(-1))
    return [ACTIONS[k] for k in torch.cat(out).tolist()]


def main():
    torch.set_num_threads(4)   # 显卡上还在训 VLM，别抢满 CPU
    tp, ts = np.load(f"{D}/test_planes.npy"), np.load(f"{D}/test_scalars.npy")
    tt = [json.loads(l) for l in open(f"{D}/test_text.jsonl", encoding="utf-8")]
    meta = {m["md5"]: m for m in map(json.loads, open(f"{SFT}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    v_human = [norm(json.loads(r["output"])["action"]) for r in tt]
    v_meta = [meta[r["md5"]] for r in tt]
    has_ted = tp[:, TEDASHI].any(axis=(1, 2))
    print(f"测试集 {len(tp)} 条；有手切信息的 {has_ted.mean():.1%}，振听标志为 1 的 {(ts[:, S['furiten']] > 0).mean():.1%}")

    variants = {"完整输入": (tp, ts)}
    p1 = tp.copy()
    p1[:, TEDASHI] = 0
    variants["清零手切"] = (p1, ts)
    s2 = ts.copy()
    s2[:, S["furiten"]] = 0
    variants["清零手切 + 振听"] = (p1, s2)

    for name in ("encoder_full_soft_e16", "encoder_full_c192b10_human"):
        ck = torch.load(f"{D}/{name}.pt", map_location="cpu", weights_only=False)
        a = ck["args"]
        model = EncoderPolicy(N_PLANES, N_SCALARS, a["ch"], a["blocks"], a["dropout"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        print(f"\n== {name}（{a.get('label', 'mortal')} 标签）")   # 早期的 checkpoint 没有 label 参数，都是 Mortal
        base = None
        for v, (pl, sc) in variants.items():
            acts = predict(model, pl, sc)
            r = test_report(acts, v_human, v_meta)
            changed = "" if base is None else f"  动作变了 {np.mean([x != y for x, y in zip(acts, base)]):.2%}"
            base = base or acts
            print(f"  {v:<12} 与牌谱 {r['human']:.2%}  与 Mortal {r['mortal']:.2%}  Q差 {r['qgap']:.3f}{changed}")


if __name__ == "__main__":
    main()
