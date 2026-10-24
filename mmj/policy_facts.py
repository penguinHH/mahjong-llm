#!/usr/bin/env python3
"""在 facts.py 的事实后面再附上编码器策略头的评价顺序（前 3 名 + 与第 1 名的 logit 差），
给理由里的"次点の○○と僅差 / より良い / より明確に良い"提供依据：编码器用 Mortal 的 Q 值软标签训过，logit 差近似 Q 差。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/policy_facts.py
输出 work/mmj/{train,test}_facts_pol.jsonl（与 {split}_facts.jsonl 行对齐），train_mm.py 用 --facts --facts_name facts_pol。
注意：训练集 10 万条在编码器的训练数据里（全量 108 万的子集），评价会比测试集更"自信"，下面打印两边的分布以便对照。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/d/mahjong-llm/tools")
from actions import ACTIONS, legal_mask  # noqa: E402
from build_explanations import act_name, name  # noqa: E402
from encoder import EncoderPolicy  # noqa: E402
from facts import CALL_NAME  # noqa: E402
from state_features import N_PLANES, N_SCALARS  # noqa: E402

D = os.path.expanduser("~/mahjong-llm/work/mmj")


def show(a):
    if a.startswith("dahai:"):
        return name(a.split(":")[1]) + "切り"
    return CALL_NAME.get(a, act_name(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default=f"{D}/encoder_full_soft_e16.pt")
    ap.add_argument("--name", default="facts_pol", help="输出 {split}_<名>.jsonl")
    args = ap.parse_args()
    ck = torch.load(args.enc, map_location="cpu", weights_only=False)
    a = ck["args"]
    use_v = a.get("value_w", 0) > 0
    model = EncoderPolicy(N_PLANES, N_SCALARS, a["ch"], a["blocks"], a["dropout"], a.get("arch", "cnn"), a.get("heads", 8),
                          value=use_v)
    model.load_state_dict(ck["state_dict"])
    model.cuda().eval()
    print(f"编码器 {args.enc}；{'价值头（Q 差）' if use_v else '策略头（logit 差）'}", flush=True)
    for split in ("train", "test"):
        pl, sc = np.load(f"{D}/{split}_planes.npy"), np.load(f"{D}/{split}_scalars.npy")
        mask = legal_mask(pl, sc)
        out, vout = [], []
        with torch.no_grad():
            for i in range(0, len(pl), 4096):
                r = model(torch.from_numpy(pl[i:i + 4096]).cuda(), torch.from_numpy(sc[i:i + 4096]).cuda(),
                          torch.from_numpy(mask[i:i + 4096]).cuda(), with_value=use_v)
                out.append((r[0] if use_v else r).float().cpu())
                if use_v:
                    vout.append(r[1].float().cpu())
        logits = torch.cat(out).numpy()
        vals = torch.cat(vout).numpy() if use_v else logits
        facts = [json.loads(l) for l in open(f"{D}/{split}_facts.jsonl", encoding="utf-8")]
        rows, gaps = [], []
        for lg, v, m, fa in zip(logits, vals, mask, facts):
            legal = np.flatnonzero(m)
            first = legal[np.argmax(lg[legal])]          # 第 1 名按策略头（动作决策），其余按价值头（有的话）排、标 Q 差
            rest = [k for k in legal[np.argsort(-v[legal])] if k != first][:2]
            parts = [show(ACTIONS[first])] + [f"{show(ACTIONS[k])}（{v[k] - v[first]:+.1f}）" for k in rest]
            if rest:
                gaps.append(v[first] - v[rest[0]])
            rows.append(fa + "評価順：" + " > ".join(parts) + "。")
        with open(f"{D}/{split}_{args.name}.jsonl", "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        g = np.array(gaps)
        print(f"[{split}] {len(rows)} 条；第 1 名与次点的分差：<0.5 {np.mean(g < .5):.1%}，0.5～3 {np.mean((g >= .5) & (g < 3)):.1%}，"
              f"≥3 {np.mean(g >= 3):.1%}，中位数 {np.median(g):.2f}")
        if split == "train":
            y = [json.loads(json.loads(l)["output"])["action"]
                 for l in open(f"{D}/train_text.jsonl", encoding="utf-8")]
            top1 = [ACTIONS[int(np.flatnonzero(m)[np.argmax(lg[m])])] for lg, m in zip(logits, mask)]
            print(f"  训练集上编码器第 1 名 = Mortal 首选：{np.mean([t == x.replace(' ', '') for t, x in zip(top1, y)]):.2%}"
                  "（与测试集的差距 = 编码器对训练数据的记忆）")
        print("  例：", rows[3][-160:])


if __name__ == "__main__":
    main()
