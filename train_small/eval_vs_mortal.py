#!/usr/bin/env python3
"""同一批 5000 条留出样本上，模型动作分别和"牌谱玩家"、"Mortal"的一致率。

    ~/mahjong-llm/venv-train/bin/python /mnt/d/mahjong-llm/train_small/eval_vs_mortal.py \
        ~/mahjong-llm/work/pred/qwen35-0.8b/generated_predictions.jsonl [更多预测文件...]

预测文件与 riichi_test5k.jsonl 行顺序相同（LLaMA-Factory do_predict 不打乱），
按 instruction 的 md5 对照 riichi_test5k_mortal_meta.jsonl（缺 Mortal 标签的行跳过）。
另报 "Q 差"：模型动作的 Q 比 Mortal 首选低多少（越小越好；不合法/格式错算缺失）。
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_acc import parse  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work/sft")


def main(paths):
    import hashlib
    src = [json.loads(l) for l in open(f"{W}/riichi_test5k.jsonl", encoding="utf-8")]
    mor = {r["md5"]: r for r in map(json.loads, open(f"{W}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    labels = [mor.get(hashlib.md5(r["instruction"].encode()).hexdigest()) for r in src]
    print(f"Mortal 标签覆盖 {sum(x is not None for x in labels)}/{len(src)}；"
          f"牌谱 vs Mortal 一致 {sum(x['human'] == x['mortal'] for x in labels if x) / sum(x is not None for x in labels):.2%}")
    for p in paths:
        preds = [json.loads(l) for l in open(p, encoding="utf-8")]
        assert len(preds) == len(src)
        n = h_hit = m_hit = 0
        gaps = []
        by = collections.defaultdict(lambda: [0, 0, 0])
        for pr, lab in zip(preds, labels):
            if lab is None:
                continue
            a = parse(pr["predict"])
            mort = lab["mortal"]
            n += 1
            h_hit += a == lab["human"]
            m_hit += a == mort
            if a in lab["q"]:
                gaps.append(max(lab["q"].values()) - lab["q"][a])
            t = lab["human"].split(":")[0]
            t = "chi" if t.startswith("chi") else t
            by[t][0] += 1
            by[t][1] += a == lab["human"]
            by[t][2] += a == mort
        gaps.sort()
        print(f"\n{p}")
        print(f"  与牌谱一致 {h_hit / n:.2%}   与 Mortal 一致 {m_hit / n:.2%}   "
              f"Q 差 平均 {sum(gaps) / len(gaps):.3f} / 中位数 {gaps[len(gaps) // 2]:.3f}（{len(gaps)} 条合法）")
        print("  按牌谱动作（牌谱 / Mortal）：" + "，".join(
            f"{t} {h / c:.1%}/{m / c:.1%}" for t, (c, h, m) in sorted(by.items(), key=lambda kv: -kv[1][0])))


if __name__ == "__main__":
    main(sys.argv[1:])
