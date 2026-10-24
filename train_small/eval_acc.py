#!/usr/bin/env python3
"""离线一致率：模型动作 vs 凤凰卓选手实际动作（留出的第 7001-8000 局）。

    ~/mahjong-llm/venv-train/bin/python /mnt/d/mahjong-llm/train_small/eval_acc.py \
        ~/mahjong-llm/work/pred/qwen25-1.5b/generated_predictions.jsonl

输入是 LLaMA-Factory do_predict 的输出（每行 prompt / predict / label）。
统计：输出格式正确率、整体一致率、按动作类型的一致率（dahai 另算"打的是同一张牌"）。
这不是论文的主指标（论文是对 Mortal 的对局成绩），只用来快速比较不同大小的模型。
"""
import collections
import json
import re
import sys


def parse(text):
    m = re.search(r'\{.*?\}', text, re.S)
    if not m:
        return None
    try:
        a = json.loads(m.group(0))["action"]
    except Exception:
        return None
    return a.replace(" ", "")


def main(path):
    n = fmt_ok = hit = 0
    by_type = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold, pred = parse(r["label"]), parse(r["predict"])
            if gold is None:
                continue
            n += 1
            fmt_ok += pred is not None
            t = gold.split(":")[0]
            ok = pred == gold
            hit += ok
            by_type[t][0] += ok
            by_type[t][1] += 1
            if not ok and pred is not None:
                confusion[(t, (pred or "").split(":")[0])] += 1
    print(f"样本 {n}  格式正确 {fmt_ok / n:.2%}  整体一致率 {hit / n:.2%}")
    for t, (h, c) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        print(f"  {t:<10} {c:>6} 条  一致率 {h / c:.2%}")
    if confusion:
        print("最常见的错误（实际动作 → 模型动作）：",
              ", ".join(f"{a}→{b}×{k}" for (a, b), k in confusion.most_common(6)))


if __name__ == "__main__":
    main(sys.argv[1])
