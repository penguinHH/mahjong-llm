#!/usr/bin/env python3
"""配对 McNemar 检验：同一批 5000 条留出样本上，两个模型的"与 Mortal 一致"差多少点才算真差。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/train_small/mcnemar.py \
        名称A=路径1 名称B=路径2 [名称C=路径3 ...] [--pairs A-B,B-C]

路径两种写法：
  普通 jsonl   LLaMA-Factory / e2e_eval / mm3 的 generated_predictions*.jsonl（与 W/mmj/test_text.jsonl 行对齐）
  enc:xxx.pt   编码器 ckpt：在真实测试张量（W/mmj/test_{planes,scalars}.npy）上重跑策略头 argmax（GPU，每个几十秒）

不加 --pairs 时对输入顺序的相邻对做检验。精确二项检验（两侧 p），纯 math.comb 实现，不依赖 scipy。
一致率的 95% 置信区间约 ±1.1～1.3 点；<1 点的差没有显著性的话不要当结论。
"""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_acc import parse  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work")
MM3 = "/mnt/d/mahjong-llm/mm3"


def binom_two_sided_p(n1, n2):
    """X~Bin(n1+n2, 0.5)，观测 (n1, n2) 的两侧精确 p。"""
    n = n1 + n2
    if n == 0:
        return 1.0
    k = min(n1, n2)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def actions_from_jsonl(path):
    return [parse(json.loads(l)["predict"]) for l in open(path, encoding="utf-8")]


def actions_from_encoder(path, device="cuda"):
    import numpy as np
    import torch
    sys.path.insert(0, MM3)
    from perception import load_encoder  # noqa: E402
    from actions import ACTIONS, legal_mask  # noqa: E402
    pl, sc = np.load(f"{W}/mmj/test_planes.npy"), np.load(f"{W}/mmj/test_scalars.npy")
    mask = legal_mask(pl, sc.round())
    enc = load_encoder(path, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(pl), 1024):
            lg = enc(torch.from_numpy(pl[i:i + 1024]).to(device), torch.from_numpy(sc[i:i + 1024]).to(device),
                     torch.from_numpy(mask[i:i + 1024]).to(device))
            for lg_, m in zip(lg.float().cpu().numpy(), mask[i:i + 1024]):
                legal = np.flatnonzero(m)
                out.append(ACTIONS[int(legal[np.argmax(lg_[legal])])] if len(legal) else None)
    del enc
    torch.cuda.empty_cache()
    return out


def main():
    args = sys.argv[1:]
    pairs_arg = None
    if "--pairs" in args:
        i = args.index("--pairs")
        pairs_arg = args[i + 1]
        args = args[:i] + args[i + 2:]
    specs = dict(a.split("=", 1) for a in args)

    src = [json.loads(l) for l in open(f"{W}/sft/riichi_test5k.jsonl", encoding="utf-8")]  # 完整 prompt，与 test_planes 行对齐
    mor = {r["md5"]: r for r in map(json.loads, open(f"{W}/sft/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    labels = [mor.get(hashlib.md5(t["instruction"].encode()).hexdigest()) for t in src]
    cov = sum(x is not None for x in labels)
    print(f"Mortal 标签覆盖 {cov}/{len(src)}")

    acts, hits = {}, {}
    for name, path in specs.items():
        if path.startswith("enc:"):
            a = actions_from_encoder(os.path.expanduser(path[4:]))
        else:
            a = actions_from_jsonl(os.path.expanduser(path))
        acts[name] = a
        hits[name] = [a is not None and a == lab["mortal"] for a, lab in zip(acts[name], labels)]
        n = sum(lab is not None for lab in labels)
        print(f"{name}: 与 Mortal 一致 {sum(hits[name]) / n:.4f}（n={n}）")

    names = list(specs)
    pairs = [(names[i], names[i + 1]) for i in range(len(names) - 1)] if pairs_arg is None else \
        [(a, b) for a, b in (p.split("-", 1) for p in pairs_arg.split(","))]
    print("\n| 对 | 一致率差 | A对B错 | A错B对 | 两侧 p | 显著(5%) |")
    print("|---|---|---|---|---|---|")
    for a, b in pairs:
        ha, hb = hits[a], hits[b]
        n10 = sum(x and not y for x, y in zip(ha, hb))
        n01 = sum(not x and y for x, y in zip(ha, hb))
        p = binom_two_sided_p(n10, n01)
        diff = (sum(ha) - sum(hb)) / cov
        print(f"| {a} - {b} | {diff:+.4f} | {n10} | {n01} | {p:.2e} | {'是' if p < 0.05 else '否'} |")


if __name__ == "__main__":
    main()
