#!/usr/bin/env python3
"""第 1 步（价值头）：冻结的编码器上接一个 46 维价值头，回归 Mortal 各合法动作的 Q 值（相对最优动作的优势，≤ 0）。

为什么：理由里的"次点の○○と僅差 / より良い / より明確に良い"由 Mortal 的 Q 值差按 0.5 / 3 两个阈值分档；
策略头只学了"哪个好"，学不到"好多少"（加编码器评价顺序后这句仍只有 68%）。价值头直接给出 Q 差的估计。

训练数据 = 全量 108 万条去掉 LLM 用的 10 万条（in100k）和验证局：这样它在 LLM 训练集上的预测也是"没见过"的，
和测试集同分布（编码器本体见过那 10 万条，这只能部分缓解，下面分别报告两边的准确率）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/train_value.py
输出 work/mmj/value_head_e16.pt
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actions import A, legal_mask  # noqa: E402
from encoder import EncoderPolicy, PolicyHead  # noqa: E402
from state_features import N_PLANES, N_SCALARS  # noqa: E402

D = os.path.expanduser("~/mahjong-llm/work/mmj")
SFT = os.path.expanduser("~/mahjong-llm/work/sft")


def level(gap):
    return np.where(gap < 0.5, 0, np.where(gap < 3, 1, 2))


def report(pred, q, logits=None):
    """pred / q：[N, 46]，非法动作为 NaN。以 Mortal 首选 a* 为"选择"：
    次点 = 其余合法动作里 Q 最高的；档位 = Q[a*] - Q[次点] 按 0.5 / 3 分档（与 build_explanations.mortal_gap 相同）。"""
    legal = ~np.isnan(q)
    ok = legal.sum(1) >= 2
    pred, q, legal = pred[ok], q[ok], legal[ok]
    qm = np.where(legal, q, -np.inf)
    best = qm.argmax(1)
    r = np.arange(len(q))
    others = qm.copy()
    others[r, best] = -np.inf
    sec = others.argmax(1)
    true_lv = level(q[r, best] - q[r, sec])
    pm = np.where(legal, pred, -np.inf)
    pm[r, best] = -np.inf
    psec = pm.argmax(1)
    plv = level(pred[r, best] - pred[r, psec])
    out = {"n": int(len(q)), "次点对": float(np.mean(psec == sec)),
           "档位对（次点给定）": float(np.mean(level(pred[r, best] - pred[r, sec]) == true_lv)),
           "整句对": float(np.mean((psec == sec) & (plv == true_lv))),
           "优势 MAE": float(np.nanmean(np.abs(np.where(legal, pred - (q - q[r, best][:, None]), np.nan))))}
    if logits is not None:
        lm = np.where(legal, logits[ok], -np.inf)
        lm[r, best] = -np.inf
        out["策略头次点对"] = float(np.mean(lm.argmax(1) == sec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default=f"{D}/encoder_full_soft_e16.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=f"{D}/value_head_e16.pt")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"

    ck = torch.load(args.enc, map_location="cpu", weights_only=False)
    a = ck["args"]
    pol = EncoderPolicy(N_PLANES, N_SCALARS, a["ch"], a["blocks"], a["dropout"])
    pol.load_state_dict(ck["state_dict"])
    pol.to(dev).eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    head = PolicyHead(a["ch"]).to(dev)       # 同样的结构：每种牌一个打牌值 + 赤五 + 9 个其它动作

    q_all = np.load(f"{D}/full_q.npy", mmap_mode="r")
    game = np.load(f"{D}/full_game.npy")
    in100k = np.load(f"{D}/full_in100k.npy")
    rng = np.random.default_rng(0)                                   # 与 train_encoder.py 同样的验证局划分
    val_games = rng.choice(game.max() + 1, int((game.max() + 1) * 0.03), replace=False)
    is_val = np.isin(game, val_games)
    keep = (game >= 0) & ~in100k & ~is_val
    tr_idx = np.flatnonzero(keep)
    planes = np.load(f"{D}/full_planes.npy", mmap_mode="r")
    scalars = np.load(f"{D}/full_scalars.npy", mmap_mode="r")
    t0 = time.time()
    TP = torch.from_numpy(np.ascontiguousarray(planes[tr_idx])).to(dev)
    TS = torch.from_numpy(np.ascontiguousarray(scalars[tr_idx])).to(dev)
    TQ = torch.from_numpy(np.ascontiguousarray(q_all[tr_idx])).to(dev).float()
    good = ~torch.isnan(TQ).all(1) & ((~torch.isnan(TQ)).sum(1) >= 2)
    TP, TS, TQ = TP[good], TS[good], TQ[good]
    print(f"训练 {len(TQ)} 条（去掉 10 万子集、验证局、没有 Q 值或只有 1 个合法动作的）；载入 {time.time() - t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    spe = math.ceil(len(TQ) / args.bs)
    total = args.epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * s / total)))

    def predict(P, S, bs=4096):
        out = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for i in range(0, len(P), bs):
                tiles, glob = pol.enc(P[i:i + bs], S[i:i + bs])
                out.append(head(tiles.float(), glob.float()).float())
        return torch.cat(out)

    for ep in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(TQ), device=dev)
        tot = 0.0
        for i in range(0, len(TQ), args.bs):
            idx = perm[i:i + args.bs]
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                tiles, glob = pol.enc(TP[idx], TS[idx])
            pred = head(tiles.float(), glob.float())
            q = TQ[idx]
            legal = ~torch.isnan(q)
            adv = q - q.nan_to_num(-1e9).amax(1, keepdim=True)          # 相对最优动作的优势（最优 = 0）
            loss = F.smooth_l1_loss(pred[legal], adv[legal])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f"ep {ep + 1}  loss {tot / len(TQ):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state_dict": head.state_dict(), "enc": args.enc, "args": vars(args)}, args.out)

    # ---------------- 评测：测试集 5000 条（Mortal 的 Q 值来自 meta），和 LLM 训练用的 10 万条
    tp, ts = np.load(f"{D}/test_planes.npy"), np.load(f"{D}/test_scalars.npy")
    tt = [json.loads(l) for l in open(f"{D}/test_text.jsonl", encoding="utf-8")]
    meta = {m["md5"]: m for m in map(json.loads, open(f"{SFT}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    qt = np.full((len(tt), 46), np.nan, np.float32)
    for i, r in enumerate(tt):
        for k, v in meta[r["md5"]]["q"].items():
            qt[i, A[k.replace(" ", "")]] = v
    XP, XS = torch.from_numpy(tp).to(dev), torch.from_numpy(ts).to(dev)
    pv = predict(XP, XS).cpu().numpy()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        lg = pol(XP, XS, torch.from_numpy(legal_mask(tp, ts)).to(dev)).float().cpu().numpy()
    print("测试集 5000：", json.dumps(report(pv, qt, lg), ensure_ascii=False))
    i100 = np.flatnonzero(in100k)[:20000]
    pv = predict(torch.from_numpy(np.ascontiguousarray(planes[i100])).to(dev),
                 torch.from_numpy(np.ascontiguousarray(scalars[i100])).to(dev)).cpu().numpy()
    print("LLM 训练集（10 万子集前 2 万）：", json.dumps(report(pv, np.asarray(q_all[i100], np.float32)), ensure_ascii=False))


if __name__ == "__main__":
    main()
