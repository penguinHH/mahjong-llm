#!/usr/bin/env python3
"""次点 / 档位分类头：理由里"次点の○○と僅差 / より良い / より明確に良い"这一句的直接预测器。

为什么：这句由 Mortal 的 Q 值决定——次点 = 选中动作以外 Q 最高的合法动作，档位 = Q[选中] - Q[次点] 按 0.5 / 3
分成三档（tools/build_explanations.py 的 mortal_gap）。原来的价值头回归各动作的优势、再按阈值切档：
回归误差 0.30 左右，落在阈值附近就切错，整句只对约 80%（tf256b6_v_x：次点 90.7%、整句 80.1%）；
理由 LLM 原样照抄，R35 里这一句精确率 79.5%，是理由里唯一不到 100% 的句型。

做法：冻结 tf256b6_v_x 编码器，接一个以"选中动作"为条件的小网络，对每个动作输出 4 维：
  [0]   次点 logit（在"合法且不是选中动作"的动作里做 softmax）
  [1:4] 该动作与选中动作的 Q 差档位（<0.5 / 0.5～3 / ≥3；对方更好时差为负，也算第 0 档，与 mortal_gap 一致）
训练时"选中动作" = 编码器策略头的首选（与推理时一样），目标用 Mortal 的 Q 值算。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/gap_head.py
输出 work/mmj/gap_head_tf256b6_v_x.pt；e2e_eval.py --gap 用它生成評価順
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actions import A, legal_mask  # noqa: E402
from encoder import EncoderPolicy  # noqa: E402

D = os.path.expanduser("~/mahjong-llm/work/mmj")
SFT = os.path.expanduser("~/mahjong-llm/work/sft")
CANON = (0.2, 1.5, 5.0)        # 評価順里写给 LLM 的各档代表分差（远离 0.5 / 3 两个阈值，LLM 照抄不会切错档）


class GapHead(nn.Module):
    """输入编码器的 tiles [B,34,C]、glob [B,C] 和选中动作 [B]，输出 [B,46,4]（见模块说明）。
    每个动作的特征：打牌 = 该牌的 token，打赤五 = 对应 5 的 token，9 个宣言 = 场况 token；再加动作身份向量。
    与选中动作的特征做 [x, c, x*c] 拼接后过 MLP——"这个动作比选中的差多少"是成对的比较。"""

    def __init__(self, ch, hidden=512, aka_pos=(4, 13, 22)):
        super().__init__()
        self.aka_pos = list(aka_pos)
        self.kind = nn.Parameter(torch.zeros(46, ch))
        nn.init.normal_(self.kind, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(3 * ch, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, tiles, glob, chosen):
        x = torch.cat([tiles, tiles[:, self.aka_pos], glob[:, None].expand(-1, 9, -1)], 1) + self.kind   # [B,46,C]
        c = x[torch.arange(len(x), device=x.device), chosen][:, None].expand_as(x)
        return self.mlp(torch.cat([x, c, x * c], -1))


def level(gap):
    """Q 差 → 档位（torch / numpy 通用）：<0.5 → 0，0.5～3 → 1，≥3 → 2。"""
    return (gap >= 0.5).long() + (gap >= 3).long() if torch.is_tensor(gap) else (gap >= 0.5) * 1 + (gap >= 3) * 1


def targets(q, legal, chosen):
    """q [B,46]（非法为 NaN）、legal [B,46]、chosen [B] → others（候选次点）、真次点 [B]、各动作档位 [B,46]。"""
    r = torch.arange(len(q), device=q.device)
    others = legal & ~torch.isnan(q)
    others[r, chosen] = False
    sec = torch.where(others, q, torch.full_like(q, -1e9)).argmax(1)
    lv = level(q[r, chosen][:, None] - q.nan_to_num(0.0))
    return others, sec, lv


def load_gap(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = GapHead(ck["ch"], ck["args"]["hidden"])
    head.load_state_dict(ck["state_dict"])
    return head.to(device).eval().requires_grad_(False)


def load_gap_encoder(path, device):
    """--finetune 训出的 checkpoint 里带着微调后的编码器（排評価順要用它）；没有就返回 None。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "enc_state" not in ck:
        return None
    enc = EncoderPolicy(**ck["enc_cfg"], value=any(k.startswith("vhead.") for k in ck["enc_state"]))
    enc.load_state_dict(ck["enc_state"])
    return enc.to(device).eval().requires_grad_(False)


@torch.no_grad()
def predict(enc, head, P, S, M, bs=4096):
    """→ 选中动作（策略头首选）[N]、头输出 [N,46,4]、价值头输出 [N,46]（对照用）。"""
    ch, out, val = [], [], []
    for i in range(0, len(P), bs):
        with torch.autocast("cuda", torch.bfloat16):
            tiles, glob = enc.enc(P[i:i + bs], S[i:i + bs])
            lg = enc.head(tiles, glob).float().masked_fill(~M[i:i + bs], -1e4)
            c = lg.argmax(1)
            o = head(tiles.float(), glob.float(), c) if head is not None else None
            v = enc.vhead(tiles, glob).float() if enc.vhead is not None else None
        ch.append(c)
        out.append(o.float() if o is not None else None)
        val.append(v)
    cat = lambda xs: torch.cat(xs) if xs[0] is not None else None  # noqa: E731
    return torch.cat(ch), cat(out), cat(val)


def report(chosen, out, val, q, legal):
    """整句对 = 次点对 且 档位对（与 eval_reason 判"Mortal評価"句的口径一致：句子只由次点和档位决定）。"""
    others, sec, lv = targets(q, legal, chosen)
    r = torch.arange(len(q), device=q.device)
    ok = others.any(1) & ~torch.isnan(q[r, chosen])
    res = {"n": int(ok.sum())}
    if out is not None:
        psec = out[..., 0].masked_fill(~others, -1e4).argmax(1)
        plv = out[..., 1:].argmax(-1)[r, psec]
        res.update({"次点对": (psec == sec)[ok].float().mean().item(),
                    "档位对（次点给定）": (out[..., 1:].argmax(-1)[r, sec] == lv[r, sec])[ok].float().mean().item(),
                    "整句对": ((psec == sec) & (plv == lv[r, sec]))[ok].float().mean().item()})
    if val is not None:
        vs = val.masked_fill(~others, -1e9).argmax(1)
        vl = level(val[r, chosen] - val[r, vs])
        res["价值头整句对（对照）"] = ((vs == sec) & (vl == lv[r, sec]))[ok].float().mean().item()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default=f"{D}/encoder_tf256b6_v_x.pt")
    ap.add_argument("--data", default="fullx", help="full / fullx（与编码器训练数据一致）")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--out", default=f"{D}/gap_head_tf256b6_v_x.pt")
    # 冻结编码器时只比价值头高 0.02 个点（测试 79.98% vs 79.96%）：读出头不是瓶颈，特征才是。
    # --finetune 把编码器主干和策略头一起训，策略头用 KL 贴住原来的分布，尽量不改变打牌决策
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--enc_lr", type=float, default=1e-4)
    ap.add_argument("--kl_w", type=float, default=1.0)
    ap.add_argument("--max_train", type=int, default=0, help=">0 时只取前 n 条训练（冒烟用）")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"

    ck = torch.load(args.enc, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    enc = EncoderPolicy(**ck["enc_cfg"], value=any(k.startswith("vhead.") for k in sd))
    enc.load_state_dict(sd)
    enc.to(dev).eval().requires_grad_(False)
    ch = ck["enc_cfg"]["ch"]
    head = GapHead(ch, args.hidden).to(dev)

    # 数据：与 train_encoder.py 同一套按局划分的验证集（只从原 7000 局里抽 3%）；验证局编码器没见过，用来选轮次
    game = np.load(f"{D}/{args.data}_game.npy")
    rng = np.random.default_rng(0)
    n_games = int(np.load(f"{D}/full_game.npy").max()) + 1
    is_val = np.isin(game, rng.choice(n_games, int(n_games * 0.03), replace=False))
    planes = np.load(f"{D}/{args.data}_planes.npy", mmap_mode="r")
    scalars = np.load(f"{D}/{args.data}_scalars.npy", mmap_mode="r")
    q_all = np.load(f"{D}/{args.data}_q.npy", mmap_mode="r")
    t0 = time.time()

    def to_gpu(idx):
        pn, sn = np.ascontiguousarray(planes[idx]), np.ascontiguousarray(scalars[idx])
        return (torch.from_numpy(pn).to(dev), torch.from_numpy(sn).to(dev), torch.from_numpy(legal_mask(pn, sn)).to(dev),
                torch.from_numpy(np.ascontiguousarray(q_all[idx])).to(dev).float())

    tr_idx = np.flatnonzero((game >= 0) & ~is_val)
    TP, TS, TM, TQ = to_gpu(tr_idx[:args.max_train] if args.max_train else tr_idx)
    VP, VS, VM, VQ = to_gpu(np.flatnonzero((game >= 0) & is_val))
    # 测试集 5000：Mortal Q 值按 md5 对 riichi_test5k_mortal_meta（与 train_encoder.py 相同）
    tp, ts = np.load(f"{D}/test_planes.npy"), np.load(f"{D}/test_scalars.npy")
    tt = [json.loads(l) for l in open(f"{D}/test_text.jsonl", encoding="utf-8")]
    meta = {m["md5"]: m for m in map(json.loads, open(f"{SFT}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    qt = np.full((len(tt), 46), np.nan, np.float32)
    for i, r in enumerate(tt):
        for k, v in meta[r["md5"]]["q"].items():
            qt[i, A[k.replace(" ", "")]] = v
    XP, XS = torch.from_numpy(tp).to(dev), torch.from_numpy(ts).to(dev)
    XM, XQ = torch.from_numpy(legal_mask(tp, ts)).to(dev), torch.from_numpy(qt).to(dev)

    # 训练集的"选中动作"只算一次（编码器冻结）；去掉选中动作没有 Q 值、或没有别的合法动作的
    TC = predict(enc, None, TP, TS, TM)[0]
    r = torch.arange(len(TC), device=dev)
    others, _, _ = targets(TQ, TM, TC)
    good = others.any(1) & ~torch.isnan(TQ[r, TC])
    TP, TS, TM, TQ, TC = TP[good], TS[good], TM[good], TQ[good], TC[good]
    print(f"训练 {len(TQ)} 条 / 验证 {len(VQ)} 条 / 测试 {len(qt)} 条；载入 {time.time() - t0:.0f}s；{vars(args)}", flush=True)
    vc, _, vv = predict(enc, None, VP, VS, VM)
    print("价值头对照：验证", json.dumps(report(vc, None, vv, VQ, VM), ensure_ascii=False), flush=True)
    XC0 = predict(enc, None, XP, XS, XM)[0]           # 原编码器在测试集上的首选（微调后对比打牌是否变了）

    groups = [{"params": head.parameters(), "lr": args.lr}]
    teacher = None
    if args.finetune:
        teacher = copy.deepcopy(enc).eval().requires_grad_(False)
        enc.enc.requires_grad_(True)
        enc.head.requires_grad_(True)
        groups.append({"params": list(enc.enc.parameters()) + list(enc.head.parameters()), "lr": args.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    spe = math.ceil(len(TQ) / args.bs)
    total = args.epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    snap = lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()}  # noqa: E731
    best, best_sd, best_enc = -1.0, None, None
    for ep in range(args.epochs):
        head.train()
        t0 = time.time()
        perm = torch.randperm(len(TQ), device=dev)
        tot_s = tot_l = tot_k = 0.0
        for i in range(0, len(TQ), args.bs):
            idx = perm[i:i + args.bs]
            kl = torch.zeros((), device=dev)
            if teacher is None:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    tiles, glob = enc.enc(TP[idx], TS[idx])
            else:
                with torch.autocast("cuda", torch.bfloat16):
                    tiles, glob = enc.enc(TP[idx], TS[idx])
                    lg = enc.head(tiles, glob).float().masked_fill(~TM[idx], -1e4)
                    with torch.no_grad():
                        t_lg = teacher(TP[idx], TS[idx], TM[idx]).float()
                kl = F.kl_div(lg.log_softmax(-1), t_lg.log_softmax(-1), log_target=True, reduction="batchmean")
            out = head(tiles.float(), glob.float(), TC[idx])
            oth, sec, lv = targets(TQ[idx], TM[idx], TC[idx])
            ls = F.cross_entropy(out[..., 0].masked_fill(~oth, -1e4), sec)
            ll = F.cross_entropy(out[..., 1:][oth], lv[oth])
            loss = ls + ll + args.kl_w * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_s += ls.item() * len(idx)
            tot_l += ll.item() * len(idx)
            tot_k += kl.item() * len(idx)
        head.eval()
        vc, vo, vv = predict(enc, head, VP, VS, VM)
        vr = report(vc, vo, vv, VQ, VM)
        vr["首选不变"] = (vc == predict(teacher, None, VP, VS, VM)[0]).float().mean().item() if teacher else 1.0
        star = vr["整句对"] > best
        if star:
            best, best_sd, best_enc = vr["整句对"], snap(head), (snap(enc) if teacher else None)
        print(f"ep {ep + 1}  次点 loss {tot_s / len(TQ):.4f}  档位 loss {tot_l / len(TQ):.4f}  KL {tot_k / len(TQ):.4f}  | 验证 "
              f"{json.dumps({k: round(v, 4) for k, v in vr.items()}, ensure_ascii=False)}  ({time.time() - t0:.0f}s)"
              + (" *" if star else ""), flush=True)

    head.load_state_dict(best_sd)
    save = {"state_dict": best_sd, "ch": ch, "enc": args.enc, "args": vars(args)}
    if best_enc is not None:                          # 微调过的编码器一起存：e2e_eval --gap 用它排評価順
        enc.load_state_dict(best_enc)
        save.update(enc_state=best_enc, enc_cfg=ck["enc_cfg"])
    torch.save(save, args.out)
    xc, xo, xv = predict(enc, head, XP, XS, XM)
    tr = report(xc, xo, xv, XQ, XM)
    tr["首选不变"] = (xc == XC0).float().mean().item()
    a_star = XQ.nan_to_num(-1e9).masked_fill(~XM, -1e9).argmax(1)
    tr["首选 = Mortal 首选"] = (xc == a_star).float().mean().item()
    print("测试集 5000（验证集最好的一轮）：", json.dumps({k: round(v, 4) for k, v in tr.items()}, ensure_ascii=False))
    print(f"已保存 {args.out}", flush=True)


if __name__ == "__main__":
    main()
