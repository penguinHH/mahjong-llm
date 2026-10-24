#!/usr/bin/env python3
"""第 0 步：只训局面编码器 + 策略头（纯 CNN，不用 LLM），模仿 Mortal。

    PY=~/mahjong-llm/venv-train/bin/python
    $PY -I /mnt/d/mahjong-llm/mmj/train_encoder.py --data 100k                       # 与 LLM 同一批 10 万条
    $PY -I /mnt/d/mahjong-llm/mmj/train_encoder.py --data full --epochs 8            # 全量 108 万条
    $PY -I /mnt/d/mahjong-llm/mmj/train_encoder.py --data full --soft_T 1 --alpha 0.5 # 加 Q 值软标签

数据：work/mmj/full_*（build_full.py）。按局随机留 3% 的局做验证集，用验证集选轮次（早停）；
测试集 work/mmj/test_*（另外 1000 局里的同一批 5000 条）只在最后用最好的一轮报一次成绩，可直接和 LLM 比。
损失 = alpha·CE(Mortal 首选, 标签平滑) + (1-alpha)·CE(softmax(Q/T) 软标签)；soft_T=0 时只用硬标签。
输出：work/mmj/encoder_<tag>.pt（验证集最好的一轮）+ encoder_<tag>.log.jsonl
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actions import ACTIONS, kind, legal_mask, norm  # noqa: E402
from encoder import EncoderPolicy  # noqa: E402
from state_features import N_PLANES, N_SCALARS  # noqa: E402

D = os.path.expanduser("~/mahjong-llm/work/mmj")
SFT = os.path.expanduser("~/mahjong-llm/work/sft")


def evaluate(model, P, S, M, bs=2048):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i in range(0, len(P), bs):
            out.append(model(P[i:i + bs], S[i:i + bs], M[i:i + bs]).argmax(-1))
    model.train()
    return torch.cat(out)


def value_report(model, P, S, qt, bs=2048):
    from train_value import report        # 与 train_value.py 同一套"次点 / 档位"口径
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i in range(0, len(P), bs):
            out.append(model(P[i:i + bs], S[i:i + bs], with_value=True)[1].float())
    model.train()
    return report(torch.cat(out).cpu().numpy(), qt)


def test_report(acts, v_human, v_meta):
    h = float(np.mean([a == x for a, x in zip(acts, v_human)]))
    m = float(np.mean([a == x["mortal"] for a, x in zip(acts, v_meta)]))
    gaps = [max(x["q"].values()) - x["q"][a] for a, x in zip(acts, v_meta) if a in x["q"]]
    by = collections.defaultdict(lambda: [0, 0, 0])
    for a, hx, x in zip(acts, v_human, v_meta):
        k = kind(hx)
        by[k][0] += 1
        by[k][1] += a == hx
        by[k][2] += a == x["mortal"]
    return {"human": h, "mortal": m, "qgap": float(np.mean(gaps)), "illegal": len(acts) - len(gaps),
            "by": {k: [c, hh / c, mm / c] for k, (c, hh, mm) in by.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["100k", "full"], default="100k")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ch", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ls", type=float, default=0.05, help="硬标签的标签平滑")
    ap.add_argument("--soft_T", type=float, default=0.0, help=">0 时加 softmax(Q/T) 软标签")
    ap.add_argument("--alpha", type=float, default=1.0, help="硬标签权重（软标签权重 = 1 - alpha）")
    ap.add_argument("--val_frac", type=float, default=0.03)
    ap.add_argument("--label", choices=["mortal", "human"], default="mortal", help="human = 模仿牌谱里的人（\"像人\"对照）")
    ap.add_argument("--arch", choices=["cnn", "tf"], default="cnn", help="tf = 34 张牌 + 场况各一个 token 的 Transformer")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--value_w", type=float, default=0.0, help=">0 时加价值头，回归 Mortal 各动作的优势（Q - 最优 Q）")
    ap.add_argument("--extra", action="store_true", help="训练数据再加 extra8000（第 8001 局以后，build_full.py --extra）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.soft_T > 0 and args.alpha == 1.0:
        args.alpha = 0.5
    assert not (args.label == "human" and args.soft_T > 0), "软标签只有 Mortal 的 Q 值"
    tag = args.tag or f"{args.data}{'+extra' if args.extra else ''}_{args.arch}{args.ch}b{args.blocks}" \
        + (f"_softT{args.soft_T:g}a{args.alpha:g}" if args.soft_T else "") + (f"_v{args.value_w:g}" if args.value_w else "") \
        + ("_human" if args.label == "human" else "")
    torch.manual_seed(0)
    dev = "cuda"

    # ---------------- 数据（全量数组里按需取 10 万子集；按局划验证集）
    pre = "fullx" if args.extra else "full"          # fullx_* = full_* 后面接上 extra8000（局编号顺延）
    y = np.load(f"{D}/{pre}_y_{args.label}.npy")
    game = np.load(f"{D}/{pre}_game.npy")
    keep = (y >= 0) & (game >= 0)
    if args.data == "100k":
        keep &= np.load(f"{D}/{pre}_in100k.npy")
    rng = np.random.default_rng(0)
    n_games = int(np.load(f"{D}/full_game.npy").max()) + 1          # 验证局只从原 7000 局里抽：加不加 extra 验证集都相同
    val_games = rng.choice(n_games, int(n_games * args.val_frac), replace=False)
    is_val = np.isin(game, val_games)
    tr_idx = np.flatnonzero(keep & ~is_val)
    va_idx = np.flatnonzero(keep & is_val)
    planes = np.load(f"{D}/{pre}_planes.npy", mmap_mode="r")
    scalars = np.load(f"{D}/{pre}_scalars.npy", mmap_mode="r")
    q = np.load(f"{D}/{pre}_q.npy", mmap_mode="r")
    t0 = time.time()

    def to_gpu(idx):
        pn, sn = np.ascontiguousarray(planes[idx]), np.ascontiguousarray(scalars[idx])   # 只从磁盘读一次
        return torch.from_numpy(pn).to(dev), torch.from_numpy(sn).to(dev), torch.from_numpy(legal_mask(pn, sn)).to(dev)

    TP, TS, TM = to_gpu(tr_idx)
    TY = torch.from_numpy(y[tr_idx].astype(np.int64)).to(dev)
    TQ = torch.from_numpy(np.ascontiguousarray(q[tr_idx])).to(dev) if args.soft_T > 0 or args.value_w > 0 else None
    VP, VS, VM = to_gpu(va_idx)
    VY = torch.from_numpy(y[va_idx].astype(np.int64)).to(dev)
    bad = (~TM.gather(1, TY[:, None])).sum().item()
    assert bad == 0, f"有 {bad} 条标签不在合法动作里"
    # 测试集：牌谱动作 + Mortal 动作 / Q 值（按 md5 对 riichi_test5k_mortal_meta）
    tp, ts = np.load(f"{D}/test_planes.npy"), np.load(f"{D}/test_scalars.npy")
    tt = [json.loads(l) for l in open(f"{D}/test_text.jsonl", encoding="utf-8")]
    meta = {m["md5"]: m for m in map(json.loads, open(f"{SFT}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    v_human = [norm(json.loads(r["output"])["action"]) for r in tt]
    v_meta = [meta[r["md5"]] for r in tt]
    XP, XS = torch.from_numpy(tp).to(dev), torch.from_numpy(ts).to(dev)
    XM = torch.from_numpy(legal_mask(tp, ts)).to(dev)
    print(f"[{tag}] 训练 {len(tr_idx)} 条 / 验证 {len(va_idx)} 条（{len(val_games)} 局）/ 测试 {len(tt)} 条；"
          f"载入 {time.time() - t0:.0f}s；{vars(args)}", flush=True)

    qt = np.full((len(tt), 46), np.nan, np.float32)            # 测试集 Mortal Q 值（价值头评测用）
    for i, x in enumerate(v_meta):
        for k, v in x["q"].items():
            qt[i, ACTIONS.index(norm(k))] = v
    model = EncoderPolicy(N_PLANES, N_SCALARS, args.ch, args.blocks, args.dropout, args.arch, args.heads,
                          value=args.value_w > 0).to(dev)
    print(f"参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    spe = math.ceil(len(TY) / args.bs)
    total, warm = args.epochs * spe, min(spe, 2000)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

    log = open(f"{D}/encoder_{tag}.log.jsonl", "w")
    best, best_ep = -1.0, 0
    for ep in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(TY), device=dev)
        tot = n = 0
        for i in range(0, len(TY), args.bs):
            idx = perm[i:i + args.bs]
            with torch.autocast("cuda", torch.bfloat16):
                out = model(TP[idx], TS[idx], TM[idx], with_value=args.value_w > 0)
            logits, vpred = (out[0].float(), out[1].float()) if args.value_w > 0 else (out.float(), None)
            logp = torch.log_softmax(logits, -1)
            m = TM[idx].float()
            # 标签平滑只摊到合法动作上（不能用 F.cross_entropy 的 label_smoothing：非法动作被压到 -1e4，loss 会爆）
            hard = F.one_hot(TY[idx], logits.shape[-1]).float() * (1 - args.ls) + args.ls * m / m.sum(-1, keepdim=True)
            loss = args.alpha * -(hard * logp.masked_fill(~TM[idx], 0)).sum(-1).mean()
            if args.soft_T > 0:
                qq = TQ[idx].float() / args.soft_T
                target = torch.softmax(qq.nan_to_num(nan=-1e4).masked_fill(~TM[idx], -1e4), -1)
                loss = loss + (1 - args.alpha) * -(target * logp.masked_fill(~TM[idx], 0)).sum(-1).mean()
            if args.value_w > 0:
                qv = TQ[idx].float()
                has = ~torch.isnan(qv)
                adv = qv - qv.nan_to_num(-1e9).amax(1, keepdim=True)
                if has.any():
                    loss = loss + args.value_w * F.smooth_l1_loss(vpred[has], adv[has])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        val = (evaluate(model, VP, VS, VM) == VY).float().mean().item()
        acts = [ACTIONS[k] for k in evaluate(model, XP, XS, XM).tolist()]
        rep = test_report(acts, v_human, v_meta)
        if args.value_w > 0:
            rep["value"] = value_report(model, XP, XS, qt)
        rec = {"epoch": ep + 1, "train_loss": tot / n, "val_mortal": val, **rep, "sec": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        flag = ""
        if val > best:
            best, best_ep, flag = val, ep + 1, " *"
            torch.save({"state_dict": model.state_dict(), "enc_cfg": model.enc.cfg, "args": vars(args),
                        "epoch": ep + 1, "metrics": rec}, f"{D}/encoder_{tag}.pt")
        vr = rep.get("value")
        print(f"ep {ep + 1:2d}  loss {rec['train_loss']:.4f}  验证({args.label}) {val:.2%}  | 测试 与牌谱 {rep['human']:.2%}"
              f"  与 Mortal {rep['mortal']:.2%}  Q差 {rep['qgap']:.3f}"
              + (f"  | 价值头 次点 {vr['次点对']:.1%} 整句 {vr['整句对']:.1%} MAE {vr['优势 MAE']:.2f}" if vr else "")
              + f"  ({rec['sec']}s){flag}", flush=True)
    r = torch.load(f"{D}/encoder_{tag}.pt", weights_only=False)["metrics"]
    print(f"\n验证集最好：第 {best_ep} 轮（验证 {best:.2%}）→ 测试 与牌谱 {r['human']:.2%}  与 Mortal {r['mortal']:.2%}"
          f"  Q差 {r['qgap']:.3f}  非法 {r['illegal']}")
    print("按牌谱动作（牌谱 / Mortal）：" + "，".join(
        f"{k} {hh:.1%}/{mm:.1%}（{c}）" for k, (c, hh, mm) in sorted(r["by"].items(), key=lambda kv: -kv[1][0])))


if __name__ == "__main__":
    main()
