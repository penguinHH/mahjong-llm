#!/usr/bin/env python3
"""Mortal 与凤凰桌玩家的一致率 —— GRPO 第 0 步：这份权重能不能当裁判。

    CUDA_VISIBLE_DEVICES= ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/tools/mortal_agreement.py \
        --logs ~/mahjong-llm/work/test1000/logs --threads 3

决策点和标签用 libriichi 的 GameplayLoader（Mortal 自己训练时取数据的方式），
过滤条件和 SFT 数据一致：只看每局第 1 名（rank 0）的决策、且合法动作 ≥ 2（--all-players 看全部 4 家）。
动作编号：0-36 打牌（34-36 为赤五）/ 37 立直 / 38-40 吃 / 41 碰 / 42 杠 / 43 和 / 44 流局 / 45 pass。
权重用 weights_only=True 加载；不 import Mortal 的 prelude.py（依赖 tensorboard）。
"""
import argparse
import collections
import glob
import os
import sys
import time

import numpy as np
import torch

HOME = os.path.expanduser("~")
TYPES = {37: "reach", 38: "chi_low", 39: "chi_mid", 40: "chi_high", 41: "pon", 42: "kan",
         43: "hora", 44: "ryukyoku", 45: "pass"}


def atype(a):
    return "dahai" if a <= 36 else TYPES[a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=f"{HOME}/mahjong-llm/work/test1000/logs")
    ap.add_argument("--state", default=f"{HOME}/mahjong-llm/work/mortal_weights/voidshine-298k/mortal_298k.pth")
    ap.add_argument("--lib-dir", default=f"{HOME}/mahjong-llm/Mortal/target/release")
    ap.add_argument("--mortal-dir", default=f"{HOME}/mahjong-llm/Mortal/mortal")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8, help="每次载入的局数（每局 4 家的 obs 约 90MB）")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--all-players", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    sys.path.insert(0, args.mortal_dir)
    sys.path.insert(0, args.lib_dir)
    from libriichi.dataset import GameplayLoader
    from model import DQN, Brain

    st = torch.load(args.state, weights_only=True, map_location="cpu")
    cfg = st["config"]
    ver = cfg["control"]["version"]
    dev = torch.device(args.device)
    brain = Brain(version=ver, conv_channels=cfg["resnet"]["conv_channels"],
                  num_blocks=cfg["resnet"]["num_blocks"]).eval().to(dev)
    dqn = DQN(version=ver).eval().to(dev)
    brain.load_state_dict(st["mortal"])
    dqn.load_state_dict(st["current_dqn"])

    files = sorted(glob.glob(os.path.join(args.logs, "*.json.gz")))
    if args.max_games:
        files = files[:args.max_games]
    loader = GameplayLoader(ver, oracle=False, always_include_kan_select=False)

    by = collections.defaultdict(lambda: [0, 0, 0])      # 类型 → [条数, top1 命中, top2 命中]
    gaps = []                                             # Q(Mortal 首选) − Q(玩家动作)
    confusion = collections.Counter()
    t0 = time.time()
    for ci in range(0, len(files), args.chunk):
        for game in loader.load_gz_log_files(files[ci:ci + args.chunk]):
            for g in game:
                pid = g.take_player_id()
                rank = g.take_grp().take_rank_by_player()[pid]
                obs, masks, acts = g.take_obs(), g.take_masks(), g.take_actions()
                if not args.all_players and rank != 0:
                    continue
                keep = [i for i, m in enumerate(masks) if m.sum() >= 2]
                if not keep:
                    continue
                x = torch.as_tensor(np.stack([obs[i] for i in keep])).to(dev)
                m = torch.as_tensor(np.stack([masks[i] for i in keep])).to(dev)
                a = torch.as_tensor([acts[i] for i in keep]).to(dev)
                with torch.inference_mode():
                    q = dqn(brain(x), m).float()
                top2 = q.topk(2, dim=-1).indices
                qa = q.gather(1, a[:, None])[:, 0]
                for i in range(len(keep)):
                    t = atype(int(a[i]))
                    hit1 = int(top2[i, 0]) == int(a[i])
                    by[t][0] += 1
                    by[t][1] += hit1
                    by[t][2] += int(a[i]) in top2[i].tolist()
                    gaps.append(float(q[i].max() - qa[i]))
                    if not hit1:
                        confusion[(t, atype(int(top2[i, 0])))] += 1
        done = min(ci + args.chunk, len(files))
        if done % 100 < args.chunk or done == len(files):
            n = sum(v[0] for v in by.values())
            print(f"  {done}/{len(files)} 局，{n} 个决策，{time.time() - t0:.0f}s", flush=True)

    n = sum(v[0] for v in by.values())
    h1 = sum(v[1] for v in by.values())
    h2 = sum(v[2] for v in by.values())
    who = "全部 4 家" if args.all_players else "每局第 1 名"
    print(f"\n{who}、合法动作 ≥ 2 的决策 {n} 个（{len(files)} 局）")
    print(f"Mortal 首选 = 玩家动作：{h1 / n:.2%}；玩家动作在 Mortal 前 2：{h2 / n:.2%}")
    for t, (c, a1, a2) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"  {t:<9} {c:>7} 条（{c / n:5.1%}）  一致率 {a1 / c:6.2%}  前2 {a2 / c:6.2%}")
    g = np.array(gaps)
    print(f"Q 差（Mortal 首选 − 玩家动作）：不一致时中位数 {np.median(g[g > 0]):.4f}，"
          f"P90 {np.percentile(g[g > 0], 90):.4f}")
    print("最常见的分歧（玩家 → Mortal）：", ", ".join(f"{a}→{b}×{k}" for (a, b), k in confusion.most_common(6)))


if __name__ == "__main__":
    main()
