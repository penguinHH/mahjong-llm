#!/usr/bin/env python3
"""给论文 SFT 样本配上 Mortal 的判断（蒸馏 / GRPO 用）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/tools/build_mortal_labels.py \
        --work ~/mahjong-llm/work/full7000 --out ~/mahjong-llm/work/distill/mortal_full7000.jsonl

逐局处理：
  1. 论文 convert_riichi(<局>-core.json) 生成该局 SFT 样本（与 sft_riichi.jsonl 逐字节相同），
     同时按同样的过滤条件（rank==0、合法动作 ≥ 2）取出对应 core 行的 obs；
  2. libriichi GameplayLoader 只载入这一局第 1 名的视角，得到 Mortal 的 obs / mask / 牌谱动作；
  3. 两边按决策顺序对齐，**牌谱动作和合法动作集合都一致**才算同一个决策点，对不上的丢弃；
  4. Mortal 算出全部合法动作的 Q 值。
输出每行：md5（instruction 的哈希，用来和 riichi_100k 等子集拼接）、human（牌谱动作）、
mortal（Mortal 首选）、q（{动作: Q}，只含合法动作）、game、player。
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HOME = os.path.expanduser("~")
PAPER = "/mnt/d/mahjong-llm/refs/LLM4CardGame"
TILES = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr"]
SPECIAL = {"reach": 37, "chi_low": 38, "chi_mid": 39, "chi_high": 40, "pon": 41, "kan": 42,
           "hora": 43, "ryukyoku": 44, "pass": 45}
ID2ACT = {i: f"dahai:{t}" for i, t in enumerate(TILES)} | {v: k for k, v in SPECIAL.items()}


def act_id(a):
    a = a.replace(" ", "")
    return TILES.index(a[6:]) if a.startswith("dahai:") else SPECIAL[a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="含 logs/ 与 logs_format4/ 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=f"{HOME}/mahjong-llm/work/mortal_weights/voidshine-298k/mortal_298k.pth")
    ap.add_argument("--lib-dir", default=f"{HOME}/mahjong-llm/Mortal/target/release")
    ap.add_argument("--mortal-dir", default=f"{HOME}/mahjong-llm/Mortal/mortal")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.mortal_dir)
    sys.path.insert(0, args.lib_dir)
    sys.path.insert(0, os.path.join(PAPER, "util", "mahjong_util"))
    sys.path.insert(0, PAPER)
    from convert_data import convert_riichi
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

    cores = sorted(glob.glob(os.path.join(args.work, "logs_format4", "*-core.json")))
    if args.max_games:
        cores = cores[:args.max_games]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".part"
    n_sft = n_ok = n_skip = 0
    t0 = time.time()
    with open(tmp, "w", encoding="utf-8") as fo:
        for gi, core in enumerate(cores, 1):
            game = os.path.basename(core)[:-len("-core.json")]          # xxx.json.gz
            items = [json.loads(s) for s in convert_riichi(core)]
            n_sft += len(items)
            if not items:
                continue
            # 论文的 rank 是"这一小局结束时的名次"（gen_action_data.py: player_ranks[at_kyoku + 1]），
            # 不是终局名次 → 一局里可能有多位玩家的样本。
            # 先取每位玩家全部"合法动作 ≥ 2"的决策（不管名次）和 Mortal 逐个对齐，再挑出论文选中的那些。
            per_player, k = {}, 0
            for r in map(json.loads, open(core, encoding="utf-8")):
                if len(r["obs"]["legal_actions"]) < 2:
                    continue
                sel = r["obs"]["rank"] <= 0
                per_player.setdefault(r["obs"]["player_id"], []).append((r, k if sel else None))
                k += sel
            assert k == len(items), (game, k, len(items))
            pids = [p for p, rs in per_player.items() if any(ix is not None for _, ix in rs)]

            raw = gzip.open(os.path.join(args.work, "logs", game), "rt", encoding="utf-8").read()
            names = json.loads(raw.split("\n", 1)[0])["names"]
            loader = GameplayLoader(ver, oracle=False, always_include_kan_select=False,
                                    player_names=[names[p] for p in pids])
            gps = {g.take_player_id(): g for g in loader.load_log(raw)}
            batch = []                                   # (item 下标, obs, mask)
            for pid in pids:
                g = gps[pid]
                rows = per_player[pid]
                mort = [(o, m, int(a)) for o, m, a in zip(g.take_obs(), g.take_masks(), g.take_actions())
                        if m.sum() >= 2]

                def same(i, j):
                    r = rows[i][0]
                    legal = {act_id(a) for a in r["obs"]["legal_actions"]}
                    return act_id(r["action"]) == mort[j][2] and legal == set(np.flatnonzero(mort[j][1]).tolist())

                i = j = 0
                while i < len(rows) and j < len(mort):
                    if same(i, j):
                        if rows[i][1] is not None:
                            batch.append((rows[i][1], pid, mort[j][0], mort[j][1]))
                        i += 1; j += 1
                    elif j + 1 < len(mort) and same(i, j + 1):
                        j += 1
                    elif i + 1 < len(rows) and same(i + 1, j):
                        i += 1
                    else:
                        i += 1; j += 1
            n_skip += len(items) - len(batch)
            if not batch:
                continue
            batch.sort(key=lambda b: b[0])               # 恢复 SFT 原顺序
            x = torch.as_tensor(np.stack([b[2] for b in batch])).to(dev)
            m = torch.as_tensor(np.stack([b[3] for b in batch])).to(dev)
            with torch.inference_mode():
                q = dqn(brain(x), m).float().cpu().numpy()
            for k, (ix, pid, _, mask) in enumerate(batch):
                legal = np.flatnonzero(mask)
                fo.write(json.dumps({
                    "md5": hashlib.md5(items[ix]["instruction"].encode()).hexdigest(),
                    "human": json.loads(items[ix]["output"])["action"],
                    "mortal": ID2ACT[int(legal[q[k][legal].argmax()])],
                    "q": {ID2ACT[int(a)]: round(float(q[k][a]), 4) for a in legal},
                    "game": game, "player": pid,
                }, ensure_ascii=False) + "\n")
            n_ok += len(batch)
            if gi % 250 == 0 or gi == len(cores):
                print(f"  {gi}/{len(cores)} 局  SFT {n_sft}  对齐 {n_ok}（{n_ok / n_sft:.2%}）  "
                      f"丢弃 {n_skip}  {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, args.out)
    print(f"完成：{n_ok}/{n_sft} 条对齐 → {args.out}")


if __name__ == "__main__":
    main()
